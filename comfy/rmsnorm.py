import mindspore
from mindspore import mint

import comfy.model_management
import numbers
import logging

RMSNorm = None

try:
    rms_norm_mindspore = mint.functional.rms_norm
    RMSNorm = mint.nn.RMSNorm
except:
    rms_norm_mindspore = None
    logging.warning("Please update mindspore to use native RMSNorm")


def rms_norm(x, weight=None, eps=1e-6):
    if rms_norm_mindspore is not None:   # and not (torch.jit.is_tracing() or torch.jit.is_scripting()):
        if weight is None:
            return rms_norm_mindspore(x, (x.shape[-1],), eps=eps)
        else:
            return rms_norm_mindspore(x, weight.shape, weight=comfy.model_management.cast_to(weight, dtype=x.dtype), eps=eps)
    else:
        r = x * mint.rsqrt(mint.mean(x**2, dim=-1, keepdim=True) + eps)
        if weight is None:
            return r
        else:
            return r * comfy.model_management.cast_to(weight, dtype=x.dtype, device=None)


if RMSNorm is None:
    class RMSNorm(mint.nn.Cell):
        def __init__(
            self,
            normalized_shape,
            eps=1e-6,
            elementwise_affine=True,
            device=None,
            dtype=None,
        ):
            factory_kwargs = {"dtype": dtype}
            super().__init__()
            if isinstance(normalized_shape, numbers.Integral):
                # mypy error: incompatible types in assignment
                normalized_shape = (normalized_shape,)  # type: ignore[assignment]
            self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
            self.eps = eps
            self.elementwise_affine = elementwise_affine
            if self.elementwise_affine:
                self.weight = mindspore.Parameter(
                    mint.empty(self.normalized_shape, **factory_kwargs)
                )
            else:
                self.register_parameter("weight", None)
            self.bias = None

        def construct(self, x):
            return rms_norm(x, self.weight, self.eps)
