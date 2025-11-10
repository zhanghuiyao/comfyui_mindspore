import mindspore
from mindspore import Tensor, mint

# from einops import rearrange

from comfy.ldm.modules.attention import optimized_attention
import comfy.model_management


def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor, mask=None, transformer_options={}) -> Tensor:
    q, k = apply_rope(q, k, pe)
    heads = q.shape[1]
    x = optimized_attention(q, k, v, heads, skip_reshape=True, mask=mask, transformer_options=transformer_options)
    return x


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0

    scale = mint.linspace(0, (dim - 2) / dim, steps=dim//2, dtype=mindspore.float64, device=None)
    omega = 1.0 / (theta**scale)
    out = mint.einsum("...n,d->...nd", pos.to(dtype=mindspore.float32, device=None), omega)
    out = mint.stack([mint.cos(out), -mint.sin(out), mint.sin(out), mint.cos(out)], dim=-1)
    
    # out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    _b, _n, _d, _ = out.shape
    _i, _j = 2, 2
    out =  out.view(_b, _n, _d, _i, _j)
    
    return out.to(dtype=mindspore.float32)

def apply_rope1(x: Tensor, freqs_cis: Tensor):
    x_ = x.to(dtype=freqs_cis.dtype).reshape(*x.shape[:-1], -1, 1, 2)

    x_out = freqs_cis[..., 0] * x_[..., 0]
    x_out = x_out.addcmul(freqs_cis[..., 1], x_[..., 1])

    return x_out.reshape(*x.shape).type_as(x)

def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor):
    return apply_rope1(xq, freqs_cis), apply_rope1(xk, freqs_cis)
