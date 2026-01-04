# Adapted from mindone.diffusers.models.attention_processors.Attention
from typing import Optional
import numpy as np

import mindspore as ms
from mindspore import ops, mint

from mindspore_patch.utils import is_ascend_310p


DTYPE_FP16_MIN = float(np.finfo(np.float16).min)


def _scaled_dot_product_attention_fa(
    query: ms.Tensor,
    key: ms.Tensor,
    value: ms.Tensor,
    attn_mask: Optional[ms.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
):
    r"""
    Perform scaled dot-product attention using the flash attention operator.

    Parameters:
        query (ms.Tensor): The query tensor.
        key (ms.Tensor): The key tensor.
        value (ms.Tensor): The value tensor.
        attn_mask (Optional[ms.Tensor], optional): The attention mask tensor. Defaults to None.
        dropout_p (float, optional): The dropout probability. Defaults to 0.0.
        is_causal (bool): Un-used. Aligned with Torch
        scale (float, optional): scaled value

    Returns:
        ms.Tensor: The result of the scaled dot-product attention.
    """
    # For most scenarios, qkv has been processed into a BNSD layout before sdp
    input_layout = "BNSD"
    head_num = query.shape[1]
    default_scale = query.shape[-1] ** -0.5

    # In case qkv is 3-dim after `head_to_batch_dim`
    if query.ndim == 3:
        input_layout = "BSH"
        head_num = 1

    # Convert to fp16 or bf16 for flash attention if needed
    original_dtype = query.dtype
    need_convert_dtype = False

    if query.dtype not in (ms.float16, ms.bfloat16):
        need_convert_dtype = True

        if is_ascend_310p():
            _dtype = ms.float16
        else:
            _dtype = ms.bfloat16

        query = query.to(_dtype)
        key = key.to(_dtype)
        value = value.to(_dtype)

    # process `attn_mask` as logic is different between PyTorch and Mindspore
    # In MindSpore, False indicates retention and True indicates discard, in PyTorch it is the opposite
    if attn_mask is not None:
        attn_mask = mint.logical_not(attn_mask) if attn_mask.dtype == ms.bool_ else attn_mask.bool()
        attn_mask = mint.broadcast_to(
            attn_mask, (attn_mask.shape[0], attn_mask.shape[1], query.shape[-2], key.shape[-2])
        )[:, :1, :, :]

    out = ops.operations.nn_ops.FlashAttentionScore(
        head_num=head_num, keep_prob=1-dropout_p, scale_value=scale or default_scale, input_layout=input_layout
    )(query, key, value, None, None, None, attn_mask)[3]

    if need_convert_dtype:
        out = out.astype(original_dtype)

    return out


def _scaled_dot_product_attention_native(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, dtype=None):
    # force dtype(fp16 or bf16) precision calculation
    ori_dtype = query.dtype
    if dtype is not None:
        query, key, value = query.astype(dtype), key.astype(dtype), value.astype(dtype)

    if attn_mask is not None:
        if attn_mask.dtype == ms.bool_:
            attn_mask = attn_mask.to(ms.float32)
            attn_mask = attn_mask.masked_fill((1 - attn_mask).to(ms.bool_), DTYPE_FP16_MIN)
        attn_mask = attn_mask.to(query.dtype)

        attn_weight = mint.nn.functional.softmax(
            mint.matmul(query, mint.swapaxes(key, -2, -1)) / (query.shape[-1] ** 0.5) + attn_mask,
            dim=-1,
            dtype=ms.float32,
        ).astype(query.dtype)
    else:
        L, S = query.shape[-2], key.shape[-2]
        attn_bias = mint.zeros((L, S), dtype=query.dtype)
        if is_causal:
            # assert attn_mask is None
            temp_mask = mint.ones((L, S), dtype=ms.bool_).tril(diagonal=0)
            attn_bias = ops.masked_fill(attn_bias, mint.logical_not(temp_mask), DTYPE_FP16_MIN)
            attn_bias = attn_bias.to(query.dtype)

        attn_weight = mint.nn.functional.softmax(
            mint.matmul(query, mint.swapaxes(key, -2, -1)) / (query.shape[-1] ** 0.5) + attn_bias,
            dim=-1,
            dtype=ms.float32,
        ).astype(query.dtype)

    attn_weight = mint.nn.Dropout(p=dropout_p)(attn_weight)

    out = mint.matmul(attn_weight, value)
    out = out.astype(ori_dtype)

    return out


if is_ascend_310p():
    scaled_dot_product_attention = _scaled_dot_product_attention_native
else:
    scaled_dot_product_attention = _scaled_dot_product_attention_fa
