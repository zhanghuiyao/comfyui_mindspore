# text_encoder.py
"""
Patch Qwen2.5-VL's attention to fix fp16 precision issues.
"""
from typing import Optional

import mindspore as ms
from mindspore import mint, nn

import mindone.transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as text_encoder_module


def eager_attention_forward(
    module: nn.Cell,
    query: ms.Tensor,
    key: ms.Tensor,
    value: ms.Tensor,
    attention_mask: Optional[ms.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = text_encoder_module.repeat_kv(key.to(ms.float32), module.num_key_value_groups)
    value_states = text_encoder_module.repeat_kv(value, module.num_key_value_groups)

    attn_weights = mint.matmul(query.to(ms.float32), key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = mint.nn.functional.softmax(attn_weights, dim=-1, dtype=ms.float32).to(query.dtype)
    attn_weights = mint.nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = mint.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


# Apply the patch to fix fp16 precision issues.
text_encoder_module.eager_attention_forward = eager_attention_forward
Qwen2_5_VLForConditionalGeneration = text_encoder_module.Qwen2_5_VLForConditionalGeneration
