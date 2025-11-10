import mindspore
from mindspore import mint

from comfy.text_encoders.bert import BertAttention
import comfy.model_management
from comfy.ldm.modules.attention import optimized_attention_for_device


class Dino2AttentionOutput(mindspore.nn.Cell):
    def __init__(self, input_dim, output_dim, layer_norm_eps, dtype, device, operations):
        super().__init__()
        self.dense = operations.Linear(input_dim, output_dim, dtype=dtype, device=None)

    def construct(self, x):
        return self.dense(x)


class Dino2AttentionBlock(mindspore.nn.Cell):
    def __init__(self, embed_dim, heads, layer_norm_eps, dtype, device, operations):
        super().__init__()
        self.attention = BertAttention(embed_dim, heads, dtype, None, operations)
        self.output = Dino2AttentionOutput(embed_dim, embed_dim, layer_norm_eps, dtype, None, operations)

    def construct(self, x, mask, optimized_attention):
        return self.output(self.attention(x, mask, optimized_attention))


class LayerScale(mindspore.nn.Cell):
    def __init__(self, dim, dtype, device, operations):
        super().__init__()
        self.lambda1 = mindspore.Parameter(mint.empty(dim, dtype=dtype))

    def construct(self, x):
        return x * comfy.model_management.cast_to_device(self.lambda1, None, x.dtype)

class Dinov2MLP(mindspore.nn.Cell):
    def __init__(self, hidden_size: int, dtype, device, operations):
        super().__init__()

        mlp_ratio = 4
        hidden_features = int(hidden_size * mlp_ratio)
        self.fc1 = operations.Linear(hidden_size, hidden_features, bias = True, device=None, dtype=dtype)
        self.fc2 = operations.Linear(hidden_features, hidden_size, bias = True, device=None, dtype=dtype)

    def construct(self, hidden_state: mindspore.Tensor) -> mindspore.Tensor:
        hidden_state = self.fc1(hidden_state)
        hidden_state = mint.functional.gelu(hidden_state)
        hidden_state = self.fc2(hidden_state)
        return hidden_state

class SwiGLUFFN(mindspore.nn.Cell):
    def __init__(self, dim, dtype, device, operations):
        super().__init__()
        in_features = out_features = dim
        hidden_features = int(dim * 4)
        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8

        self.weights_in = operations.Linear(in_features, 2 * hidden_features, bias=True, device=None, dtype=dtype)
        self.weights_out = operations.Linear(hidden_features, out_features, bias=True, device=None, dtype=dtype)

    def construct(self, x):
        x = self.weights_in(x)
        x1, x2 = x.chunk(2, dim=-1)
        x = mint.functional.silu(x1) * x2
        return self.weights_out(x)


class Dino2Block(mindspore.nn.Cell):
    def __init__(self, dim, num_heads, layer_norm_eps, dtype, device, operations, use_swiglu_ffn):
        super().__init__()
        self.attention = Dino2AttentionBlock(dim, num_heads, layer_norm_eps, dtype, None, operations)
        self.layer_scale1 = LayerScale(dim, dtype, None, operations)
        self.layer_scale2 = LayerScale(dim, dtype, None, operations)
        if use_swiglu_ffn:
            self.mlp = SwiGLUFFN(dim, dtype, None, operations)
        else:
            self.mlp = Dinov2MLP(dim, dtype, None, operations)
        self.norm1 = operations.LayerNorm(dim, eps=layer_norm_eps, dtype=dtype, device=None)
        self.norm2 = operations.LayerNorm(dim, eps=layer_norm_eps, dtype=dtype, device=None)

    def construct(self, x, optimized_attention):
        x = x + self.layer_scale1(self.attention(self.norm1(x), None, optimized_attention))
        x = x + self.layer_scale2(self.mlp(self.norm2(x)))
        return x


class Dino2Encoder(mindspore.nn.Cell):
    def __init__(self, dim, num_heads, layer_norm_eps, num_layers, dtype, device, operations, use_swiglu_ffn):
        super().__init__()
        self.layer = mindspore.nn.CellList([Dino2Block(dim, num_heads, layer_norm_eps, dtype, None, operations, use_swiglu_ffn = use_swiglu_ffn)
                                          for _ in range(num_layers)])

    def construct(self, x, intermediate_output=None):
        optimized_attention = optimized_attention_for_device(None, False, small_input=True)

        if intermediate_output is not None:
            if intermediate_output < 0:
                intermediate_output = len(self.layer) + intermediate_output

        intermediate = None
        for i, layer in enumerate(self.layer):
            x = layer(x, optimized_attention)
            if i == intermediate_output:
                intermediate = x.clone()
        return x, intermediate


class Dino2PatchEmbeddings(mindspore.nn.Cell):
    def __init__(self, dim, num_channels=3, patch_size=14, image_size=518, dtype=None, device=None, operations=None):
        super().__init__()
        self.projection = operations.Conv2d(
            in_channels=num_channels,
            out_channels=dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
            dtype=dtype,
            device=None
        )

    def construct(self, pixel_values):
        return self.projection(pixel_values).flatten(2).transpose(1, 2)


class Dino2Embeddings(mindspore.nn.Cell):
    def __init__(self, dim, dtype, device, operations):
        super().__init__()
        patch_size = 14
        image_size = 518

        self.patch_embeddings = Dino2PatchEmbeddings(dim, patch_size=patch_size, image_size=image_size, dtype=dtype, device=None, operations=operations)
        self.position_embeddings = mindspore.Parameter(mint.empty(1, (image_size // patch_size) ** 2 + 1, dim, dtype=dtype, device=None))
        self.cls_token = mindspore.Parameter(mint.empty(1, 1, dim, dtype=dtype, device=None))
        self.mask_token = mindspore.Parameter(mint.empty(1, dim, dtype=dtype, device=None))

    def construct(self, pixel_values):
        x = self.patch_embeddings(pixel_values)
        # TODO: mask_token?
        x = mint.cat((self.cls_token.to(dtype=x.dtype).expand((x.shape[0], -1, -1)), x), dim=1)
        x = x + comfy.model_management.cast_to_device(self.position_embeddings, None, x.dtype)
        return x


class Dinov2Model(mindspore.nn.Cell):
    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        num_layers = config_dict["num_hidden_layers"]
        dim = config_dict["hidden_size"]
        heads = config_dict["num_attention_heads"]
        layer_norm_eps = config_dict["layer_norm_eps"]
        use_swiglu_ffn = config_dict["use_swiglu_ffn"]

        self.embeddings = Dino2Embeddings(dim, dtype, None, operations)
        self.encoder = Dino2Encoder(dim, heads, layer_norm_eps, num_layers, dtype, None, operations, use_swiglu_ffn = use_swiglu_ffn)
        self.layernorm = operations.LayerNorm(dim, eps=layer_norm_eps, dtype=dtype, device=None)

    def construct(self, pixel_values, attention_mask=None, intermediate_output=None):
        x = self.embeddings(pixel_values)
        x, i = self.encoder(x, intermediate_output=intermediate_output)
        x = self.layernorm(x)
        pooled_output = x[:, 0, :]
        return x, i, pooled_output, None
