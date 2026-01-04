import mindspore as ms
from mindspore import nn, mint
import mindspore.mint.nn.functional as F
from typing import Tuple, Optional
import math
from .model import WanModel, sinusoidal_embedding_1d
from comfy.ldm.modules.attention import optimized_attention

class CausalConv1d(nn.Cell):

    def __init__(self, chan_in, chan_out, kernel_size=3, stride=1, dilation=1, pad_mode="replicate", operations=None, **kwargs):
        super().__init__()

        self.pad_mode = pad_mode
        padding = (kernel_size - 1, 0)  # T
        self.time_causal_padding = padding

        self.conv = operations.Conv1d(chan_in, chan_out, kernel_size, stride=stride, dilation=dilation, **kwargs)

    def construct(self, x):
        x = F.pad(x, self.time_causal_padding, mode=self.pad_mode)
        return self.conv(x)


class FaceEncoder(nn.Cell):
    def __init__(self, in_dim: int, hidden_dim: int, num_heads=int, dtype=None, device=None, operations=None):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()

        self.num_heads = num_heads
        self.conv1_local = CausalConv1d(in_dim, 1024 * num_heads, 3, stride=1, operations=operations, **factory_kwargs)
        self.norm1 = operations.LayerNorm(hidden_dim // 8, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.act = nn.SiLU()
        self.conv2 = CausalConv1d(1024, 1024, 3, stride=2, operations=operations, **factory_kwargs)
        self.conv3 = CausalConv1d(1024, 1024, 3, stride=2, operations=operations, **factory_kwargs)

        self.out_proj = operations.Linear(1024, hidden_dim, **factory_kwargs)
        self.norm1 = operations.LayerNorm(1024, elementwise_affine=False, eps=1e-6, **factory_kwargs)

        self.norm2 = operations.LayerNorm(1024, elementwise_affine=False, eps=1e-6, **factory_kwargs)

        self.norm3 = operations.LayerNorm(1024, elementwise_affine=False, eps=1e-6, **factory_kwargs)

        self.padding_tokens = ms.Parameter(mint.empty(1, 1, 1, hidden_dim, **factory_kwargs))

    def construct(self, x):

        x = x.transpose(1, 2) # rearrange(x, "b t c -> b c t")
        b, c, t = x.shape

        x = self.conv1_local(x)
        # rearrange(x, "b (n c) t -> (b n) t c", n=self.num_heads)
        x = x.reshape(-1, self.num_heads, x.shape[1] // self.num_heads, x.shape[2]).reshape(-1, x.shape[-2], x.shape[-1])
        
        x = self.norm1(x)
        x = self.act(x)
        x = x.transpose(1, 2) # rearrange(x, "b t c -> b c t")
        x = self.conv2(x)
        x = x.transpose(1, 2) # rearrange(x, "b c t -> b t c")
        x = self.norm2(x)
        x = self.act(x)
        x = x.transpose(1, 2) # rearrange(x, "b t c -> b c t")
        x = self.conv3(x)
        x = x.transpose(1, 2) # rearrange(x, "b c t -> b t c")
        x = self.norm3(x)
        x = self.act(x)
        x = self.out_proj(x)
        x = x.reshape(b, -1, x.shape[-2], x.shape[-1]).transpose(1, 2) # rearrange(x, "(b n) t c -> b t n c", b=b)
        
        padding = self.padding_tokens.to(dtype=x.dtype).repeat(b, x.shape[1], 1, 1)
        x = mint.cat([x, padding], dim=-2)
        x_local = x.copy()

        return x_local


def get_norm_layer(norm_layer, operations=None):
    """
    Get the normalization layer.

    Args:
        norm_layer (str): The type of normalization layer.

    Returns:
        norm_layer (nn.Cell): The normalization layer.
    """
    if norm_layer == "layer":
        return operations.LayerNorm
    elif norm_layer == "rms":
        return operations.RMSNorm
    else:
        raise NotImplementedError(f"Norm layer {norm_layer} is not implemented")


class FaceAdapter(nn.Cell):
    def __init__(
        self,
        hidden_dim: int,
        heads_num: int,
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        num_adapter_layers: int = 1,
        dtype=None, device=None, operations=None
    ):

        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()
        self.hidden_size = hidden_dim
        self.heads_num = heads_num
        self.fuser_blocks = nn.CellList(
            [
                FaceBlock(
                    self.hidden_size,
                    self.heads_num,
                    qk_norm=qk_norm,
                    qk_norm_type=qk_norm_type,
                    operations=operations,
                    **factory_kwargs,
                )
                for _ in range(num_adapter_layers)
            ]
        )

    def construct(
        self,
        x: ms.Tensor,
        motion_embed: ms.Tensor,
        idx: int,
        freqs_cis_q: Tuple[ms.Tensor, ms.Tensor] = None,
        freqs_cis_k: Tuple[ms.Tensor, ms.Tensor] = None,
    ) -> ms.Tensor:

        return self.fuser_blocks[idx](x, motion_embed, freqs_cis_q, freqs_cis_k)



class FaceBlock(nn.Cell):
    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        qk_scale: float = None,
        dtype: Optional[ms.Type] = None,
        device = None,
        operations=None
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.deterministic = False
        self.hidden_size = hidden_size
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        self.scale = qk_scale or head_dim**-0.5

        self.linear1_kv = operations.Linear(hidden_size, hidden_size * 2, **factory_kwargs)
        self.linear1_q = operations.Linear(hidden_size, hidden_size, **factory_kwargs)

        self.linear2 = operations.Linear(hidden_size, hidden_size, **factory_kwargs)

        qk_norm_layer = get_norm_layer(qk_norm_type, operations=operations)
        self.q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )
        self.k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )

        self.pre_norm_feat = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)

        self.pre_norm_motion = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)

    def construct(
        self,
        x: ms.Tensor,
        motion_vec: ms.Tensor,
        motion_mask: Optional[ms.Tensor] = None,
        # use_context_parallel=False,
    ) -> ms.Tensor:

        B, T, N, C = motion_vec.shape
        T_comp = T

        x_motion = self.pre_norm_motion(motion_vec)
        x_feat = self.pre_norm_feat(x)

        kv = self.linear1_kv(x_motion)
        q = self.linear1_q(x_feat)

        # k, v = rearrange(kv, "B L N (K H D) -> K B L N H D", K=2, H=self.heads_num)
        B, L, N, _ = kv.shape
        D = kv.shape[3] // (2 * self.heads_num)
        reshaped = kv.reshape(B, L, N, 2, self.heads_num, D)
        k, v = reshaped[:, :, :, 0], reshaped[:, :, :, 1]

        # q = rearrange(q, "B S (H D) -> B S H D", H=self.heads_num)
        q = q.reshape(q.shape[0], q.shape[1], self.heads_num, -1)

        # Apply QK-Norm if needed.
        q = self.q_norm(q).to(v)
        k = self.k_norm(k).to(v)

        # k = rearrange(k, "B L N H D -> (B L) N H D")
        # v = rearrange(v, "B L N H D -> (B L) N H D")
        _, _, N, H, D = k.shape
        k = k.reshape(-1, N, H, D)
        _, _, N, H, D = v.shape
        v = v.reshape(-1, N, H, D)

        # q = rearrange(q, "B (L S) H D -> (B L) S (H D)", L=T_comp)
        B, LS, H, D = q.shape
        L, S = T_comp, LS // T_comp
        q = q.reshape(B, L, -1, H, D).reshape(-1, S, H, D).reshape(B*L, S, -1)

        attn = optimized_attention(q, k, v, heads=self.heads_num)

        # attn = rearrange(attn, "(B L) S C -> B (L S) C", L=T_comp)
        BL, S, C = attn.shape
        B, L = BL // T_comp, T_comp
        attn = attn.reshape(B, L, S, C).reshape(B, -1, C)

        output = self.linear2(attn)

        if motion_mask is not None:
            # output = output * rearrange(motion_mask, "B T H W -> B (T H W)").unsqueeze(-1)
            output = output * motion_mask.reshape(motion_mask.shape[0], -1).unsqueeze(-1)

        return output

# https://github.com/XPixelGroup/BasicSR/blob/8d56e3a045f9fb3e1d8872f92ee4a4f07f886b0a/basicsr/ops/upfirdn2d/upfirdn2d.py#L162
def upfirdn2d_native(input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1):
    _, minor, in_h, in_w = input.shape
    kernel_h, kernel_w = kernel.shape

    out = input.view(-1, minor, in_h, 1, in_w, 1)
    out = F.pad(out, [0, up_x - 1, 0, 0, 0, up_y - 1, 0, 0])
    out = out.view(-1, minor, in_h * up_y, in_w * up_x)

    out = F.pad(out, [max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)])
    out = out[:, :, max(-pad_y0, 0): out.shape[2] - max(-pad_y1, 0), max(-pad_x0, 0): out.shape[3] - max(-pad_x1, 0)]

    out = out.reshape([-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1])
    w = mint.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    out = F.conv2d(out, w)
    out = out.reshape(-1, minor, in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1, in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1)
    return out[:, :, ::down_y, ::down_x]

def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    return upfirdn2d_native(input, kernel, up, up, down, down, pad[0], pad[1], pad[0], pad[1])

# https://github.com/XPixelGroup/BasicSR/blob/8d56e3a045f9fb3e1d8872f92ee4a4f07f886b0a/basicsr/ops/fused_act/fused_act.py#L81
class FusedLeakyReLU(nn.Cell):
    def __init__(self, channel, negative_slope=0.2, scale=2 ** 0.5, dtype=None, device=None):
        super().__init__()
        self.bias = ms.Parameter(mint.empty(1, channel, 1, 1, dtype=dtype))
        self.negative_slope = negative_slope
        self.scale = scale

    def construct(self, input):
        return fused_leaky_relu(input, self.bias.to(dtype=input.dtype), self.negative_slope, self.scale)

def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2 ** 0.5):
    return F.leaky_relu(input + bias, negative_slope) * scale

class Blur(nn.Cell):
    def __init__(self, kernel, pad, dtype=None, device=None):
        super().__init__()
        kernel = ms.tensor(kernel, dtype=dtype)
        kernel = kernel[None, :] * kernel[:, None]
        kernel = kernel / kernel.sum()
        self.register_buffer('kernel', kernel)
        self.pad = pad

    def construct(self, input):
        return upfirdn2d(input, self.kernel.to(dtype=input.dtype), pad=self.pad)

#https://github.com/XPixelGroup/BasicSR/blob/8d56e3a045f9fb3e1d8872f92ee4a4f07f886b0a/basicsr/archs/stylegan2_arch.py#L590
class ScaledLeakyReLU(nn.Cell):
    def __init__(self, negative_slope=0.2):
        super().__init__()
        self.negative_slope = negative_slope

    def construct(self, input):
        return F.leaky_relu(input, negative_slope=self.negative_slope)

# https://github.com/XPixelGroup/BasicSR/blob/8d56e3a045f9fb3e1d8872f92ee4a4f07f886b0a/basicsr/archs/stylegan2_arch.py#L605
class EqualConv2d(nn.Cell):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True, dtype=None, device=None, operations=None):
        super().__init__()
        self.weight = ms.Parameter(mint.empty(out_channel, in_channel, kernel_size, kernel_size, dtype=dtype))
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)
        self.stride = stride
        self.padding = padding
        self.bias = ms.Parameter(mint.empty(out_channel, dtype=dtype)) if bias else None

    def construct(self, input):
        if self.bias is None:
            bias = None
        else:
            bias = self.bias.to(dtype=input.dtype)

        return F.conv2d(input, self.weight.to(dtype=input.dtype) * self.scale, bias=bias, stride=self.stride, padding=self.padding)

# https://github.com/XPixelGroup/BasicSR/blob/8d56e3a045f9fb3e1d8872f92ee4a4f07f886b0a/basicsr/archs/stylegan2_arch.py#L134
class EqualLinear(nn.Cell):
    def __init__(self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None, dtype=None, device=None, operations=None):
        super().__init__()
        self.weight = ms.Parameter(mint.empty(out_dim, in_dim, dtype=dtype))
        self.bias = ms.Parameter(mint.empty(out_dim, dtype=dtype)) if bias else None
        self.activation = activation
        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def construct(self, input):
        if self.bias is None:
            bias = None
        else:
            bias = self.bias.to(dtype=input.dtype) * self.lr_mul

        if self.activation:
            out = F.linear(input, self.weight.to(dtype=input.dtype) * self.scale)
            return fused_leaky_relu(out, bias)
        return F.linear(input, self.weight.to(dtype=input.dtype) * self.scale, bias=bias)

# https://github.com/XPixelGroup/BasicSR/blob/8d56e3a045f9fb3e1d8872f92ee4a4f07f886b0a/basicsr/archs/stylegan2_arch.py#L654
class ConvLayer(nn.SequentialCell):
    def __init__(self, in_channel, out_channel, kernel_size, downsample=False, blur_kernel=[1, 3, 3, 1], bias=True, activate=True, dtype=None, device=None, operations=None):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            layers.append(Blur(blur_kernel, pad=((p + 1) // 2, p // 2)))
            stride, padding = 2, 0
        else:
            stride, padding = 1, kernel_size // 2

        layers.append(EqualConv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias and not activate, dtype=dtype, operations=operations))

        if activate:
            layers.append(FusedLeakyReLU(out_channel) if bias else ScaledLeakyReLU(0.2))

        super().__init__(*layers)

# https://github.com/XPixelGroup/BasicSR/blob/8d56e3a045f9fb3e1d8872f92ee4a4f07f886b0a/basicsr/archs/stylegan2_arch.py#L704
class ResBlock(nn.Cell):
    def __init__(self, in_channel, out_channel, dtype=None, device=None, operations=None):
        super().__init__()
        self.conv1 = ConvLayer(in_channel, in_channel, 3, dtype=dtype, operations=operations)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True, dtype=dtype, operations=operations)
        self.skip = ConvLayer(in_channel, out_channel, 1, downsample=True, activate=False, bias=False, dtype=dtype, operations=operations)

    def construct(self, input):
        out = self.conv2(self.conv1(input))
        skip = self.skip(input)
        return (out + skip) / math.sqrt(2)


class EncoderApp(nn.Cell):
    def __init__(self, w_dim=512, dtype=None, device=None, operations=None):
        super().__init__()
        kwargs = {"device": device, "dtype": dtype, "operations": operations}

        self.convs = nn.CellList([
            ConvLayer(3, 32, 1, **kwargs), ResBlock(32, 64, **kwargs),
            ResBlock(64, 128, **kwargs), ResBlock(128, 256, **kwargs),
            ResBlock(256, 512, **kwargs), ResBlock(512, 512, **kwargs),
            ResBlock(512, 512, **kwargs), ResBlock(512, 512, **kwargs),
            EqualConv2d(512, w_dim, 4, padding=0, bias=False, **kwargs)
        ])

    def construct(self, x):
        h = x
        for conv in self.convs:
            h = conv(h)
        return h.squeeze(-1).squeeze(-1)

class Encoder(nn.Cell):
    def __init__(self, dim=512, motion_dim=20, dtype=None, device=None, operations=None):
        super().__init__()
        self.net_app = EncoderApp(dim, dtype=dtype, operations=operations)
        self.fc = nn.SequentialCell(*[EqualLinear(dim, dim, dtype=dtype, operations=operations) for _ in range(4)] + [EqualLinear(dim, motion_dim, dtype=dtype, operations=operations)])

    def encode_motion(self, x):
        return self.fc(self.net_app(x))

class Direction(nn.Cell):
    def __init__(self, motion_dim, dtype=None, device=None, operations=None):
        super().__init__()
        self.weight = ms.Parameter(mint.empty(512, motion_dim, dtype=dtype))
        self.motion_dim = motion_dim

    def construct(self, input):
        stabilized_weight = self.weight.to(dtype=input.dtype) + 1e-8 * mint.eye(512, self.motion_dim, dtype=input.dtype)
        Q, _ = mint.linalg.qr(stabilized_weight.float())
        if input is None:
            return Q
        return mint.sum(input.unsqueeze(-1) * Q.T.to(input.dtype), dim=1)

class Synthesis(nn.Cell):
    def __init__(self, motion_dim, dtype=None, device=None, operations=None):
        super().__init__()
        self.direction = Direction(motion_dim, dtype=dtype, operations=operations)

class Generator(nn.Cell):
    def __init__(self, style_dim=512, motion_dim=20, dtype=None, device=None, operations=None):
        super().__init__()
        self.enc = Encoder(style_dim, motion_dim, dtype=dtype, operations=operations)
        self.dec = Synthesis(motion_dim, dtype=dtype, operations=operations)

    def get_motion(self, img):
        motion_feat = self.enc.encode_motion(img)
        return self.dec.direction(motion_feat)

class AnimateWanModel(WanModel):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    def __init__(self,
                 model_type='animate',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6,
                 flf_pos_embed_token_number=None,
                 motion_encoder_dim=512,
                 image_model=None,
                 device=None,
                 dtype=None,
                 operations=None,
                 ):

        super().__init__(model_type='i2v', patch_size=patch_size, text_len=text_len, in_dim=in_dim, dim=dim, ffn_dim=ffn_dim, freq_dim=freq_dim, text_dim=text_dim, out_dim=out_dim, num_heads=num_heads, num_layers=num_layers, window_size=window_size, qk_norm=qk_norm, cross_attn_norm=cross_attn_norm, eps=eps, flf_pos_embed_token_number=flf_pos_embed_token_number, image_model=image_model, dtype=dtype, operations=operations)

        self.pose_patch_embedding = operations.Conv3d(
            16, dim, kernel_size=patch_size, stride=patch_size, dtype=dtype
        )

        self.motion_encoder = Generator(style_dim=512, motion_dim=20, dtype=dtype, operations=operations)

        self.face_adapter = FaceAdapter(
            heads_num=self.num_heads,
            hidden_dim=self.dim,
            num_adapter_layers=self.num_layers // 5,
            dtype=dtype, operations=operations
        )

        self.face_encoder = FaceEncoder(
            in_dim=motion_encoder_dim,
            hidden_dim=self.dim,
            num_heads=4,
            dtype=dtype, operations=operations
        )

    def after_patch_embedding(self, x, pose_latents, face_pixel_values):
        if pose_latents is not None:
            pose_latents = self.pose_patch_embedding(pose_latents)
            x[:, :, 1:pose_latents.shape[2] + 1] += pose_latents[:, :, :x.shape[2] - 1]

        if face_pixel_values is None:
            return x, None

        b, c, T, h, w = face_pixel_values.shape
        # rearrange(face_pixel_values, "b c t h w -> (b t) c h w")
        face_pixel_values = face_pixel_values.transpose(1, 2).reshape(-1, face_pixel_values.shape[-3], face_pixel_values.shape[-2], face_pixel_values.shape[-1])
        encode_bs = 8
        face_pixel_values_tmp = []
        for i in range(math.ceil(face_pixel_values.shape[0] / encode_bs)):
            face_pixel_values_tmp.append(self.motion_encoder.get_motion(face_pixel_values[i * encode_bs: (i + 1) * encode_bs]))

        motion_vec = mint.cat(face_pixel_values_tmp)

        # motion_vec = rearrange(motion_vec, "(b t) c -> b t c", t=T)
        motion_vec = motion_vec.reshape(-1, T, motion_vec.shape[-1])
        motion_vec = self.face_encoder(motion_vec)

        B, L, H, C = motion_vec.shape
        pad_face = mint.zeros((B, 1, H, C)).type_as(motion_vec)
        motion_vec = mint.cat([pad_face, motion_vec], dim=1)

        if motion_vec.shape[1] < x.shape[2]:
            B, L, H, C = motion_vec.shape
            pad = mint.zeros((B, x.shape[2] - motion_vec.shape[1], H, C)).type_as(motion_vec)
            motion_vec = mint.cat([motion_vec, pad], dim=1)
        else:
            motion_vec = motion_vec[:, :x.shape[2]]
        return x, motion_vec

    def forward_orig(
        self,
        x,
        t,
        context,
        clip_fea=None,
        pose_latents=None,
        face_pixel_values=None,
        freqs=None,
        transformer_options={},
        **kwargs,
    ):
        # embeddings
        x = self.patch_embedding(x.float()).to(x.dtype)
        x, motion_vec = self.after_patch_embedding(x, pose_latents, face_pixel_values)
        grid_sizes = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)

        # time embeddings
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).to(dtype=x[0].dtype))
        e = e.reshape(t.shape[0], -1, e.shape[-1])
        e0 = self.time_projection(e).unflatten(2, (6, self.dim))

        full_ref = None
        if self.ref_conv is not None:
            full_ref = kwargs.get("reference_latent", None)
            if full_ref is not None:
                full_ref = self.ref_conv(full_ref).flatten(2).transpose(1, 2)
                x = mint.concat((full_ref, x), dim=1)

        # context
        context = self.text_embedding(context)

        context_img_len = None
        if clip_fea is not None:
            if self.img_emb is not None:
                context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
                context = mint.concat([context_clip, context], dim=1)
            context_img_len = clip_fea.shape[-2]

        patches_replace = transformer_options.get("patches_replace", {})
        blocks_replace = patches_replace.get("dit", {})
        for i, block in enumerate(self.blocks):
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"] = block(args["img"], context=args["txt"], e=args["vec"], freqs=args["pe"], context_img_len=context_img_len, transformer_options=args["transformer_options"])
                    return out
                out = blocks_replace[("double_block", i)]({"img": x, "txt": context, "vec": e0, "pe": freqs, "transformer_options": transformer_options}, {"original_block": block_wrap})
                x = out["img"]
            else:
                x = block(x, e=e0, freqs=freqs, context=context, context_img_len=context_img_len, transformer_options=transformer_options)

            if i % 5 == 0 and motion_vec is not None:
                x = x + self.face_adapter.fuser_blocks[i // 5](x, motion_vec)

        # head
        x = self.head(x, e)

        if full_ref is not None:
            x = x[:, full_ref.shape[1]:]

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return x
