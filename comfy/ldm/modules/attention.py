import math
import sys

import mindspore
from mindspore import mint
from mindspore.mint import functional as F
from mindspore.mint import nn, einsum

from typing import Optional, Any, Callable, Union
import logging
import functools

from .diffusionmodules.util import AlphaBlender, timestep_embedding

from comfy import model_management

from mindspore_patch.utils import dtype_to_max

SAGE_ATTENTION_IS_AVAILABLE = False
FLASH_ATTENTION_IS_AVAILABLE = False  # TODO: mindspore support fa


REGISTERED_ATTENTION_FUNCTIONS = {}
def register_attention_function(name: str, func: Callable):
    # avoid replacing existing functions
    if name not in REGISTERED_ATTENTION_FUNCTIONS:
        REGISTERED_ATTENTION_FUNCTIONS[name] = func
    else:
        logging.warning(f"Attention function {name} already registered, skipping registration.")

def get_attention_function(name: str, default: Any=...) -> Union[Callable, None]:
    if name == "optimized":
        return optimized_attention
    elif name not in REGISTERED_ATTENTION_FUNCTIONS:
        if default is ...:
            raise KeyError(f"Attention function {name} not found.")
        else:
            return default
    return REGISTERED_ATTENTION_FUNCTIONS[name]

from comfy.cli_args import args
import comfy.ops
ops = comfy.ops.disable_weight_init

FORCE_UPCAST_ATTENTION_DTYPE = model_management.force_upcast_attention_dtype()

def get_attn_precision(attn_precision, current_dtype):
    if args.dont_upcast_attention:
        return None

    if FORCE_UPCAST_ATTENTION_DTYPE is not None and current_dtype in FORCE_UPCAST_ATTENTION_DTYPE:
        return FORCE_UPCAST_ATTENTION_DTYPE[current_dtype]
    return attn_precision

def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d


# feedforward
class GEGLU(mindspore.nn.Cell):
    def __init__(self, dim_in, dim_out, dtype=None, operations=ops):
        super().__init__()
        self.proj = operations.Linear(dim_in, dim_out * 2, dtype=dtype)

    def construct(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(mindspore.nn.Cell):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0., dtype=None, operations=ops):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = mindspore.nn.SequentialCell([
            operations.Linear(dim, inner_dim, dtype=dtype),
            nn.GELU()
        ]) if not glu else GEGLU(dim, inner_dim, dtype=dtype, operations=operations)

        self.net = mindspore.nn.SequentialCell([
            project_in,
            nn.Dropout(dropout),
            operations.Linear(inner_dim, dim_out, dtype=dtype)
        ])

    def construct(self, x):
        return self.net(x)

def Normalize(in_channels, dtype=None):
    return mint.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True, dtype=dtype)


def wrap_attn(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        remove_attn_wrapper_key = False
        try:
            if "_inside_attn_wrapper" not in kwargs:
                transformer_options = kwargs.get("transformer_options", None)
                remove_attn_wrapper_key = True
                kwargs["_inside_attn_wrapper"] = True
                if transformer_options is not None:
                    if "optimized_attention_override" in transformer_options:
                        return transformer_options["optimized_attention_override"](func, *args, **kwargs)
            return func(*args, **kwargs)
        finally:
            if remove_attn_wrapper_key:
                del kwargs["_inside_attn_wrapper"]
    return wrapper

@wrap_attn
def attention_basic(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
    attn_precision = get_attn_precision(attn_precision, q.dtype)

    if skip_reshape:
        b, _, _, dim_head = q.shape
    else:
        b, _, dim_head = q.shape
        dim_head //= heads

    scale = dim_head ** -0.5

    h = heads
    if skip_reshape:
         q, k, v = map(
            lambda t: t.reshape(b * heads, -1, dim_head),
            (q, k, v),
        )
    else:
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, -1, heads, dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * heads, -1, dim_head)
            .contiguous(),
            (q, k, v),
        )

    # force cast to fp32 to avoid overflowing
    if attn_precision == mindspore.float32:
        sim = einsum('b i d, b j d -> b i j', q.float(), k.float()) * scale
    else:
        sim = einsum('b i d, b j d -> b i j', q, k) * scale

    del q, k

    if exists(mask):
        if mask.dtype == mindspore.bool:
            
            # mask = rearrange(mask, 'b ... -> b (...)') #TODO: check if this bool part matches mindspore attention
            b = mask.shape[0]
            mask = mask.view(b, -1)

            max_neg_value = -dtype_to_max(sim.dtype)
            
            # mask = repeat(mask, 'b j -> (b h) () j', h=h)
            _b, _j = mask.shape
            mask = mask[:, None, :].expand((_b*h, 1, _j))

            sim.masked_fill_(~mask, max_neg_value)
        else:
            if len(mask.shape) == 2:
                bs = 1
            else:
                bs = mask.shape[0]
            mask = mask.reshape(bs, -1, mask.shape[-2], mask.shape[-1]).expand((b, heads, -1, -1)).reshape(-1, mask.shape[-2], mask.shape[-1])
            sim.add_(mask)

    # attention, what we cannot get enough of
    sim = sim.softmax(dim=-1)

    out = einsum('b i j, b j d -> b i d', sim.to(v.dtype), v)

    if skip_output_reshape:
        out = (
            out.unsqueeze(0)
            .reshape(b, heads, -1, dim_head)
        )
    else:
        out = (
            out.unsqueeze(0)
            .reshape(b, heads, -1, dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, -1, heads * dim_head)
        )
    return out

@wrap_attn
def attention_sub_quad(query, key, value, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
    raise NotImplementedError

@wrap_attn
def attention_split(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
    raise NotImplementedError

BROKEN_XFORMERS = False

@wrap_attn
def attention_mindspore(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
    if skip_reshape:
        b, _, _, dim_head = q.shape
    else:
        b, _, dim_head = q.shape
        dim_head //= heads
        q, k, v = map(
            lambda t: t.view(b, -1, heads, dim_head).transpose(1, 2),
            (q, k, v),
        )

    if mask is not None:
        # add a batch dimension if there isn't already one
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        # add a heads dimension if there isn't already one
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)

    out = comfy.ops.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
    if not skip_output_reshape:
        out = (
            out.transpose(1, 2).reshape(b, -1, heads * dim_head)
        )

    return out

@wrap_attn
def attention_sage(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
    raise NotImplementedError


def flash_attn_wrapper(q: mindspore.Tensor, k: mindspore.Tensor, v: mindspore.Tensor,
                dropout_p: float = 0.0, causal: bool = False) -> mindspore.Tensor:
    assert False, f"Could not define flash_attn_wrapper: do not support fa now."

@wrap_attn
def attention_flash(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
    raise NotImplementedError


optimized_attention = attention_basic

if model_management.sage_attention_enabled():
    logging.info("Using sage attention")
    raise NotImplementedError
elif model_management.xformers_enabled():
    logging.info("Using xformers attention")
    raise NotImplementedError
elif model_management.flash_attention_enabled():
    logging.info("Using Flash Attention")
    raise NotImplementedError
elif model_management.mindspore_attention_enabled():
    logging.info("Using mindspore attention")
    optimized_attention = attention_mindspore
else:
    if args.use_split_cross_attention:
        logging.info("Using split optimization for attention")
        raise NotImplementedError
    else:
        logging.info("Using sub quadratic optimization for attention, if you have memory or speed issues try using: --use-split-cross-attention")
        raise NotImplementedError

optimized_attention_masked = optimized_attention


# register core-supported attention functions
if SAGE_ATTENTION_IS_AVAILABLE:
    # register_attention_function("sage", attention_sage)
    raise NotImplementedError
if FLASH_ATTENTION_IS_AVAILABLE:
    # register_attention_function("flash", attention_flash)
    raise NotImplementedError
if model_management.xformers_enabled():
    raise NotImplementedError
register_attention_function("mindspore", attention_mindspore)
# register_attention_function("sub_quad", attention_sub_quad)
# register_attention_function("split", attention_split)


def optimized_attention_for_device(device=None, mask=False, small_input=False):
    if small_input:
        if model_management.mindspore_attention_enabled():
            return attention_mindspore #TODO: need to confirm but this is probably slightly faster for small inputs in all cases
        else:
            return attention_basic

    if mask:
        return optimized_attention_masked

    return optimized_attention


class CrossAttention(mindspore.nn.Cell):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0., attn_precision=None, dtype=None, operations=ops):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.attn_precision = attn_precision

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = operations.Linear(query_dim, inner_dim, bias=False, dtype=dtype)
        self.to_k = operations.Linear(context_dim, inner_dim, bias=False, dtype=dtype)
        self.to_v = operations.Linear(context_dim, inner_dim, bias=False, dtype=dtype)

        self.to_out = mindspore.nn.SequentialCell([operations.Linear(inner_dim, query_dim, dtype=dtype), nn.Dropout(dropout)])

    def construct(self, x, context=None, value=None, mask=None, transformer_options={}):
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        if value is not None:
            v = self.to_v(value)
            del value
        else:
            v = self.to_v(context)

        if mask is None:
            out = optimized_attention(q, k, v, self.heads, attn_precision=self.attn_precision, transformer_options=transformer_options)
        else:
            out = optimized_attention_masked(q, k, v, self.heads, mask, attn_precision=self.attn_precision, transformer_options=transformer_options)
        return self.to_out(out)


class BasicTransformerBlock(mindspore.nn.Cell):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True, ff_in=False, inner_dim=None,
                 disable_self_attn=False, disable_temporal_crossattention=False, switch_temporal_ca_to_sa=False, attn_precision=None, dtype=None, operations=ops):
        super().__init__()

        self.ff_in = ff_in or inner_dim is not None
        if inner_dim is None:
            inner_dim = dim

        self.is_res = inner_dim == dim
        self.attn_precision = attn_precision

        if self.ff_in:
            self.norm_in = operations.LayerNorm(dim, dtype=dtype)
            self.ff_in = FeedForward(dim, dim_out=inner_dim, dropout=dropout, glu=gated_ff, dtype=dtype, operations=operations)

        self.disable_self_attn = disable_self_attn
        self.attn1 = CrossAttention(query_dim=inner_dim, heads=n_heads, dim_head=d_head, dropout=dropout,
                              context_dim=context_dim if self.disable_self_attn else None, attn_precision=self.attn_precision, dtype=dtype, operations=operations)  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward(inner_dim, dim_out=dim, dropout=dropout, glu=gated_ff, dtype=dtype, operations=operations)

        if disable_temporal_crossattention:
            if switch_temporal_ca_to_sa:
                raise ValueError
            else:
                self.attn2 = None
        else:
            context_dim_attn2 = None
            if not switch_temporal_ca_to_sa:
                context_dim_attn2 = context_dim

            self.attn2 = CrossAttention(query_dim=inner_dim, context_dim=context_dim_attn2,
                                heads=n_heads, dim_head=d_head, dropout=dropout, attn_precision=self.attn_precision, dtype=dtype, operations=operations)  # is self-attn if context is none
            self.norm2 = operations.LayerNorm(inner_dim, dtype=dtype)

        self.norm1 = operations.LayerNorm(inner_dim, dtype=dtype)
        self.norm3 = operations.LayerNorm(inner_dim, dtype=dtype)
        self.n_heads = n_heads
        self.d_head = d_head
        self.switch_temporal_ca_to_sa = switch_temporal_ca_to_sa

    def construct(self, x, context=None, transformer_options={}):
        extra_options = {}
        block = transformer_options.get("block", None)
        block_index = transformer_options.get("block_index", 0)
        transformer_patches = {}
        transformer_patches_replace = {}

        for k in transformer_options:
            if k == "patches":
                transformer_patches = transformer_options[k]
            elif k == "patches_replace":
                transformer_patches_replace = transformer_options[k]
            else:
                extra_options[k] = transformer_options[k]

        extra_options["n_heads"] = self.n_heads
        extra_options["dim_head"] = self.d_head
        extra_options["attn_precision"] = self.attn_precision

        if self.ff_in:
            x_skip = x
            x = self.ff_in(self.norm_in(x))
            if self.is_res:
                x += x_skip

        n = self.norm1(x)
        if self.disable_self_attn:
            context_attn1 = context
        else:
            context_attn1 = None
        value_attn1 = None

        if "attn1_patch" in transformer_patches:
            patch = transformer_patches["attn1_patch"]
            if context_attn1 is None:
                context_attn1 = n
            value_attn1 = context_attn1
            for p in patch:
                n, context_attn1, value_attn1 = p(n, context_attn1, value_attn1, extra_options)

        if block is not None:
            transformer_block = (block[0], block[1], block_index)
        else:
            transformer_block = None
        attn1_replace_patch = transformer_patches_replace.get("attn1", {})
        block_attn1 = transformer_block
        if block_attn1 not in attn1_replace_patch:
            block_attn1 = block

        if block_attn1 in attn1_replace_patch:
            if context_attn1 is None:
                context_attn1 = n
                value_attn1 = n
            n = self.attn1.to_q(n)
            context_attn1 = self.attn1.to_k(context_attn1)
            value_attn1 = self.attn1.to_v(value_attn1)
            n = attn1_replace_patch[block_attn1](n, context_attn1, value_attn1, extra_options)
            n = self.attn1.to_out(n)
        else:
            n = self.attn1(n, context=context_attn1, value=value_attn1, transformer_options=transformer_options)

        if "attn1_output_patch" in transformer_patches:
            patch = transformer_patches["attn1_output_patch"]
            for p in patch:
                n = p(n, extra_options)

        x = n + x
        if "middle_patch" in transformer_patches:
            patch = transformer_patches["middle_patch"]
            for p in patch:
                x = p(x, extra_options)

        if self.attn2 is not None:
            n = self.norm2(x)
            if self.switch_temporal_ca_to_sa:
                context_attn2 = n
            else:
                context_attn2 = context
            value_attn2 = None
            if "attn2_patch" in transformer_patches:
                patch = transformer_patches["attn2_patch"]
                value_attn2 = context_attn2
                for p in patch:
                    n, context_attn2, value_attn2 = p(n, context_attn2, value_attn2, extra_options)

            attn2_replace_patch = transformer_patches_replace.get("attn2", {})
            block_attn2 = transformer_block
            if block_attn2 not in attn2_replace_patch:
                block_attn2 = block

            if block_attn2 in attn2_replace_patch:
                if value_attn2 is None:
                    value_attn2 = context_attn2
                n = self.attn2.to_q(n)
                context_attn2 = self.attn2.to_k(context_attn2)
                value_attn2 = self.attn2.to_v(value_attn2)
                n = attn2_replace_patch[block_attn2](n, context_attn2, value_attn2, extra_options)
                n = self.attn2.to_out(n)
            else:
                n = self.attn2(n, context=context_attn2, value=value_attn2, transformer_options=transformer_options)

        if "attn2_output_patch" in transformer_patches:
            patch = transformer_patches["attn2_output_patch"]
            for p in patch:
                n = p(n, extra_options)

        x = n + x
        if self.is_res:
            x_skip = x
        x = self.ff(self.norm3(x))
        if self.is_res:
            x = x_skip + x

        return x


class SpatialTransformer(mindspore.nn.Cell):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None,
                 disable_self_attn=False, use_linear=False,
                 use_checkpoint=True, attn_precision=None, dtype=None, operations=ops):
        super().__init__()
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim] * depth
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = operations.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True, dtype=dtype)
        if not use_linear:
            self.proj_in = operations.Conv2d(in_channels,
                                     inner_dim,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0, dtype=dtype)
        else:
            self.proj_in = operations.Linear(in_channels, inner_dim, dtype=dtype)

        self.transformer_blocks = mindspore.nn.CellList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim[d],
                                   disable_self_attn=disable_self_attn, checkpoint=use_checkpoint, attn_precision=attn_precision, dtype=dtype, operations=operations)
                for d in range(depth)]
        )
        if not use_linear:
            self.proj_out = operations.Conv2d(inner_dim,in_channels,
                                                  kernel_size=1,
                                                  stride=1,
                                                  padding=0, dtype=dtype)
        else:
            self.proj_out = operations.Linear(in_channels, inner_dim, dtype=dtype)
        self.use_linear = use_linear

    def construct(self, x, context=None, transformer_options={}):
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context] * len(self.transformer_blocks)
        b, c, h, w = x.shape
        transformer_options["activations_shape"] = list(x.shape)
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = x.movedim(1, 3).flatten(1, 2).contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            transformer_options["block_index"] = i
            x = block(x, context=context[i], transformer_options=transformer_options)
        if self.use_linear:
            x = self.proj_out(x)
        x = x.reshape(x.shape[0], h, w, x.shape[-1]).movedim(3, 1).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in


class SpatialVideoTransformer(SpatialTransformer):
    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.0,
        use_linear=False,
        context_dim=None,
        use_spatial_context=False,
        timesteps=None,
        merge_strategy: str = "fixed",
        merge_factor: float = 0.5,
        time_context_dim=None,
        ff_in=False,
        checkpoint=False,
        time_depth=1,
        disable_self_attn=False,
        disable_temporal_crossattention=False,
        max_time_embed_period: int = 10000,
        attn_precision=None,
        dtype=None, operations=ops
    ):
        super().__init__(
            in_channels,
            n_heads,
            d_head,
            depth=depth,
            dropout=dropout,
            use_checkpoint=checkpoint,
            context_dim=context_dim,
            use_linear=use_linear,
            disable_self_attn=disable_self_attn,
            attn_precision=attn_precision,
            dtype=dtype, operations=operations
        )
        self.time_depth = time_depth
        self.depth = depth
        self.max_time_embed_period = max_time_embed_period

        time_mix_d_head = d_head
        n_time_mix_heads = n_heads

        time_mix_inner_dim = int(time_mix_d_head * n_time_mix_heads)

        inner_dim = n_heads * d_head
        if use_spatial_context:
            time_context_dim = context_dim

        self.time_stack = mindspore.nn.CellList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    n_time_mix_heads,
                    time_mix_d_head,
                    dropout=dropout,
                    context_dim=time_context_dim,
                    # timesteps=timesteps,
                    checkpoint=checkpoint,
                    ff_in=ff_in,
                    inner_dim=time_mix_inner_dim,
                    disable_self_attn=disable_self_attn,
                    disable_temporal_crossattention=disable_temporal_crossattention,
                    attn_precision=attn_precision,
                    dtype=dtype, operations=operations
                )
                for _ in range(self.depth)
            ]
        )

        assert len(self.time_stack) == len(self.transformer_blocks)

        self.use_spatial_context = use_spatial_context
        self.in_channels = in_channels

        time_embed_dim = self.in_channels * 4
        self.time_pos_embed = mindspore.nn.SequentialCell([
            operations.Linear(self.in_channels, time_embed_dim, dtype=dtype),
            nn.SiLU(),
            operations.Linear(time_embed_dim, self.in_channels, dtype=dtype),
        ])

        self.time_mixer = AlphaBlender(
            alpha=merge_factor, merge_strategy=merge_strategy
        )

    def construct(
        self,
        x: mindspore.Tensor,
        context: Optional[mindspore.Tensor] = None,
        time_context: Optional[mindspore.Tensor] = None,
        timesteps: Optional[int] = None,
        image_only_indicator: Optional[mindspore.Tensor] = None,
        transformer_options={}
    ) -> mindspore.Tensor:
        _, _, h, w = x.shape
        transformer_options["activations_shape"] = list(x.shape)
        x_in = x
        spatial_context = None
        if exists(context):
            spatial_context = context

        if self.use_spatial_context:
            assert (
                context.ndim == 3
            ), f"n dims of spatial context should be 3 but are {context.ndim}"

            if time_context is None:
                time_context = context
            time_context_first_timestep = time_context[::timesteps]
            
            # time_context = repeat(
            #     time_context_first_timestep, "b ... -> (b n) ...", n=h * w
            # )
            _shape = time_context_first_timestep.shape
            time_context = time_context.expand((_shape[0]*h*w, *_shape[1:]))

        elif time_context is not None and not self.use_spatial_context:

            # time_context = repeat(time_context, "b ... -> (b n) ...", n=h * w)
            _shape = time_context.shape
            time_context = time_context.expand((_shape[0]*h*w, *_shape[1:]))

            if time_context.ndim == 2:
                # time_context = rearrange(time_context, "b c -> b 1 c")
                time_context = time_context[:, None, :]

        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        
        # x = rearrange(x, "b c h w -> b (h w) c")
        b, c, _, _ = x.shape
        x = x.permute(0, 2, 3, 1).view(b, -1, c)

        if self.use_linear:
            x = self.proj_in(x)

        num_frames = mint.arange(timesteps)

        # num_frames = repeat(num_frames, "t -> b t", b=x.shape[0] // timesteps)
        # num_frames = rearrange(num_frames, "b t -> (b t)")
        b, t = x.shape[0] // timesteps, num_frames.shape[0]
        num_frames = num_frames[None, :].expand((b, t)).view(-1)

        t_emb = timestep_embedding(num_frames, self.in_channels, repeat_only=False, max_period=self.max_time_embed_period).to(x.dtype)
        emb = self.time_pos_embed(t_emb)
        emb = emb[:, None, :]

        for it_, (block, mix_block) in enumerate(
            zip(self.transformer_blocks, self.time_stack)
        ):
            transformer_options["block_index"] = it_
            x = block(
                x,
                context=spatial_context,
                transformer_options=transformer_options,
            )

            x_mix = x
            x_mix = x_mix + emb

            B, S, C = x_mix.shape
            
            # x_mix = rearrange(x_mix, "(b t) s c -> (b s) t c", t=timesteps)
            _, s, c = x_mix.shape
            t = timesteps
            x_mix = x_mix.view(-1, t, s, c).permute(0, 2, 1, 3).view(-1, t, c)

            x_mix = mix_block(x_mix, context=time_context, transformer_options=transformer_options)
            
            # x_mix = rearrange(
            #     x_mix, "(b s) t c -> (b t) s c", s=S, b=B // timesteps, c=C, t=timesteps
            # )
            _, t, c = x_mix.shape
            s, b, c, t = S, B // timesteps, C, timesteps
            x_mix = x_mix.view(b, s, t, c).permute(0, 2, 1, 3).view(b*t, s, c)


            x = self.time_mixer(x_spatial=x, x_temporal=x_mix, image_only_indicator=image_only_indicator)

        if self.use_linear:
            x = self.proj_out(x)
        
        # x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        b, _, c = x.shape
        x = x.view(b, h, w, c).permute(0, 3, 1, 2)

        if not self.use_linear:
            x = self.proj_out(x)
        out = x + x_in
        return out
