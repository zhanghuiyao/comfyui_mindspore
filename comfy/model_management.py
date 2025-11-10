"""
    This file is part of ComfyUI.
    Copyright (C) 2024 Comfy

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import psutil
import logging
from enum import Enum
from comfy.cli_args import args, PerformanceFeature
import sys
import importlib
import platform
import weakref
import gc

import mindspore
from mindspore import mint

from mindspore_patch.utils import dtype_to_size


class VRAMState(Enum):
    DISABLED = 0    #No vram present: no need to move models to vram
    NO_VRAM = 1     #Very low vram: enable all the options to save vram
    LOW_VRAM = 2
    NORMAL_VRAM = 3
    HIGH_VRAM = 4
    SHARED = 5      #No dedicated vram: memory shared between CPU and GPU but models still need to be moved between both.

class CPUState(Enum):
    GPU = 0
    CPU = 1
    MPS = 2

# Determine VRAM State
vram_state = VRAMState.NORMAL_VRAM
set_vram_to = VRAMState.NORMAL_VRAM
cpu_state = CPUState.GPU

total_vram = 0

def get_supported_float8_types():
    float8_types = []
    try:
        float8_types.append(mindspore.float8_e4m3fn)
    except:
        pass
    try:
        float8_types.append(mindspore.float8_e4m3fnuz)
    except:
        pass
    try:
        float8_types.append(mindspore.float8_e5m2)
    except:
        pass
    try:
        float8_types.append(mindspore.float8_e5m2fnuz)
    except:
        pass
    try:
        float8_types.append(mindspore.float8_e8m0fnu)
    except:
        pass
    return float8_types

FLOAT8_TYPES = get_supported_float8_types()

mindspore_version = ""
try:
    mindspore_version = mindspore.version.__version__
    temp = mindspore_version.split(".")
    mindspore_version_numeric = (int(temp[0]), int(temp[1]))
except:
    pass

lowvram_available = True
if args.deterministic:
    logging.info("Using deterministic algorithms for mindspore")
    mindspore.set_deterministic(True)

directml_enabled = False
if args.directml is not None:
    raise NotImplementedError

try:
    npu_available = mindspore.hal.is_available("Ascend")
except:
    pass

xpu_available = False
mlu_available = False
ixuca_available = False

if args.cpu:
    cpu_state = CPUState.CPU

def is_intel_xpu():
    global cpu_state
    global xpu_available
    if cpu_state == CPUState.GPU:
        if xpu_available:
            return True
    return False

def is_ascend_npu():
    global npu_available
    if npu_available:
        return True
    return False

def is_mlu():
    global mlu_available
    if mlu_available:
        return True
    return False

def is_ixuca():
    global ixuca_available
    if ixuca_available:
        return True
    return False

def get_mindspore_device():
    return None

def get_total_memory(dev=None, mindspore_total_too=False):
    global directml_enabled

    if not is_ascend_npu():
        mem_total = psutil.virtual_memory().total
        mem_total_torch = mem_total
    else:
        stats = mindspore.runtime.memory_stats()
        mem_reserved = stats['total_reserved_memory']
        mem_total = mem_total_torch = mem_reserved

    if mindspore_total_too:
        return (mem_total, mem_total_torch)
    else:
        return mem_total

def mac_version():
    try:
        return tuple(int(n) for n in platform.mac_ver()[0].split("."))
    except:
        return None

total_vram = get_total_memory() / (1024 * 1024)
total_ram = psutil.virtual_memory().total / (1024 * 1024)
logging.info("Total VRAM {:0.0f} MB, total RAM {:0.0f} MB".format(total_vram, total_ram))

try:
    logging.info("mindspore version: {}".format(mindspore_version))
    mac_ver = mac_version()
    if mac_ver is not None:
        logging.info("Mac Version {}".format(mac_ver))
except:
    pass

OOM_EXCEPTION = None #OutOfMemoryError

XFORMERS_VERSION = ""
XFORMERS_ENABLED_VAE = True
XFORMERS_IS_AVAILABLE = False

def is_nvidia():
    return False

def is_amd():
    return False

def amd_min_version(device=None, min_rdna_version=0):
    return False

MIN_WEIGHT_MEMORY_RATIO = 0.4
if is_nvidia():
    MIN_WEIGHT_MEMORY_RATIO = 0.0

ENABLE_MINDSPORE_ATTENTION = False
if args.use_mindspore_cross_attention:
    ENABLE_MINDSPORE_ATTENTION = True
    XFORMERS_IS_AVAILABLE = False

try:
    if ENABLE_MINDSPORE_ATTENTION == False and args.use_split_cross_attention == False and args.use_quad_cross_attention == False:
        ENABLE_MINDSPORE_ATTENTION = True
except:
    pass


SUPPORT_FP8_OPS = False  #args.supports_fp8_compute

AMD_RDNA2_AND_OLDER_ARCH = []  #["gfx1030", "gfx1031", "gfx1010", "gfx1011", "gfx1012", "gfx906", "gfx900", "gfx803"]

if ENABLE_MINDSPORE_ATTENTION:
    pass

PRIORITIZE_FP16 = False  # TODO: remove and replace with something that shows exactly which dtype is faster than the other

if args.lowvram:
    set_vram_to = VRAMState.LOW_VRAM
    lowvram_available = True
elif args.novram:
    set_vram_to = VRAMState.NO_VRAM
elif args.highvram or args.gpu_only:
    vram_state = VRAMState.HIGH_VRAM

FORCE_FP32 = False
if args.force_fp32:
    logging.info("Forcing FP32, if this improves things please report it.")
    FORCE_FP32 = True

if lowvram_available:
    if set_vram_to in (VRAMState.LOW_VRAM, VRAMState.NO_VRAM):
        vram_state = set_vram_to


if cpu_state != CPUState.GPU:
    vram_state = VRAMState.DISABLED

if cpu_state == CPUState.MPS:
    vram_state = VRAMState.SHARED

logging.info(f"Set vram state to: {vram_state.name}")

DISABLE_SMART_MEMORY = args.disable_smart_memory

if DISABLE_SMART_MEMORY:
    logging.info("Disabling smart memory management")

def get_mindspore_device_name(device):    
    if is_ascend_npu():
        return "Ascend"
    else:
        return "CPU"

try:
    logging.info("Device: {}".format(get_mindspore_device_name(None)))
except:
    logging.warning("Could not pick default device.")


current_loaded_models = []

def module_size(module):
    module_mem = 0
    sd = module.state_dict()
    for k in sd:
        t = sd[k]
        module_mem += t.nelement() * dtype_to_size(t.dtype)
    return module_mem

class LoadedModel:
    def __init__(self, model):
        self._set_model(model)
        self.device = model.load_device
        self.real_model = None
        self.currently_used = True
        self.model_finalizer = None
        self._patcher_finalizer = None

    def _set_model(self, model):
        self._model = weakref.ref(model)
        if model.parent is not None:
            self._parent_model = weakref.ref(model.parent)
            self._patcher_finalizer = weakref.finalize(model, self._switch_parent)

    def _switch_parent(self):
        model = self._parent_model()
        if model is not None:
            self._set_model(model)

    @property
    def model(self):
        return self._model()

    def model_memory(self):
        return None  #self.model.model_size()

    def model_loaded_memory(self):
        return None  #self.model.loaded_size()

    def model_offloaded_memory(self):
        return None  #self.model.model_size() - self.model.loaded_size()

    def model_memory_required(self, device):
        return None  #self.model_memory()

    def model_load(self, lowvram_model_memory=0, force_patch_weights=False):
        # self.model.model_patches_to(self.device)
        self.model.model_patches_to(self.model.model_dtype())

        # if self.model.loaded_size() > 0:
        use_more_vram = lowvram_model_memory
        if use_more_vram == 0:
            use_more_vram = 1e32
        self.model_use_more_vram(use_more_vram, force_patch_weights=force_patch_weights)
        real_model = self.model.model

        if is_intel_xpu() and not args.disable_ipex_optimize and 'ipex' in globals() and real_model is not None:
            raise NotImplementedError

        self.real_model = weakref.ref(real_model)
        self.model_finalizer = weakref.finalize(real_model, cleanup_models)
        return real_model

    def should_reload_model(self, force_patch_weights=False):
        if force_patch_weights and self.model.lowvram_patch_counter() > 0:
            return True
        return False

    def model_unload(self, memory_to_free=None, unpatch_weights=True):
        if memory_to_free is not None:
            if memory_to_free < self.model.loaded_size():
                freed = self.model.partially_unload(self.model.offload_device, memory_to_free)
                if freed >= memory_to_free:
                    return False
        self.model.detach(unpatch_weights)
        self.model_finalizer.detach()
        self.model_finalizer = None
        self.real_model = None
        return True

    def model_use_more_vram(self, extra_memory, force_patch_weights=False):
        return self.model.partially_load(self.device, extra_memory, force_patch_weights=force_patch_weights)

    def __eq__(self, other):
        return self.model is other.model

    def __del__(self):
        if self._patcher_finalizer is not None:
            self._patcher_finalizer.detach()

    def is_dead(self):
        return self.real_model() is not None and self.model is None


def use_more_memory(extra_memory, loaded_models, device):
    for m in loaded_models:
        if m.device == device:
            extra_memory -= m.model_use_more_vram(extra_memory)
            if extra_memory <= 0:
                break

def offloaded_memory(loaded_models, device):
    offloaded_mem = 0
    # for m in loaded_models:
    #     if m.device == device:
    #         offloaded_mem += m.model_offloaded_memory()
    return offloaded_mem

WINDOWS = any(platform.win32_ver())

EXTRA_RESERVED_VRAM = 400 * 1024 * 1024
if WINDOWS:
    # EXTRA_RESERVED_VRAM = 600 * 1024 * 1024 #Windows is higher because of the shared vram issue
    # if total_vram > (15 * 1024):  # more extra reserved vram on 16GB+ cards
    #     EXTRA_RESERVED_VRAM += 100 * 1024 * 1024
    pass

if args.reserve_vram is not None:
    EXTRA_RESERVED_VRAM = args.reserve_vram * 1024 * 1024 * 1024
    logging.debug("Reserving {}MB vram for other applications.".format(EXTRA_RESERVED_VRAM / (1024 * 1024)))

def extra_reserved_memory():
    return EXTRA_RESERVED_VRAM

def minimum_inference_memory():
    return (1024 * 1024 * 1024) * 0.8 + extra_reserved_memory()

def free_memory(memory_required, device, keep_loaded=[]):
    cleanup_models_gc()
    unloaded_model = []
    can_unload = []
    unloaded_models = []

    for i in range(len(current_loaded_models) -1, -1, -1):
        shift_model = current_loaded_models[i]
        if shift_model.device == device:
            if shift_model not in keep_loaded and not shift_model.is_dead():
                can_unload.append((-shift_model.model_offloaded_memory(), sys.getrefcount(shift_model.model), shift_model.model_memory(), i))
                shift_model.currently_used = False

    for x in sorted(can_unload):
        i = x[-1]
        memory_to_free = None
        if not DISABLE_SMART_MEMORY:
            free_mem = get_free_memory(device)
            if free_mem > memory_required:
                break
            memory_to_free = memory_required - free_mem
        logging.debug(f"Unloading {current_loaded_models[i].model.model.__class__.__name__}")
        if current_loaded_models[i].model_unload(memory_to_free):
            unloaded_model.append(i)

    for i in sorted(unloaded_model, reverse=True):
        unloaded_models.append(current_loaded_models.pop(i))

    if len(unloaded_model) > 0:
        soft_empty_cache()
    else:
        if vram_state != VRAMState.HIGH_VRAM:
            mem_free_total, mem_free_torch = get_free_memory(device, mindspore_free_too=True)
            if mem_free_torch > mem_free_total * 0.25:
                soft_empty_cache()
    return unloaded_models

def load_models_gpu(models, memory_required=0, force_patch_weights=False, minimum_memory_required=None, force_full_load=False):
    raise NotImplementedError

def load_model_gpu(model):
    return load_models_gpu([model])

def loaded_models(only_currently_used=False):
    output = []
    for m in current_loaded_models:
        if only_currently_used:
            if not m.currently_used:
                continue

        output.append(m.model)
    return output


def cleanup_models_gc():
    do_gc = False
    for i in range(len(current_loaded_models)):
        cur = current_loaded_models[i]
        if cur.is_dead():
            logging.info("Potential memory leak detected with model {}, doing a full garbage collect, for maximum performance avoid circular references in the model code.".format(cur.real_model().__class__.__name__))
            do_gc = True
            break

    if do_gc:
        gc.collect()
        soft_empty_cache()

        for i in range(len(current_loaded_models)):
            cur = current_loaded_models[i]
            if cur.is_dead():
                logging.warning("WARNING, memory leak with model {}. Please make sure it is not being referenced from somewhere.".format(cur.real_model().__class__.__name__))



def cleanup_models():
    to_delete = []
    for i in range(len(current_loaded_models)):
        if current_loaded_models[i].real_model() is None:
            to_delete = [i] + to_delete

    for i in to_delete:
        x = current_loaded_models.pop(i)
        del x

def dtype_size(dtype):
    dtype_size = 4
    if dtype == mindspore.float16 or dtype == mindspore.bfloat16:
        dtype_size = 2
    elif dtype == mindspore.float32:
        dtype_size = 4
    return dtype_size

def unet_offload_device():
    return None

def unet_inital_load_device(parameters, dtype):
    return None

def maximum_vram_for_weights(device=None):
    return (get_total_memory() * 0.88 - minimum_inference_memory())

def unet_dtype(device=None, model_params=0, supported_dtypes=[mindspore.float16, mindspore.bfloat16, mindspore.float32], weight_dtype=None):
    if model_params < 0:
        model_params = 1000000000000000000000
    if args.fp32_unet:
        return mindspore.float32
    if args.fp64_unet:
        return mindspore.float64
    if args.bf16_unet:
        return mindspore.bfloat16
    if args.fp16_unet:
        return mindspore.float16
    if args.fp8_e4m3fn_unet:
        return mindspore.float8_e4m3fn
    if args.fp8_e5m2_unet:
        return mindspore.float8_e5m2
    if args.fp8_e8m0fnu_unet:
        return mindspore.float8_e8m0fnu

    fp8_dtype = None
    if weight_dtype in FLOAT8_TYPES:
        fp8_dtype = weight_dtype

    if fp8_dtype is not None:
        if supports_fp8_compute(None): #if fp8 compute is supported the casting is most likely not expensive
            return fp8_dtype

        free_model_memory = maximum_vram_for_weights(None)
        if model_params * 2 > free_model_memory:
            return fp8_dtype

    if PRIORITIZE_FP16 or weight_dtype == mindspore.float16:
        if mindspore.float16 in supported_dtypes and should_use_fp16(device=None, model_params=model_params):
            return mindspore.float16

    for dt in supported_dtypes:
        if dt == mindspore.float16 and should_use_fp16(device=None, model_params=model_params):
            if mindspore.float16 in supported_dtypes:
                return mindspore.float16
        if dt == mindspore.bfloat16 and should_use_bf16(None, model_params=model_params):
            if mindspore.bfloat16 in supported_dtypes:
                return mindspore.bfloat16

    for dt in supported_dtypes:
        if dt == mindspore.float16 and should_use_fp16(device=None, model_params=model_params, manual_cast=True):
            if mindspore.float16 in supported_dtypes:
                return mindspore.float16
        if dt == mindspore.bfloat16 and should_use_bf16(None, model_params=model_params, manual_cast=True):
            if mindspore.bfloat16 in supported_dtypes:
                return mindspore.bfloat16

    return mindspore.float32

# None means no manual cast
def unet_manual_cast(weight_dtype, inference_device, supported_dtypes=[mindspore.float16, mindspore.bfloat16, mindspore.float32]):
    if weight_dtype == mindspore.float32 or weight_dtype == mindspore.float64:
        return None

    fp16_supported = should_use_fp16(None, prioritize_performance=False)
    if fp16_supported and weight_dtype == mindspore.float16:
        return None

    bf16_supported = should_use_bf16(None)
    if bf16_supported and weight_dtype == mindspore.bfloat16:
        return None

    fp16_supported = should_use_fp16(None, prioritize_performance=True)
    if PRIORITIZE_FP16 and fp16_supported and mindspore.float16 in supported_dtypes:
        return mindspore.float16

    for dt in supported_dtypes:
        if dt == mindspore.float16 and fp16_supported:
            return mindspore.float16
        if dt == mindspore.bfloat16 and bf16_supported:
            return mindspore.bfloat16

    return mindspore.float32

def text_encoder_offload_device():
    return None

def text_encoder_device():
    return None

def text_encoder_initial_device(load_device, offload_device, model_size=0):
    return None

def text_encoder_dtype(device=None):
    if args.fp8_e4m3fn_text_enc:
        return mindspore.float8_e4m3fn
    elif args.fp8_e5m2_text_enc:
        return mindspore.float8_e5m2
    elif args.fp16_text_enc:
        return mindspore.float16
    elif args.bf16_text_enc:
        return mindspore.bfloat16
    elif args.fp32_text_enc:
        return mindspore.float32

    if is_device_cpu(None):
        return mindspore.float16

    return mindspore.float16


def intermediate_device():
    return None

def vae_device():
    return None

def vae_offload_device():
    return None

def vae_dtype(device=None, allowed_dtypes=[]):
    if args.fp16_vae:
        return mindspore.float16
    elif args.bf16_vae:
        return mindspore.bfloat16
    elif args.fp32_vae:
        return mindspore.float32

    for d in allowed_dtypes:
        if d == mindspore.float16 and should_use_fp16(None):
            return d

        if d == mindspore.bfloat16 and should_use_bf16(None):
            return d

    return mindspore.float32

def get_autocast_device(dev):
    return None

def supports_dtype(device, dtype): #TODO
    if dtype == mindspore.float32:
        return True
    # if is_device_cpu(device):
    #     return False
    if dtype == mindspore.float16:
        return True
    if dtype == mindspore.bfloat16:
        return True
    return False

def supports_cast(device, dtype): #TODO
    if dtype == mindspore.float32:
        return True
    if dtype == mindspore.float16:
        return True
    if directml_enabled: #TODO: test this
        return False
    if dtype == mindspore.bfloat16:
        return True
    if is_device_mps(None):
        return False
    if is_ascend_npu():
        return False
    if dtype == mindspore.float8_e4m3fn:
        return True
    if dtype == mindspore.float8_e5m2:
        return True
    return False

def pick_weight_dtype(dtype, fallback_dtype, device=None):
    if dtype is None:
        dtype = fallback_dtype
    elif dtype_size(dtype) > dtype_size(fallback_dtype):
        dtype = fallback_dtype

    if not supports_cast(None, dtype):
        dtype = fallback_dtype

    return dtype

def device_supports_non_blocking(device):
    if args.force_non_blocking:
        return True
    if is_device_mps(None):
        return False #pytorch bug? mps doesn't support non blocking
    if is_intel_xpu(): #xpu does support non blocking but it is slower on iGPUs for some reason so disable by default until situation changes
        return False
    if args.deterministic: #TODO: figure out why deterministic breaks non blocking from gpu to cpu (previews)
        return False
    if directml_enabled:
        return False
    return True

def force_channels_last():
    if args.force_channels_last:
        return True

    #TODO
    return False


STREAMS = {}
NUM_STREAMS = 1
if args.async_offload:
    NUM_STREAMS = 2
    logging.info("Using async weight offloading with {} streams".format(NUM_STREAMS))

def current_stream(device):
    if is_ascend_npu():
        return mindspore.runtime.current_stream()
    else:
        return None

stream_counters = {}
def get_offload_stream(device):
    return None

def sync_stream(device, stream):
    if stream is None or current_stream(None) is None:
        return
    current_stream(None).wait_stream(stream)

def cast_to(weight, dtype=None, device=None, non_blocking=False, copy=False, stream=None):
    if not copy:
        if dtype is None or weight.dtype == dtype:
            return weight
    if stream is not None:
        with stream:
            return weight.to(dtype=dtype, copy=copy)
    return weight.to(dtype=dtype, copy=copy)

def cast_to_device(tensor, device, dtype, copy=False):
    non_blocking = device_supports_non_blocking(None)
    return cast_to(tensor, dtype=dtype, device=None, non_blocking=non_blocking, copy=copy)


PINNED_MEMORY = {}
TOTAL_PINNED_MEMORY = 0
MAX_PINNED_MEMORY = -1
if not args.disable_pinned_memory:
    pass


def pin_memory(tensor):
    return True

def unpin_memory(tensor):
    return True

def sage_attention_enabled():
    return args.use_sage_attention

def flash_attention_enabled():
    return args.use_flash_attention

def xformers_enabled():
    global directml_enabled
    global cpu_state
    if cpu_state != CPUState.GPU:
        return False
    if is_intel_xpu():
        return False
    if is_ascend_npu():
        return False
    if is_mlu():
        return False
    if is_ixuca():
        return False
    if directml_enabled:
        return False
    return XFORMERS_IS_AVAILABLE


def xformers_enabled_vae():
    enabled = xformers_enabled()
    if not enabled:
        return False

    return XFORMERS_ENABLED_VAE

def mindspore_attention_enabled():
    global ENABLE_MINDSPORE_ATTENTION
    return ENABLE_MINDSPORE_ATTENTION

def mindspore_attention_enabled():
    global ENABLE_MINDSPORE_ATTENTION
    return ENABLE_MINDSPORE_ATTENTION

def mindspore_attention_enabled_vae():
    if is_amd():
        return False  # enabling pytorch attention on AMD currently causes crash when doing high res
    return mindspore_attention_enabled()

def mindspore_attention_flash_attention():
    global ENABLE_MINDSPORE_ATTENTION
    if ENABLE_MINDSPORE_ATTENTION:
        #TODO: more reliable way of checking for flash attention?
        if is_nvidia():
            return True
        if is_intel_xpu():
            return True
        if is_ascend_npu():
            return True
        if is_mlu():
            return True
        if is_amd():
            return True #if you have pytorch attention enabled on AMD it probably supports at least mem efficient attention
        if is_ixuca():
            return True
    return False

def force_upcast_attention_dtype():
    upcast = args.force_upcast_attention

    macos_version = mac_version()
    if macos_version is not None and ((14, 5) <= macos_version):  # black image bug on recent versions of macOS, I don't think it's ever getting fixed
        upcast = True

    if upcast:
        return {mindspore.float16: mindspore.float32}
    else:
        return None

def get_free_memory(dev=None, mindspore_free_too=False):
    global directml_enabled

    if not is_ascend_npu():
        mem_free_total = psutil.virtual_memory().available
        mem_free_mindspore = mem_free_total
    else:
        stats = mindspore.runtime.memory_stats()
        mem_active = stats['total_reserved_memory']
        mem_reserved = stats['total_allocated_memory']
        mem_free_total = mem_reserved - mem_active
        mem_free_mindspore = mem_free_total

    if mindspore_free_too:
        return (mem_free_total, mem_free_mindspore)
    else:
        return mem_free_total

def cpu_mode():
    global cpu_state
    return False  #cpu_state == CPUState.CPU

def mps_mode():
    global cpu_state
    return False  #cpu_state == CPUState.MPS

def is_device_type(device, type):
    return False

def is_device_cpu(device):
    return is_device_type(None, 'cpu')

def is_device_mps(device):
    return is_device_type(None, 'mps')

def is_device_xpu(device):
    return is_device_type(None, 'xpu')

def is_device_cuda(device):
    return is_device_type(None, 'cuda')

def is_directml_enabled():
    global directml_enabled
    if directml_enabled:
        return True

    return False

def should_use_fp16(device=None, model_params=0, prioritize_performance=True, manual_cast=False):
    if args.force_fp16:
        return True

    if FORCE_FP32:
        return False

    if is_directml_enabled():
        return True

    if cpu_mode():
        return False

    if is_intel_xpu():
        return True

    if is_ascend_npu():
        return True

    if is_mlu():
        return True

    if is_ixuca():
        return True

    if manual_cast:
        free_model_memory = maximum_vram_for_weights(None)
        if (not prioritize_performance) or model_params * 4 > free_model_memory:
            return True

    return True

def should_use_bf16(device=None, model_params=0, prioritize_performance=True, manual_cast=False):
    if FORCE_FP32:
        return False

    if directml_enabled:
        return False

    if cpu_mode():
        return False

    if is_intel_xpu():
        return True

    if is_ascend_npu():
        return True

    if is_ixuca():
        return True

    if is_amd():
        if manual_cast:
            return True
        return False

    bf16_works = True if is_ascend_npu() else False
    if bf16_works and manual_cast:
        free_model_memory = maximum_vram_for_weights(None)
        if (not prioritize_performance) or model_params * 4 > free_model_memory:
            return True

    return False

def supports_fp8_compute(device=None):
    return False

def extended_fp16_support():
    return True

def soft_empty_cache(force=False):
    if is_ascend_npu():
        mindspore.runtime.empty_cache()

def unload_all_models():
    free_memory(1e30, get_mindspore_device())


#TODO: might be cleaner to put this somewhere else
import threading

class InterruptProcessingException(Exception):
    pass

interrupt_processing_mutex = threading.RLock()

interrupt_processing = False
def interrupt_current_processing(value=True):
    global interrupt_processing
    global interrupt_processing_mutex
    with interrupt_processing_mutex:
        interrupt_processing = value

def processing_interrupted():
    global interrupt_processing
    global interrupt_processing_mutex
    with interrupt_processing_mutex:
        return interrupt_processing

def throw_exception_if_processing_interrupted():
    global interrupt_processing
    global interrupt_processing_mutex
    with interrupt_processing_mutex:
        if interrupt_processing:
            interrupt_processing = False
            raise InterruptProcessingException()
