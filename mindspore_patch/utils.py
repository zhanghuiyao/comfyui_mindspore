import numpy as np

import mindspore

_DTYPE_2_STRING = {
    mindspore.float16: "float16",
    mindspore.bfloat16: "bfloat16",
    mindspore.float32: "float32",
    mindspore.float64: "float64",
    mindspore.uint8: "uint8",
    mindspore.int8: "int8",
    mindspore.int16: "int16",
    mindspore.int32: "int32",
    mindspore.int64: "int64",
    mindspore.bool_: "bool",
}

_STRING_2_DTYPE = {
    "float16": mindspore.float16,
    "bfloat16": mindspore.bfloat16,
    "float32": mindspore.float32,
    "float64": mindspore.float64,
    "uint8": mindspore.uint8,
    "int8": mindspore.int8,
    "int16": mindspore.int16,
    "int32": mindspore.int32,
    "int64": mindspore.int64,
    "bool": mindspore.bool_,
}


_MIN_FP16 = mindspore.tensor(np.finfo(np.float16).min, dtype=mindspore.float16)
_MIN_FP32 = mindspore.tensor(np.finfo(np.float32).min, dtype=mindspore.float32)
_MIN_FP64 = mindspore.tensor(np.finfo(np.float64).min, dtype=mindspore.float64)
_MIN_BF16 = mindspore.tensor(float.fromhex("-0x1.fe00000000000p+127"), dtype=mindspore.bfloat16)
_MAX_FP16 = mindspore.tensor(np.finfo(np.float16).max, dtype=mindspore.float16)
_MAX_FP32 = mindspore.tensor(np.finfo(np.float32).max, dtype=mindspore.float32)
_MAX_FP64 = mindspore.tensor(np.finfo(np.float64).max, dtype=mindspore.float64)
_MAX_BF16 = mindspore.tensor(float.fromhex("0x1.fe00000000000p+127"), dtype=mindspore.bfloat16)

_BITS_FP16 = np.finfo(np.float16).bits
_BITS_FP32 = np.finfo(np.float32).bits
_BITS_FP64 = np.finfo(np.float64).bits
_BITS_BF16 = 16

_DTYPE_2_MIN = {
    mindspore.float16: _MIN_FP16,
    mindspore.float32: _MIN_FP32,
    mindspore.float64: _MIN_FP64,
    mindspore.bfloat16: _MIN_BF16,
}

_DTYPE_2_MAX = {
    mindspore.float16: _MAX_FP16,
    mindspore.float32: _MAX_FP32,
    mindspore.float64: _MAX_FP64,
    mindspore.bfloat16: _MAX_BF16,
}

_DTYPE_2_BITS = {
    mindspore.float16: _BITS_FP16,
    mindspore.float32: _BITS_FP32,
    mindspore.float64: _BITS_FP64,
    mindspore.bfloat16: _BITS_BF16,
}

_DTYPE_2_SIZE = {
    mindspore.float8_e4m3fn: 1,
    mindspore.float8_e5m2: 1,
    mindspore.float16: 2,
    mindspore.float32: 4,
    mindspore.float64: 8,
    mindspore.bfloat16: 2,
    mindspore.uint8: 1,
    mindspore.uint16: 2,
    mindspore.int16: 2,
    mindspore.int32: 4,
    mindspore.int64: 8
}

TORCH_TO_MINDSPORE_DTYPE_MAP = {
    "torch.float32": mindspore.float32,
    "torch.bfloat16": mindspore.bfloat16,
    "torch.float16": mindspore.float16,
}


def dtype_to_min(dtype):
    return _DTYPE_2_MIN.get(dtype, "others dtype")


def dtype_to_max(dtype):
    return _DTYPE_2_MAX.get(dtype, "others dtype")


def dtype_to_bits(dtype):
    return _DTYPE_2_BITS.get(dtype, "others dtype")

def dtype_to_size(dtype):
    return _DTYPE_2_SIZE.get(dtype, "others dtype")

def dtype_to_str(dtype):
    return _DTYPE_2_STRING.get(dtype, "others dtype")


def str_to_dtype(dtype):
    return _STRING_2_DTYPE.get(dtype, "others dtype")

