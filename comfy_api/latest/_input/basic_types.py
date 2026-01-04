import mindspore
from typing import TypedDict, List, Optional

ImageInput = mindspore.Tensor
"""
An image in format [B, H, W, C] where B is the batch size, C is the number of channels,
"""

MaskInput = mindspore.Tensor
"""
A mask in format [B, H, W] where B is the batch size
"""

class AudioInput(TypedDict):
    """
    TypedDict representing audio input.
    """

    waveform: mindspore.Tensor
    """
    Tensor in the format [B, C, T] where B is the batch size, C is the number of channels,
    """

    sample_rate: int

class LatentInput(TypedDict):
    """
    TypedDict representing latent input.
    """

    samples: mindspore.Tensor
    """
    Tensor in the format [B, C, H, W] where B is the batch size, C is the number of channels,
    H is the height, and W is the width.
    """

    noise_mask: Optional[MaskInput]
    """
    Optional noise mask tensor in the same format as samples.
    """

    batch_index: Optional[List[int]]
