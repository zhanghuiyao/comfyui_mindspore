#!/usr/bin/env python3
"""
Tiny AutoEncoder for Stable Diffusion
(DNN for encoding / decoding SD's latent space)
"""
import mindspore
import mindspore.mint.nn as nn
from mindspore import mint

import comfy.utils
import comfy.ops

def conv(n_in, n_out, **kwargs):
    return comfy.ops.disable_weight_init.Conv2d(n_in, n_out, 3, padding=1, **kwargs)

class Clamp(mindspore.nn.Cell):
    def construct(self, x):
        return mint.tanh(x / 3) * 3

class Block(mindspore.nn.Cell):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = mindspore.nn.SequentialCell([conv(n_in, n_out), nn.ReLU(), conv(n_out, n_out), nn.ReLU(), conv(n_out, n_out)])
        self.skip = comfy.ops.disable_weight_init.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.fuse = nn.ReLU()
    def construct(self, x):
        return self.fuse(self.conv(x) + self.skip(x))

def Encoder(latent_channels=4):
    return mindspore.nn.SequentialCell([
        conv(3, 64), Block(64, 64),
        conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
        conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
        conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
        conv(64, latent_channels),
    ])


def Decoder(latent_channels=4):
    return mindspore.nn.SequentialCell([
        Clamp(), conv(latent_channels, 64), nn.ReLU(),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), conv(64, 3),
    ])

class TAESD(mindspore.nn.Cell):
    latent_magnitude = 3
    latent_shift = 0.5

    def __init__(self, encoder_path=None, decoder_path=None, latent_channels=4):
        """Initialize pretrained TAESD on the given device from the given checkpoints."""
        super().__init__()
        self.taesd_encoder = Encoder(latent_channels=latent_channels)
        self.taesd_decoder = Decoder(latent_channels=latent_channels)
        self.vae_scale = mindspore.Parameter(mindspore.tensor(1.0))
        self.vae_shift = mindspore.Parameter(mindspore.tensor(0.0))
        if encoder_path is not None:
            self.taesd_encoder.load_state_dict(comfy.utils.load_mindspore_file(encoder_path, safe_load=True))
        if decoder_path is not None:
            self.taesd_decoder.load_state_dict(comfy.utils.load_mindspore_file(decoder_path, safe_load=True))

    @staticmethod
    def scale_latents(x):
        """raw latents -> [0, 1]"""
        return x.div(2 * TAESD.latent_magnitude).add(TAESD.latent_shift).clamp(0, 1)

    @staticmethod
    def unscale_latents(x):
        """[0, 1] -> raw latents"""
        return x.sub(TAESD.latent_shift).mul(2 * TAESD.latent_magnitude)

    def decode(self, x):
        x_sample = self.taesd_decoder((x - self.vae_shift) * self.vae_scale)
        x_sample = x_sample.sub(0.5).mul(2)
        return x_sample

    def encode(self, x):
        return (self.taesd_encoder(x * 0.5 + 0.5) / self.vae_scale) + self.vae_shift
