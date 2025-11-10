from ..diffusionmodules.upscaling import ImageConcatWithNoiseAugmentation
from ..diffusionmodules.openaimodel import Timestep
import mindspore
from mindspore import mint

class CLIPEmbeddingNoiseAugmentation(ImageConcatWithNoiseAugmentation):
    def __init__(self, *args, clip_stats_path=None, timestep_dim=256, **kwargs):
        super().__init__(*args, **kwargs)
        if clip_stats_path is None:
            clip_mean, clip_std = mint.zeros(timestep_dim), mint.ones(timestep_dim)
        else:
            # clip_mean, clip_std = torch.load(clip_stats_path, map_location="cpu")
            raise NotImplementedError
        self.register_buffer("data_mean", clip_mean[None, :], persistent=False)
        self.register_buffer("data_std", clip_std[None, :], persistent=False)
        self.time_embed = Timestep(timestep_dim)

    def scale(self, x):
        # re-normalize to centered mean and unit variance
        x = (x - self.data_mean) * 1. / self.data_std
        return x

    def unscale(self, x):
        # back to original data stats
        x = (x * self.data_std) + self.data_mean
        return x

    def construct(self, x, noise_level=None, seed=None):
        if noise_level is None:
            noise_level = mint.randint(0, self.max_noise_level, (x.shape[0],)).long()
        else:
            assert isinstance(noise_level, mindspore.Tensor)
        x = self.scale(x)
        z = self.q_sample(x, noise_level, seed=seed)
        z = self.unscale(z)
        noise_level = self.time_embed(noise_level)
        return z, noise_level
