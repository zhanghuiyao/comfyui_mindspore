import mindspore
from mindspore import mint
import numpy as np
from functools import partial

from .util import extract_into_tensor, make_beta_schedule


class AbstractLowScaleModel(mindspore.nn.Cell):
    # for concatenating a downsampled image to the latent representation
    def __init__(self, noise_schedule_config=None):
        super(AbstractLowScaleModel, self).__init__()
        if noise_schedule_config is not None:
            self.register_schedule(**noise_schedule_config)

    def register_schedule(self, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                   cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_mindspore = partial(mindspore.tensor, dtype=mindspore.float32)

        self.register_buffer('betas', to_mindspore(betas))
        self.register_buffer('alphas_cumprod', to_mindspore(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_mindspore(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_mindspore(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_mindspore(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_mindspore(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_mindspore(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_mindspore(np.sqrt(1. / alphas_cumprod - 1)))

    def q_sample(self, x_start, t, noise=None, seed=None):
        if noise is None:
            if seed is None:
                noise = mint.randn_like(x_start)
            else:
                noise = mint.randn(x_start.size(), dtype=x_start.dtype, generator=mindspore.manual_seed(seed))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def construct(self, x):
        return x, None

    def decode(self, x):
        return x


class SimpleImageConcat(AbstractLowScaleModel):
    # no noise level conditioning
    def __init__(self):
        super(SimpleImageConcat, self).__init__(noise_schedule_config=None)
        self.max_noise_level = 0

    def construct(self, x):
        # fix to constant noise level
        return x, mint.zeros(x.shape[0]).long()


class ImageConcatWithNoiseAugmentation(AbstractLowScaleModel):
    def __init__(self, noise_schedule_config, max_noise_level=1000, to_cuda=False):
        super().__init__(noise_schedule_config=noise_schedule_config)
        self.max_noise_level = max_noise_level

    def construct(self, x, noise_level=None, seed=None):
        if noise_level is None:
            noise_level = mint.randint(0, self.max_noise_level, (x.shape[0],)).long()
        else:
            assert isinstance(noise_level, mindspore.Tensor)
        z = self.q_sample(x, noise_level, seed=seed)
        return z, noise_level



