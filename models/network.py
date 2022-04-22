import math
import torch
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm

from core.base_network import BaseNetwork
from .sr3_modules.unet import UNet
class Network(BaseNetwork):
    def __init__(self, unet, beta_schedule, **kwargs):
        super(Network, self).__init__(**kwargs)
        self.denoise_fn = UNet(**unet)
        self.beta_schedule = beta_schedule

    def set_loss(self, loss_fn):
        self.loss_fn = loss_fn

    def set_new_noise_schedule(self, device=torch.device('cuda'), phase='train'):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)

        betas = make_schedule(**self.beta_schedule[phase])
        alphas = 1. - betas
        alphas = alphas.detach().cpu().numpy() if isinstance(alphas, torch.Tensor) else alphas
        gammas = np.cumprod(alphas, axis=0)

        timesteps, = alphas.shape
        self.num_timesteps = int(timesteps)

        self.register_buffer('gammas', to_torch(gammas))
        self.register_buffer('sqrt_gammas', to_torch(np.sqrt(gammas)))
        self.register_buffer('sqrt_one_minus_gammas', to_torch(np.sqrt(1-gammas)))

        self.register_buffer('one_minus_alphas', to_torch(1-alphas))
        self.register_buffer('one_div_sqrt_alphas', to_torch(np.sqrt(1./alphas)))
        self.register_buffer('sqrt_one_minus_alphas', to_torch(np.sqrt(1-alphas)))


    @torch.no_grad()
    def restoration(self, y_cond, y_t=None, y_0=None, mask=None, sample_num=8):
        b, *_ = y_cond.shape
        sample_inter = (self.num_timesteps//sample_num)
        y_t = default(y_t, lambda: torch.randn_like(y_cond))
        ret_img = y_t
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            t = torch.full((b,), i, device=y_cond.device, dtype=torch.long)
            noise_pred = self.denoise_fn(torch.cat([y_cond, y_t], dim=1), extract(self.gammas, t))
            u_t = extract(self.one_div_sqrt_alphas, t) * (
                y_t - ((extract(self.one_minus_alphas, t) / extract(self.sqrt_one_minus_gammas, t)) * noise_pred)
            )
            if i>0:
                var_t =  extract(self.sqrt_one_minus_alphas, t) * torch.rand_like(y_cond)
                y_t = u_t + var_t
            else:
                y_t = u_t
            if mask is not None:
                y_t = y_0*(1.-mask) + mask*y_t
            if t % sample_inter == 0:
                ret_img = torch.cat([ret_img, y_t], dim=0)
        return ret_img

    
    def forward(self, y_0, y_cond, mask=None, noise=None):
        b, *_ = y_0.shape
        t = torch.randint(0, self.num_timesteps, (b,), device=y_0.device).long()
        noise = default(noise, lambda: torch.randn_like(y_0))

        y_noisy = extract(self.sqrt_gammas, t) * y_0 + extract(self.sqrt_one_minus_gammas, t) * noise
        if mask is not None:
            noise_pred = self.denoise_fn(torch.cat([y_cond, y_noisy*mask+(1.-mask)*y_0], dim=1), extract(self.gammas, t))
            loss = self.loss_fn(mask*noise, mask*noise_pred)
        else:
            noise_pred = self.denoise_fn(torch.cat([y_cond, y_noisy], dim=1), extract(self.gammas, t))
            loss = self.loss_fn(noise, noise_pred)
        return loss


def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def extract(a, t, x_shape=(1,1,1,1)):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def make_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end, n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup(linear_start, linear_end, n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup(linear_start, linear_end, n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep, 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas

def _warmup(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas