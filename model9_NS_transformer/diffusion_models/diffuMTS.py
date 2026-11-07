"""D3U diffusion schedule and PatchDN denoiser used by the main experiment."""

import argparse
import torch
from torch import nn
import yaml
from model9_NS_transformer.diffusion_models.diffusion_utils import make_beta_schedule
from model9_NS_transformer.denoise_models.PatchDN import PatchDN


def dict2namespace(config):
    return argparse.Namespace(**{key: dict2namespace(value) if isinstance(value, dict) else value
                                 for key, value in config.items()})


def extract(input, t, x):
    out = torch.gather(input, 0, t.to(input.device))
    return out.reshape(t.shape[0], *([1] * (x.ndim - 1)))


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        with open(configs.diffusion_config_dir, "r") as f:
            config = yaml.safe_load(f)
            diffusion_config = dict2namespace(config)

        diffusion_config.diffusion.timesteps = configs.timesteps
        self.args = configs
        self.diffusion_config = diffusion_config

        self.model_var_type = diffusion_config.model.var_type
        self.num_timesteps = diffusion_config.diffusion.timesteps
        self.dataset_object = None

        #beta_schedule: linear ,beta_start: 0.0001, beta_end: 0.02
        betas = make_beta_schedule(schedule=diffusion_config.diffusion.beta_schedule, num_timesteps=self.num_timesteps,
                                   start=diffusion_config.diffusion.beta_start, end=diffusion_config.diffusion.beta_end)
        betas = self.betas = betas.float()
        self.betas=betas
        self.betas_sqrt = torch.sqrt(betas)
        alphas = 1.0 - betas
        self.alphas = alphas
        self.one_minus_betas_sqrt = torch.sqrt(alphas)
        alphas_cumprod = alphas.cumprod(dim=0)
        self.alphas_cumprod= alphas_cumprod
        self.alphas_bar_sqrt = torch.sqrt(alphas_cumprod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_cumprod)
        if diffusion_config.diffusion.beta_schedule == "cosine":
            self.one_minus_alphas_bar_sqrt *= 0.9999  # avoid division by 0 for 1/sqrt(alpha_bar_t) during inference
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1), alphas_cumprod[:-1]], dim=0
        )
        self.alphas_cumprod=alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev

        self.sqrt_one_minus_betas_sqrt = torch.sqrt(1./alphas)
        self.sqrt_recip_alphas_cumprod =torch.sqrt(1. / alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod=torch.sqrt(1. / alphas_cumprod - 1)

        self.posterior_mean_coef1 = (
                betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
                torch.sqrt(alphas) * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        )
        self.posterior_mean_coef3 = (
                betas /torch.sqrt(1 - alphas_cumprod) / torch.sqrt( alphas)
        )

        posterior_variance = (
                betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_log_variance_clipped=torch.log(posterior_variance.clamp(min=1e-20))

        self.posterior_variance = posterior_variance
        self.logvar = betas.log()



        self.tau = None  # precision fo test NLL computation

        if self.args.denoise_model != 'PatchDN':
            raise ValueError('This release includes PatchDN.')
        self.diffussion_model = PatchDN(MTS_args=self.args, depth=self.args.depth, mlp_ratio=1.0)

    
        a = 0


    def q_sample(self, y, t, noise=None):
        #y=y_0(真实), y_T_mean（NSformer学出来的）, self.model.alphas_bar_sqrt,self.model.one_minus_alphas_bar_sqrt, t（时间步骤), noise=e（高斯噪声）
        """
        y_0_hat: prediction of pre-trained guidance model; can be extended to represent
            any prior mean setting at timestep T.
        """
        #y[batch,len,var]
        if noise is None:
            noise = torch.randn_like(y)
        sqrt_alpha_bar_t = extract(self.alphas_bar_sqrt, t, y).to(y.device)
        
        sqrt_one_minus_alpha_bar_t = extract(self.one_minus_alphas_bar_sqrt, t, y).to(y.device)
        
        # q(y_t | y_0, x)
        y_t = sqrt_alpha_bar_t * y + sqrt_one_minus_alpha_bar_t * noise

        return y_t


    def forward(self, y_t, t, enc_out):
        return self.diffussion_model(y_t, t, enc_out)
