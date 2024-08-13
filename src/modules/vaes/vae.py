# MIT License
#
# Copyright (c) 2023 Christopher Friesen
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import Optional, Callable, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import numpy as np

from modules.module import DualDiffusionModule, DualDiffusionModuleConfig
from modules.mp_tools import MPConv, normalize, mp_silu
from modules.formats.format import DualDiffusionFormat

class IsotropicGaussianDistribution:

    def __init__(self, parameters: torch.Tensor, logvar: torch.Tensor, deterministic: bool = False) -> None:

        self.deterministic = deterministic
        self.parameters = self.mean = parameters
        self.logvar = torch.clamp(logvar, -30.0, 20.0)
        
        if self.deterministic:
            self.var = self.std = torch.zeros_like(
                self.mean, device=self.parameters.device, dtype=self.parameters.dtype
            )
        else:
            self.std = torch.exp(0.5 * self.logvar)
            self.var = torch.exp(self.logvar)
    
    def sample(self, noise_fn: Optional[Callable[..., torch.Tensor]] = None, **kwargs) -> torch.Tensor:
        noise_fn = noise_fn or torch.randn
        noise = noise_fn(self.mean.shape,
                         device=self.parameters.device,
                         dtype=self.parameters.dtype,
                         **kwargs)
        return self.mean + self.std * noise

    def mode(self) -> torch.Tensor:
        return self.mean
    
    def kl(self, other: Optional["IsotropicGaussianDistribution"] = None) -> torch.Tensor:
        if self.deterministic:
            return torch.Tensor([0.], device=self.parameters.device, dtype=self.parameters.dtype)
        
        reduction_dims = tuple(range(0, len(self.mean.shape)))
        if other is None:
            return 0.5 * torch.mean(torch.pow(self.mean, 2) + self.var - 1. - self.logvar, dim=reduction_dims)
        else:
            return 0.5 * torch.mean(
                (self.mean - other.mean).square() / other.var
                + self.var / other.var - 1. - self.logvar + other.logvar,
                dim=reduction_dims,
            )

@dataclass
class DualDiffusionVAEConfig(DualDiffusionModuleConfig, ABC):

    in_channels:     int = 2,
    out_channels:    int = 2,
    latent_channels: int = 4,
    label_dim:    int = 1,
    dropout:    float = 0.,
    target_snr: float = 16.,

class DualDiffusionVAE(DualDiffusionModule, ABC):

    module_name: str = "vae"

    @abstractmethod
    def __init__(self, config: DualDiffusionVAEConfig):
        super().__init__()
        self.config = config

    def get_class_embeddings(self, class_labels: torch.Tensor) -> torch.Tensor:
        return mp_silu(self.emb_label(normalize(class_labels).to(device=self.device, dtype=self.dtype)))

    def get_recon_loss_logvar(self) -> torch.Tensor:
        return self.recon_loss_logvar
    
    def get_target_snr(self) -> float:
        return self.config.target_snr
    
    def encode(self, x: torch.Tensor,
               class_embeddings: torch.Tensor,
               format: DualDiffusionFormat) -> IsotropicGaussianDistribution:
        
        x = torch.cat((x, torch.ones_like(x[:, :1]), format.get_ln_freqs(x)), dim=1)
        for name, block in self.enc.items():
            x = block(x) if 'conv' in name else block(x, class_embeddings, None, None)

        latents = self.conv_latents_out(x, gain=self.latents_out_gain)
        noise_logvar = torch.tensor(np.log(1 / (self.config.target_snr**2 + 1)), device=x.device, dtype=x.dtype)
        return IsotropicGaussianDistribution(latents, noise_logvar)
    
    def decode(self, x: torch.Tensor,
               class_embeddings: torch.Tensor,
               format: DualDiffusionFormat) -> torch.Tensor:
        
        x = torch.cat((x, torch.ones_like(x[:, :1]), format.get_ln_freqs(x)), dim=1)
        x = self.conv_latents_in(x)
        for _, block in self.dec.items():
            x = block(x, class_embeddings, None, None)

        return self.conv_out(x, gain=self.out_gain)
    
    def get_latent_shape(self, sample_shape: Union[torch.Size, tuple[int, int, int, int]]) -> torch.Size:
        if len(sample_shape) == 4:
            return (sample_shape[0],
                    self.config.latent_channels,
                    sample_shape[2] // 2 ** (self.config.num_levels-1),
                    sample_shape[3] // 2 ** (self.config.num_levels-1))
        else:
            raise ValueError(f"Invalid sample shape: {sample_shape}")
    
    @torch.no_grad()
    def normalize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, MPConv):
                module.normalize_weights()