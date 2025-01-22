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

from typing import Optional, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from numpy import ndarray

from modules.module import DualDiffusionModule, DualDiffusionModuleConfig
from modules.formats.format import DualDiffusionFormat
from utils.dual_diffusion_utils import tensor_to_img

class LatentsDistribution(ABC):
    
    @abstractmethod
    def sample(self) -> torch.Tensor:
        pass

    @abstractmethod
    def mode(self) -> torch.Tensor:
        pass
    
    @abstractmethod
    def kl(self, other: Optional["LatentsDistribution"] = None) -> torch.Tensor:
        pass

class IsotropicGaussianDistribution(LatentsDistribution):

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
    
    def sample(self) -> torch.Tensor:
        return self.mean + self.std * torch.randn_like(self.mean)

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

class DegenerateDistribution(LatentsDistribution):

    def __init__(self, parameters: torch.Tensor) -> None:
        self.parameters = self.mean = parameters
    
    def sample(self) -> torch.Tensor:
        return self.mean

    def mode(self) -> torch.Tensor:
        return self.mean
    
    def kl(self, other: Optional["IsotropicGaussianDistribution"] = None) -> torch.Tensor:

        reduction_dims = tuple(range(0, len(self.mean.shape)))
        if other is None:
            return 0.5 * torch.mean(torch.pow(self.mean, 2), dim=reduction_dims)
        else:
            return 0.5 * torch.mean((self.mean - other.mean).square(), dim=reduction_dims)
        
@dataclass
class DualDiffusionVAEConfig(DualDiffusionModuleConfig, ABC):

    in_channels: int     = 2
    in_num_freqs: int    = 256
    in_channels_emb: int = 512
    out_channels: int    = 2
    latent_channels: int = 4
    dropout: float       = 0.

    latents_img_channel_order: Optional[tuple[int]] = None

class DualDiffusionVAE(DualDiffusionModule, ABC):

    module_name: str = "vae"
    
    @abstractmethod
    def get_embeddings(self, emb_in: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def get_recon_loss_logvar(self) -> torch.Tensor:
        pass
    
    @abstractmethod
    def get_latent_shape(self, sample_shape: Union[torch.Size, tuple[int, int, int, int]]) -> torch.Size:
        pass

    @abstractmethod
    def get_sample_shape(self, latent_shape: Union[torch.Size, tuple[int, int, int, int]]) -> torch.Size:
        pass

    @abstractmethod
    def encode(self, x: torch.Tensor,
               embeddings: torch.Tensor,
               format: DualDiffusionFormat) -> LatentsDistribution:
        pass
    
    @abstractmethod
    def decode(self, x: torch.Tensor,
               embeddings: torch.Tensor,
               format: DualDiffusionFormat) -> torch.Tensor:
        pass

    def compile(self, **kwargs) -> None:
        if type(self).supports_compile == True:
            super().compile(**kwargs)
            self.encode = torch.compile(self.encode, **kwargs)
            self.decode = torch.compile(self.decode, **kwargs)

    def latents_to_img(self, latents: torch.Tensor) -> ndarray:
        if latents.shape[1] == 8:
            latents = torch.cat((latents[:, 0::2], latents[:, 1::2]), dim=2)
        elif latents.shape[1] > 4:
            raise ValueError(f"Error: latents_to_img shape has invalid num channels: {latents.shape}")
        
        return tensor_to_img(latents, flip_y=True, channel_order=self.config.latents_img_channel_order)