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
from utils.dual_diffusion_utils import tensor_to_img


def top_pca_components(x: torch.Tensor, n_pca: int = 4) -> torch.Tensor:
    B, C, H, W = x.shape

    result = torch.zeros((B, n_pca, H, W), device=x.device, dtype=x.dtype)

    for b in range(B):
        # reshape to (H*W, C) for this batch item
        x_b = x[b].reshape(C, H*W).transpose(0, 1)  # shape: (H*W, C)
        
        # center the data
        mean_b = torch.mean(x_b, dim=0, keepdim=True)
        x_b_centered = x_b - mean_b
        
        # do PCA using torch.pca_lowrank
        U, S, V = torch.pca_lowrank(x_b_centered, q=n_pca)

        # project data onto the top n_pca principal components
        projected_data = torch.matmul(x_b_centered, V)  # shape: (H*W, n_pca)

        # reshape back to (n_pca, H, W)
        result[b] = projected_data.transpose(0, 1).reshape(n_pca, H, W)
    
    return result

def pca_align_ref(pca: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    
    ref = torch.nn.functional.interpolate(ref, pca.shape[2:], mode="area").mean(dim=1, keepdim=True)

    alignment0 = ( pca * ref).mean(dim=(2,3), keepdim=True)
    alignment1 = (-pca * ref).mean(dim=(2,3), keepdim=True)

    return torch.where(alignment0 > alignment1, -pca, pca)

@dataclass
class DualDiffusionDAEConfig(DualDiffusionModuleConfig, ABC):

    in_channels: int     = 2
    in_channels_emb: int = 1024
    in_num_freqs: int    = 256
    out_channels: int    = 2
    latent_channels: int = 4

    latents_img_split_stereo: bool = True
    latents_img_use_pca: bool      = True
    latents_img_channel_order: Optional[tuple[int]] = (1, 3, 2, 0)
    latents_img_flip_stereo: bool = False
    latents_img_brightness: Optional[float] = 1
    latents_img_contrast: Optional[float] = 0.94
    latents_img_saturation: Optional[float] = 1.13
    latents_img_gamma: Optional[float] = 1.46

class DualDiffusionDAE(DualDiffusionModule, ABC):

    module_name: str = "dae"
    
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
    def get_mel_spec_shape(self, latent_shape: Union[torch.Size, tuple[int, int, int, int]]) -> torch.Size:
        pass

    @abstractmethod
    def encode(self, x: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        pass
    
    @abstractmethod
    def decode(self, x: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        pass
    
    def tiled_encode(self, x: torch.Tensor, embeddings: torch.Tensor, max_chunk: int = 6144, overlap: int = 256) -> torch.Tensor:
        raise NotImplementedError(f"tiled_encode not implemented for {type(self).__name__}")
    
    def tiled_decode(self, x: torch.Tensor, embeddings: torch.Tensor, max_chunk: int = 6144, overlap: int = 256) -> torch.Tensor:
        raise NotImplementedError(f"tiled_decode not implemented for {type(self).__name__}")
    
    @abstractmethod
    def forward(self, samples: torch.Tensor, embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass

    def compile(self, **kwargs) -> None:
        if type(self).supports_compile == True:
            super().compile(**kwargs)
            self.encode = torch.compile(self.encode, **kwargs)
            self.decode = torch.compile(self.decode, **kwargs)
            self.forward = torch.compile(self.forward, **kwargs)

    def latents_to_img(self, latents: torch.Tensor, img_split_stereo: Optional[bool] = None, align_ref: Optional[torch.Tensor] = None) -> ndarray:
        
        if img_split_stereo is None:
            img_split_stereo = self.config.latents_img_split_stereo
        if img_split_stereo == True:
            if self.config.latents_img_flip_stereo == True:
                latents = latents.clone().detach()
                latents[:, 1::2] = latents[:, 1::2].flip(dims=(2,))

            latents = torch.cat((latents[:, 0::2], latents[:, 1::2]), dim=2)
        
        if self.config.latents_img_use_pca == True:
            pca = top_pca_components(latents.float())
            if align_ref is not None:
                pca = pca_align_ref(pca, align_ref)
            latents = pca
        
        return tensor_to_img(latents, flip_y=True, channel_order=self.config.latents_img_channel_order,
            brightness=self.config.latents_img_brightness, contrast=self.config.latents_img_contrast,
            saturation=self.config.latents_img_saturation, gamma=self.config.latents_img_gamma)
        
    def ddec_cond_to_img(self, ddec_cond: torch.Tensor, align_ref: Optional[torch.Tensor] = None) -> ndarray:
        
        if ddec_cond.shape[1] > 3:
            pca = top_pca_components(ddec_cond.float())
            if align_ref is not None:
                pca = pca_align_ref(pca, align_ref)
            return tensor_to_img(pca, flip_y=True, channel_order=self.config.latents_img_channel_order)
        else:
            return tensor_to_img(ddec_cond, flip_y=True)