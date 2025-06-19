# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Improved diffusion model architecture proposed in the paper
"Analyzing and Improving the Training Dynamics of Diffusion Models"."""

# Modifications under MIT License
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

from dataclasses import dataclass
from typing import Union, Literal, Optional

import torch

from modules.unets.unet import DualDiffusionUNet, DualDiffusionUNetConfig
from modules.mp_tools import MPFourier, mp_silu, normalize, mp_sum, mp_cat
from modules.formats.raw import RawFormat
from modules.daes.dae_edm2_k1 import MPConv2D
from utils.resample import FilteredDownsample1D, FilteredUpsample1D
from utils.resample import FilteredDownsample2D, FilteredUpsample2D


@dataclass
class DDec_UNet_K1_Config(DualDiffusionUNetConfig):

    in_channels: int     = 4
    out_channels: int    = 4
    in_channels_emb: int = 0 # unused

    sigma_max: float = 12
    sigma_min: float = 0.00008
    in_num_freqs: int = 1
    
    resample_beta: float = 3.437
    resample_k_size: int = 23
    resample_factor: int = 2
    downsample_type: Literal["1d", "2d"] = "2d"

    model_channels: int   = 32
    logvar_channels: int  = 192
    channel_mult_emb: int = 4
    channel_mult_enc: list[int] = (1,2,3,4)
    channel_mult_dec: list[int] = (1,2,3,4)
    num_layers_per_block: list[int] = (3,3,3,3)
    kernel_enc: list[int] = (3,3)
    kernel_dec: list[int] = (3,3)
    mlp_multiplier: int = 1
    mlp_groups: int     = 1

    label_balance: float = 0.5
    cat_balance: float   = 0.5
    res_balance: float   = 0.3

class Block2D(torch.nn.Module):

    def __init__(self,
        level: int,                             # Resolution level.
        in_channels: int,                       # Number of input channels.
        out_channels: int,                      # Number of output channels.
        label_channels: int,
        emb_channels: int,                      # Number of embedding channels.
        flavor: Literal["enc", "dec"] = "enc",
        resample: Optional[torch.nn.Module] = None,
        res_balance: float     = 0.3,
        clip_act: float        = 256,      # Clip output activations. None = do not clip.
        mlp_multiplier: int    = 1,        # Multiplier for the number of channels in the MLP.
        mlp_groups: int        = 1,        # Number of groups for the MLP.
        kernel: tuple[int, int] = (3,3),   # Kernel size for the convolutional layers.
    ) -> None:
        super().__init__()

        self.level = level
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.flavor = flavor
        self.res_balance = res_balance
        self.clip_act = clip_act

        self.resample = resample if resample is not None else torch.nn.Identity()
            
        self.conv_res0 = MPConv2D(out_channels if flavor == "enc" else in_channels,
                    out_channels * mlp_multiplier, kernel=kernel, groups=mlp_groups)
        self.conv_res1 = MPConv2D(out_channels * mlp_multiplier, out_channels, kernel=kernel, groups=mlp_groups)        

        if in_channels != out_channels or mlp_groups > 1:
            self.conv_skip = MPConv2D(in_channels, out_channels, kernel=(1,1), groups=1)
        else:
            self.conv_skip = None

        self.emb_gain = torch.nn.Parameter(torch.zeros([])) if emb_channels != 0 else None
        self.emb_linear = MPConv2D(emb_channels, out_channels * mlp_multiplier,
            kernel=(1,1), groups=1) if emb_channels != 0 else None
        
        if label_channels > 0:
            self.emb_label = MPConv2D(label_channels, emb_channels, kernel=(1,1))
            self.u_embedding = torch.nn.Parameter(torch.zeros(1, emb_channels, 1, 1))
        else:
            self.emb_label = None
            self.u_embedding = None
    
    def get_embeddings(self, emb_in: torch.Tensor, conditioning_mask: torch.Tensor) -> torch.Tensor:
        c_embedding: torch.Tensor = self.emb_label(emb_in)
        return torch.where(conditioning_mask, c_embedding, self.u_embedding.to(dtype=torch.bfloat16))
        
    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        
        x = self.resample(x)
        
        if self.flavor == "enc":
            if self.conv_skip is not None:
                x = self.conv_skip(x)
            x = normalize(x, dim=1) # pixel norm

        y = self.conv_res0(mp_silu(x))

        if self.emb_linear is not None:
            c = self.emb_linear(emb, gain=self.emb_gain) + 1.
            y = mp_silu(y * c)
        else:
            y = mp_silu(y)

        y = self.conv_res1(y)
        
        if self.flavor == "dec" and self.conv_skip is not None:
            x = self.conv_skip(x)
        
        x = mp_sum(x, y, self.res_balance)
        
        if self.clip_act is not None:
            x = x.clip_(-self.clip_act, self.clip_act)

        return x
    
class DDec_UNet_K1(DualDiffusionUNet):

    def __init__(self, config: DDec_UNet_K1_Config) -> None:
        super().__init__()
        self.config = config

        block_kwargs = {"mlp_multiplier": config.mlp_multiplier,
                        "mlp_groups": config.mlp_groups,
                        "res_balance": config.res_balance}
        
        enc_channels = [config.model_channels * m for m in config.channel_mult_enc]
        dec_channels = [config.model_channels * m for m in config.channel_mult_dec]

        cemb = config.model_channels * config.channel_mult_emb

        self.num_levels = len(config.channel_mult_dec)
        self.total_downsample_ratio = config.resample_factor ** (self.num_levels - 1)

        # embedding
        self.emb_fourier = MPFourier(cemb)
        self.emb_noise = MPConv2D(cemb, cemb, kernel=())

        # training uncertainty estimation
        self.logvar_fourier = MPFourier(config.logvar_channels)
        self.logvar_linear = MPConv2D(config.logvar_channels, 1, kernel=(), disable_weight_norm=True)

        assert len(enc_channels) == len(dec_channels) == len(config.num_layers_per_block)

        if config.downsample_type == "1d":
            downsample_cls = FilteredDownsample1D
            upsample_cls = FilteredUpsample1D
        else:
            downsample_cls = FilteredDownsample2D
            upsample_cls = FilteredUpsample2D

        self.downsample = downsample_cls(k_size=config.resample_k_size,
                        beta=config.resample_beta, factor=config.resample_factor)
        self.upsample = upsample_cls(
            k_size=config.resample_k_size * config.resample_factor + config.resample_k_size % config.resample_factor,
            beta=config.resample_beta, factor=config.resample_factor)

        # encoder
        self.enc = torch.nn.ModuleDict()
        cout = enc_channels[0]

        self.conv_in = MPConv2D(config.in_channels + 1, cout, kernel=config.kernel_enc)

        for level, channels in enumerate(enc_channels):
    
            clabel = channels if level == 0 else 0

            if level == 0:
                self.enc[f"block{level}_in"] = Block2D(
                    level, cout, channels, clabel, cemb, flavor="enc", kernel=config.kernel_enc, **block_kwargs)
            else:
                self.enc[f"block{level}_down"] = Block2D(level, cout, channels, clabel, cemb,
                    flavor="enc", resample=self.downsample, kernel=config.kernel_enc, **block_kwargs)
            
            for idx in range(config.num_layers_per_block[level]):
                self.enc[f"block{level}_layer{idx}"] = Block2D(
                    level, channels, channels, clabel, cemb, flavor="enc", kernel=config.kernel_enc, **block_kwargs)
            
            cout = channels

        # decoder
        self.dec = torch.nn.ModuleDict()
        skips = [block.out_channels for block in self.enc.values()]
        cout = enc_channels[-1]

        for level in reversed(range(0, self.num_levels)):
            channels = dec_channels[level]
            clabel = channels if level == 0 else 0

            if level == self.num_levels - 1:
                self.dec[f"block{level}_in"] = Block2D(
                    level, cout, channels, clabel, cemb, flavor="dec", **block_kwargs, kernel=config.kernel_dec)
            else:
                self.dec[f"block{level}_up"] = Block2D(level, cout , channels, clabel,
                    cemb, flavor="dec", resample=self.upsample, **block_kwargs, kernel=config.kernel_dec)
            
            for idx in range(config.num_layers_per_block[level] + 1):
                self.dec[f"block{level}_layer{idx}"] = Block2D(
                    level, channels + skips.pop(), channels, clabel, cemb, flavor="dec", **block_kwargs, kernel=config.kernel_dec)

            cout = channels

        self.conv_out = MPConv2D(cout, config.out_channels, kernel=config.kernel_dec)
        self.conv_out_gain = torch.nn.Parameter(torch.zeros([]))

    def get_embeddings(self, emb_in: list[torch.Tensor], conditioning_mask: torch.Tensor) -> list[torch.Tensor]:
        
        conditioning_mask = conditioning_mask[:, None, None, None]
        embeddings = []

        for name, block in self.enc.items():
            
            if block.emb_label is not None:
                embeddings.append(block.get_embeddings(
                    emb_in.to(dtype=torch.bfloat16), conditioning_mask))

        for name, block in self.dec.items():

            if block.emb_label is not None:
                embeddings.append(block.get_embeddings(
                    emb_in.to(dtype=torch.bfloat16), conditioning_mask))

        embeddings.reverse()
        return embeddings

    def get_sigma_loss_logvar(self, sigma: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.logvar_linear(self.logvar_fourier(sigma.flatten().log() / 4)).view(-1, 1, 1, 1).float()
    
    def get_latent_shape(self, latent_shape: Union[torch.Size, tuple[int, int, int, int]]) -> torch.Size:
        raise NotImplementedError()
        return latent_shape[0:2] + ((latent_shape[2] // 2**(self.num_levels-1)) * 2**(self.num_levels-1),
                                    (latent_shape[3] // 2**(self.num_levels-1)) * 2**(self.num_levels-1))
  
    def forward(self, x_in: torch.Tensor,
                sigma: torch.Tensor,
                format: RawFormat,
                embeddings: list[torch.Tensor],
                x_ref: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        with torch.no_grad():
            sigma = sigma.view(-1, 1, 1, 1)
            
            # preconditioning weights
            c_skip = self.config.sigma_data ** 2 / (sigma ** 2 + self.config.sigma_data ** 2)
            c_out = sigma * self.config.sigma_data / (sigma ** 2 + self.config.sigma_data ** 2).sqrt()
            c_in = 1 / (self.config.sigma_data ** 2 + sigma ** 2).sqrt()
            c_noise = (sigma.flatten().log() / 4).to(self.dtype)

            x = (c_in * x_in).to(dtype=torch.bfloat16)

        embeddings = [*embeddings]

        # embedding
        emb_noise = self.emb_noise(self.emb_fourier(c_noise))[:, :, None, None].to(dtype=torch.bfloat16)
        
        x = self.conv_in(torch.cat((x, torch.ones_like(x[:, :1])), dim=1))
        skips = []
        
        for name, block in self.enc.items():
            
            if block.emb_label is not None:               
                emb = mp_silu(mp_sum(emb_noise, embeddings.pop().to(dtype=torch.bfloat16), t=self.config.label_balance))
            else:
                emb = emb_noise

            #print(name, x.shape, emb.shape)
            x = block(x, emb)
            skips.append(x)

        for name, block in self.dec.items():
            
            if block.emb_label is not None:
                emb = mp_silu(mp_sum(emb_noise, embeddings.pop().to(dtype=torch.bfloat16), t=self.config.label_balance))
            else:
                emb = emb_noise
            
            #print(name, x.shape, emb.shape, skips[-1].shape if "layer" in name else "")

            if "layer" in name:
                x = mp_cat(x, skips.pop(), t=self.config.cat_balance)

            x = block(x, emb)

        x: torch.Tensor = self.conv_out(x, gain=self.conv_out_gain)
        D_x: torch.Tensor = c_skip * x_in.float() + c_out * x.float()

        return D_x