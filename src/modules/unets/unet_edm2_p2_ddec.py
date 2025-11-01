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
from typing import Union, Optional, Literal

import torch

from modules.unets.unet import DualDiffusionUNet, DualDiffusionUNetConfig
from modules.mp_tools import MPConv, MPFourier, mp_silu, mp_sum, normalize, mp_cat, resample_2d
from modules.formats.format import DualDiffusionFormat


@dataclass
class UNetConfig(DualDiffusionUNetConfig):

    in_channels:  int = 1
    out_channels: int = 1
    in_channels_emb: int = 0
    in_channels_x_ref: int = 2048
    in_num_freqs: int = 256

    sigma_max: float  = 11.
    sigma_min: float  = 0.0002
    sigma_data: float = 1.

    mp_fourier_ln_sigma_offset: float = -0.2
    mp_fourier_bandwidth:       float = 1

    model_channels: int     = 32             # Base multiplier for the number of channels.
    logvar_channels: int    = 192            # Number of channels for training uncertainty estimation.
    channel_mult: list[int] = (1,2,3,4,5,6)  # Per-resolution multipliers for the number of channels.
    channel_mult_noise: Optional[int] = 6    # Multiplier for noise embedding dimensionality.
    channel_mult_emb: Optional[int]   = 6    # Multiplier for final embedding dimensionality.
    num_layers_per_block: int = 3            # Number of resnet blocks per resolution.
    label_balance: float      = 0.5          # Balance between noise embedding (0) and class embedding (1).
    concat_balance: float     = 0.5          # Balance between skip connections (0) and main path (1).
    res_balance: float        = 0.3          # Balance between main branch (0) and residual branch (1).
    stereo_balance: float     = 0.3          
    mlp_multiplier: int       = 1            # Multiplier for the number of channels in the MLP.
    mlp_groups: int           = 1            # Number of groups for the MLPs.
    emb_linear_groups: int    = 1

class Block(torch.nn.Module):

    def __init__(self,
        level: int,                        # Resolution level.
        in_channels: int,                  # Number of input channels.
        out_channels: int,                 # Number of output channels.
        emb_channels: int,                 # Number of embedding channels.
        num_freqs: int,                    # Number of frequencies in x_ref.
        flavor: Literal["enc", "dec"] = "enc",
        resample_mode: Literal["keep", "up", "down"] = "keep",
        dropout: float         = 0.,       # Dropout probability.
        res_balance: float     = 0.3,      # Balance between main branch (0) and residual branch (1).
        stereo_balance: float  = 0.3,
        clip_act: float        = 256,      # Clip output activations. None = do not clip.
        mlp_multiplier: int    = 1,        # Multiplier for the number of channels in the MLP.
        mlp_groups: int        = 1,        # Number of groups for the MLP.
        emb_linear_groups: int = 1
    ) -> None:
        super().__init__()

        self.level = level
        self.out_channels = out_channels
        self.dropout = dropout
        self.res_balance = res_balance
        self.stereo_balance = stereo_balance
        self.clip_act = clip_act
        self.flavor = flavor
        self.resample_mode = resample_mode
        self.num_freqs = num_freqs
        
        inner_channels = out_channels * mlp_multiplier

        if in_channels != out_channels:
            self.conv_skip = MPConv(in_channels, out_channels, kernel=(1,1), groups=1)
        else:
            self.conv_skip = None
            
        self.conv_res0 = MPConv(out_channels if flavor == "enc" else in_channels,
                                inner_channels, kernel=(3,3), groups=mlp_groups)
        self.conv_res1 = MPConv(inner_channels, out_channels, kernel=(3,3), groups=mlp_groups)
        self.conv_stereo = MPConv(inner_channels, inner_channels, kernel=(1,1), groups=1)
        
        self.emb_gain = torch.nn.Parameter(torch.zeros([]))
        self.emb_linear = MPConv(emb_channels, inner_channels,
            kernel=(1,1), groups=emb_linear_groups)

    def forward(self, x0: torch.Tensor, x1: torch.Tensor, emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        
        x0 = resample_2d(x0, mode=self.resample_mode)
        x1 = resample_2d(x1, mode=self.resample_mode)

        if self.flavor == "enc":

            if self.conv_skip is not None:
                x0 = self.conv_skip(x0)
                x1 = self.conv_skip(x1)

            x0 = normalize(x0, dim=1)
            x1 = normalize(x1, dim=1)

        y0 = self.conv_res0(mp_silu(x0))
        y1 = self.conv_res0(mp_silu(x1))

        z0 = self.conv_stereo(y0)
        z1 = self.conv_stereo(y1)
        y0 = mp_sum(y0, z1, t=self.stereo_balance)
        y1 = mp_sum(y1, z0, t=self.stereo_balance)

        c = self.emb_linear(emb, gain=self.emb_gain) + 1.
        y0 = mp_silu(y0 * c)
        y1 = mp_silu(y1 * c)

        y0 = self.conv_res1(y0)
        y1 = self.conv_res1(y1)

        if self.flavor == "dec" and self.conv_skip is not None:
            x0 = self.conv_skip(x0)
            x1 = self.conv_skip(x1)

        x0 = mp_sum(x0, y0, t=self.res_balance)
        x1 = mp_sum(x1, y1, t=self.res_balance)
        
        if self.clip_act is not None:
            x0 = x0.clip_(-self.clip_act, self.clip_act)
            x1 = x1.clip_(-self.clip_act, self.clip_act)

        return x0, x1

class UNet(DualDiffusionUNet):

    def __init__(self, config: UNetConfig) -> None:
        super().__init__()
        self.config = config

        block_kwargs = {"dropout": config.dropout,
                        "mlp_multiplier": config.mlp_multiplier,
                        "mlp_groups": config.mlp_groups,
                        "emb_linear_groups": config.emb_linear_groups,
                        "res_balance": config.res_balance,
                        "stereo_balance": config.stereo_balance}

        cblock = [config.model_channels * x for x in config.channel_mult]
        cnoise = config.model_channels * config.channel_mult_noise if config.channel_mult_noise is not None else max(cblock)
        cemb = config.model_channels * config.channel_mult_emb if config.channel_mult_emb is not None else max(cblock)
        cdata = config.in_channels

        assert config.in_channels_x_ref % config.in_num_freqs == 0
        self.ref_channels_per_freq = config.in_channels_x_ref // config.in_num_freqs

        self.num_levels = len(config.channel_mult)
        self.downsample_ratio = 2 ** (self.num_levels - 1)

        # Embedding.
        self.emb_fourier = MPFourier(cnoise, bandwidth=config.mp_fourier_bandwidth)
        self.emb_noise = MPConv(cnoise, cemb, kernel=())

        if config.in_channels_emb > 0:
            self.emb_label = MPConv(config.in_channels_emb, cemb, kernel=())
            self.emb_label_unconditional = MPConv(1, cemb, kernel=())
        else:
            self.emb_label = None
            self.emb_label_unconditional = None

        # Training uncertainty estimation.
        self.logvar_fourier = MPFourier(config.logvar_channels)
        self.logvar_linear = MPConv(config.logvar_channels, 1, kernel=(), disable_weight_norm=True)
        self.logvar_linear.weight.data.fill_(0)

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = cdata + self.ref_channels_per_freq + 1

        for level, channels in enumerate(cblock):
            
            num_freqs = config.in_num_freqs // 2**level

            if level == 0:
                cin = cout
                cout = channels
                self.enc[f"conv_in"] = MPConv(cin, cout, kernel=(3,3))
            else:
                self.enc[f"block{level}_down"] = Block(
                    level, cout, cout, cemb, num_freqs, flavor="enc", resample_mode="down", **block_kwargs)
            
            for idx in range(config.num_layers_per_block):
                cin = cout
                cout = channels
                self.enc[f"block{level}_layer{idx}"] = Block(
                    level, cin, cout, cemb, num_freqs, flavor="enc", **block_kwargs)
                
        # Decoder.
        self.dec = torch.nn.ModuleDict()
        skips = [block.out_channels for block in self.enc.values()]
        
        for level, channels in reversed(list(enumerate(cblock))):
            
            num_freqs = config.in_num_freqs // 2**level

            if level == len(cblock) - 1:
                self.dec[f"block{level}_in0"] = Block(level, cout, cout, cemb, num_freqs, flavor="dec", **block_kwargs)
                self.dec[f"block{level}_in1"] = Block(level, cout, cout, cemb, num_freqs, flavor="dec", **block_kwargs)
            else:
                self.dec[f"block{level}_up"] = Block(level, cout, cout, cemb, num_freqs, flavor="dec", resample_mode="up", **block_kwargs)

            for idx in range(config.num_layers_per_block + 1):
                cin = cout + skips.pop()
                cout = channels
                self.dec[f"block{level}_layer{idx}"] = Block(level, cin, cout, cemb, num_freqs, flavor="dec", **block_kwargs)
                
        self.out_gain = torch.nn.Parameter(torch.zeros([]))
        self.conv_out = MPConv(cout, config.out_channels, kernel=(3,3))

    def get_embeddings(self, emb_in: torch.Tensor, conditioning_mask: torch.Tensor) -> torch.Tensor:
        if self.config.in_channels_emb > 0:
            u_embedding = self.emb_label_unconditional(torch.ones(1, device=self.device, dtype=self.dtype))
            c_embedding = self.emb_label(normalize(emb_in).to(device=self.device, dtype=self.dtype))
            return mp_sum(u_embedding, c_embedding, t=conditioning_mask.unsqueeze(1).to(self.device, self.dtype))
        else:
            return None
        
    def get_sigma_loss_logvar(self, sigma: Optional[torch.Tensor] = None) -> torch.Tensor:
        ln_sigma = sigma.flatten().log() - self.config.mp_fourier_ln_sigma_offset
        return self.logvar_linear(self.logvar_fourier(ln_sigma / 4)).view(-1, 1, 1, 1).float()
    
    def get_latent_shape(self, latent_shape: Union[torch.Size, tuple[int, int, int, int]]) -> torch.Size:
        return latent_shape
    
    def forward(self, x_in: torch.Tensor,
                sigma: torch.Tensor,
                format: DualDiffusionFormat,
                embeddings: torch.Tensor,
                x_ref: Optional[torch.Tensor] = None,
                perturbed_input: Optional[torch.Tensor] = None) -> torch.Tensor:

        with torch.no_grad():
            sigma = sigma.view(-1, 1, 1, 1)
            
            # Preconditioning weights.
            c_skip = self.config.sigma_data ** 2 / (sigma ** 2 + self.config.sigma_data ** 2)
            c_out = sigma * self.config.sigma_data / (sigma ** 2 + self.config.sigma_data ** 2).sqrt()
            c_in = 1 / (self.config.sigma_data ** 2 + sigma ** 2).sqrt()
            ln_sigma = sigma.flatten().log() - self.config.mp_fourier_ln_sigma_offset
            c_noise = ln_sigma / 4

            if perturbed_input is not None:
                x = (c_in * perturbed_input).to(dtype=torch.bfloat16)
            else:
                x = (c_in * x_in).to(dtype=torch.bfloat16)

        # Embedding.
        emb: torch.Tensor = self.emb_noise(self.emb_fourier(c_noise))
        if self.config.in_channels_emb > 0:
            emb = mp_silu(mp_sum(emb, embeddings.to(dtype=emb.dtype), t=self.config.label_balance))
        emb = emb[:, :, None, None].to(dtype=torch.bfloat16)

        assert x.shape[2] == x_ref.shape[2] == 2
        assert x.shape[1] == self.config.in_num_freqs
        assert x_ref.shape[1] == self.ref_channels_per_freq * self.config.in_num_freqs

        x_ref = x_ref.view(x_ref.shape[0], self.ref_channels_per_freq, self.config.in_num_freqs, 2, x_ref.shape[3]).to(dtype=x.dtype)
        x_ref0, x_ref1 = x_ref.unbind(dim=3)

        x = x.unsqueeze(1)
        x0, x1 = x.unbind(dim=3)
        x0 = torch.cat((x0, x_ref0, torch.ones_like(x0[:, :1])), dim=1)
        x1 = torch.cat((x1, x_ref1, torch.ones_like(x1[:, :1])), dim=1)

        skips0 = []; skips1 = []
        for name, block in self.enc.items():
            if "conv" in name:
                x0 = block(x0)
                x1 = block(x1)
            else:
                x0, x1 = block(x0, x1, emb)

            skips0.append(x0)
            skips1.append(x1)

        # Decoder.
        for name, block in self.dec.items():
            if "layer" in name:
                x0 = mp_cat(x0, skips0.pop(), t=self.config.concat_balance)
                x1 = mp_cat(x1, skips1.pop(), t=self.config.concat_balance)
            x0, x1 = block(x0, x1, emb)

        x0 = self.conv_out(x0, gain=self.out_gain)
        x1 = self.conv_out(x1, gain=self.out_gain)
        
        x = torch.stack((x0, x1), dim=3).squeeze(1)

        D_x = c_skip * x_in + c_out * x.float()

        return D_x
