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
from typing import Optional, Literal

import torch

from modules.cnets.cnet import DualDiffusionCNet, DualDiffusionCNetConfig
from modules.mp_tools import mp_cat, mp_silu, mp_sum, normalize, resample_2d
from modules.daes.dae_edm2_2psd_a1 import MPConv2D
from modules.formats.format import DualDiffusionFormat


@dataclass
class CNet_2PSD_A1_Config(DualDiffusionCNetConfig):

    in_channels:  int   = 4
    out_channels: int   = 4
    x_ref_channels: int = 4

    in_num_freqs: int = 1024

    model_channels: int  = 32                # Base multiplier for the number of channels.
    channel_mult: list[int]    = (1,2,4,8)   # Per-resolution multipliers for the number of channels.
    double_midblock: bool      = False
    num_layers_per_block: int = 3            # Number of resnet blocks per resolution.
    label_balance: float      = 0.5          # Balance between noise embedding (0) and class embedding (1).
    concat_balance: float     = 0.5          # Balance between skip connections (0) and main path (1).
    res_balance: float        = 0.3          # Balance between main branch (0) and residual branch (1).
    mlp_multiplier: int    = 4               # Multiplier for the number of channels in the MLP.
    mlp_groups: int        = 4               # Number of groups for the MLPs.
    add_constant_channel: bool = True

class Block(torch.nn.Module):

    def __init__(self,
        level: int,                             # Resolution level.
        in_channels: int,                       # Number of input channels.
        out_channels: int,                      # Number of output channels.
        num_freqs: int,
        flavor: Literal["enc", "dec"] = "enc",
        resample_mode: Literal["keep", "up", "down"] = "keep",
        res_balance: float     = 0.3,      # Balance between main branch (0) and residual branch (1).
        clip_act: float        = 256,      # Clip output activations. None = do not clip.
        mlp_multiplier: int    = 1,        # Multiplier for the number of channels in the MLP.
        mlp_groups: int        = 1,        # Number of groups for the MLP.
    ) -> None:
        super().__init__()

        self.level = level
        self.num_freqs = num_freqs
        self.out_channels = out_channels
        self.flavor = flavor
        self.resample_mode = resample_mode
        self.res_balance = res_balance
        self.clip_act = clip_act

        assert num_freqs >= 1

        if num_freqs == 1:
            if resample_mode != "keep":
                self.resample_mode += "_w"
            kernel = (1,3)
        else:
            kernel = (3,3)

        self.conv_res0 = MPConv2D(out_channels if flavor == "enc" else in_channels,
                out_channels * mlp_multiplier, kernel=kernel, groups=mlp_groups)

        self.conv_res1 = MPConv2D(out_channels * mlp_multiplier, out_channels, kernel=kernel, groups=mlp_groups)

        if in_channels != out_channels or mlp_groups > 1:
            self.conv_skip = MPConv2D(in_channels, out_channels, kernel=(1,1), groups=1)
        else:
            self.conv_skip = None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = resample_2d(x, mode=self.resample_mode)

        if self.flavor == "enc":
            if self.conv_skip is not None:
                x = self.conv_skip(x)
            x = normalize(x, dim=1) # pixel norm

        y = self.conv_res0(mp_silu(x))
        y = self.conv_res1(mp_silu(y))

        if self.flavor == "dec" and self.conv_skip is not None:
            x = self.conv_skip(x)
        x = mp_sum(x, y, t=self.res_balance)
        
        if self.clip_act is not None:
            x = x.clip_(-self.clip_act, self.clip_act)

        return x

class CNet_2PSD_A1(DualDiffusionCNet):

    def __init__(self, config: CNet_2PSD_A1_Config) -> None:
        super().__init__()
        self.config = config

        block_kwargs = {"mlp_multiplier": config.mlp_multiplier,
                        "mlp_groups": config.mlp_groups,
                        "res_balance": config.res_balance}

        cblock = [config.model_channels * x for x in config.channel_mult]
        self.num_levels = len(config.channel_mult)

        # Training uncertainty estimation.
        self.loss_logvar = torch.nn.Parameter(torch.zeros(self.config.out_channels))

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = config.in_channels + config.x_ref_channels + int(config.add_constant_channel)

        for level, channels in enumerate(cblock):
            
            num_freqs = max(config.in_num_freqs // 2**level, 1)

            if level == 0:
                cin = cout
                cout = channels
                self.enc[f"conv_in"] = MPConv2D(cin, cout, kernel=(3,3))
            else:
                self.enc[f"block{level}_down"] = Block(
                    level, cout, cout, num_freqs, flavor="enc", resample_mode="down", **block_kwargs)
            
            for idx in range(config.num_layers_per_block):
                cin = cout
                cout = channels
                self.enc[f"block{level}_layer{idx}"] = Block(
                    level, cin, cout, num_freqs, flavor="enc", **block_kwargs)

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        skips = [block.out_channels for block in self.enc.values()]
        
        for level, channels in reversed(list(enumerate(cblock))):
            
            num_freqs = max(config.in_num_freqs // 2**level, 1)

            if level == len(cblock) - 1:
                self.dec[f"block{level}_in0"] = Block(
                    level, cout, cout, num_freqs, flavor="dec", **block_kwargs)
                if config.double_midblock == True:
                    self.dec[f"block{level}_in1"] = Block(
                        level, cout, cout, num_freqs, **block_kwargs)
            else:
                self.dec[f"block{level}_up"] = Block(
                    level, cout, cout, num_freqs, flavor="dec", resample_mode="up", **block_kwargs)

            for idx in range(config.num_layers_per_block + 1):
                cin = cout + skips.pop()
                cout = channels
                self.dec[f"block{level}_layer{idx}"] = Block(
                    level, cin, cout, num_freqs, flavor="dec", **block_kwargs)
                
        self.out_gain = torch.nn.Parameter(torch.zeros([]))
        self.conv_out = MPConv2D(cout, config.out_channels, kernel=(3,3))

    def get_loss_logvar(self) -> torch.Tensor:
        return self.loss_logvar.view(1,-1, 1, 1)

    def forward(self, x_in: torch.Tensor,
                format: DualDiffusionFormat,
                x_ref: Optional[torch.Tensor] = None) -> torch.Tensor:

        x_in = (x_in + torch.randn_like(x_in)) / 2**0.5
        
        # Encoder.
        inputs = (x_in, x_ref, torch.ones_like(x_in[:, :1])) if self.config.add_constant_channel else (x_in, x_ref)
        x = torch.cat(inputs, dim=1)

        skips = []
        for name, block in self.enc.items():
            x = block(x) if "conv" in name else block(x)
            skips.append(x)

        # Decoder.
        for name, block in self.dec.items():
            if "layer" in name:
                x = mp_cat(x, skips.pop(), t=self.config.concat_balance)
            x = block(x)

        x: torch.Tensor = self.conv_out(x, gain=self.out_gain)
        D_x = (x_in.float() + x.float()) / 2**0.5

        return D_x
    
