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
from typing import Literal

import torch

from modules.phaseret.phaseret import DualDiffusionPhaseRet, DualDiffusionPhaseRetConfig
from modules.mp_tools import MPConv, mp_cat, mp_silu, mp_sum, normalize, resample_2d


@dataclass
class PhaseRetConfig(DualDiffusionPhaseRetConfig):

    in_channels:  int = 9
    out_channels: int = 4

    out_num_freqs: int = 256
    in_psd_freqs: int  = 512

    model_channels: int  = 64                # Base multiplier for the number of channels.
    channel_mult: list[int]    = (1,2,3,4)   # Per-resolution multipliers for the number of channels.
    double_midblock: bool      = False
    midblock_attn: bool        = True

    channels_per_head: int    = 64           # Number of channels per attention head.
    num_layers_per_block: int = 3            # Number of resnet blocks per resolution.
    concat_balance: float     = 0.5          # Balance between main path   (0) skip connections (1).
    res_balance: float        = 0.3          # Balance between main branch (0) and residual branch (1).
    attn_balance: float       = 0.3          # Balance between main branch (0) and self-attention (1).
    attn_levels: list[int]    = (3,)         # List of resolution levels to use self-attention.
    mlp_multiplier: int    = 2               # Multiplier for the number of channels in the MLP.
    mlp_groups: int        = 1               # Number of groups for the MLPs.

class Block(torch.nn.Module):

    def __init__(self,
        level: int,                             # Resolution level.
        in_channels: int,                       # Number of input channels.
        out_channels: int,                      # Number of output channels.
        num_freqs: int,         
        flavor: Literal["enc", "dec"] = "enc",
        resample_mode: Literal["keep", "up", "down"] = "keep",
        res_balance: float     = 0.3,      # Balance between main branch (0) and residual branch (1).
        attn_balance: float    = 0.3,      # Balance between main branch (0) and self-attention (1).
        clip_act: float        = 256,      # Clip output activations. None = do not clip.
        mlp_multiplier: int    = 1,        # Multiplier for the number of channels in the MLP.
        mlp_groups: int        = 1,        # Number of groups for the MLP.
        channels_per_head: int = 64,       # Number of channels per attention head.
        use_attention: bool    = False     # Use self-attention in this block.
    ) -> None:
        super().__init__()

        self.level = level
        self.num_freqs = num_freqs
        self.use_attention = use_attention
        self.num_heads = out_channels // channels_per_head
        self.out_channels = out_channels
        self.flavor = flavor
        self.resample_mode = resample_mode
        self.res_balance = res_balance
        self.attn_balance = attn_balance
        self.clip_act = clip_act
        
        self.conv_res0 = MPConv(out_channels if flavor == "enc" else in_channels,
                                out_channels * mlp_multiplier, kernel=(3,3), groups=mlp_groups)
        self.conv_res1 = MPConv(out_channels * mlp_multiplier, out_channels, kernel=(3,3), groups=mlp_groups)

        if in_channels != out_channels:
            self.conv_skip = MPConv(in_channels, out_channels, kernel=(1,1), groups=1)
        else:
            self.conv_skip = None
        
        if self.use_attention:
            self.attn_q = MPConv(out_channels, out_channels, kernel=(1,1))
            self.attn_k = MPConv(out_channels, out_channels, kernel=(1,1))
            self.attn_v = MPConv(out_channels, out_channels, kernel=(1,1))
            self.attn_proj = MPConv(out_channels, out_channels, kernel=(1,1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        if self.flavor == "enc":
            if self.conv_skip is not None:
                x = self.conv_skip(x)

        x = resample_2d(x, mode=self.resample_mode)

        if self.flavor == "enc":
            x = normalize(x, dim=1) # pixel norm

        y = self.conv_res0(mp_silu(x))
        y = self.conv_res1(mp_silu(y))

        if self.flavor == "dec" and self.conv_skip is not None:
            x = self.conv_skip(x)
        x = mp_sum(x, y, t=self.res_balance)
        
        if self.use_attention:

            q: torch.Tensor = self.attn_q(x)
            k: torch.Tensor = self.attn_k(x)
            v: torch.Tensor = self.attn_v(x)
            q = q.reshape(q.shape[0], self.num_heads, -1, x.shape[2] * x.shape[3])
            k = k.reshape(k.shape[0], self.num_heads, -1, x.shape[2] * x.shape[3])
            v = v.reshape(v.shape[0], self.num_heads, -1, x.shape[2] * x.shape[3])
            q = normalize(q, dim=2).transpose(-1, -2)
            k = normalize(k, dim=2).transpose(-1, -2)
            v = normalize(v, dim=2).transpose(-1, -2)

            y = torch.nn.functional.scaled_dot_product_attention(q, k, v).transpose(-1, -2)

            y = self.attn_proj(y.reshape(*x.shape))
            x = mp_sum(x, y, t=self.attn_balance)

        if self.clip_act is not None:
            x = x.clip_(-self.clip_act, self.clip_act)

        return x

class PhaseRet(DualDiffusionPhaseRet):

    def __init__(self, config: PhaseRetConfig) -> None:
        super().__init__()
        self.config = config

        block_kwargs = {"mlp_multiplier": config.mlp_multiplier,
                        "mlp_groups": config.mlp_groups,
                        "res_balance": config.res_balance,
                        "attn_balance": config.attn_balance,
                        "channels_per_head": config.channels_per_head}

        cblock = [config.model_channels * x for x in config.channel_mult]

        self.num_levels = len(config.channel_mult)
        
        assert config.in_psd_freqs % config.out_num_freqs == 0
        self.psd_freqs_per_freq = config.in_psd_freqs // config.out_num_freqs
        assert self.psd_freqs_per_freq in (1, 2)
        
        self.recon_logvar = torch.nn.Parameter(torch.zeros([]))
        
        self.enc = torch.nn.ModuleDict()
        cin = config.in_channels

        for level, channels in enumerate(cblock):
            
            num_freqs = config.out_num_freqs // 2**level
            cout = channels

            if level == 0:
                self.enc[f"conv_in"] = MPConv(cin, cout, kernel=(3,3))
            else:
                self.enc[f"block{level}_down"] = Block(level, cin, cout, num_freqs,
                    use_attention=level in config.attn_levels, flavor="enc", resample_mode="down", **block_kwargs)
            
            for idx in range(config.num_layers_per_block):
                cin = cout
                cout = channels
                self.enc[f"block{level}_layer{idx}"] = Block(level, cin, cout, num_freqs,
                    use_attention=level in config.attn_levels, flavor="enc", **block_kwargs)

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        skips = [block.out_channels for block in self.enc.values()]
        
        for level, channels in reversed(list(enumerate(cblock))):
            
            num_freqs = config.out_num_freqs // 2**level

            if level == len(cblock) - 1:
                self.dec[f"block{level}_in0"] = Block(level, cout, cout, num_freqs,
                    use_attention=config.midblock_attn, flavor="dec", **block_kwargs)
                if config.double_midblock == True:
                    self.dec[f"block{level}_in1"] = Block(level, cout, cout, num_freqs,
                        use_attention=config.midblock_attn, flavor="dec", **block_kwargs)
            else:
                self.dec[f"block{level}_up"] = Block(level, cout, cout, num_freqs,
                    use_attention=level in config.attn_levels, flavor="dec", resample_mode="up", **block_kwargs)

            for idx in range(config.num_layers_per_block + 1):
                cin = cout + skips.pop()
                cout = channels
                self.dec[f"block{level}_layer{idx}"] = Block(level, cin, cout, num_freqs,
                    use_attention=level in config.attn_levels, flavor="dec", **block_kwargs)
                
        self.out_gain = torch.nn.Parameter(torch.zeros([]))
        self.conv_out = MPConv(cout, config.out_channels, kernel=(5,5))

    def get_recon_loss_logvar(self) -> torch.Tensor:
        return getattr(self, "recon_logvar", None)

    def forward(self, x_in: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        x = x_in.to(dtype=torch.bfloat16)

        # encoder
        skips = []
        for name, block in self.enc.items():
            
            x = block(x)
                
            if "conv" in name and self.psd_freqs_per_freq == 2:
                x = torch.nn.functional.avg_pool2d(x, kernel_size=(2,1), stride=(2,1))

            skips.append(x)

        # decoder
        for name, block in self.dec.items():
            if "layer" in name:
                x = mp_cat(x, skips.pop(), t=self.config.concat_balance)
            x = block(x)

        x: torch.Tensor = self.conv_out(x, gain=self.out_gain).float()

        return x, self.get_recon_loss_logvar()
    