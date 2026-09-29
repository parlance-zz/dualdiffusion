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

from modules.ddecs.ddec import DualDiffusionDecoder, DualDiffusionDecoderConfig
from modules.mp_tools import MPFourier, MPConv, mp_cat, mp_silu, mp_sum, normalize, resample_2d
from modules.formats.ms_mdct_dual_7 import MS_MDCT_DualFormat


@dataclass
class DiffusionDecoderConfig(DualDiffusionDecoderConfig):

    in_channels:  int = 4
    out_channels: int = 4
    in_channels_emb: int = 0
    in_channels_x_ref: int = 2

    step_balance: list[float] = (0, 1/7, 2/7, 3/7, 4/7, 5/7, 6/7)
    in_balance: float    = 0.5
    label_balance: float = 0.5
    skip_balance: float  = 0.5

    in_num_freqs: int = 64
    in_psd_num_freqs: list[int] = (64, 128, 256, 512)
    in_num_mdct_levels: int = 4

    model_channels: int  = 64                # Base multiplier for the number of channels.
    channel_mult: list[int]    = (1,1,1,1)   # Per-resolution multipliers for the number of channels.
    double_midblock: bool      = False
    midblock_attn: bool        = False
    channel_mult_noise: Optional[int] = 2    # Multiplier for noise embedding dimensionality.
    channel_mult_emb: Optional[int]   = 2    # Multiplier for final embedding dimensionality.
    channels_per_head: int    = 64           # Number of channels per attention head.
    num_layers_per_block: int = 3            # Number of resnet blocks per resolution.
    label_balance: float      = 0.5          # Balance between noise embedding (0) and class embedding (1).
    concat_balance: float     = 0.5          # Balance between skip connections (0) and main path (1).
    res_balance: float        = 0.3          # Balance between main branch (0) and residual branch (1).
    attn_balance: float       = 0.3          # Balance between main branch (0) and self-attention (1).
    attn_levels: list[int]    = ()           # List of resolution levels to use self-attention.
    mlp_multiplier: int    = 2               # Multiplier for the number of channels in the MLP.
    mlp_groups: int        = 1               # Number of groups for the MLPs.
    emb_linear_groups: int = 1

class Block(torch.nn.Module):

    def __init__(self,
        level: int,                             # Resolution level.
        in_channels: int,                       # Number of input channels.
        out_channels: int,                      # Number of output channels.
        emb_channels: int,                      # Number of embedding channels.
        num_freqs: int,         
        flavor: Literal["enc", "dec"] = "enc",
        resample_mode: Literal["keep", "up", "down"] = "keep",
        res_balance: float     = 0.3,      # Balance between main branch (0) and residual branch (1).
        attn_balance: float    = 0.3,      # Balance between main branch (0) and self-attention (1).
        clip_act: float        = 256,      # Clip output activations. None = do not clip.
        mlp_multiplier: int    = 1,        # Multiplier for the number of channels in the MLP.
        mlp_groups: int        = 1,        # Number of groups for the MLP.
        emb_linear_groups: int = 1,
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

        self.emb_gain = torch.nn.Parameter(torch.zeros([]))
        self.emb_linear = MPConv(emb_channels, out_channels * mlp_multiplier,
            kernel=(1,1), groups=emb_linear_groups) if emb_channels != 0 else None
        
        if self.use_attention:
            self.attn_q = MPConv(out_channels, out_channels, kernel=(1,1))
            self.attn_k = MPConv(out_channels, out_channels, kernel=(1,1))
            self.attn_v = MPConv(out_channels, out_channels, kernel=(1,1))
            self.attn_proj = MPConv(out_channels, out_channels, kernel=(1,1))

            self.emb_gain_qkv = torch.nn.Parameter(torch.zeros([]))
            self.emb_linear_qkv = MPConv(emb_channels, out_channels, kernel=(1,1))

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        
        x = resample_2d(x, mode=self.resample_mode)

        if self.flavor == "enc":
            if self.conv_skip is not None:
                x = self.conv_skip(x)
            x = normalize(x, dim=1) # pixel norm

        y = self.conv_res0(mp_silu(x))

        c = self.emb_linear(emb, gain=self.emb_gain) + 1.
        y = mp_silu(y * c)

        y = self.conv_res1(y)

        if self.flavor == "dec" and self.conv_skip is not None:
            x = self.conv_skip(x)
        x = mp_sum(x, y, t=self.res_balance)
        
        if self.use_attention:
            c = self.emb_linear_qkv(emb, gain=self.emb_gain_qkv) + 1.
            y = x * c

            q: torch.Tensor = self.attn_q(y)
            k: torch.Tensor = self.attn_k(y)
            v: torch.Tensor = self.attn_v(y)
            q = q.reshape(q.shape[0], self.num_heads, -1, y.shape[2] * y.shape[3])
            k = k.reshape(k.shape[0], self.num_heads, -1, y.shape[2] * y.shape[3])
            v = v.reshape(v.shape[0], self.num_heads, -1, y.shape[2] * y.shape[3])
            q = normalize(q, dim=2).transpose(-1, -2)
            k = normalize(k, dim=2).transpose(-1, -2)
            v = normalize(v, dim=2).transpose(-1, -2)

            y = torch.nn.functional.scaled_dot_product_attention(q, k, v).transpose(-1, -2)

            y = self.attn_proj(y.reshape(*x.shape))
            x = mp_sum(x, y, t=self.attn_balance)

        if self.clip_act is not None:
            x = x.clip_(-self.clip_act, self.clip_act)

        return x

class DiffusionDecoder(DualDiffusionDecoder):

    def __init__(self, config: DiffusionDecoderConfig) -> None:
        super().__init__()
        self.config = config

        block_kwargs = {"mlp_multiplier": config.mlp_multiplier,
                        "mlp_groups": config.mlp_groups,
                        "emb_linear_groups": config.emb_linear_groups,
                        "res_balance": config.res_balance,
                        "attn_balance": config.attn_balance,
                        "channels_per_head": config.channels_per_head}

        cblock = [config.model_channels * x for x in config.channel_mult]
        cnoise = config.model_channels * config.channel_mult_noise if config.channel_mult_noise is not None else max(cblock)
        cemb = config.model_channels * config.channel_mult_emb if config.channel_mult_emb is not None else max(cblock)

        self.num_levels = len(config.channel_mult)
        self.num_psd_levels = len(config.in_psd_num_freqs)
        assert self.num_psd_levels <= self.num_levels
        assert config.in_num_mdct_levels <= self.num_levels
        assert config.in_num_mdct_levels > 0
        assert config.in_channels == config.out_channels
        
        self.psd_freqs_per_freq: list[int] = []
        for i in range(self.num_psd_levels):
            level_freqs = config.in_num_freqs * (2 ** i)
            assert config.in_psd_num_freqs[i] % level_freqs == 0
            self.psd_freqs_per_freq.append(config.in_psd_num_freqs[i] // level_freqs)

        # Embedding.
        self.emb_fourier = MPFourier(cnoise)
        self.emb_noise = MPConv(cnoise, cemb, kernel=())
        self.emb_label = MPConv(config.in_channels_emb, cemb, kernel=()) if config.in_channels_emb > 0 else None

        # Training uncertainty estimation.
        #self.error_logvar = torch.nn.Parameter(torch.zeros(1))

        # Encoder.
        self.x_ref_in_gain = torch.nn.Parameter(torch.ones(self.num_psd_levels) * 0.5)
        self.conv_x_ref_in = torch.nn.ModuleDict()
        for i in range(self.num_psd_levels):
            self.conv_x_ref_in[f"conv_x_ref_in{i}"] = MPConv(config.in_channels_x_ref, cblock[i], kernel=(3,3), bias=True)

        self.conv_in = torch.nn.ModuleDict()
        for i in range(config.in_num_mdct_levels):
            self.conv_in[f"conv_in{i}"] = MPConv(config.in_channels, cblock[i], kernel=(3,3), bias=True)

        self.enc = torch.nn.ModuleDict()
        cin = cblock[0]

        for level, channels in enumerate(cblock):
            
            num_freqs = config.in_num_freqs * (2 ** level)
            cout = channels

            if level == 0:    
                self.enc[f"block{level}_in"] = Block(level, cin, cout, cemb, num_freqs,
                    use_attention=level in config.attn_levels, flavor="enc", **block_kwargs)
            else:
                self.enc[f"block{level}_down"] = Block(level, cin, cout, cemb, num_freqs,
                    use_attention=level in config.attn_levels, flavor="enc", resample_mode="up_down", **block_kwargs)
            
            for idx in range(config.num_layers_per_block):
                cin = cout
                cout = channels
                self.enc[f"block{level}_layer{idx}"] = Block(level, cin, cout, cemb, num_freqs,
                    use_attention=level in config.attn_levels, flavor="enc", **block_kwargs)

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        skips = [block.out_channels for block in self.enc.values()]
        
        for level, channels in reversed(list(enumerate(cblock))):
            
            num_freqs = config.in_num_freqs * (2 ** level)

            if level == len(cblock) - 1:
                self.dec[f"block{level}_in0"] = Block(level, cout, cout, cemb, num_freqs,
                    use_attention=config.midblock_attn, flavor="dec", **block_kwargs)
                if config.double_midblock == True:
                    self.dec[f"block{level}_in1"] = Block(level, cout, cout, cemb, num_freqs,
                        use_attention=config.midblock_attn, flavor="dec", **block_kwargs)
            else:
                self.dec[f"block{level}_up"] = Block(level, cout, cout, cemb, num_freqs,
                    use_attention=level in config.attn_levels, flavor="dec", resample_mode="down_up", **block_kwargs)

            for idx in range(config.num_layers_per_block + 1):
                cin = cout + skips.pop()
                cout = channels
                self.dec[f"block{level}_layer{idx}"] = Block(level, cin, cout, cemb, num_freqs,
                    use_attention=level in config.attn_levels, flavor="dec", **block_kwargs)

            if level < config.in_num_mdct_levels:
                self.dec[f"conv_out{level}"] = MPConv(cout, config.out_channels, kernel=(3,3))

        self.out_gain = torch.nn.Parameter(torch.ones(config.in_num_mdct_levels))

    def get_embeddings(self, emb_in: torch.Tensor) -> torch.Tensor:
        if self.emb_label is not None:
            return mp_silu(self.emb_label(normalize(emb_in).to(device=self.device, dtype=self.dtype)))
        else:
            return None
        
    def get_latent_shape(self, latent_shape: Union[torch.Size, tuple[int, int, int, int]]) -> torch.Size:
        return latent_shape[0:3] + ( (latent_shape[3] // 2**(self.num_levels-1)) * 2**(self.num_levels-1), )

    def get_state_shape(self, format: MS_MDCT_DualFormat, x_ref: Union[torch.Tensor, list[torch.Tensor]]) -> torch.Size:
        state_shape = format.flatten_mdct_phase_psd(x_ref).shape
        return state_shape[0:1] + (self.config.out_channels,) + state_shape[2:]

    def step(self, x: torch.Tensor, step: int, format: MS_MDCT_DualFormat,
        x_ref: Union[torch.Tensor, list[torch.Tensor]], embeddings: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        c_step = torch.tensor((step / self.config.num_steps) * 3, device=x.device).expand(x.shape[0])
        emb = self.emb_fourier(c_step)

        x_out: list[torch.Tensor] = []
        x_in = format.unflatten_mdct_phase_psd(x.to(dtype=torch.bfloat16))
        assert len(x_in) == self.config.in_num_mdct_levels

        x_ref = [x.to(dtype=torch.bfloat16) for x in x_ref]
        assert len(x_ref) == self.num_psd_levels
        
        # embedding
        emb = self.emb_noise(emb)
        if self.config.in_channels_emb > 0:
            emb = mp_silu(mp_sum(emb, embeddings, t=self.config.label_balance))
        emb = emb[:, :, None, None].to(dtype=torch.bfloat16)

        x = self.conv_in["conv_in0"](x_in[0])

        # encoder
        skips = []
        for name, block in self.enc.items():

            x = block(x, emb)

            if "down" in name and block.level < self.config.in_num_mdct_levels:
                x = mp_sum(x, self.conv_in[f"conv_in{block.level}"](x_in[block.level]), t=self.config.in_balance)

            if ("down" in name or "in" in name) and block.level < self.num_psd_levels:
                x = x + self.conv_x_ref_in[f"conv_x_ref_in{block.level}"](x_ref[block.level]) * self.x_ref_in_gain[block.level]

            skips.append(x)

        # decoder
        for name, block in self.dec.items():
            if "conv_out" in name:
                x_out.append(block(x, gain=self.out_gain[-1 - len(x_out)]))
            else:
                if "layer" in name:
                    x = mp_cat(x, skips.pop(), t=self.config.skip_balance)

                x = block(x, emb)

        x_out.reverse()
        x = format.flatten_mdct_phase_psd(x_out)

        return x.float()
