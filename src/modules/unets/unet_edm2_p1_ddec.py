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
from typing import Union, Optional

import torch

from modules.unets.unet import DualDiffusionUNet, DualDiffusionUNetConfig
from modules.daes.dae_edm2_p1 import _rope_tables_for_stereo
from modules.mp_tools import MPConv, MPFourier, mp_silu, mp_sum, normalize
from modules.formats.format import DualDiffusionFormat
from modules.rope import _rope_pair_rotate_partial
from modules.sliding_attention import SlidingWindowAttention
from modules.daes.dae_edm2_p1 import MPConvS


@dataclass
class UNetConfig(DualDiffusionUNetConfig):

    in_channels:  int = 256
    out_channels: int = 256
    in_channels_emb: int = 1024

    sigma_max: float  = 11.
    sigma_min: float  = 0.0002
    sigma_data: float = 1.

    mp_fourier_ln_sigma_offset: float = -0.5
    mp_fourier_bandwidth:       float = 1

    model_channels: int  = 2048              # Base multiplier for the number of channels.
    logvar_channels: int = 192               # Number of channels for training uncertainty estimation.
    channel_mult: list[int]    = (1,)        # Per-resolution multipliers for the number of channels.
    channel_mult_noise: Optional[int] = 1    # Multiplier for noise embedding dimensionality.
    channel_mult_emb: Optional[int]   = 1    # Multiplier for final embedding dimensionality.
    use_skips: bool     = True
    use_conv_skip: bool = True
    channels_per_head: int    = 128          # Number of channels per attention head.
    rope_channels: int        = 112
    rope_base: float          = 10000.
    attention_window_size: int = 8
    num_layers_per_block: int = 9            # Number of resnet blocks per resolution.
    label_balance: float      = 0.5          # Balance between noise embedding (0) and class embedding (1).
    res_balance: float        = 0.5          # Balance between main branch (0) and residual branch (1).
    attn_balance: float       = 0.5          # Balance between main branch (0) and self-attention (1).
    attn_levels: list[int]    = ()           # List of resolution levels to use self-attention.
    mlp_multiplier: int    = 1               # Multiplier for the number of channels in the MLP.
    mlp_groups: int        = 4               # Number of groups for the MLPs.
    emb_linear_groups: int = 1

    input_skip_t: float = 0.5

class Block(torch.nn.Module):

    def __init__(self,
        level: int,                        # Resolution level.
        in_channels: int,                  # Number of input channels.
        out_channels: int,                 # Number of output channels.
        skip_channels: int,
        emb_channels: int,                 # Number of embedding channels.
        dropout: float         = 0.,       # Dropout probability.
        res_balance: float     = 0.3,      # Balance between main branch (0) and residual branch (1).
        attn_balance: float    = 0.3,      # Balance between main branch (0) and self-attention (1).
        clip_act: float        = 256,      # Clip output activations. None = do not clip.
        mlp_multiplier: int    = 4,        # Multiplier for the number of channels in the MLP.
        mlp_groups: int        = 4,        # Number of groups for the MLP.
        emb_linear_groups: int = 4,
        channels_per_head: int = 64,       # Number of channels per attention head.
        use_attention: bool    = False,    # Use self-attention in this block.
        attention_window_size: int = 8
    ) -> None:
        super().__init__()

        assert out_channels % channels_per_head == 0

        self.level = level
        self.use_attention = use_attention
        self.num_heads = out_channels // channels_per_head
        self.out_channels = out_channels
        self.dropout = dropout
        self.res_balance = res_balance
        self.attn_balance = attn_balance
        self.clip_act = clip_act
        
        inner_channels = out_channels * mlp_multiplier

        if skip_channels > 0:
            self.conv_skip = MPConv(in_channels + skip_channels, in_channels, kernel=(1,1))
        else:
            self.conv_skip = torch.nn.Identity()

        self.conv_res0 = MPConvS(in_channels, inner_channels,  kernel=(1,3), groups=mlp_groups)
        self.conv_res1 = MPConvS(inner_channels, out_channels, kernel=(1,3), groups=mlp_groups)
        
        self.emb_gain = torch.nn.Parameter(torch.zeros([]))
        self.emb_linear = MPConv(emb_channels, inner_channels, kernel=(1,1), groups=emb_linear_groups)

        if self.use_attention == True:
            self.attn_q = MPConv(out_channels, out_channels, kernel=(1,1))
            self.attn_k = MPConv(out_channels, out_channels, kernel=(1,1))
            self.attn_v = MPConv(out_channels, out_channels, kernel=(1,1))
            self.attn_proj = MPConv(out_channels, out_channels, kernel=(1,1))

            self.emb_gain_qkv = torch.nn.Parameter(torch.zeros([]))
            self.emb_linear_qkv = MPConv(emb_channels, out_channels, kernel=(1,1), groups=1)

            self.sliding_attn = SlidingWindowAttention(attention_window_size, causal=False, head_dim=channels_per_head)

    def forward(self, x: torch.Tensor, emb: torch.Tensor, rope_tables: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        
        x = self.conv_skip(x)

        if self.use_attention == True:
            
            x = x.transpose(-1, -2)

            c = self.emb_linear_qkv(emb, gain=self.emb_gain_qkv) + 1.
            y = x * c

            # bizarrely this is way faster than doing a single qkv projection, and uses less memory (with compile)
            q: torch.Tensor = self.attn_q(y)
            k: torch.Tensor = self.attn_k(y)
            v: torch.Tensor = self.attn_v(y)
            q = q.reshape(q.shape[0], self.num_heads, -1, y.shape[2] * y.shape[3])
            k = k.reshape(k.shape[0], self.num_heads, -1, y.shape[2] * y.shape[3])
            v = v.reshape(v.shape[0], self.num_heads, -1, y.shape[2] * y.shape[3])
            q = normalize(q, dim=2)
            k = normalize(k, dim=2)
            v = normalize(v, dim=2)

            q_rot = _rope_pair_rotate_partial(q.transpose(-1, -2), rope_tables)
            k_rot = _rope_pair_rotate_partial(k.transpose(-1, -2), rope_tables)

            #y = torch.nn.functional.scaled_dot_product_attention(q_rot, k_rot, v.transpose(-1, -2)).transpose(-1, -2)
            y = self.sliding_attn(q_rot, k_rot, v.transpose(-1, -2)).transpose(-1, -2)

            y = self.attn_proj(y.reshape(*x.shape))
            x = mp_sum(x, y, t=self.attn_balance).transpose(-1, -2)

        y = self.conv_res0(x)

        c = self.emb_linear(emb, gain=self.emb_gain) + 1.
        y = mp_silu(normalize(y * c, dim=1))

        if self.dropout != 0 and self.training == True: # magnitude preserving fix for dropout
            y = torch.nn.functional.dropout(y, p=self.dropout) * (1. - self.dropout)**0.5

        y: torch.Tensor = self.conv_res1(y)
        
        x = mp_sum(x, y, t=self.res_balance)

        if self.clip_act is not None:
            x = x.clip_(-self.clip_act, self.clip_act)

        return x

class UNet(DualDiffusionUNet):

    def __init__(self, config: UNetConfig) -> None:
        super().__init__()
        self.config = config

        block_kwargs = {"dropout": config.dropout,
                        "mlp_multiplier": config.mlp_multiplier,
                        "mlp_groups": config.mlp_groups,
                        "emb_linear_groups": config.emb_linear_groups,
                        "res_balance": config.res_balance,
                        "attn_balance": config.attn_balance,
                        "channels_per_head": config.channels_per_head,
                        "attention_window_size": config.attention_window_size}

        cblock = [config.model_channels * x for x in config.channel_mult]
        cnoise = config.model_channels * config.channel_mult_noise if config.channel_mult_noise is not None else max(cblock)
        cemb = config.model_channels * config.channel_mult_emb if config.channel_mult_emb is not None else max(cblock)
        cdata = config.in_channels

        self.num_levels = len(config.channel_mult)

        assert self.num_levels == 1
        assert config.rope_channels % 2 == 0
        assert config.rope_channels <= config.channels_per_head
        if config.input_skip_t > 0:
            assert cblock[0] >= cdata

        # Embedding.
        self.emb_fourier = MPFourier(cnoise, bandwidth=config.mp_fourier_bandwidth)
        self.emb_noise = MPConv(cnoise, cemb, kernel=())
        self.emb_cond = MPConv(cemb, cemb, kernel=())

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
        self.dec = torch.nn.ModuleDict()
        cout = cdata + 1 # 1 extra const channel

        for level, channels in enumerate(cblock):
            
            cin = cout
            cout = channels
            self.dec[f"conv_in"] = MPConvS(cin, cout, kernel=(1,3))
            
            for idx in range(config.num_layers_per_block):

                cin = cout
                cout = channels

                if config.use_skips == True and config.use_conv_skip == True and idx >= config.num_layers_per_block / 2:
                    cskip = channels
                else:
                    cskip = 0

                self.dec[f"block{level}_layer{idx}"] = Block(level, cin, cout, cskip, cemb, use_attention=level in config.attn_levels, **block_kwargs)
                
        self.out_gain = torch.nn.Parameter(torch.zeros([]))
        self.conv_out = MPConvS(cout, config.out_channels, kernel=(1,3))

    def get_embeddings(self, emb_in: torch.Tensor, conditioning_mask: torch.Tensor) -> torch.Tensor:
        if self.config.in_channels_emb > 0:
            u_embedding = self.emb_label_unconditional(torch.ones(1, device=self.device, dtype=self.dtype))
            c_embedding = self.emb_label(normalize(emb_in).to(device=self.device, dtype=self.dtype))
            return mp_sum(u_embedding, c_embedding, t=conditioning_mask.unsqueeze(1).to(self.device, self.dtype))
        else:
            return emb_in
        
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
            emb = mp_sum(emb, embeddings.to(dtype=emb.dtype), t=self.config.label_balance)
        emb = self.emb_cond(mp_silu(emb))[..., None, None]
        emb = mp_silu(normalize(emb + x_ref.to(dtype=emb.dtype), dim=1)).to(dtype=torch.bfloat16)
      
        rope_tables = _rope_tables_for_stereo(x, self.config.rope_channels, self.config.rope_base)

        # Encoder.
        x_input = x
        x = torch.cat((x_input, torch.ones_like(x[:, :1])), dim=1)

        idx = 0; skips = []
        for name, block in self.dec.items():
            if "conv" in name:
                x = block(x)
            else:
                if self.config.use_skips == True and idx >= self.config.num_layers_per_block / 2:
                    if self.config.use_conv_skip == True:
                        x = torch.cat((x, skips.pop()), dim=1)
                    else:
                        x = mp_sum(x, skips.pop(), t=0.5)

                if self.config.input_skip_t > 0:
                    x[:, :x_input.shape[1]] = mp_sum(x[:, :x_input.shape[1]], x_input, t=self.config.input_skip_t)

                x = block(x, emb, rope_tables)

                if self.config.use_skips == True and idx < self.config.num_layers_per_block / 2 - 0.5:
                    skips.append(x)
                
                idx += 1

        x: torch.Tensor = self.conv_out(x, gain=self.out_gain)
        D_x = c_skip * x_in + c_out * x.float()

        return D_x
