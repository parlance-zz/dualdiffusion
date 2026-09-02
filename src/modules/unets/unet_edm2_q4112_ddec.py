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
from modules.formats.ms_mdct_dual_9 import MS_MDCT_DualFormat, patch_ms_psd, unpatch_ms_psd
from modules.mp_tools import MPFourier, MPConv, AdaptiveGroupBalance, mp_silu, mp_sum, normalize, normalize_groups


@dataclass
class UNetConfig(DualDiffusionUNetConfig):

    in_channels:  int = 2
    out_channels: int = 2
    in_channels_emb: int = 0
    in_channels_x_ref: int = 3

    in_num_freqs: int = 64
    in_psd_num_freqs: list[int] = (64, 128, 256, 512)

    sigma_max: float  = 50
    sigma_min: float  = 1e-3
    sigma_data: float = 1

    adg_min_balance: Optional[float]  = 0.1
    adg_max_balance: Optional[float]  = 0.9
    adg_weight_decay: Optional[float] = None

    model_channels: int  = 1024                # Base multiplier for the number of channels.
    logvar_channels: int = 192                 # Number of channels for training uncertainty estimation.
    channel_mult: list[int] = (1,)             # Per-resolution multipliers for the number of channels.
    channel_mult_noise: Optional[float] = 0.25 # Multiplier for noise embedding dimensionality.
    channel_mult_emb: Optional[float]   = 1    # Multiplier for final embedding dimensionality.
    channels_per_head: int      = 128          # Number of channels per attention head.
    num_layers_per_block: int = 8           # Number of resnet blocks per resolution.
    label_balance: float      = 0.5          # Balance between noise embedding (0) and class embedding (1).
    balance_logits_offset: float = -4
    mlp_multiplier: int    = 1               # Multiplier for the number of channels in the MLP.
    mlp_groups: int        = 8              # Number of groups for the MLPs.
    emb_linear_groups: int = 8

class Block(torch.nn.Module):

    def __init__(self,
        in_channels: int,                  # Number of input channels.
        out_channels: int,                 # Number of output channels.
        emb_channels: int,                 # Number of embedding channels.
        num_freqs: int,
        dropout: float         = 0.,       # Dropout probability.
        balance_logits_offset: float = -4, # Offset for the balance logits before sigmoid.
        clip_act: float        = 256,      # Clip output activations. None = do not clip.
        mlp_multiplier: int    = 2,        # Multiplier for the number of channels in the MLP.
        mlp_groups: int        = 16,        # Number of groups for the MLP.
        emb_linear_groups: int = 16,
        channels_per_head: int = 128,       # Number of channels per attention head.
        adg_min_balance: Optional[float]  = 0.1,
        adg_max_balance: Optional[float]  = 0.9,
        adg_weight_decay: Optional[float] = None,
    ) -> None:
        super().__init__()
        assert out_channels % channels_per_head == 0

        self.num_heads = out_channels // mlp_groups // channels_per_head
        self.channels_per_head = channels_per_head
        self.mlp_groups = mlp_groups
        self.out_channels = out_channels
        self.dropout = dropout
        self.balance_logits_offset = balance_logits_offset
        self.clip_act = clip_act
        self.num_freqs = num_freqs

        inner_channels = out_channels * mlp_multiplier

        assert self.num_heads == 1
        assert emb_channels % emb_linear_groups == 0
        assert inner_channels % mlp_groups == 0
        assert inner_channels % emb_linear_groups == 0
        assert out_channels % mlp_groups == 0
        assert in_channels % mlp_groups == 0

        self.conv_res0 = MPConv(in_channels, inner_channels,  kernel=(3,3), groups=mlp_groups)
        self.conv_res1 = MPConv(inner_channels, out_channels, kernel=(3,3), groups=mlp_groups)
        
        self.emb_gain = torch.nn.Parameter(torch.zeros([]))
        self.emb_linear = MPConv(emb_channels, inner_channels, kernel=(1,1), groups=emb_linear_groups)
        self.emb_res_balance = AdaptiveGroupBalance(emb_channels, mlp_groups, balance_logits_offset,
            min_balance=adg_min_balance, max_balance=adg_max_balance, weight_decay=adg_weight_decay)
    
        self.attn_q = MPConv(out_channels, out_channels, kernel=(1,1), groups=mlp_groups)
        self.attn_k = MPConv(out_channels, out_channels, kernel=(1,1), groups=mlp_groups)
        self.attn_v = MPConv(out_channels, out_channels, kernel=(1,1), groups=mlp_groups)
        self.attn_proj = MPConv(out_channels, out_channels, kernel=(1,1), groups=mlp_groups)

        self.emb_gain_qkv = torch.nn.Parameter(torch.zeros([]))
        self.emb_linear_qkv = MPConv(emb_channels, out_channels, kernel=(1,1), groups=emb_linear_groups)
        self.emb_attn_balance = AdaptiveGroupBalance(emb_channels, mlp_groups, balance_logits_offset,
            min_balance=adg_min_balance, max_balance=adg_max_balance, weight_decay=adg_weight_decay)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:

        y = self.conv_res0(x)

        c = self.emb_linear(emb, gain=self.emb_gain) + 1.
        y = mp_silu(normalize_groups(y * c, groups=self.mlp_groups))

        if self.dropout != 0 and self.training == True: # magnitude preserving fix for dropout
            y = torch.nn.functional.dropout(y, p=self.dropout) * (1. - self.dropout)**0.5

        y: torch.Tensor = self.conv_res1(y)
        x = self.emb_res_balance(x, y, emb)

        c = self.emb_linear_qkv(emb, gain=self.emb_gain_qkv) + 1.
        y = x * c

        B, C, H, W = y.shape

        q: torch.Tensor = self.attn_q(y).permute(0, 3, 2, 1)
        k: torch.Tensor = self.attn_k(y).permute(0, 3, 2, 1)
        v: torch.Tensor = self.attn_v(y).permute(0, 3, 2, 1)
        q = q.reshape(B, W, H, self.mlp_groups, self.channels_per_head)
        k = k.reshape(B, W, H, self.mlp_groups, self.channels_per_head)
        v = v.reshape(B, W, H, self.mlp_groups, self.channels_per_head)
        q = normalize(q, dim=4)
        k = normalize(k, dim=4)
        v = normalize(v, dim=4)

        y = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        y = y.permute(0, 3, 4, 2, 1).reshape(B, C, H, W)

        y = self.attn_proj(y)
        x = self.emb_attn_balance(x, y, emb)

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
                        "balance_logits_offset": config.balance_logits_offset,
                        "channels_per_head": config.channels_per_head,
                        "adg_min_balance": config.adg_min_balance,
                        "adg_max_balance": config.adg_max_balance,
                        "adg_weight_decay": config.adg_weight_decay}

        cblock = [config.model_channels * x for x in config.channel_mult]
        cnoise = int(config.model_channels * config.channel_mult_noise) if config.channel_mult_noise is not None else max(cblock)
        cemb = int(config.model_channels * config.channel_mult_emb) if config.channel_mult_emb is not None else max(cblock)

        self.num_levels = len(config.channel_mult)
        self.num_psd_levels = len(config.in_psd_num_freqs)
        
        assert self.num_levels == 1
        assert cnoise % 2 == 0
        assert cemb % config.emb_linear_groups == 0

        self.psd_freqs_per_freq = 2 ** (self.num_psd_levels - 1)
        for i in range(self.num_psd_levels):
            assert config.in_psd_num_freqs[i] % self.psd_freqs_per_freq == 0
        
        # Embedding.
        self.emb_fourier = MPFourier(cnoise)
        self.emb_noise = MPConv(cnoise, cemb, kernel=())
        self.emb_label = MPConv(config.in_channels_emb, cemb, kernel=()) if config.in_channels_emb > 0 else None

        # Training uncertainty estimation.
        self.logvar_fourier = MPFourier(config.logvar_channels)
        self.logvar_linear = MPConv(config.logvar_channels, 1, kernel=(), disable_weight_norm=True)
        self.logvar_linear.weight.data.fill_(0)

        assert config.in_channels_x_ref > 0
        self.emb_x_ref = MPConv(config.in_channels_x_ref * self.psd_freqs_per_freq * self.num_psd_levels, cemb, kernel=(1,1), bias=True)
        self.conv_in = MPConv(config.in_channels * self.psd_freqs_per_freq * self.num_psd_levels, cblock[0], kernel=(1,1), bias=True)

        self.dec = torch.nn.ModuleDict()
        for idx in range(config.num_layers_per_block):
            self.dec[f"block0_layer{idx}"] = Block(
                cblock[0], cblock[0], cemb, config.in_num_freqs, **block_kwargs)
        
        self.out_gain = torch.nn.Parameter(torch.zeros([]))
        self.conv_out = MPConv(cblock[0], config.out_channels * self.psd_freqs_per_freq * self.num_psd_levels, kernel=(1,1))

    def get_embeddings(self, emb_in: torch.Tensor, conditioning_mask: torch.Tensor) -> torch.Tensor:
        if self.config.in_channels_emb > 0:
            c_embedding = self.emb_label(normalize(emb_in).to(device=self.device, dtype=self.dtype))
            u_embedding = torch.zeros_like(c_embedding)
            return mp_sum(u_embedding, c_embedding, t=conditioning_mask.unsqueeze(1).to(u_embedding.dtype))
        else:
            return None
        
    def get_sigma_loss_logvar(self, sigma: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.logvar_linear(self.logvar_fourier(sigma.flatten().log() / 4)).view(-1, 1, 1, 1).float()
    
    def get_latent_shape(self, latent_shape: Union[torch.Size, tuple[int, int, int, int]]) -> torch.Size:
        return latent_shape

    def forward(self, x_in: torch.Tensor,
                sigma: torch.Tensor,
                format: MS_MDCT_DualFormat,
                embeddings: torch.Tensor,
                x_ref: Optional[torch.Tensor] = None,
                perturbed_input: Optional[torch.Tensor] = None,
                conditioning_mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        with torch.no_grad():
            sigma = sigma.view(-1, 1, 1, 1)
            
            # Preconditioning weights.
            c_skip = self.config.sigma_data ** 2 / (sigma ** 2 + self.config.sigma_data ** 2)
            c_out = sigma * self.config.sigma_data / (sigma ** 2 + self.config.sigma_data ** 2).sqrt()
            c_in = 1 / (self.config.sigma_data ** 2 + sigma ** 2).sqrt()
            c_noise = (sigma.flatten().log() / 4).to(self.dtype)

            if perturbed_input is not None:
                x = (c_in * perturbed_input).to(dtype=torch.bfloat16)
            else:
                x = (c_in * x_in).to(dtype=torch.bfloat16)

            emb = self.emb_fourier(c_noise)

        x = patch_ms_psd(format.unflatten_mdct_phase_psd(x), self.num_psd_levels)
        x_ref = patch_ms_psd(x_ref, self.num_psd_levels).to(dtype=torch.bfloat16)

        # embedding
        if conditioning_mask is not None: # nuisance due to ddp wrapper limitations
            assert self.training == True
            embeddings = self.get_embeddings(embeddings, conditioning_mask)
        else:
            assert self.training == False

        emb = self.emb_noise(emb)
        if self.config.in_channels_emb > 0:
            emb = mp_silu(mp_sum(emb, embeddings, t=self.config.label_balance))
        emb = emb[:, :, None, None].to(dtype=torch.bfloat16)
        emb = mp_silu(mp_sum(emb, self.emb_x_ref(x_ref), t=0.5))

        x = self.conv_in(x)

        for name, block in self.dec.items():
            x = block(x, emb)

        x: torch.Tensor = self.conv_out(x, gain=self.out_gain)
        x = format.flatten_mdct_phase_psd(unpatch_ms_psd(x, self.num_psd_levels))

        D_x: torch.Tensor = c_skip * x_in.float() + c_out * x.float()
        return D_x, self.get_sigma_loss_logvar(sigma)
