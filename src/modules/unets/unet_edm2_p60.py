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

# for backwards compatibility with p6

from dataclasses import dataclass
from typing import Union, Optional

import torch

from modules.unets.unet import DualDiffusionUNet, DualDiffusionUNetConfig
from modules.mp_tools import MPConv, MPFourier, AdaptiveGroupBalance, mp_silu, mp_sum, normalize, normalize_groups
from modules.rope import _rope_pair_rotate_partial, _rope_tables_for_seq
from modules.formats.format import DualDiffusionFormat


@dataclass
class UNetConfig(DualDiffusionUNetConfig):

    in_channels:  int = 256
    out_channels: int = 256
    in_channels_emb: int = 1024
    in_num_freqs: int = 1

    sigma_max: float  = 200
    sigma_min: float  = 0.08
    sigma_data: float = 1.

    mp_fourier_ln_sigma_offset: float = 0
    mp_fourier_bandwidth:       float = 1

    adg_min_balance: Optional[float]  = 0.1
    adg_max_balance: Optional[float]  = 0.9
    adg_weight_decay: Optional[float] = None

    model_channels: int  = 4096                # Base multiplier for the number of channels.
    logvar_channels: int = 192                 # Number of channels for training uncertainty estimation.
    channel_mult: list[int] = (1,)             # Per-resolution multipliers for the number of channels.
    channel_mult_noise: Optional[float] = 0.25 # Multiplier for noise embedding dimensionality.
    channel_mult_emb: Optional[float]   = 1    # Multiplier for final embedding dimensionality.
    use_skips: bool = False
    channels_per_head: int      = 128          # Number of channels per attention head.
    rope_channels: int          = 112
    rope_base: float            = 10000.
    rope_scale: Optional[float] = None
    num_layers_per_block: int = 20           # Number of resnet blocks per resolution.
    label_balance: float      = 0.5          # Balance between noise embedding (0) and class embedding (1).
    balance_logits_offset: float = -4
    block_kernel_size: int = 3
    mlp_multiplier: int    = 3               # Multiplier for the number of channels in the MLP.
    mlp_groups: int        = 32              # Number of groups for the MLPs.
    emb_linear_groups: int = 32

class Block(torch.nn.Module):

    def __init__(self,
        level: int,                        # Resolution level.
        in_channels: int,                  # Number of input channels.
        out_channels: int,                 # Number of output channels.
        skip_channels: int,
        emb_channels: int,                 # Number of embedding channels.
        dropout: float         = 0.,       # Dropout probability.
        balance_logits_offset: float = -2, # Offset for the balance logits before sigmoid.
        clip_act: float        = 256,      # Clip output activations. None = do not clip.
        mlp_multiplier: int    = 3,        # Multiplier for the number of channels in the MLP.
        mlp_groups: int        = 32,        # Number of groups for the MLP.
        emb_linear_groups: int = 32,
        channels_per_head: int = 128,       # Number of channels per attention head.
        global_attention: bool = False,
        adg_min_balance: Optional[float]  = 0.1,
        adg_max_balance: Optional[float]  = 0.9,
        adg_weight_decay: Optional[float] = None,
        block_kernel_size: int = 3
    ) -> None:
        super().__init__()

        assert out_channels % channels_per_head == 0

        self.level = level
        self.num_heads = out_channels // mlp_groups // channels_per_head
        self.channels_per_head = channels_per_head
        self.mlp_groups = mlp_groups
        self.out_channels = out_channels
        self.dropout = dropout
        self.balance_logits_offset = balance_logits_offset
        self.clip_act = clip_act
        self.global_attention = global_attention

        inner_channels = out_channels * mlp_multiplier

        assert self.num_heads == 1
        assert emb_channels % emb_linear_groups == 0
        assert inner_channels % mlp_groups == 0
        assert inner_channels % emb_linear_groups == 0
        assert out_channels % mlp_groups == 0
        assert in_channels % mlp_groups == 0

        if skip_channels > 0:
            self.conv_skip = MPConv(skip_channels, out_channels, kernel=(1,1), groups=mlp_groups)
            self.skip_balance = AdaptiveGroupBalance(emb_channels, mlp_groups, balance_logits_offset,
                min_balance=adg_min_balance, max_balance=adg_max_balance, weight_decay=adg_weight_decay)
        else:
            self.conv_skip = None
            self.skip_balance = None

        self.conv_res0 = MPConv(in_channels, inner_channels,  kernel=(1,block_kernel_size), groups=mlp_groups)
        self.conv_res1 = MPConv(inner_channels, out_channels, kernel=(1,block_kernel_size), groups=mlp_groups)
        
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

    def forward(self, x: torch.Tensor, emb: torch.Tensor, skip: torch.Tensor, rope_tables: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:

        c = self.emb_linear_qkv(emb, gain=self.emb_gain_qkv) + 1.
        y = x * c

        B, C, H, W = y.shape

        if self.global_attention == True:

            q: torch.Tensor = self.attn_q(y)
            k: torch.Tensor = self.attn_k(y)
            v: torch.Tensor = self.attn_v(y)
            q = q.reshape(B, self.mlp_groups, self.channels_per_head, H*W)
            k = k.reshape(B, self.mlp_groups, self.channels_per_head, H*W)
            v = v.reshape(B, self.mlp_groups, self.channels_per_head, H*W)
            q = normalize(q, dim=2).transpose(-1, -2)
            k = normalize(k, dim=2).transpose(-1, -2)
            v = normalize(v, dim=2).transpose(-1, -2)

            q_rot = _rope_pair_rotate_partial(q, rope_tables)
            k_rot = _rope_pair_rotate_partial(k, rope_tables)

            y = torch.nn.functional.scaled_dot_product_attention(q_rot, k_rot, v)
            y = y.transpose(-1, -2).reshape(B, C, H, W)
        else:
            q: torch.Tensor = self.attn_q(y).permute(0, 3, 2, 1)
            k: torch.Tensor = self.attn_k(y).permute(0, 3, 2, 1)
            v: torch.Tensor = self.attn_v(y).permute(0, 3, 2, 1)
            q = q.reshape(B, W, 1, self.mlp_groups, self.channels_per_head)
            k = k.reshape(B, W, 1, self.mlp_groups, self.channels_per_head)
            v = v.reshape(B, W, 1, self.mlp_groups, self.channels_per_head)
            q = normalize(q, dim=4)
            k = normalize(k, dim=4)
            v = normalize(v, dim=4)

            y = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            y = y.permute(0, 3, 4, 2, 1).reshape(B, C, H, W)

        y = self.attn_proj(y)
        x = self.emb_attn_balance(x, y, emb)

        y = self.conv_res0(x)

        c = self.emb_linear(emb, gain=self.emb_gain) + 1.
        y = mp_silu(normalize_groups(y * c, groups=self.mlp_groups))

        if self.dropout != 0 and self.training == True: # magnitude preserving fix for dropout
            y = torch.nn.functional.dropout(y, p=self.dropout) * (1. - self.dropout)**0.5

        if self.conv_skip is not None:
            skip = self.conv_skip(skip)
            x = self.skip_balance(x, skip, emb)

        y: torch.Tensor = self.conv_res1(y)
        x = self.emb_res_balance(x, y, emb)

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
                        "adg_weight_decay": config.adg_weight_decay,
                        "block_kernel_size": config.block_kernel_size}

        cblock = [config.model_channels * x for x in config.channel_mult]
        cnoise = int(config.model_channels * config.channel_mult_noise) if config.channel_mult_noise is not None else max(cblock)
        cemb = int(config.model_channels * config.channel_mult_emb) if config.channel_mult_emb is not None else max(cblock)
        cdata = config.in_channels

        self.num_levels = len(config.channel_mult)

        assert self.num_levels == 1
        assert cnoise % 2 == 0
        assert cemb % config.emb_linear_groups == 0
        assert config.rope_channels % 2 == 0
        assert config.rope_channels <= config.channels_per_head

        # embedding
        self.emb_fourier = MPFourier(cnoise, bandwidth=config.mp_fourier_bandwidth)
        self.emb_noise = MPConv(cnoise, cemb, kernel=())

        if config.in_channels_emb > 0:
            self.emb_label = MPConv(config.in_channels_emb, cemb, kernel=())
        else:
            self.emb_label = None

        # training uncertainty estimation
        self.logvar_fourier = MPFourier(config.logvar_channels)
        self.logvar_linear = MPConv(config.logvar_channels, 1, kernel=(), disable_weight_norm=True)
        self.logvar_linear.weight.data.fill_(0)

        # decoder
        self.dec = torch.nn.ModuleDict()
        cout = cdata

        for level, channels in enumerate(cblock):
            
            cin = cout
            cout = channels
            self.dec[f"conv_in"] = MPConv(cin, cout, kernel=(1,1), bias=True)
            
            for idx in range(config.num_layers_per_block):

                cin = cout
                cout = channels

                if config.use_skips == True and idx >= config.num_layers_per_block / 2:
                    cskip = channels
                else:
                    cskip = 0

                global_attention = ((idx % 2) == 0)
                self.dec[f"block{level}_layer{idx}"] = Block(level, cin, cout, cskip, cemb, global_attention=global_attention, **block_kwargs)
                
        self.out_gain = torch.nn.Parameter(torch.zeros([]))
        self.conv_out = MPConv(cout, config.out_channels, kernel=(1,1))

    def get_embeddings(self, emb_in: torch.Tensor, conditioning_mask: torch.Tensor) -> torch.Tensor:
        if self.config.in_channels_emb > 0:
            c_embedding = self.emb_label(normalize(emb_in).to(device=self.device, dtype=self.dtype))
            u_embedding = torch.zeros_like(c_embedding)
            return mp_sum(u_embedding, c_embedding, t=conditioning_mask.unsqueeze(1).to(u_embedding.dtype))
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
                perturbed_input: Optional[torch.Tensor] = None,
                conditioning_mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        with torch.no_grad():
            sigma = sigma.view(-1, 1, 1, 1)
            
            # preconditioning weights
            c_skip = self.config.sigma_data ** 2 / (sigma ** 2 + self.config.sigma_data ** 2)
            c_out = sigma * self.config.sigma_data / (sigma ** 2 + self.config.sigma_data ** 2).sqrt()
            c_in = 1 / (self.config.sigma_data ** 2 + sigma ** 2).sqrt()
            
            # improved lipschitz behavior at low sigma, improved conditioning resolution at snr ~= 1
            #sigma_center = torch.e ** self.config.mp_fourier_ln_sigma_offset
            #c_noise = 2 * (sigma_center / sigma.flatten()).atan() #0 <= c_noise <= pi

            ln_sigma = sigma.flatten().log() - self.config.mp_fourier_ln_sigma_offset
            c_noise = ln_sigma / 4

        if perturbed_input is not None:
            x = (c_in * perturbed_input).to(dtype=torch.bfloat16)
        else:
            x = (c_in * x_in).to(dtype=torch.bfloat16)

        og_x_shape = x.shape
        x = x.permute(0, 2, 1, 3).reshape(x.shape[0], x.shape[1]*x.shape[2], 1, x.shape[3])

        # nuisance due to ddp wrapper limitations
        if conditioning_mask is not None:
            assert self.training == True
            embeddings = self.get_embeddings(embeddings, conditioning_mask)
        else:
            assert self.training == False
        
        # Embedding.
        emb: torch.Tensor = self.emb_noise(self.emb_fourier(c_noise))
        emb = mp_sum(emb, embeddings.to(dtype=emb.dtype), t=self.config.label_balance)
        emb = mp_silu(emb).unsqueeze(2).unsqueeze(3).to(dtype=torch.bfloat16)

        # build rope tables
        rope_tables = _rope_tables_for_seq(x.shape[3], self.config.rope_channels,
            self.config.rope_base, scale=self.config.rope_scale, device=x.device, dtype=x.dtype)

        idx = 0; skips = []
        for name, block in self.dec.items():
            if "conv" in name:
                x = block(x)
            else:
                skip = None

                if self.config.use_skips == True:
                    if idx < self.config.num_layers_per_block / 2 - 0.5:
                        skips.append(x)
                    elif idx >= self.config.num_layers_per_block / 2:
                        skip = skips.pop()

                x = block(x, emb, skip, rope_tables)

                idx += 1

        x: torch.Tensor = self.conv_out(x, gain=self.out_gain)
        x = x.reshape(og_x_shape[0], og_x_shape[2], og_x_shape[1], og_x_shape[3]).permute(0, 2, 1, 3).contiguous()
        
        D_x = c_skip * x_in + c_out * x.float()

        if x_ref is not None:
            #D_x = mp_sum(x_ref[:, :-1].float(), D_x, t=x_ref[:, -1:].float())
            raise NotImplementedError()

        return D_x, self.get_sigma_loss_logvar(sigma)
