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
from numpy import ndarray

from modules.daes.dae import DualDiffusionDAE, DualDiffusionDAEConfig
from modules.mp_tools import MPConv, AdaptiveGroupBalance, mp_silu, normalize, resample_1d, normalize_groups
from utils.resample import FilteredDownsample1D


@dataclass
class DAE_Config(DualDiffusionDAEConfig):

    in_channels:  int = 512
    out_channels: int = 512
    in_channels_emb: int = 0
    latent_channels: int = 256
    in_num_freqs: int = 256

    model_channels: int   = 4096              # Base multiplier for the number of channels.
    channel_mult_enc: int = 1 
    channel_mult_dec: list[int] = (1,1,1,1)   # Per-resolution multipliers for the number of channels.
    channel_mult_emb: Optional[int] = 1       # Multiplier for final embedding dimensionality.
    channels_per_head: int    = 128           # Number of channels per attention head.
    attn_logit_scale: float   = 1
    num_enc_layers: int = 8
    num_dec_layers_per_block: int = 2        # Number of resnet blocks per resolution.
    balance_logits_offset: float = -1.75
    mlp_multiplier: int    = 2               # Multiplier for the number of channels in the MLP.
    mlp_groups: int        = 32              # Number of groups for the MLPs.
    emb_linear_groups: int = 32

class LatentStatsTracker(torch.nn.Module):

    def __init__(self, num_channels: int, momentum: float = 0.99, eps: float = 1e-6) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.momentum = momentum
        self.eps = eps
        
        self.mean: torch.Tensor
        self.register_buffer("mean", torch.zeros(num_channels))
        self.var: torch.Tensor
        self.register_buffer("var", torch.ones(num_channels))

        self.global_mean: torch.Tensor
        self.register_buffer("global_mean", torch.zeros(1))
        self.global_var: torch.Tensor
        self.register_buffer("global_var", torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self.training == True:
            dx = x.detach().to(dtype=self.mean.dtype)

            per_channel_mean = dx.mean(dim=(0,2,3))
            self.mean.lerp_(per_channel_mean, 1. - self.momentum)
            per_channel_var = dx.var(dim=(0,2,3))
            self.var.lerp_(per_channel_var, 1. - self.momentum)

            global_mean = dx.mean()
            self.global_mean.lerp_(global_mean, 1. - self.momentum)
            global_var = dx.var()
            self.global_var.lerp_(global_var, 1. - self.momentum)

        return x
    
    def remove_mean(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean[None, :, None, None].detach()).to(dtype=x.dtype)
    
    def add_mean(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mean[None, :, None, None].detach().to(dtype=x.dtype)
    
    def unscale(self, x: torch.Tensor) -> torch.Tensor:
        std = (self.var + self.eps).pow(0.5)
        return (x / std[None, :, None, None].detach()).to(dtype=x.dtype)
    
    def rescale(self, x: torch.Tensor) -> torch.Tensor:
        std = (self.var + self.eps).pow(0.5)
        return (x * std[None, :, None, None].detach()).to(dtype=x.dtype)
    
class Block(torch.nn.Module):

    def __init__(self,
        level: int,                             # Resolution level.
        in_channels: int,                       # Number of input channels.
        out_channels: int,                      # Number of output channels.
        emb_channels: int,                      # Number of embedding channels.
        flavor: Literal["enc", "dec"],
        resample_mode: Literal["keep", "up", "down"] = "keep",
        dropout: float         = 0.,       # Dropout probability.
        balance_logits_offset: float = -2,
        clip_act: float        = 256,      # Clip output activations. None = do not clip.
        mlp_multiplier: int    = 4,        # Multiplier for the number of channels in the MLP.
        mlp_groups: int        = 4,        # Number of groups for the MLP.
        emb_linear_groups: int = 4,
        channels_per_head: int = 64,       # Number of channels per attention head.
        attn_logit_scale: float = 1.
    ) -> None:
        super().__init__()

        assert out_channels % channels_per_head == 0

        self.level = level
        self.num_heads = out_channels // mlp_groups // channels_per_head
        self.channels_per_head = channels_per_head
        self.mlp_groups = mlp_groups
        self.out_channels = out_channels
        self.flavor = flavor
        self.resample_mode = resample_mode
        self.dropout = dropout
        self.balance_logits_offset = balance_logits_offset
        self.clip_act = clip_act
        self.attn_logit_scale = attn_logit_scale
        
        inner_channels = out_channels * mlp_multiplier

        assert self.num_heads == 1
        assert emb_channels % emb_linear_groups == 0
        assert inner_channels % mlp_groups == 0
        assert inner_channels % emb_linear_groups == 0
        assert out_channels % mlp_groups == 0
        assert in_channels % mlp_groups == 0

        self.conv_res0 = MPConv(in_channels, inner_channels,  kernel=(1,3), groups=mlp_groups)
        self.conv_res1 = MPConv(inner_channels, out_channels, kernel=(1,3), groups=mlp_groups)

        if emb_channels > 0:
            self.emb_gain = torch.nn.Parameter(torch.zeros([]))
            self.emb_linear = MPConv(emb_channels, inner_channels, kernel=(1,1), groups=emb_linear_groups)
        else:
            self.emb_gain = None
            self.emb_linear = None

        self.emb_attn_balance = AdaptiveGroupBalance(emb_channels, mlp_groups, balance_logits_offset, min_balance=0.01)
        self.emb_res_balance  = AdaptiveGroupBalance(emb_channels, mlp_groups, balance_logits_offset, min_balance=0.01)

        self.attn_q = MPConv(out_channels, out_channels, kernel=(1,1), groups=mlp_groups)
        self.attn_k = MPConv(out_channels, out_channels, kernel=(1,1), groups=mlp_groups)
        self.attn_v = MPConv(out_channels, out_channels, kernel=(1,1), groups=mlp_groups)
        self.attn_proj = MPConv(out_channels, out_channels, kernel=(1,1), groups=mlp_groups)

        if emb_channels > 0:
            self.emb_gain_qkv = torch.nn.Parameter(torch.zeros([]))
            self.emb_linear_qkv = MPConv(emb_channels, out_channels, kernel=(1,1), groups=emb_linear_groups)
        else:
            self.emb_gain_qkv = None
            self.emb_linear_qkv = None

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        
        x = resample_1d(x, self.resample_mode)

        if self.emb_linear_qkv is not None:
            c = self.emb_linear_qkv(emb, gain=self.emb_gain_qkv) + 1.
            y = x * c
        else:
            y = x

        B, C, H, W = y.shape

        q: torch.Tensor = self.attn_q(y).permute(0, 3, 2, 1)
        k: torch.Tensor = self.attn_k(y).permute(0, 3, 2, 1)
        v: torch.Tensor = self.attn_v(y).permute(0, 3, 2, 1)
        q = q.reshape(B, W, 1, self.mlp_groups, self.channels_per_head)
        k = k.reshape(B, W, 1, self.mlp_groups, self.channels_per_head)
        v = v.reshape(B, W, 1, self.mlp_groups, self.channels_per_head)
        q = normalize(q, dim=4)
        k = normalize(k, dim=4)
        v = normalize(v, dim=4)

        y = torch.nn.functional.scaled_dot_product_attention(q, k, v,
            scale=self.attn_logit_scale / self.channels_per_head**0.5)
        
        y = y.permute(0, 3, 4, 2, 1).reshape(B, C, H, W)

        y = self.attn_proj(y)
        x = self.emb_attn_balance(x, y, emb)

        y = self.conv_res0(x)

        if self.emb_linear is not None:
            c = self.emb_linear(emb, gain=self.emb_gain) + 1.
            y = y * c

        y = mp_silu(normalize_groups(y, groups=self.mlp_groups))
        
        if self.dropout != 0 and self.training == True: # magnitude preserving fix for dropout
            y = torch.nn.functional.dropout(y, p=self.dropout) * (1. - self.dropout)**0.5

        y: torch.Tensor = self.conv_res1(y)
        x = self.emb_res_balance(x, y, emb)

        if self.clip_act is not None:
            x = x.clip(-self.clip_act, self.clip_act)

        return x

class DAE(DualDiffusionDAE):

    def __init__(self, config: DAE_Config) -> None:
        super().__init__()
        self.config = config

        block_kwargs = {"mlp_multiplier": config.mlp_multiplier,
                        "mlp_groups": config.mlp_groups,
                        "emb_linear_groups": config.emb_linear_groups,
                        "balance_logits_offset": config.balance_logits_offset,
                        "channels_per_head": config.channels_per_head,
                        "attn_logit_scale": config.attn_logit_scale}

        cenc = config.model_channels * config.channel_mult_enc
        cdec = [config.model_channels * x for x in config.channel_mult_dec]
        cemb = int(config.model_channels * config.channel_mult_emb) if config.channel_mult_emb is not None else max(cdec)
        cdata = config.in_channels

        self.num_levels = len(config.channel_mult_dec)
        self.downsample_ratio = 2 ** (self.num_levels - 1)

        if config.in_channels_emb > 0:
            self.emb_label = MPConv(config.in_channels_emb, cemb, kernel=())
        else:
            self.emb_label = None
            cemb = 0

        # encoder
        self.enc = torch.nn.ModuleDict()
        self.enc[f"conv_in"] = MPConv(cdata, cenc, kernel=(1,1), bias=True)

        for idx in range(config.num_enc_layers):
            self.enc[f"block_0_layer{idx}"] = Block(0, cenc, cenc, cemb, flavor="enc", **block_kwargs)

        self.conv_latents_out = MPConv(cenc, config.latent_channels, kernel=(1,1))
        self.conv_latents_out_gain = torch.nn.Parameter(torch.ones([]))

        self.downsample = FilteredDownsample1D(beta=6.95, k_size=31, factor=2)

        self.latents_stats_tracker = LatentStatsTracker(config.latent_channels)
        self.conv_latents_in  = MPConv(config.latent_channels, cdec[-1], kernel=(1,1), bias=True)

        # decoder
        self.dec = torch.nn.ModuleDict()
        cin = cdec[-1]

        for level in reversed(range(0, self.num_levels)):
            
            cout = cdec[level]

            if level == self.num_levels - 1:
                self.dec[f"block{level}_in0"] = Block(level, cin, cout, cemb, flavor="dec", **block_kwargs)
            else:
                self.dec[f"block{level}_up"] = Block(level, cin, cout, cemb, flavor="dec", resample_mode="up", **block_kwargs)

            for idx in range(config.num_dec_layers_per_block):
                self.dec[f"block{level}_layer{idx}"] = Block(level, cout, cout, cemb, flavor="dec", **block_kwargs)

            cin = cout

        self.conv_out = MPConv(cout, config.out_channels, kernel=(1,1))
        self.conv_out_gain = torch.nn.Parameter(torch.ones([]))

    def get_embeddings(self, emb_in: torch.Tensor) -> torch.Tensor:
        if self.emb_label is not None:
            return self.emb_label(normalize(emb_in).to(device=self.device, dtype=self.dtype))
        else:
            return None
        
    def get_recon_loss_logvar(self) -> torch.Tensor:
        return self.recon_loss_logvar
    
    def get_latent_shape(self, mdct_shape: Union[torch.Size, tuple[int, int, int, int]]) -> torch.Size:
        if len(mdct_shape) == 4:
            return (mdct_shape[0], self.config.latent_channels, mdct_shape[2],
                    mdct_shape[3] // self.downsample_ratio)
        else:
            raise ValueError(f"Invalid sample shape: {mdct_shape}")
    
    def get_mel_spec_shape(self, latent_shape: Union[torch.Size, tuple[int, int, int, int]]) -> torch.Size:
        if len(latent_shape) == 4:
            return (latent_shape[0], self.config.in_channels // 2, latent_shape[2],
                    latent_shape[3] * self.downsample_ratio)
        else:
            raise ValueError(f"Invalid latent shape: {latent_shape}")

    def encode(self, x: torch.Tensor, embeddings: torch.Tensor, training: bool = False) -> torch.Tensor:        

        if embeddings is not None:
            emb = mp_silu(embeddings[..., None, None]).to(dtype=torch.bfloat16)
        else:
            emb = None

        x = x.permute(0, 2, 1, 3).reshape(x.shape[0], x.shape[1]*x.shape[2], 1, x.shape[3]).to(dtype=torch.bfloat16)

        for name, block in self.enc.items():
            x = block(x) if "conv" in name else block(x, emb)

        x = normalize_groups(x, groups=self.config.mlp_groups)
        latents = self.conv_latents_out(x, gain=self.conv_latents_out_gain).float()

        full_latents = latents
        for _ in range(self.num_levels - 1):
            latents = self.downsample(latents)

        latents = latents.to(dtype=x.dtype)
        self.latents_stats_tracker(latents)
        if training == True:
            return latents
        else:
            return latents, full_latents

    def decode(self, x: torch.Tensor, embeddings: torch.Tensor, training: bool = False) -> torch.Tensor:

        if embeddings is not None:
            emb = mp_silu(embeddings[..., None, None]).to(dtype=torch.bfloat16)
        else:
            emb = None
        
        x = self.conv_latents_in(x)

        for name, block in self.dec.items():
            x = block(x, emb)

        out: torch.Tensor = self.conv_out(x, gain=self.conv_out_gain)
        return out

    def forward(self, samples: torch.Tensor, audio_embeddings: torch.Tensor,
            latents_sigma: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, ...]:

        dae_embeddings = self.get_embeddings(audio_embeddings)
        pre_norm_latents = self.encode(samples, dae_embeddings, training=True)
        
        latents = pre_norm_latents

        if latents_sigma is not None:
            latents = (latents + latents_sigma.view(-1, 1, 1, 1) * torch.randn_like(latents)) / (1 + latents_sigma.view(-1, 1, 1, 1)**2)**0.5
        
        out = self.decode(latents, dae_embeddings, training=True)
        return latents, out, pre_norm_latents

    def tiled_encode(self, x: torch.Tensor, embeddings: torch.Tensor, max_chunk: int = 6144, overlap: int = 256) -> torch.Tensor:
        raise NotImplementedError()

    def latents_to_img(self, latents: torch.Tensor) -> ndarray:
        
        latents = latents.reshape(latents.shape[0], latents.shape[1] // 4, 4, latents.shape[3])
        latents = latents.permute(0, 2, 1, 3).contiguous()
        
        return super().latents_to_img(latents, img_split_stereo=False)