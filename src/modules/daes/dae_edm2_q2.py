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

from modules.daes.dae import DualDiffusionDAE, DualDiffusionDAEConfig
from modules.mp_tools import MPConv, mp_silu, mp_sum, normalize, resample_2d, normalize_groups, AdaptiveGroupBalance


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

class MPConvS(torch.nn.Module):

    def __init__(self, in_channels: int, out_channels: int,
                 kernel: tuple[int, int], groups: int = 1, stride: int = 1,
                 disable_weight_norm: bool = False, bias: bool = False) -> None:
        
        super().__init__()

        self.conv0 = MPConv(in_channels, out_channels, kernel, groups, stride, disable_weight_norm, bias)
        self.conv1 = MPConv(in_channels, out_channels, kernel, groups, stride, disable_weight_norm, bias)

    def forward(self, x: torch.Tensor, gain: Union[float, torch.Tensor] = 1) -> torch.Tensor:
        
        x0 = x[0::2]
        x1 = x[1::2]

        y0 = self.conv0(x1) + self.conv1(x0)
        y1 = self.conv0(x0) + self.conv1(x1)

        z = torch.stack((y0, y1), dim=1).reshape(x.shape)

        return z * (gain / 2**0.5)
    
@dataclass
class DAE_Config(DualDiffusionDAEConfig):

    in_channels: int     = 1
    in_channels_emb: int = 1024
    in_num_freqs: int    = 256
    out_channels: int    = 1
    latent_channels: int = 4

    model_channels: int       = 32           # Base multiplier for the number of channels.
    channel_mult_enc: int     = 1            
    channel_mult_dec: list[int] = (1,2,4,8)  
    channel_mult_emb: int     = 4            # Multiplier for final embedding dimensionality.
    channels_per_head: int    = 64           # Number of channels per attention head.
    num_enc_layers: int       = 6            # Number of resnet blocks per resolution.
    num_dec_layers_per_block: int = 3        # Number of resnet blocks per resolution.
    res_balance: float        = 0.3          # Balance between main branch (0) and residual branch (1).
    attn_balance: float       = 0.3          # Balance between main branch (0) and self-attention (1).
    attn_levels: list[int]    = ()           # List of resolution levels to use self-attention.
    balance_logits_offset: float = -1.75
    mlp_multiplier: int    = 2               # Multiplier for the number of channels in the MLP.
    mlp_groups: int        = 1               # Number of groups for the MLPs.
    emb_linear_groups: int = 1
    add_constant_channel: bool = True
    add_pixel_norm: bool       = False
    
class Block(torch.nn.Module):

    def __init__(self,
        level: int,                             # Resolution level.
        in_channels: int,                       # Number of input channels.
        out_channels: int,                      # Number of output channels.
        emb_channels: int,                      # Number of embedding channels.
        flavor: Literal["enc", "dec"] = "enc",
        resample_mode: Literal["keep", "up", "down"] = "keep",
        dropout: float         = 0.,       # Dropout probability.
        balance_logits_offset: float = -2,
        res_balance: float     = 0.3,      # Balance between main branch (0) and residual branch (1).
        attn_balance: float    = 0.3,      # Balance between main branch (0) and self-attention (1).
        clip_act: float        = 256,      # Clip output activations. None = do not clip.
        mlp_multiplier: int    = 1,        # Multiplier for the number of channels in the MLP.
        mlp_groups: int        = 1,        # Number of groups for the MLP.
        emb_linear_groups: int = 1,
        channels_per_head: int = 64,       # Number of channels per attention head.
        use_attention: bool    = False,    # Use self-attention in this block.
        use_pixel_norm: bool   = False,
    ) -> None:
        super().__init__()

        self.level = level
        self.use_attention = use_attention
        self.use_pixel_norm = use_pixel_norm
        self.num_heads = out_channels // channels_per_head
        self.out_channels = out_channels
        self.flavor = flavor
        self.resample_mode = resample_mode
        self.dropout = dropout
        self.res_balance = res_balance
        self.attn_balance = attn_balance
        self.clip_act = clip_act
        self.mlp_groups = mlp_groups

        self.conv_res0 = MPConv(out_channels if flavor == "enc" else in_channels,
                        out_channels * mlp_multiplier, kernel=(3,3), groups=mlp_groups)
        self.conv_res1 = MPConv(out_channels * mlp_multiplier,
                    out_channels, kernel=(3,3), groups=mlp_groups)

        if in_channels != out_channels or mlp_groups > 1:
            self.conv_skip = MPConvS(in_channels, out_channels, kernel=(1,1), groups=1)
        else:
            self.conv_skip = None

        if emb_channels > 0:
            self.emb_gain = torch.nn.Parameter(torch.zeros([]))
            self.emb_linear = MPConv(emb_channels, out_channels * mlp_multiplier,
                kernel=(1,1), groups=emb_linear_groups) if emb_channels != 0 else None
            
            self.emb_res_balance  = AdaptiveGroupBalance(emb_channels, mlp_groups, balance_logits_offset)
        else:
            self.emb_gain = self.emb_linear = None
        
        if self.use_attention == True:
            raise NotImplementedError()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        
        if self.resample_mode == "up":
            x = resample_2d(x, "up")

        if self.flavor == "enc":
            if self.conv_skip is not None:
                x = self.conv_skip(x)
            if self.use_pixel_norm == True:
                x = normalize_groups(x, groups=self.mlp_groups)

        #y = self.conv_res0(mp_silu(x))
        y = self.conv_res0(x)

        if self.emb_linear is not None:
            c = self.emb_linear(emb, gain=self.emb_gain) + 1.
            y = y * c

        y = mp_silu(normalize_groups(y, groups=self.mlp_groups))

        if self.dropout != 0 and self.training == True: # magnitude preserving fix for dropout
            y = torch.nn.functional.dropout(y, p=self.dropout) * (1. - self.dropout)**0.5

        y = self.conv_res1(y)

        if self.flavor == "dec" and self.conv_skip is not None:
            x = self.conv_skip(x)

        if self.emb_res_balance is not None:
            x = self.emb_res_balance(x, y, emb)
        else:
            x = mp_sum(x, y, t=self.res_balance)
        
        if self.use_attention == True:
            raise NotImplementedError()

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
                        "res_balance": config.res_balance,
                        "attn_balance": config.attn_balance,
                        "channels_per_head": config.channels_per_head,
                        "use_pixel_norm": config.add_pixel_norm,
                        "balance_logits_offset": config.balance_logits_offset}
        
        cemb = config.model_channels * config.channel_mult_emb * config.mlp_multiplier if config.in_channels_emb > 0 else 0

        self.num_levels = len(config.channel_mult_dec)
        self.downsample_ratio = 2 ** (self.num_levels - 1)
        self.out_gain = torch.nn.Parameter(torch.ones([]))
        self.recon_loss_logvar = torch.nn.Parameter(torch.zeros([]))

        # embedding
        if config.in_channels_emb > 0:
            self.emb_label = MPConv(config.in_channels_emb, cemb, kernel=())
            self.emb_dim = cemb
        else:
            cemb = 0
            self.emb_label = None
            self.emb_dim = 0

        in_channels  = 1 + int(config.add_constant_channel)
        out_channels = 1
        enc_channels = config.model_channels * config.channel_mult_enc
        dec_channels = [config.model_channels * m for m in config.channel_mult_dec]
        level = 0

        self.latents_stats_tracker = LatentStatsTracker(config.latent_channels * 2)

        # encoder
        self.enc = torch.nn.ModuleDict()
        self.enc[f"conv_in"] = MPConv(in_channels, enc_channels, kernel=(5,5))
        
        for idx in range(config.num_enc_layers):
            self.enc[f"block{level}_layer{idx}"] = Block(level, enc_channels, enc_channels, cemb,
                use_attention=level in config.attn_levels, flavor="enc", **block_kwargs)

        self.conv_latents_out = MPConvS(enc_channels, config.latent_channels, kernel=(3,3))
        self.conv_latents_in = MPConvS(config.latent_channels + int(config.add_constant_channel), dec_channels[-1], kernel=(3,3))

        # decoder
        self.dec = torch.nn.ModuleDict()
        cin = dec_channels[-1]

        for level in reversed(range(0, self.num_levels)):
            
            cout = dec_channels[level]

            if level == self.num_levels - 1:
                self.dec[f"block{level}_in0"] = Block(level, cin, cout, cemb,
                    use_attention=level in config.attn_levels, flavor="dec", **block_kwargs)
            else:
                self.dec[f"block{level}_up"] = Block(level, cin, cout, cemb,
                    use_attention=level in config.attn_levels, flavor="dec", resample_mode="up", **block_kwargs)
                
            for idx in range(config.num_dec_layers_per_block):
                self.dec[f"block{level}_layer{idx}"] = Block(level, cout, cout, cemb,
                    use_attention=level in config.attn_levels, flavor="dec", **block_kwargs)

            cin = cout

        self.conv_out = MPConv(cout, out_channels, kernel=(5,5))
            
    def get_embeddings(self, emb_in: torch.Tensor) -> torch.Tensor:
        if self.emb_label is not None:
            return self.emb_label(normalize(emb_in).to(device=self.device, dtype=self.dtype))
        else:
            return None
    
    def get_recon_loss_logvar(self) -> torch.Tensor:
        return self.recon_loss_logvar
    
    def get_latent_shape(self, mel_spec_shape: Union[torch.Size, tuple[int, int, int, int]]) -> torch.Size:
        if len(mel_spec_shape) == 4:
            return (mel_spec_shape[0], self.config.latent_channels * 2,
                    mel_spec_shape[2] // 2 ** (self.num_levels-1),
                    mel_spec_shape[3] // 2 ** (self.num_levels-1))
        else:
            raise ValueError(f"Invalid sample shape: {mel_spec_shape}")
        
    def get_mel_spec_shape(self, latent_shape: Union[torch.Size, tuple[int, int, int, int]]) -> torch.Size:
        if len(latent_shape) == 4:
            return (latent_shape[0], 2,
                    latent_shape[2] * 2 ** (self.num_levels-1),
                    latent_shape[3] * 2 ** (self.num_levels-1))
        else:
            raise ValueError(f"Invalid latent shape: {latent_shape}")
        
    def encode(self, x: torch.Tensor, embeddings: torch.Tensor, training: bool = False) -> torch.Tensor:

        B,C,H,W = x.shape
        x = x.reshape(B*2, C//2, H, W)
        x = torch.cat((x, torch.ones_like(x[:, :1])), dim=1)

        if embeddings is not None:
            embeddings = embeddings[:, :, None, None]

        for name, block in self.enc.items():
            x = block(x) if "conv" in name else block(x, embeddings)
        
        latents = self.conv_latents_out(x)
        latents = torch.nn.functional.avg_pool2d(latents, self.downsample_ratio)

        B,C,H,W = latents.shape
        latents = latents.reshape(B//2, 2, C, H, W).transpose(1, 2).reshape(B//2, C*2, H, W)

        if training == True:
            self.latents_stats_tracker(latents)
        
        return latents

    def decode(self, x: torch.Tensor, embeddings: torch.Tensor, training: bool = False) -> torch.Tensor:

        B,C,H,W = x.shape
        x = x.reshape(B, C//2, 2, H, W).transpose(1, 2).reshape(B*2, C//2, H, W)

        x = torch.cat((x, torch.ones_like(x[:, :1])), dim=1)
        x = self.conv_latents_in(x)

        if embeddings is not None:
            embeddings = embeddings[:, :, None, None]

        for block in self.dec.values():
            x = block(x, embeddings)

        x: torch.Tensor = self.conv_out(x, gain=self.out_gain)

        B,C,H,W = x.shape
        x = x.reshape(B//2, 2, C, H, W)

        return x
    
    def forward(self, samples: torch.Tensor, dae_embeddings: torch.Tensor, latents_sigma: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        pre_norm_latents = self.encode(samples, dae_embeddings, training=True)

        if latents_sigma is not None:
            pre_norm_latents = pre_norm_latents + latents_sigma * torch.randn_like(pre_norm_latents)
        
        latents = pre_norm_latents

        reconstructed = self.decode(latents, dae_embeddings, training=True)
        return latents, reconstructed, pre_norm_latents

    def tiled_encode(self, x: torch.Tensor, embeddings: torch.Tensor, max_chunk: int = 6144, overlap: int = 256) -> torch.Tensor:

        x_w = x.shape[-1]
        ds = self.downsample_ratio
        
        assert max_chunk % ds == 0, "max_chunk must be divisible by downsample ratio"
        assert overlap % ds == 0, "overlap must be divisible by downsample ratio"
        assert x_w % ds == 0, "sample length must be divisible by downsample ratio"

        if x_w <= max_chunk:
            return self.encode(x, embeddings)
        
        min_chunk_len = overlap * 3
        out_overlap = overlap // ds
        
        latents_shape = (x.shape[0], self.config.latent_channels*2, x.shape[-2] // ds, x.shape[-1] // ds)
        latents = torch.zeros(latents_shape, device=x.device, dtype=x.dtype)
        
        # encode latents in overlapping chunks
        for w_start in range(0, x_w, max_chunk - overlap*2):

            if w_start >= x_w:
                break
                
            # sample boundaries including overlap
            chunk_start = max(0, w_start)
            chunk_end = min(x_w, w_start + max_chunk)
            
            # if last chunk is too small, extend it to the left
            if chunk_end - chunk_start < min_chunk_len:
                chunk_start -= min_chunk_len - (chunk_end - chunk_start)

            chunk = x[:, :, :, chunk_start:chunk_end]
            latents_chunk = self.encode(chunk, embeddings, normalize_latents=False)
            
            # latent boundaries including overlap
            out_start = chunk_start // ds
            out_end = chunk_end // ds
            
            # first chunk: keep left edge, other chunks: discard left overlap
            is_first_chunk = (w_start == 0)
            valid_start = 0 if is_first_chunk else out_overlap
            
            # last chunk: keep right edge, other chunks: discard right overlap
            is_last_chunk = (chunk_end == x_w)
            valid_end = latents_chunk.shape[3] if is_last_chunk else latents_chunk.shape[3] - out_overlap
            
            # latent boundaries excluding overlap
            dest_start = out_start if is_first_chunk else out_start + out_overlap
            dest_end = out_end if is_last_chunk else out_end - out_overlap
            
            latents[:, :, :, dest_start:dest_end] = latents_chunk[:, :, :, valid_start:valid_end]
        
        return normalize(latents)