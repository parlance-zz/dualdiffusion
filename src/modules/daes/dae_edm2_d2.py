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
from typing import Union, Literal

import torch

from modules.daes.dae import DualDiffusionDAE, DualDiffusionDAEConfig
from modules.mp_tools import (
    mp_silu, mp_sum, normalize, MPConv3D, channel_to_space3d,
    wavelet_decompose2d, wavelet_recompose2d
)
from utils.dual_diffusion_utils import tensor_4d_to_5d, tensor_5d_to_4d

import math

@dataclass
class DAE_D2_Config(DualDiffusionDAEConfig):

    in_channels: int     = 1
    in_channels_emb: int = 1024
    in_num_freqs: int    = 256
    out_channels: int    = 1
    latent_channels: int = 4

    model_channels: int       = 16           # Base multiplier for the number of channels.
    noise_channels: int       = 32
    downsample_ratio: int     = 8
    channel_mult_enc: int     = 1      # Multiplier for final embedding dimensionality.
    channel_mult_dec: int     = 8      # Multiplier for final embedding dimensionality.
    channel_mult_emb: int     = 4      # Multiplier for final embedding dimensionality.
    channels_per_head: int    = 64           # Number of channels per attention head.
    num_enc_layers_per_block: int = 4        # Number of resnet blocks per resolution.
    num_dec_layers_per_block: int = 4        # Number of resnet blocks per resolution.
    res_balance: float        = 0.5          # Balance between main branch (0) and residual branch (1).
    attn_balance: float       = 0.5          # Balance between main branch (0) and self-attention (1).
    attn_levels: list[int]    = ()           # List of resolution levels to use self-attention.
    mlp_multiplier: int   = 2                # Multiplier for the number of channels in the MLP.
    mlp_groups: int       = 1                # Number of groups for the MLPs.
    emb_linear_groups: int = 1
    add_constant_channel: bool = True
    add_pixel_norm: bool       = False

    wavelet_rescale_factors: list[float] = (0.60, 0.74, 0.90, 0.98)
    
class Block(torch.nn.Module):

    def __init__(self,
        level: int,                             # Resolution level.
        in_channels: int,                       # Number of input channels.
        out_channels: int,                      # Number of output channels.
        emb_channels: int,                      # Number of embedding channels.
        flavor: Literal["enc", "dec"] = "enc",
        resample_mode: Literal["keep", "up", "down"] = "keep",
        dropout: float         = 0.,       # Dropout probability.
        res_balance: float     = 0.3,      # Balance between main branch (0) and residual branch (1).
        attn_balance: float    = 0.3,      # Balance between main branch (0) and self-attention (1).
        clip_act: float        = 256,      # Clip output activations. None = do not clip.
        mlp_multiplier: int    = 1,        # Multiplier for the number of channels in the MLP.
        mlp_groups: int        = 1,        # Number of groups for the MLP.
        emb_linear_groups: int = 1,
        channels_per_head: int = 64,       # Number of channels per attention head.
        use_attention: bool    = False,    # Use self-attention in this block.
        use_pixel_norm: bool   = False,
        noise_channels: int    = 0
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
        self.noise_channels = noise_channels

        if resample_mode == "up":
            in_channels = in_channels // 4 + noise_channels

        self.conv_res0 = MPConv3D(out_channels if flavor == "enc" else in_channels,
                        out_channels * mlp_multiplier, kernel=(2,3,3), groups=mlp_groups)
        self.conv_res1 = MPConv3D(out_channels * mlp_multiplier,
                    out_channels, kernel=(2,3,3), groups=mlp_groups)

        if in_channels != out_channels or mlp_groups > 1:
            self.conv_skip = MPConv3D(in_channels, out_channels, kernel=(1,1,1), groups=1)
        else:
            self.conv_skip = None

        self.emb_gain = torch.nn.Parameter(torch.zeros([]))
        self.emb_linear = MPConv3D(emb_channels, out_channels * mlp_multiplier,
            kernel=(1,1,1), groups=emb_linear_groups) if emb_channels != 0 else None
        
        if self.use_attention:
            self.emb_gain_qk = torch.nn.Parameter(torch.zeros([]))
            self.emb_gain_v = torch.nn.Parameter(torch.zeros([]))
            if emb_channels != 0:
                self.emb_linear_qk = MPConv3D(emb_channels, out_channels, kernel=(1,1,1), groups=1)
                self.emb_linear_v = MPConv3D(emb_channels, out_channels, kernel=(1,1,1), groups=1)
            else:
                self.emb_linear_qk = None
                self.emb_linear_v = None

            self.attn_qk = MPConv3D(out_channels, out_channels * 2, kernel=(1,1,1))
            self.attn_v = MPConv3D(out_channels, out_channels, kernel=(1,1,1))
            self.attn_proj = MPConv3D(out_channels, out_channels, kernel=(1,1,1))

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        
        if self.resample_mode == "up":
            x = channel_to_space3d(x)
            noise = torch.randn_like(x[:, :self.noise_channels])
            x = torch.cat((x, noise), dim=1)

        if self.flavor == "enc":
            if self.conv_skip is not None:
                x = self.conv_skip(x)
            if self.use_pixel_norm == True:
                x = normalize(x, dim=1)

        y = self.conv_res0(mp_silu(x))

        if self.emb_linear is not None:
            c = self.emb_linear(emb, gain=self.emb_gain) + 1.
            y = mp_silu(y * c)
        else:
            y = mp_silu(y)

        if self.dropout != 0 and self.training == True: # magnitude preserving fix for dropout
            y = torch.nn.functional.dropout(y, p=self.dropout) * (1. - self.dropout)**0.5

        y = self.conv_res1(y)

        if self.flavor == "dec" and self.conv_skip is not None:
            x = self.conv_skip(x)
        x = mp_sum(x, y, t=self.res_balance)
        
        if self.use_attention:

            c = self.emb_linear_qk(emb, gain=self.emb_gain_qk) + 1.

            qk = self.attn_qk(x * c)
            qk = qk.reshape(qk.shape[0], self.num_heads, -1, 2, y.shape[2] * y.shape[3])
            q, k = normalize(qk, dim=2).unbind(3)

            v = self.attn_v(x)
            v = v.reshape(v.shape[0], self.num_heads, -1, y.shape[2] * y.shape[3])
            v = normalize(v, dim=2)

            y = torch.nn.functional.scaled_dot_product_attention(q.transpose(-1, -2),
                                                                 k.transpose(-1, -2),
                                                                 v.transpose(-1, -2)).transpose(-1, -2)
            y = y.reshape(*x.shape)

            c = self.emb_linear_v(emb, gain=self.emb_gain_v) + 1.
            y = mp_silu(y * c)

            y = self.attn_proj(y)
            x = mp_sum(x, y, t=self.attn_balance)

        if self.clip_act is not None:
            x = x.clip_(-self.clip_act, self.clip_act)

        if self.clip_act is not None:
            x = x.clip_(-self.clip_act, self.clip_act)
        return x

class DAE_D2(DualDiffusionDAE):

    supports_channels_last: Union[bool, Literal["3d"]] = "3d"

    def __init__(self, config: DAE_D2_Config) -> None:
        super().__init__()
        self.config = config

        block_kwargs = {"mlp_multiplier": config.mlp_multiplier,
                        "mlp_groups": config.mlp_groups,
                        "emb_linear_groups": config.emb_linear_groups,
                        "res_balance": config.res_balance,
                        "attn_balance": config.attn_balance,
                        "channels_per_head": config.channels_per_head,
                        "use_pixel_norm": config.add_pixel_norm}
        
        cemb = config.model_channels * config.channel_mult_emb * config.mlp_multiplier if config.in_channels_emb > 0 else 0

        self.num_levels = int(math.log2(config.downsample_ratio)) + 1
        self.out_gain = torch.nn.Parameter(torch.ones([]))
        self.recon_loss_logvar = torch.nn.Parameter(torch.zeros([]))

        # Embedding.
        if config.in_channels_emb > 0:
            self.emb_label = MPConv3D(config.in_channels_emb, cemb, kernel=())
            self.emb_dim = cemb
        else:
            cemb = 0
            self.emb_label = None
            self.emb_dim = 0

        in_channels = 1 + int(config.add_constant_channel)
        out_channels = 1
        enc_channels = config.model_channels * config.channel_mult_enc
        dec_channels = config.model_channels * config.channel_mult_dec
        level = 0

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        self.enc[f"conv_in"] = MPConv3D(in_channels, enc_channels, kernel=(2,3,3))
        
        for idx in range(config.num_enc_layers_per_block):
            self.enc[f"block{level}_layer{idx}"] = Block(level, enc_channels, enc_channels, cemb,
                use_attention=level in config.attn_levels, flavor="enc", **block_kwargs)

        self.conv_latents_out = MPConv3D(enc_channels, config.latent_channels, kernel=(2,3,3))
        self.conv_latents_in = MPConv3D(config.latent_channels + int(config.add_constant_channel), dec_channels, kernel=(2,3,3))

        # Decoder.
        self.dec = torch.nn.ModuleDict()

        noise_channels = config.noise_channels
        for level in reversed(range(0, self.num_levels)):
            
            if level == self.num_levels - 1:
                self.dec[f"block{level}_in0"] = Block(level, dec_channels, dec_channels, cemb,
                    use_attention=level in config.attn_levels, flavor="dec", **block_kwargs)
            else:
                self.dec[f"block{level}_up"] = Block(level, dec_channels, dec_channels, cemb, noise_channels=noise_channels,
                    use_attention=level in config.attn_levels, flavor="dec", resample_mode="up", **block_kwargs)
                noise_channels //= 2
                
            for idx in range(config.num_dec_layers_per_block):
                self.dec[f"block{level}_layer{idx}"] = Block(level, dec_channels, dec_channels, cemb,
                    use_attention=level in config.attn_levels, flavor="dec", **block_kwargs)
        
        self.conv_out = MPConv3D(dec_channels, out_channels, kernel=(2,3,3))
            
    def get_embeddings(self, emb_in: torch.Tensor) -> torch.Tensor:
        if self.emb_label is not None:
            return self.emb_label(normalize(emb_in).to(device=self.device, dtype=self.dtype))
        else:
            return None
    
    def get_recon_loss_logvar(self) -> torch.Tensor:
        return self.recon_loss_logvar
    
    def get_latent_shape(self, sample_shape: Union[torch.Size, tuple[int, int, int, int]]) -> torch.Size:
        if len(sample_shape) == 4:
            return (sample_shape[0], self.config.latent_channels * 2,
                    sample_shape[2] // 2 ** (self.num_levels-1),
                    sample_shape[3] // 2 ** (self.num_levels-1))
        else:
            raise ValueError(f"Invalid sample shape: {sample_shape}")
        
    def get_mel_spec_shape(self, latent_shape: Union[torch.Size, tuple[int, int, int, int]]) -> torch.Size:
        if len(latent_shape) == 4:
            return (latent_shape[0], 2,
                    latent_shape[2] * 2 ** (self.num_levels-1),
                    latent_shape[3] * 2 ** (self.num_levels-1))
        else:
            raise ValueError(f"Invalid latent shape: {latent_shape}")
        
    def encode(self, x: torch.Tensor, embeddings: torch.Tensor, normalize_latents: bool = True) -> torch.Tensor:

        x = tensor_4d_to_5d(x, num_channels=1).to(memory_format=torch.channels_last_3d)
        x = torch.cat((x, torch.ones_like(x[:, :1])), dim=1)

        if embeddings is not None:
            embeddings = embeddings.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        for name, block in self.enc.items():
            x = block(x) if "conv" in name else block(x, embeddings)
        
        latents = tensor_5d_to_4d(self.conv_latents_out(x)).to(memory_format=torch.channels_last)
        latents = torch.nn.functional.avg_pool2d(latents, self.config.downsample_ratio)
        if normalize_latents == True:
            return normalize(latents)
        else:
            return latents

    def decode(self, x: torch.Tensor, embeddings: torch.Tensor, training: bool = False) -> torch.Tensor:

        x = tensor_4d_to_5d(x, num_channels=self.config.latent_channels).to(memory_format=torch.channels_last_3d)
        x = torch.cat((x, torch.ones_like(x[:, :1])), dim=1)
        x = self.conv_latents_in(x)

        if embeddings is not None:
            embeddings = embeddings.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        for block in self.dec.values():
            x = block(x, embeddings)

        dec_out = tensor_5d_to_4d(self.conv_out(x, gain=self.out_gain)).to(memory_format=torch.channels_last)

        # renormalize wavelet levels by measured relative variance, but only in inference
        if training == False and len(self.config.wavelet_rescale_factors) > 0: 
            
            wavelets = wavelet_decompose2d(dec_out, num_levels=len(self.config.wavelet_rescale_factors))
            for i in range(len(self.config.wavelet_rescale_factors)):
                wavelets[i] /= self.config.wavelet_rescale_factors[i]**0.5

            dec_out = wavelet_recompose2d(wavelets)

        return dec_out
    
    def forward(self, samples: torch.Tensor, dae_embeddings: torch.Tensor,
            add_latents_noise: float = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        latents = self.encode(samples, dae_embeddings, normalize_latents=False)
        latents_pre_norm_std = latents.std(dim=(1,2,3))
        latents = normalize(latents)
        if add_latents_noise > 0:
            latents = normalize(latents + torch.randn_like(latents))
        
        reconstructed = self.decode(latents, dae_embeddings, training=True)

        return latents, reconstructed, latents_pre_norm_std

    def tiled_encode(self, x: torch.Tensor, embeddings: torch.Tensor, max_chunk: int = 6144, overlap: int = 256) -> torch.Tensor:

        x_w = x.shape[-1]
        ds = self.config.downsample_ratio
        
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