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

from modules.daes.dae import DualDiffusionDAE, DualDiffusionDAEConfig
from modules.daes.dae_edm2_g1 import MPConv3D_E
from modules.mp_tools import (
    mp_silu, mp_sum, normalize, resample_3d, MPConv3D
)
from utils.dual_diffusion_utils import tensor_4d_to_5d, tensor_5d_to_4d


@dataclass
class DAE_H1_Config(DualDiffusionDAEConfig):

    in_channels: int     = 1
    in_channels_emb: int = 0
    in_num_freqs: int    = 256
    out_channels: int    = 32
    latent_channels: int = 4

    model_channels: int       = 32           # Base multiplier for the number of channels.
    channel_mult: list[int]   = (1,2,4,8)    # Per-resolution multipliers for the number of channels.
    double_midblock: bool     = True
    midblock_attn: bool       = True
    channel_mult_emb: Optional[int] = 0      # Multiplier for final embedding dimensionality.
    channels_per_head: int    = 64           # Number of channels per attention head.
    num_layers_per_block: int = 3            # Number of resnet blocks per resolution.
    res_balance: float        = 0.3          # Balance between main branch (0) and residual branch (1).
    attn_balance: float       = 0.3          # Balance between main branch (0) and self-attention (1).
    attn_levels: list[int]    = ()           # List of resolution levels to use self-attention.
    mlp_multiplier: int   = 2                # Multiplier for the number of channels in the MLP.
    mlp_groups: int       = 1                # Number of groups for the MLPs.
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
        res_balance: float     = 0.3,      # Balance between main branch (0) and residual branch (1).
        attn_balance: float    = 0.3,      # Balance between main branch (0) and self-attention (1).
        clip_act: float        = 256,      # Clip output activations. None = do not clip.
        mlp_multiplier: int    = 1,        # Multiplier for the number of channels in the MLP.
        mlp_groups: int        = 1,        # Number of groups for the MLP.
        emb_linear_groups: int = 1,
        channels_per_head: int = 64,       # Number of channels per attention head.
        use_attention: bool    = False,    # Use self-attention in this block.
        use_pixel_norm: bool   = False
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
        
        conv_class = MPConv3D_E if flavor == "enc" else MPConv3D

        self.conv_res0 = conv_class(out_channels if flavor == "enc" else in_channels,
                        out_channels * mlp_multiplier, kernel=(2,3,3), groups=mlp_groups)
        self.conv_res1 = conv_class(out_channels * mlp_multiplier,
                    out_channels, kernel=(2,3,3), groups=mlp_groups)

        if in_channels != out_channels or mlp_groups > 1:
            self.conv_skip = MPConv3D(in_channels, out_channels, kernel=(1,1,1), groups=1)
        else:
            self.conv_skip = None

        self.emb_gain = torch.nn.Parameter(torch.zeros([]))
        self.emb_linear = MPConv3D(emb_channels, out_channels * mlp_multiplier,
            kernel=(1,1,1), groups=emb_linear_groups) if emb_channels != 0 else None
        
        if self.use_attention == True:
            self.attn_qkv = MPConv3D(out_channels, out_channels * 3, kernel=(1,1,1))
            self.attn_proj = MPConv3D(out_channels, out_channels, kernel=(1,1,1))

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        
        x = resample_3d(x, mode=self.resample_mode)

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
        
        if self.use_attention == True:

            qkv: torch.Tensor = self.attn_qkv(x)
            
            #                 b  z  w, c, h
            qkv = qkv.permute(0, 2, 4, 1, 3)
            qkv = qkv.reshape(qkv.shape[0]*qkv.shape[1]*qkv.shape[2], self.num_heads, -1, 3, qkv.shape[4])
            q, k, v = normalize(qkv, dim=2).unbind(3)

            y = torch.nn.functional.scaled_dot_product_attention(q.transpose(-1, -2),
                                                                 k.transpose(-1, -2),
                                                                 v.transpose(-1, -2)).transpose(-1, -2)
            
            #             b         , z         , w         , c         , h
            y = y.reshape(x.shape[0], x.shape[2], x.shape[4], x.shape[1], x.shape[3])
            #             b, c, z, h, w
            y = y.permute(0, 3, 1, 4, 2).contiguous(memory_format=torch.channels_last_3d)

            y = self.attn_proj(mp_silu(y))
            x = mp_sum(x, y, t=self.attn_balance)

        if self.clip_act is not None:
            x = x.clip_(-self.clip_act, self.clip_act)

        return x

class DAE_H1(DualDiffusionDAE):

    supports_channels_last: Union[bool, Literal["3d"]] = "3d"

    def __init__(self, config: DAE_H1_Config) -> None:
        super().__init__()
        self.config = config

        block_kwargs = {"mlp_multiplier": config.mlp_multiplier,
                        "mlp_groups": config.mlp_groups,
                        "emb_linear_groups": config.emb_linear_groups,
                        "res_balance": config.res_balance,
                        "attn_balance": config.attn_balance,
                        "channels_per_head": config.channels_per_head,
                        "use_pixel_norm": config.add_pixel_norm}
        
        cblock = [config.model_channels * x for x in config.channel_mult]
        cemb = config.model_channels * config.channel_mult_emb if config.channel_mult_emb is not None else max(cblock)
        cemb *= self.config.mlp_multiplier

        self.num_levels = len(config.channel_mult)
        self.downsample_ratio = 2 ** (self.num_levels - 1)
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

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = 1 + int(config.add_constant_channel)
        for level, channels in enumerate(cblock):
            
            if level == 0:
                cin = cout
                cout = channels
                self.enc[f"conv_in"] = MPConv3D(cin, cout, kernel=(2,3,3))
            else:
                self.enc[f"block{level}_down"] = Block(level, cout, cout, cemb,
                    use_attention=level in config.attn_levels, flavor="enc", resample_mode="down", **block_kwargs)
            
            for idx in range(config.num_layers_per_block):
                cin = cout
                cout = channels
                self.enc[f"block{level}_layer{idx}"] = Block(level, cin, cout, cemb,
                    use_attention=level in config.attn_levels, flavor="enc", **block_kwargs)

        self.conv_latents_out = MPConv3D(cout, config.latent_channels, kernel=(2,3,3))
        self.conv_latents_in = MPConv3D(config.latent_channels + int(config.add_constant_channel), cout, kernel=(2,3,3))

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        for level, channels in reversed(list(enumerate(cblock))):
            
            if level == len(cblock) - 1:

                self.dec[f"block{level}_in0"] = Block(level, cout, cout, cemb,
                    use_attention=config.midblock_attn, flavor="dec", **block_kwargs)
                
                if config.double_midblock == True:
                    self.dec[f"block{level}_in1"] = Block(level, cout, cout, cemb,
                        use_attention=config.midblock_attn, flavor="dec", **block_kwargs)
            else:
                self.dec[f"block{level}_up"] = Block(level, cout, cout, cemb,
                    use_attention=level in config.attn_levels, flavor="dec", resample_mode="up", **block_kwargs)
                
            for idx in range(config.num_layers_per_block + 1):
                cin = cout
                cout = channels
                self.dec[f"block{level}_layer{idx}"] = Block(level, cin, cout, cemb,
                    use_attention=level in config.attn_levels, flavor="dec", **block_kwargs)
        
        self.conv_out = MPConv3D(cout, config.out_channels, kernel=(2,3,3))
            
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
        
        latents = normalize(self.conv_latents_out(x))  if normalize_latents else self.conv_latents_out(x)
        return tensor_5d_to_4d(latents).to(memory_format=torch.channels_last)
    
    def decode(self, x: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:

        x = tensor_4d_to_5d(x, num_channels=self.config.latent_channels).to(memory_format=torch.channels_last_3d)
        x = torch.cat((x, torch.ones_like(x[:, :1])), dim=1)
        x = self.conv_latents_in(x)

        if embeddings is not None:
            embeddings = embeddings.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        for block in self.dec.values():
            x = block(x, embeddings)

        return tensor_5d_to_4d(self.conv_out(x.square(), gain=self.out_gain)).to(memory_format=torch.channels_last)
    
    def forward(self, samples: torch.Tensor, dae_embeddings: torch.Tensor, add_latents_noise: float = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        pre_norm_latents = self.encode(samples, dae_embeddings, normalize_latents=False)
        latents = normalize(pre_norm_latents)
        if add_latents_noise > 0:
            latents = normalize(latents + torch.randn_like(latents) * add_latents_noise)
        
        reconstructed = self.decode(latents, dae_embeddings)
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