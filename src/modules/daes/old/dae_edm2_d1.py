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
from modules.mp_tools import (
    mp_silu, mp_sum, normalize, resample_3d, MPConv3D,
    wavelet_decompose2d, wavelet_recompose2d
)
from utils.dual_diffusion_utils import tensor_4d_to_5d, tensor_5d_to_4d


@dataclass
class DAE_D1_Config(DualDiffusionDAEConfig):

    in_channels: int     = 1
    in_channels_emb: int = 1024
    in_num_freqs: int    = 256
    out_channels: int    = 1
    latent_channels: int = 4

    model_channels: int       = 32           # Base multiplier for the number of channels.
    channel_mult: list[int]   = (1,2,4,4)    # Per-resolution multipliers for the number of channels.
    double_midblock: bool     = True
    midblock_attn: bool       = False
    channel_mult_emb: Optional[int] = 4      # Multiplier for final embedding dimensionality.
    channels_per_head: int    = 64           # Number of channels per attention head.
    num_layers_per_block: int = 2            # Number of resnet blocks per resolution.
    res_balance: float        = 0.4          # Balance between main branch (0) and residual branch (1).
    attn_balance: float       = 0.4          # Balance between main branch (0) and self-attention (1).
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

class DAE_D1(DualDiffusionDAE):

    supports_channels_last: Union[bool, Literal["3d"]] = "3d"

    def __init__(self, config: DAE_D1_Config) -> None:
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
        
        self.conv_out = MPConv3D(cout, 1, kernel=(2,3,3))
            
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
        
    def encode(self, x: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:

        x = tensor_4d_to_5d(x, num_channels=1).to(memory_format=torch.channels_last_3d)
        x = torch.cat((x, torch.ones_like(x[:, :1])), dim=1)

        if embeddings is not None:
            embeddings = embeddings.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        for name, block in self.enc.items():
            x = block(x) if "conv" in name else block(x, embeddings)
        
        latents = normalize(self.conv_latents_out(x))
        return tensor_5d_to_4d(latents).to(memory_format=torch.channels_last)
    
    def decode(self, x: torch.Tensor, embeddings: torch.Tensor, training: bool = False) -> torch.Tensor:

        x = tensor_4d_to_5d(x, num_channels=self.config.latent_channels).to(memory_format=torch.channels_last_3d)
        x = torch.cat((x, torch.ones_like(x[:, :1])), dim=1)
        x = self.conv_latents_in(x)

        if embeddings is not None:
            embeddings = embeddings.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        for block in self.dec.values():
            x = block(x, embeddings)

        dec_out = tensor_5d_to_4d(self.conv_out(x, gain=self.out_gain)).to(memory_format=torch.channels_last)

        # renormalize wavelet levels by measured relative var, but only in inference
        if training == False and len(self.config.wavelet_rescale_factors) > 0: 
            
            wavelets = wavelet_decompose2d(dec_out, num_levels=len(self.config.wavelet_rescale_factors))
            for i in range(len(self.config.wavelet_rescale_factors)):
                wavelets[i] /= self.config.wavelet_rescale_factors[i]**0.5

            dec_out = wavelet_recompose2d(wavelets)

        return dec_out
    
    def forward(self, samples: torch.Tensor, dae_embeddings: torch.Tensor,
            add_latents_noise: float = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        latents = self.encode(samples, dae_embeddings)
        latents_pre_norm_std = latents.std(dim=(1,2,3))
        if add_latents_noise > 0:
            latents = normalize(latents + torch.randn_like(latents) * latents_pre_norm_std.detach().view(-1, 1, 1, 1) * add_latents_noise)
        
        reconstructed = self.decode(latents, dae_embeddings, training=True)

        return latents, reconstructed, latents_pre_norm_std
