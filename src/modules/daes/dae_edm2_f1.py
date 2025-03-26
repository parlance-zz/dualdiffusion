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

from utils.dual_diffusion_utils import tensor_4d_to_5d, tensor_5d_to_4d
from modules.daes.dae import DualDiffusionDAE, DualDiffusionDAEConfig
from modules.mp_tools import (
    mp_silu, mp_sum, normalize, MPConv3D, channel_to_space3d,
    wavelet_decompose2d, wavelet_recompose2d, resample_3d
)


@dataclass
class DAE_F1_Config(DualDiffusionDAEConfig):

    in_channels: int     = 1
    in_channels_emb: int = 1024
    in_num_freqs: int    = 256
    out_channels: int    = 1
    latent_channels: int = 4

    model_channels: int         = 32         # Base multiplier for the number of channels.
    channel_mult_enc: list[int] = (4,4,4,4)
    channel_mult_dec: list[int] = (4,4,4,4)  # Per-resolution multipliers for the number of channels.
    channel_mult_emb: int     = 4
    channels_per_head: int    = 64           # Number of channels per attention head.
    num_enc_layers_per_block: int = 3       
    num_dec_layers_per_block: int = 4
    res_balance: float        = 0.3          # Balance between main branch (0) and residual branch (1).
    attn_balance: float       = 0.3          # Balance between main branch (0) and self-attention (1).
    attn_levels: list[int]    = ()           # List of resolution levels to use self-attention.
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
        res_kernel: tuple[int, int, int] = (2,3,3),
    ) -> None:
        super().__init__()

        self.level = level
        self.use_attention = use_attention
        self.use_pixel_norm = use_pixel_norm
        self.num_heads = out_channels // channels_per_head
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.flavor = flavor
        self.dropout = dropout
        self.res_balance = res_balance
        self.attn_balance = attn_balance
        self.clip_act = clip_act

        self.conv_res0 = MPConv3D(in_channels, out_channels * mlp_multiplier, kernel=res_kernel, groups=mlp_groups)
        self.conv_res1 = MPConv3D(out_channels * mlp_multiplier, out_channels, kernel=res_kernel, groups=mlp_groups)

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

        if self.flavor == "enc":
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

class DAE_F1(DualDiffusionDAE):

    supports_channels_last: Union[bool, Literal["3d"]] = "3d"

    def __init__(self, config: DAE_F1_Config) -> None:
        super().__init__()
        self.config = config

        block_kwargs = {"mlp_multiplier": config.mlp_multiplier,
                        "mlp_groups": config.mlp_groups,
                        "emb_linear_groups": config.emb_linear_groups,
                        "res_balance": config.res_balance,
                        "attn_balance": config.attn_balance,
                        "channels_per_head": config.channels_per_head,
                        "use_pixel_norm": config.add_pixel_norm}
        
        self.num_levels = len(config.channel_mult_dec)

        cemb = config.model_channels * config.channel_mult_emb * config.mlp_multiplier if config.in_channels_emb > 0 else 0
        cenc = [config.model_channels * m for m in config.channel_mult_enc]
        cdec = [config.model_channels * m for m in config.channel_mult_dec]

        assert len(cenc) == len(cdec) == self.num_levels

        self.total_recon_loss_logvar = torch.nn.Parameter(torch.zeros([]))
        self.level_recon_loss_logvar = torch.nn.Parameter(torch.zeros(self.num_levels))

        # Embedding.
        if config.in_channels_emb > 0:
            self.emb_label = MPConv3D(config.in_channels_emb, cemb, kernel=())
            self.emb_dim = cemb
        else:
            cemb = self.emb_dim = 0
            self.emb_label = None

        in_channels, out_channels = 1 + int(config.add_constant_channel), 1

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cin = cenc[0]
        
        for level, cout in enumerate(cenc):
            
            self.enc[f"block{level}_conv_in"] = MPConv3D(in_channels, cin, kernel=(2,3,3))

            if level == 0:
                self.enc[f"block{level}_in"] = Block(level, cin, cout, cemb,
                    use_attention=level in config.attn_levels, flavor="dec", **block_kwargs)
            else:
                self.enc[f"block{level}_down"] = Block(level, cin, cout, cemb,
                    use_attention=level in config.attn_levels, flavor="dec", **block_kwargs)
                
            for idx in range(config.num_enc_layers_per_block):
                self.enc[f"block{level}_layer{idx}"] = Block(level, cout, cout, cemb,
                    use_attention=level in config.attn_levels, flavor="enc", **block_kwargs)

            self.enc[f"block{level}_conv_latents_out"] = MPConv3D(cout, config.latent_channels, kernel=(2,3,3))

            cin = cout

        self.conv_latents_in = MPConv3D(config.latent_channels + int(config.add_constant_channel), cdec[-1], kernel=(2,3,3))

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        cin = cdec[-1]

        for level, cout in zip(reversed(range(self.num_levels)), reversed(cdec)):

            if level == self.num_levels - 1:
                self.dec[f"block{level}_in"] = Block(level, cin, cout, cemb,
                    use_attention=level in config.attn_levels, flavor="dec", **block_kwargs)
            else:
                self.dec[f"block{level}_up"] = Block(level, cin, cout, cemb,
                    use_attention=level in config.attn_levels, flavor="dec", **block_kwargs)
                
            for idx in range(config.num_dec_layers_per_block):
                self.dec[f"block{level}_layer{idx}"] = Block(level, cout, cout, cemb,
                    use_attention=level in config.attn_levels, flavor="dec", **block_kwargs)

            self.dec[f"block{level}_conv_out"] = MPConv3D(cout, out_channels, kernel=(2,3,3), out_gain_param=True)
            
    def get_embeddings(self, emb_in: torch.Tensor) -> torch.Tensor:
        if self.emb_label is not None:
            return self.emb_label(normalize(emb_in).to(device=self.device, dtype=self.dtype))
        else:
            return None
    
    def get_recon_loss_logvar(self) -> torch.Tensor:
        return self.total_recon_loss_logvar
    
    def get_latent_shape(self, sample_shape: Union[torch.Size, tuple[int, int, int, int]]) -> torch.Size:
        if len(sample_shape) == 4:
            return (sample_shape[0], self.config.latent_channels * 2,
                    sample_shape[2] // 2 ** (self.num_levels-1),
                    sample_shape[3] // 2 ** (self.num_levels-1))
        else:
            raise ValueError(f"Invalid sample shape: {sample_shape}")
        
    def get_sample_shape(self, latent_shape: Union[torch.Size, tuple[int, int, int, int]]) -> torch.Size:
        if len(latent_shape) == 4:
            return (latent_shape[0], 2,
                    latent_shape[2] * 2 ** (self.num_levels-1),
                    latent_shape[3] * 2 ** (self.num_levels-1))
        else:
            raise ValueError(f"Invalid latent shape: {latent_shape}")
        
    def encode(self, x: torch.Tensor, embeddings: torch.Tensor, return_pre_norm_latents: bool = False) -> torch.Tensor:

        with torch.no_grad():
            x_wavelets = wavelet_decompose2d(x, self.num_levels)

            for i, wavelet in enumerate(x_wavelets):
                wavelet = tensor_4d_to_5d(wavelet, num_channels=1).to(memory_format=torch.channels_last_3d)

                if self.config.add_constant_channel == True:
                    wavelet = torch.cat((wavelet, torch.ones_like(wavelet[:, :1])), dim=1)

                x_wavelets[i] = wavelet.detach()

            x_wavelets.reverse()

        if embeddings is not None:
            embeddings = embeddings.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        latents_wavelets = []
        for name, block in self.enc.items():
            
            if "conv_in" in name:
                conv_in = block(x_wavelets.pop())
            elif "in" in name:
                x = block(conv_in, embeddings)
            elif "down" in name:
                x = block(resample_3d(x, "down") + conv_in, embeddings)
            elif "conv_latents_out" in name:
                latents_wavelets.append(tensor_5d_to_4d(block(x)))
            else:
                x = block(x, embeddings)
        
        pre_norm_latents = latents_wavelets.pop()
        for i in range(1, self.num_levels):
            pre_norm_latents = pre_norm_latents + torch.nn.functional.avg_pool2d(latents_wavelets.pop(), 2**i)

        latents = normalize(pre_norm_latents - pre_norm_latents.mean(dim=(1,2,3), keepdim=True))

        if return_pre_norm_latents == True:
            return latents, pre_norm_latents
        else:
            return latents

    def decode(self, x: torch.Tensor, embeddings: torch.Tensor,
            return_training_output: bool = False) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:

        x = tensor_4d_to_5d(x, num_channels=self.config.latent_channels).to(memory_format=torch.channels_last_3d)

        if self.config.add_constant_channel == True:
            x = torch.cat((x, torch.ones_like(x[:, :1])), dim=1)

        x = self.conv_latents_in(x)

        if embeddings is not None:
            embeddings = embeddings.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        dec_outputs = []
        for name, block in self.dec.items():

            if "up" in name:
                x = block(resample_3d(x, "up"), embeddings)
            elif "conv_out" in name:   
                dec_outputs.append(tensor_5d_to_4d(block(x)))
            else:
                x = block(x, embeddings)

        dec_outputs.reverse()

        if return_training_output == True:
            return dec_outputs
        else:

            for i in range(self.num_levels):
                out_var = dec_outputs[i].var(dim=(1,2,3), keepdim=True)
                target_var = out_var + self.level_recon_loss_logvar[i].exp().detach()
                dec_outputs[i] *= (target_var / out_var)**0.5
                
            return wavelet_recompose2d(dec_outputs)
    
    def forward(self, samples: torch.Tensor, dae_embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        latents, pre_norm_latents = self.encode(samples, dae_embeddings, return_pre_norm_latents=True)
        dec_outputs = self.decode(latents, dae_embeddings, return_training_output=True)

        return (latents, pre_norm_latents, dec_outputs)
