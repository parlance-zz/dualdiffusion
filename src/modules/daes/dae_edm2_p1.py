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

from modules.daes.dae import DualDiffusionDAE, DualDiffusionDAEConfig, top_pca_components
from modules.mp_tools import MPConv, mp_silu, mp_sum, normalize, resample_1d
from modules.formats.frequency_scale import get_mel_density
from modules.rope import _rope_pair_rotate_partial, _rope_tables_for_seq


def _rope_tables_for_stereo(x: torch.Tensor, rope_channels: int, rope_base: float) -> tuple[torch.Tensor, torch.Tensor]:

    rope_tables_cos, rope_tables_sin = _rope_tables_for_seq(x.shape[3], rope_channels, rope_base, device=x.device, dtype=x.dtype)

    # expand for stereo in h dim
    rope_tables_cos = rope_tables_cos.repeat(1, 1, 2, 1)
    rope_tables_sin = rope_tables_sin.repeat(1, 1, 2, 1)

    # add values for stereo differentiation
    rope_tables_cos = torch.cat((rope_tables_cos,  torch.ones_like(rope_tables_cos[..., 0:2])), dim=-1)
    rope_tables_sin = torch.cat((rope_tables_sin, -torch.ones_like(rope_tables_sin[..., 0:2])), dim=-1)

    return (rope_tables_cos, rope_tables_sin)

@dataclass
class DAE_Config(DualDiffusionDAEConfig):

    in_channels:  int = 256
    out_channels: int = 256
    in_channels_emb: int = 1024
    latent_channels: int = 64
    in_num_freqs: int = 128

    mp_fourier_ln_sigma_offset: float = -0.7
    mp_fourier_bandwidth:       float = 1

    model_channels: int   = 1024              # Base multiplier for the number of channels.
    logvar_channels: int  = 192               # Number of channels for training uncertainty estimation.
    channel_mult_enc: int = 1
    channel_mult_dec: list[int] = (1,1,1,1,1) # Per-resolution multipliers for the number of channels.
    channel_mult_emb: Optional[int] = 1       # Multiplier for final embedding dimensionality.
    channels_per_head: int    = 64           # Number of channels per attention head.
    rope_channels: int        = 48
    rope_base: float          = 10000.
    num_enc_layers: int       = 8
    num_dec_layers_per_block: int = 2        # Number of resnet blocks per resolution.
    res_balance: float        = 0.3          # Balance between main branch (0) and residual branch (1).
    attn_balance: float       = 0.3          # Balance between main branch (0) and self-attention (1).
    attn_levels: list[int]    = ()           # List of resolution levels to use self-attention.
    mlp_multiplier: int    = 2               # Multiplier for the number of channels in the MLP.
    mlp_groups: int        = 8               # Number of groups for the MLPs.
    emb_linear_groups: int = 2

class Block(torch.nn.Module):

    def __init__(self,
        level: int,                             # Resolution level.
        in_channels: int,                       # Number of input channels.
        out_channels: int,                      # Number of output channels.
        emb_channels: int,                      # Number of embedding channels.
        flavor: Literal["enc", "dec"],
        resample_mode: Literal["keep", "up", "down"] = "keep",
        dropout: float         = 0.,       # Dropout probability.
        res_balance: float     = 0.5,      # Balance between main branch (0) and residual branch (1).
        attn_balance: float    = 0.5,      # Balance between main branch (0) and self-attention (1).
        clip_act: float        = 256,      # Clip output activations. None = do not clip.
        mlp_multiplier: int    = 4,        # Multiplier for the number of channels in the MLP.
        mlp_groups: int        = 4,        # Number of groups for the MLP.
        emb_linear_groups: int = 4,
        channels_per_head: int = 64,       # Number of channels per attention head.
        use_attention: bool    = False,    # Use self-attention in this block.
    ) -> None:
        super().__init__()

        assert out_channels % channels_per_head == 0

        self.level = level
        self.use_attention = use_attention
        self.num_heads = out_channels // channels_per_head
        self.out_channels = out_channels
        self.flavor = flavor
        self.resample_mode = resample_mode
        self.dropout = dropout
        self.res_balance = res_balance
        self.attn_balance = attn_balance
        self.clip_act = clip_act
        
        inner_channels = out_channels * mlp_multiplier

        if in_channels != out_channels or mlp_groups > 1:
            self.conv_skip = MPConv(in_channels, out_channels, kernel=(1,1), groups=1)
        else:
            self.conv_skip = torch.nn.Identity()

        self.conv_res0 = MPConv(out_channels, inner_channels, kernel=(1,3), groups=mlp_groups)
        self.conv_res1 = MPConv(inner_channels, out_channels, kernel=(1,3), groups=mlp_groups)
        #self.conv_stereo0 = MPConv(inner_channels, inner_channels, kernel=(1,1), groups=1)
        self.conv_stereo1 = MPConv(out_channels, out_channels, kernel=(1,1), groups=1)
        
        self.emb_gain = torch.nn.Parameter(torch.zeros([]))
        self.emb_linear = MPConv(emb_channels, inner_channels, kernel=(1,1), groups=emb_linear_groups)

        if self.use_attention == True:
            self.attn_q = MPConv(out_channels, out_channels, kernel=(1,1))
            self.attn_k = MPConv(out_channels, out_channels, kernel=(1,1))
            self.attn_v = MPConv(out_channels, out_channels, kernel=(1,1))
            self.attn_proj = MPConv(out_channels, out_channels, kernel=(1,1))

            self.emb_gain_qkv = torch.nn.Parameter(torch.zeros([]))
            self.emb_linear_qkv = MPConv(emb_channels, out_channels, kernel=(1,1), groups=emb_linear_groups)

    def forward(self, x: torch.Tensor, emb: torch.Tensor, rope_tables: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        
        x = resample_1d(x, self.resample_mode)
        x = normalize(self.conv_skip(x), dim=1)

        if self.use_attention == True:

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

            y = torch.nn.functional.scaled_dot_product_attention(q_rot, k_rot, v.transpose(-1, -2)).transpose(-1, -2)

            y = self.attn_proj(y.reshape(*x.shape))
            x = mp_sum(x, y, t=self.attn_balance)

        y = self.conv_res0(mp_silu(x))
        #stereo: torch.Tensor = self.conv_stereo0(y).flip(dims=(2,))
        #y = mp_sum(y, stereo, t=0.5)

        c = self.emb_linear(emb, gain=self.emb_gain) + 1.
        y = mp_silu(normalize(y * c, dim=1))

        if self.dropout != 0 and self.training == True: # magnitude preserving fix for dropout
            y = torch.nn.functional.dropout(y, p=self.dropout) * (1. - self.dropout)**0.5

        y: torch.Tensor = self.conv_res1(y)
        stereo: torch.Tensor = self.conv_stereo1(y).flip(dims=(2,))
        y = normalize(mp_sum(y, stereo, t=0.5), dim=1)
        
        x = mp_sum(x, y, t=self.res_balance)

        if self.clip_act is not None:
            x = x.clip_(-self.clip_act, self.clip_act)

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
                        "channels_per_head": config.channels_per_head}

        cenc = config.model_channels * config.channel_mult_enc
        cblock = [config.model_channels * x for x in config.channel_mult_dec]
        cemb = config.model_channels * config.channel_mult_emb if config.channel_mult_emb is not None else max(cblock)
        cdata = config.in_channels

        self.num_levels = len(config.channel_mult_dec)
        self.downsample_ratio = 2 ** (self.num_levels - 1)

        assert config.rope_channels % 2 == 0
        assert config.rope_channels <= config.channels_per_head

        self.emb_label = MPConv(config.in_channels_emb, cemb, kernel=())
        self.recon_loss_logvar = torch.nn.Parameter(torch.zeros([]))

        # encoder
        self.enc = torch.nn.ModuleDict()
        self.enc[f"conv_in"] = MPConv(cdata + 1, cenc, kernel=(3,3))

        for idx in range(config.num_enc_layers):
            self.enc[f"block_0_layer{idx}"] = Block(0, cenc, cenc, cemb,
                flavor="enc", use_attention=False, **block_kwargs)
                
        self.conv_latents_out = MPConv(cenc, config.latent_channels, kernel=(3,3))
        self.conv_latents_out_gain = torch.nn.Parameter(torch.ones([]))
        self.conv_latents_in  = MPConv(config.latent_channels + 1, cblock[-1], kernel=(3,3))

        # decoder
        self.dec = torch.nn.ModuleDict()
        cin = cblock[-1]

        for level in reversed(range(0, self.num_levels)):
            
            cout = cblock[level]

            if level == self.num_levels - 1:
                self.dec[f"block{level}_in0"] = Block(level, cin, cout, cemb, flavor="dec",
                    use_attention=level in config.attn_levels, **block_kwargs)
            else:
                self.dec[f"block{level}_up"] = Block(level, cin, cout, cemb, flavor="dec",
                    use_attention=level in config.attn_levels, resample_mode="up", **block_kwargs)

            for idx in range(config.num_dec_layers_per_block):
                self.dec[f"block{level}_layer{idx}"] = Block(level, cout, cout, cemb, flavor="dec",
                    use_attention=level in config.attn_levels, **block_kwargs)

            cin = cout

        self.conv_cond_out = MPConv(cout, config.out_channels, kernel=(3,3))
        self.conv_cond_out_gain = torch.nn.Parameter(torch.zeros([]))

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

        rope_tables = _rope_tables_for_stereo(x, self.config.rope_channels, self.config.rope_base)
        x = torch.cat((x, torch.ones_like(x[:, :1])), dim=1).to(dtype=torch.bfloat16)

        for name, block in self.enc.items():
            x = block(x) if "conv" in name else block(x, emb, rope_tables)

        latents = self.conv_latents_out(x, gain=self.conv_latents_out_gain)
        latents = torch.nn.functional.avg_pool2d(latents, (1, self.downsample_ratio))

        if training == True:
            return latents
        else:
            return normalize(latents, dim=1)

    def decode(self, x: torch.Tensor, embeddings: torch.Tensor, training: bool = False) -> torch.Tensor:

        if embeddings is not None:
            emb = mp_silu(embeddings[..., None, None]).to(dtype=torch.bfloat16)
        
        rope_tables = _rope_tables_for_stereo(x, self.config.rope_channels, self.config.rope_base)

        x = torch.cat((x, torch.ones_like(x[:, :1])), dim=1).to(dtype=torch.bfloat16)
        x = self.conv_latents_in(x)

        for block in self.dec.values():
            x = block(x, emb, rope_tables)

        cond_out = self.conv_cond_out(x, gain=self.conv_cond_out_gain)
        return cond_out

    def forward(self, samples: torch.Tensor, dae_embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        pre_norm_latents = self.encode(samples, dae_embeddings, training=True)
        latents = normalize(pre_norm_latents, dim=1)
        
        cond_out = self.decode(latents, dae_embeddings, training=True)
        return latents, cond_out, pre_norm_latents

    def tiled_encode(self, x: torch.Tensor, embeddings: torch.Tensor, max_chunk: int = 6144, overlap: int = 256) -> torch.Tensor:
        pass

    def latents_to_img(self, latents: torch.Tensor) -> ndarray:
        
        #latents = top_pca_components(latents, n_pca = latents.shape[1])
        
        latents = latents.view(latents.shape[0], 4, latents.shape[1] // 4, 2, latents.shape[3])
        latents = latents.permute(0, 1, 3, 2, 4)
        latents = latents.reshape(latents.shape[0], 8, latents.shape[3], latents.shape[4])

        #latents = normalize(latents, dim=3)
        #hz = torch.linspace(0, 16000, steps=latents.shape[2], device=latents.device)
        #mel_density = get_mel_density(hz)
        #latents /= mel_density[None, None, :, None]

        return super().latents_to_img(latents)