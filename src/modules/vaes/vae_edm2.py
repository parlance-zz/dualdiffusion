# MIT License
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

from typing import Optional, Union, Literal
from dataclasses import dataclass

import torch
import numpy as np

from modules.formats.format import DualDiffusionFormat
from modules.vaes.vae import DualDiffusionVAEConfig, DualDiffusionVAE, IsotropicGaussianDistribution
from modules.mp_tools import MPConv, normalize, resample, mp_silu, mp_sum

@dataclass
class DualDiffusionVAE_EDM2Config(DualDiffusionVAEConfig):

    model_channels: int       = 256          # Base multiplier for the number of channels.
    channel_mult: list[int]   = (1,2,3,4)    # Per-resolution multipliers for the number of channels.
    channel_mult_emb: Optional[int] = None   # Multiplier for final embedding dimensionality.
    channels_per_head: int    = 64           # Number of channels per attention head.
    num_layers_per_block: int = 2            # Number of resnet blocks per resolution.
    res_balance: float        = 0.3          # Balance between main branch (0) and residual branch (1).
    attn_balance: float       = 0.3          # Balance between main branch (0) and self-attention (1).
    mlp_multiplier: int = 1                  # Multiplier for the number of channels in the MLP.
    mlp_groups: int     = 1                  # Number of groups for the MLPs.
    add_mid_block_attention: bool = False    # Add attention layers in decoder mid-block

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
        channels_per_head: int = 64,       # Number of channels per attention head.
        use_attention: bool    = False,    # Use self-attention in this block.
    ) -> None:
        super().__init__()

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
        
        self.conv_res0 = MPConv(out_channels if flavor == "enc" else in_channels,
                                out_channels * mlp_multiplier, kernel=(3,3), groups=mlp_groups)
        self.conv_res1 = MPConv(out_channels * mlp_multiplier, out_channels, kernel=(3,3), groups=mlp_groups)
        self.conv_skip = MPConv(in_channels, out_channels, kernel=(1,1), groups=1) if in_channels != out_channels else None

        self.emb_gain = torch.nn.Parameter(torch.zeros([]))
        self.emb_linear = MPConv(emb_channels, out_channels * mlp_multiplier,
                                 kernel=(), groups=mlp_groups) if emb_channels != 0 else None

        if self.use_attention:
            self.emb_gain_qk = torch.nn.Parameter(torch.zeros([]))
            self.emb_gain_v = torch.nn.Parameter(torch.zeros([]))
            self.emb_linear_qk = MPConv(emb_channels, out_channels, kernel=(1,1), groups=1) if emb_channels != 0 else None
            self.emb_linear_v = MPConv(emb_channels, out_channels, kernel=(1,1), groups=1) if emb_channels != 0 else None

            self.attn_qk = MPConv(out_channels, out_channels * 2, kernel=(1,1))
            self.attn_v = MPConv(out_channels, out_channels, kernel=(1,1))
            self.attn_proj = MPConv(out_channels, out_channels, kernel=(1,1))

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        
        x = resample(x, mode=self.resample_mode)

        if self.flavor == "enc":
            if self.conv_skip is not None:
                x = self.conv_skip(x)
            x = normalize(x, dim=1) # pixel norm

        y = self.conv_res0(mp_silu(x))

        c = self.emb_linear(emb, gain=self.emb_gain) + 1.
        y = mp_silu(y * c)

        if self.dropout != 0: # magnitude preserving fix for dropout
            if self.training:
                y = torch.nn.functional.dropout(y, p=self.dropout)
            else:
                y *= 1. - self.dropout

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
        return x

class AutoencoderKL_EDM2(DualDiffusionVAE):

    def __init__(self, config: DualDiffusionVAE_EDM2Config) -> None:
        super().__init__()
        self.config = config

        block_kwargs = {"dropout": config.dropout,
                        "mlp_multiplier": config.mlp_multiplier,
                        "mlp_groups": config.mlp_groups,
                        "res_balance": config.res_balance,
                        "attn_balance": config.attn_balance,
                        "channels_per_head": config.channels_per_head}
        
        cblock = [config.model_channels * x for x in config.channel_mult]
        cemb = config.model_channels * config.channel_mult_emb if config.channel_mult_emb is not None else max(cblock)

        self.num_levels = len(config.channel_mult)

        target_noise_std = (1 / (config.target_snr**2 + 1))**0.5
        target_sample_std = (1 - target_noise_std**2)**0.5
        self.latents_out_gain = torch.nn.Parameter(torch.tensor(target_sample_std))
        self.out_gain = torch.nn.Parameter(torch.ones([]))
        
        # Embedding.
        self.emb_label = MPConv(config.label_dim, cemb, kernel=())

        # Training uncertainty estimation.
        self.recon_loss_logvar = torch.nn.Parameter(torch.zeros(1))
        self.latents_logvar = torch.nn.Parameter(torch.zeros(1)) # currently unused
        
        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = config.in_channels + 2 # 1 extra const channel, 1 pos embedding channel
        for level, channels in enumerate(cblock):
            
            if level == 0:
                cin = cout
                cout = channels
                self.enc[f"conv_in"] = MPConv(cin, cout, kernel=(3,3))
            else:
                self.enc[f"block{level}_down"] = Block(level, cout, cout, cemb, use_attention=False,
                                                       flavor="enc", resample_mode="down", **block_kwargs)
            
            for idx in range(config.num_layers_per_block):
                cin = cout
                cout = channels
                self.enc[f"block{level}_layer{idx}"] = Block(level, cin, cout, cemb, use_attention=False,
                                                             flavor="enc", **block_kwargs)

        self.conv_latents_out = MPConv(cout, config.latent_channels, kernel=(3,3))
        self.conv_latents_in = MPConv(config.latent_channels + 2, cout, kernel=(3,3)) # 1 extra const channel, 1 pos embedding channel

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        for level, channels in reversed(list(enumerate(cblock))):
            
            if level == len(cblock) - 1:
                self.dec[f"block{level}_in0"] = Block(level, cout, cout, cemb, flavor="dec",
                                                      use_attention=config.add_mid_block_attention, **block_kwargs)
                self.dec[f"block{level}_in1"] = Block(level, cout, cout, cemb, flavor="dec",
                                                      use_attention=config.add_mid_block_attention, **block_kwargs)
            else:
                self.dec[f"block{level}_up"] = Block(level, cout, cout, cemb, flavor="dec",
                                                     resample_mode="up", **block_kwargs)
            for idx in range(config.num_layers_per_block + 1):
                cin = cout
                cout = channels
                self.dec[f"block{level}_layer{idx}"] = Block(level, cin, cout, cemb, flavor="dec",
                                                             use_attention=False, **block_kwargs)
        self.conv_out = MPConv(cout, config.out_channels, kernel=(3,3))

    def get_class_embeddings(self, class_labels: torch.Tensor) -> torch.Tensor:
        return mp_silu(self.emb_label(normalize(class_labels).to(device=self.device, dtype=self.dtype)))

    def get_recon_loss_logvar(self) -> torch.Tensor:
        return self.recon_loss_logvar
    
    def get_target_snr(self) -> float:
        return self.config.target_snr
    
    def get_latent_shape(self, sample_shape: Union[torch.Size, tuple[int, int, int, int]]) -> torch.Size:
        if len(sample_shape) == 4:
            return (sample_shape[0],
                    self.config.latent_channels,
                    sample_shape[2] // 2 ** (self.num_levels-1),
                    sample_shape[3] // 2 ** (self.num_levels-1))
        else:
            raise ValueError(f"Invalid sample shape: {sample_shape}")
        
    def encode(self, x: torch.Tensor,
               class_embeddings: torch.Tensor,
               format: DualDiffusionFormat) -> IsotropicGaussianDistribution:
        
        x = torch.cat((x, torch.ones_like(x[:, :1]), format.get_ln_freqs(x)), dim=1)
        for name, block in self.enc.items():
            x = block(x) if "conv" in name else block(x, class_embeddings)

        latents = self.conv_latents_out(x, gain=self.latents_out_gain)
        noise_logvar = torch.tensor(np.log(1 / (self.config.target_snr**2 + 1)), device=x.device, dtype=x.dtype)
        return IsotropicGaussianDistribution(latents, noise_logvar)
    
    def decode(self, x: torch.Tensor,
               class_embeddings: torch.Tensor,
               format: DualDiffusionFormat) -> torch.Tensor:
        
        x = torch.cat((x, torch.ones_like(x[:, :1]), format.get_ln_freqs(x)), dim=1)
        x = self.conv_latents_in(x)
        for _, block in self.dec.items():
            x = block(x, class_embeddings, None, None)

        return self.conv_out(x, gain=self.out_gain)