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

from modules.unets.unet import DualDiffusionUNet, DualDiffusionUNetConfig
from modules.mp_tools import MPFourier, mp_cat, mp_silu, mp_sum, normalize, resample_2d
from modules.formats.format import DualDiffusionFormat


class MPConv2D(torch.nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel: tuple[int, int],
                 groups: int = 1, disable_weight_norm: bool = False) -> None:
        
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.disable_weight_norm = disable_weight_norm
        
        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels // groups, *kernel))
        if self.weight.numel() == 0:
            raise ValueError(f"Invalid weight shape: {self.weight.shape}")
        
        if self.weight.ndim == 4:
            if kernel[1] // 2 != 0:
                self.padding = torch.nn.ReflectionPad2d((kernel[1] // 2, kernel[1] // 2, 0, 0))
            else:
                self.padding = torch.nn.Identity()
        else:
            self.padding = None

    def forward(self, x: torch.Tensor, gain: Union[float, torch.Tensor] = 1) -> torch.Tensor:
        
        w = self.weight.float()
        if self.training == True and self.disable_weight_norm == False:
            w = normalize(w) # traditional weight normalization
            
        w = w * (gain / w[0].numel()**0.5) # magnitude-preserving scaling
        w = w.to(x.dtype)

        if w.ndim == 2:
            return x @ w.t()
        
        return torch.nn.functional.conv2d(self.padding(x), w, padding=(w.shape[-2]//2, 0), groups=self.groups)

    @torch.no_grad()
    def normalize_weights(self) -> None:
        if self.disable_weight_norm == False:
            self.weight.copy_(normalize(self.weight))

@dataclass
class DDec_MDCT_UNet_B1_Config(DualDiffusionUNetConfig):

    in_channels:  int = 4
    out_channels: int = 4
    in_channels_emb: int = 0

    sigma_max: float = 11
    sigma_min: float = 0.0002

    in_num_freqs: int = 256
    in_psd_freqs: int = 4096
    x_ref_balance: float = 0.5

    model_channels: int  = 64                # Base multiplier for the number of channels.
    logvar_channels: int = 192               # Number of channels for training uncertainty estimation.
    channel_mult: list[int]    = (1,2,3,4)   # Per-resolution multipliers for the number of channels.
    double_midblock: bool      = True
    midblock_attn: bool        = False
    channel_mult_noise: Optional[int] = 3    # Multiplier for noise embedding dimensionality.
    channel_mult_emb: Optional[int]   = 3    # Multiplier for final embedding dimensionality.
    channels_per_head: int    = 64           # Number of channels per attention head.
    num_layers_per_block: int = 3            # Number of resnet blocks per resolution.
    label_balance: float      = 0.5          # Balance between noise embedding (0) and class embedding (1).
    concat_balance: float     = 0.5          # Balance between skip connections (0) and main path (1).
    res_balance: float        = 0.3          # Balance between main branch (0) and residual branch (1).
    attn_balance: float       = 0.3          # Balance between main branch (0) and self-attention (1).
    attn_levels: list[int]    = ()           # List of resolution levels to use self-attention.
    mlp_multiplier: int    = 2               # Multiplier for the number of channels in the MLP.
    mlp_groups: int        = 1               # Number of groups for the MLPs.
    emb_linear_groups: int = 1
    add_constant_channel: bool = True

class Block(torch.nn.Module):

    def __init__(self,
        level: int,                             # Resolution level.
        in_channels: int,                       # Number of input channels.
        out_channels: int,                      # Number of output channels.
        emb_channels: int,                      # Number of embedding channels.
        num_freqs: int,         
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
        use_attention: bool    = False     # Use self-attention in this block.
    ) -> None:
        super().__init__()

        self.level = level
        self.num_freqs = num_freqs
        self.use_attention = use_attention
        self.num_heads = out_channels // channels_per_head
        self.out_channels = out_channels
        self.flavor = flavor
        self.resample_mode = resample_mode
        self.dropout = dropout
        self.res_balance = res_balance
        self.attn_balance = attn_balance
        self.clip_act = clip_act
        
        self.conv_res0 = MPConv2D(out_channels if flavor == "enc" else in_channels,
                                out_channels * mlp_multiplier, kernel=(3,3), groups=mlp_groups)
        self.conv_res1 = MPConv2D(out_channels * mlp_multiplier, out_channels, kernel=(3,3), groups=mlp_groups)

        if in_channels != out_channels or mlp_groups > 1:
            self.conv_skip = MPConv2D(in_channels, out_channels, kernel=(1,1), groups=1)
        else:
            self.conv_skip = None

        self.emb_gain = torch.nn.Parameter(torch.zeros([]))
        self.emb_linear = MPConv2D(emb_channels, out_channels * mlp_multiplier,
            kernel=(1,1), groups=emb_linear_groups) if emb_channels != 0 else None
        
        if self.use_attention == True:
            raise NotImplementedError()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        
        x = resample_2d(x, mode=self.resample_mode).to(dtype=torch.bfloat16)

        if self.flavor == "enc":
            if self.conv_skip is not None:
                x = self.conv_skip(x)
            x = normalize(x, dim=1) # pixel norm

        y = self.conv_res0(mp_silu(x))

        c = self.emb_linear(emb, gain=self.emb_gain) + 1.
        y = mp_silu(y * c)

        if self.dropout != 0 and self.training == True: # magnitude preserving fix for dropout
            y = torch.nn.functional.dropout(y, p=self.dropout) * (1. - self.dropout)**0.5

        y = self.conv_res1(y)

        if self.flavor == "dec" and self.conv_skip is not None:
            x = self.conv_skip(x)
        x = mp_sum(x, y, t=self.res_balance)
        
        if self.use_attention:
            pass

        if self.clip_act is not None:
            x = x.clip_(-self.clip_act, self.clip_act)

        return x

class DDec_MDCT_UNet_B1(DualDiffusionUNet):

    def __init__(self, config: DDec_MDCT_UNet_B1_Config) -> None:
        super().__init__()
        self.config = config

        block_kwargs = {"dropout": config.dropout,
                        "mlp_multiplier": config.mlp_multiplier,
                        "mlp_groups": config.mlp_groups,
                        "emb_linear_groups": config.emb_linear_groups,
                        "res_balance": config.res_balance,
                        "attn_balance": config.attn_balance,
                        "channels_per_head": config.channels_per_head}

        cblock = [config.model_channels * x for x in config.channel_mult]
        cnoise = config.model_channels * config.channel_mult_noise if config.channel_mult_noise is not None else max(cblock)
        cemb = config.model_channels * config.channel_mult_emb if config.channel_mult_emb is not None else max(cblock)
        cemb *= self.config.mlp_multiplier

        self.num_levels = len(config.channel_mult)

        assert config.in_psd_freqs % config.in_num_freqs == 0
        self.psd_freqs_per_freq = config.in_psd_freqs // config.in_num_freqs
        
        # Embedding.
        self.emb_fourier = MPFourier(cnoise)
        self.emb_noise = MPConv2D(cnoise, cemb, kernel=())
        self.emb_label = MPConv2D(config.in_channels_emb, cemb, kernel=()) if config.in_channels_emb > 0 else None
        self.emb_label_unconditional = MPConv2D(1, cemb, kernel=()) if config.in_channels_emb > 0 else None

        # Training uncertainty estimation.
        self.logvar_fourier = MPFourier(config.logvar_channels)
        self.logvar_linear = MPConv2D(config.logvar_channels, 1, kernel=(), disable_weight_norm=True)

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = config.in_channels + self.psd_freqs_per_freq * 2 + int(config.add_constant_channel)

        for level, channels in enumerate(cblock):
            
            num_freqs = config.in_num_freqs // 2**level

            if level == 0:
                cin = cout
                cout = channels
                self.enc[f"conv_in"] = MPConv2D(cin, cout, kernel=(3,3))
            else:
                self.enc[f"block{level}_down"] = Block(level, cout, cout, cemb, num_freqs,
                    use_attention=level in config.attn_levels, flavor="enc", resample_mode="down", **block_kwargs)
            
            for idx in range(config.num_layers_per_block):
                cin = cout
                cout = channels
                self.enc[f"block{level}_layer{idx}"] = Block(level, cin, cout, cemb, num_freqs,
                    use_attention=level in config.attn_levels, flavor="enc", **block_kwargs)

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        skips = [block.out_channels for block in self.enc.values()]
        
        for level, channels in reversed(list(enumerate(cblock))):
            
            num_freqs = config.in_num_freqs // 2**level

            if level == len(cblock) - 1:
                self.dec[f"block{level}_in0"] = Block(level, cout, cout, cemb, num_freqs,
                    use_attention=config.midblock_attn, flavor="dec", **block_kwargs)
                if config.double_midblock == True:
                    self.dec[f"block{level}_in1"] = Block(level, cout, cout, cemb, num_freqs,
                        use_attention=config.midblock_attn, flavor="dec", **block_kwargs)
            else:
                self.dec[f"block{level}_up"] = Block(level, cout, cout, cemb, num_freqs,
                    use_attention=level in config.attn_levels, flavor="dec", resample_mode="up", **block_kwargs)

            for idx in range(config.num_layers_per_block + 1):
                cin = cout + skips.pop()
                cout = channels
                self.dec[f"block{level}_layer{idx}"] = Block(level, cin, cout, cemb, num_freqs,
                    use_attention=level in config.attn_levels, flavor="dec", **block_kwargs)
        
        self.out_gain = torch.nn.Parameter(torch.zeros([]))
        self.conv_out = MPConv2D(cout, config.out_channels, kernel=(5,5))

    def get_embeddings(self, emb_in: torch.Tensor, conditioning_mask: torch.Tensor) -> torch.Tensor:
        if self.config.in_channels_emb > 0:
            u_embedding = self.emb_label_unconditional(torch.ones(1, device=self.device, dtype=self.dtype))
            c_embedding = self.emb_label(normalize(emb_in).to(device=self.device, dtype=self.dtype))
            return mp_sum(u_embedding, c_embedding, t=conditioning_mask.unsqueeze(1).to(self.device, self.dtype))
        else:
            return None
        
    def get_sigma_loss_logvar(self, sigma: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.logvar_linear(self.logvar_fourier(sigma.flatten().log() / 4)).view(-1, 1, 1, 1).float()
    
    def get_latent_shape(self, latent_shape: Union[torch.Size, tuple[int, int, int, int]]) -> torch.Size:
        return latent_shape[0:2] + ((latent_shape[2] // 2**(self.num_levels-1)) * 2**(self.num_levels-1),
                                    (latent_shape[3] // 2**(self.num_levels-1)) * 2**(self.num_levels-1))

    def forward(self, x_in: torch.Tensor,
                sigma: torch.Tensor,
                format: DualDiffusionFormat,
                embeddings: torch.Tensor,
                x_ref: Optional[torch.Tensor] = None) -> torch.Tensor:

        with torch.no_grad():
            sigma = sigma.view(-1, 1, 1, 1)
            
            # Preconditioning weights.
            c_skip = self.config.sigma_data ** 2 / (sigma ** 2 + self.config.sigma_data ** 2)
            c_out = sigma * self.config.sigma_data / (sigma ** 2 + self.config.sigma_data ** 2).sqrt()
            c_in = 1 / (self.config.sigma_data ** 2 + sigma ** 2).sqrt()
            c_noise = (sigma.flatten().log() / 4).to(self.dtype)

            x_ref = x_ref.view(x_ref.shape[0], x_ref.shape[1], self.config.in_num_freqs, self.psd_freqs_per_freq, x_ref.shape[3])
            x_ref = x_ref.permute(0, 3, 1, 2, 4).to(dtype=torch.bfloat16)
            x_ref = x_ref.reshape(x_ref.shape[0], x_ref.shape[1] * x_ref.shape[2], x_ref.shape[3], x_ref.shape[4])

            x = (c_in * x_in).to(dtype=torch.bfloat16)
 
        # Embedding.
        emb = self.emb_noise(self.emb_fourier(c_noise))
        if self.config.in_channels_emb > 0:
            emb = mp_silu(mp_sum(emb, embeddings, t=self.config.label_balance))
        emb = emb.unsqueeze(-1).unsqueeze(-1).to(dtype=torch.bfloat16)

        # Encoder.
        x = mp_cat(x, x_ref, t=self.config.x_ref_balance)
        if self.config.add_constant_channel == True:
            x = torch.cat((x, torch.ones_like(x[:, :1])), dim=1)

        skips = []
        for name, block in self.enc.items():
            x = block(x) if "conv" in name else block(x, emb)
            skips.append(x)

        # Decoder.
        for name, block in self.dec.items():
            if "layer" in name:
                x = mp_cat(x, skips.pop(), t=self.config.concat_balance)
            x = block(x, emb)

        x: torch.Tensor = self.conv_out(x, gain=self.out_gain)
        D_x: torch.Tensor = c_skip * x_in.float() + c_out * x.float()

        return D_x
    
