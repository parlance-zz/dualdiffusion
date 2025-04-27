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

from modules.discs.disc import DualDiffusionDiscriminator, DualDiffusionDiscriminatorConfig
from modules.mp_tools import mp_silu, normalize, resample_3d
from utils.dual_diffusion_utils import tensor_4d_to_5d, tensor_5d_to_4d, tensor_random_cat


@dataclass
class Discriminator_J3_Config(DualDiffusionDiscriminatorConfig):

    in_channels_emb: int = 1024
    in_num_freqs: int    = 256

    model_channels: int         = 32         # Base multiplier for the number of channels.
    channel_mult_emb: int       = 12         # Multiplier for final embedding dimensionality.
    num_layers: int             = 6          
    mlp_multiplier: int    = 2               # Multiplier for the number of channels in the MLP.
    mlp_groups: int        = 1               # Number of groups for the MLPs.

class MPConv3D_E(torch.nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel: tuple[int, int, int],
                 groups: int = 1, disable_weight_norm: bool = False) -> None:
        
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.disable_weight_norm = disable_weight_norm
        
        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels // groups, *kernel))
        if self.weight.numel() == 0:
            raise ValueError(f"Invalid weight shape: {self.weight.shape}")
        
        if self.weight.ndim == 5:
            pad_z, pad_w = kernel[0] // 2, kernel[2] // 2
            if pad_w != 0 or pad_z != 0:
                self.padding = torch.nn.ReflectionPad3d((kernel[2] // 2, kernel[2] // 2, 0, 0, 0, kernel[0] // 2))
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
        
        return torch.nn.functional.conv3d(self.padding(x), w,
            padding=(0, w.shape[-2]//2, 0), groups=self.groups)

    @torch.no_grad()
    def normalize_weights(self) -> None:
        if self.disable_weight_norm == False:
            self.weight.copy_(normalize(self.weight))

class Block(torch.nn.Module):

    def __init__(self,
        level: int,                             # Resolution level.
        in_channels: int,                       # Number of input channels.
        out_channels: int,                      # Number of output channels.
        emb_channels: int,
        flavor: Literal["enc", "dec"] = "enc",
        resample_mode: Literal["keep", "up", "down"] = "keep",
        clip_act: float        = 256,      # Clip output activations. None = do not clip.
        mlp_multiplier: int    = 2,        # Multiplier for the number of channels in the MLP.
        mlp_groups: int        = 1,        # Number of groups for the MLP.
        kernel: tuple[int, int] = (1,3,3),   # Kernel size for the convolutional layers.
    ) -> None:
        super().__init__()

        self.level = level
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.flavor = flavor
        self.resample_mode = resample_mode
        self.clip_act = clip_act
        self.kernel = kernel

        self.conv_res0 = MPConv3D_E(in_channels, out_channels * mlp_multiplier,
                                    kernel=kernel, groups=mlp_groups)
        self.conv_res1 = MPConv3D_E(out_channels * mlp_multiplier,
                        out_channels, kernel=kernel, groups=mlp_groups)
        
        self.emb_gain = torch.nn.Parameter(torch.zeros([]))
        self.emb_linear = MPConv3D_E(emb_channels, out_channels * mlp_multiplier,
                                     kernel=(1,1,1), groups=1) if emb_channels != 0 else None
        
        if in_channels != out_channels or mlp_groups > 1:
            self.conv_skip = MPConv3D_E(in_channels, out_channels, kernel=(1,1,1), groups=1)
        else:
            self.conv_skip = None

        self.res_balance = torch.nn.Parameter(-torch.ones([]) * 0.7)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        
        x = resample_3d(x, mode=self.resample_mode)

        y = self.conv_res0(mp_silu(x))

        if self.emb_linear is not None:
            y = y * (self.emb_linear(emb, gain=self.emb_gain) + 1)

        y = self.conv_res1(mp_silu(y))
        
        if self.conv_skip is not None:
            x = self.conv_skip(x)
        
        t = self.res_balance.sigmoid()
        x = torch.lerp(x, y, t) / ((1 - t) ** 2 + t ** 2) ** 0.5
        
        if self.clip_act is not None:
            x = x.clip_(-self.clip_act, self.clip_act)

        kl_dims = (1,2,3,4)
        x_mean = x.mean(dim=kl_dims)
        x_var = x.var(dim=kl_dims).clip(min=1e-2)
        kld = x_mean.square() + x_var - 1 - x_var.log()

        return x, kld

class Discriminator(torch.nn.Module):

    def __init__(self, model_channels: int, emb_channels: int,
            num_layers: int, block_kwargs: dict, kernel: tuple[int, int] = (1,3,3)) -> None:
        
        super().__init__()

        self.input_gain = torch.nn.Parameter(torch.ones([]))
        self.input_shift = torch.nn.Parameter(torch.zeros([]))
        self.conv_in = MPConv3D_E(2 + 1, model_channels, kernel=kernel)

        self.disc = torch.nn.ModuleDict()
        for idx in range(num_layers):
            self.disc[f"layer{idx}"] = Block(0, model_channels, model_channels, emb_channels,
                                                            kernel=kernel, **block_kwargs)

        self.conv_out = MPConv3D_E(model_channels, 1, kernel=kernel)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        
        x = torch.cat((x, torch.ones_like(x[:, :1])), dim=1)
        x = self.conv_in(x, gain=self.input_gain) + self.input_shift

        hidden_kld = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

        for block in self.disc.values():
            x, kld = block(x, emb)
            hidden_kld = hidden_kld + kld

        x = self.conv_out(x)
        return x, hidden_kld
    
class Discriminator_J3(DualDiffusionDiscriminator):

    supports_channels_last: Union[bool, Literal["3d"]] = "3d"

    def __init__(self, config: Discriminator_J3_Config) -> None:
        super().__init__()
        self.config = config

        block_kwargs = {"mlp_multiplier": config.mlp_multiplier,
                        "mlp_groups": config.mlp_groups}

        cemb = config.model_channels * config.channel_mult_emb if config.in_channels_emb > 0 else 0

        # embedding
        if cemb > 0:
            self.emb_label = MPConv3D_E(config.in_channels_emb, cemb, kernel=())
            self.emb_dim = cemb
        else:
            self.emb_label = None
            self.emb_dim = 0

        self.disc = Discriminator(config.model_channels, cemb, config.num_layers, block_kwargs, kernel=(1,3,3))

    def get_embeddings(self, emb_in: torch.Tensor) -> torch.Tensor:
        if self.emb_label is not None:
            return self.emb_label(normalize(emb_in).to(device=self.device, dtype=self.dtype))
        else:
            return None

    def forward(self, samples: torch.Tensor, target: torch.Tensor, embeddings: torch.Tensor, training: bool = False) -> tuple[torch.Tensor, ...]:
        
        if embeddings is not None:
            embeddings = embeddings.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            
        if training == True:
            x, labels = tensor_random_cat(samples, target)
        else:
            x = torch.cat((samples, target), dim=1)

        y, hidden_kld = self.disc(tensor_4d_to_5d(x, num_channels=2), embeddings)
        logits = tensor_5d_to_4d(y)

        if training == False:
            return logits
        else:
            bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none").mean(dim=(1,2,3))
            return bce_loss, hidden_kld