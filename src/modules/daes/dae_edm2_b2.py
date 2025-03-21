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

from modules.daes.dae import DualDiffusionDAEConfig, DualDiffusionDAE
from modules.mp_tools import MPConv3D, normalize, resample_3d, mp_silu, mp_sum
from utils.dual_diffusion_utils import tensor_4d_to_5d, tensor_5d_to_4d


@dataclass
class DualDiffusionDAE_EDM2_B2_Config(DualDiffusionDAEConfig):

    model_channels: int       = 32           # Base multiplier for the number of channels.
    latent_channels: int      = 4
    in_channels_emb: int      = 0
    channel_mult: list[int]   = (1,2,3,5)    # Per-resolution multipliers for the number of channels.
    channel_mult_emb: Optional[int] = 5      # Multiplier for final embedding dimensionality.
    num_layers_per_block: int = 3            # Number of resnet blocks per resolution.
    res_balance: float        = 0.3          # Balance between main branch (0) and residual branch (1).
    mlp_multiplier: int   = 2                # Multiplier for the number of channels in the MLP.
    mlp_groups: int       = 1                # Number of groups for the MLPs.
  
class Block(torch.nn.Module):

    def __init__(self,
        level: int,                             # Resolution level.
        in_channels: int,                       # Number of input channels.
        out_channels: int,                      # Number of output channels.
        emb_channels: int,                      # Number of embedding channels.
        flavor: Literal["enc", "dec"] = "enc",
        resample_mode: Literal["keep", "up", "down"] = "keep",
        res_balance: float     = 0.3,      # Balance between main branch (0) and residual branch (1).
        clip_act: float        = 256,      # Clip output activations. None = do not clip.
        mlp_multiplier: int    = 1,        # Multiplier for the number of channels in the MLP.
        mlp_groups: int        = 1,        # Number of groups for the MLP.

    ) -> None:
        super().__init__()

        self.level = level
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.flavor = flavor
        self.resample_mode = resample_mode
        self.res_balance = res_balance
        self.clip_act = clip_act
        
        self.conv_res0 = MPConv3D(out_channels if flavor == "enc" else in_channels,
                                out_channels * mlp_multiplier, kernel=(1,3,3), groups=mlp_groups)
        self.conv_res1 = MPConv3D(out_channels * mlp_multiplier, out_channels, kernel=(1,3,3), groups=mlp_groups)
        self.conv_skip = MPConv3D(in_channels, out_channels, kernel=(2,1,1), groups=1) if in_channels != out_channels else None

        self.emb_gain = torch.nn.Parameter(torch.zeros([]))
        self.emb_linear = MPConv3D(emb_channels, out_channels * mlp_multiplier,
                                 kernel=(1,1,1), groups=1) if emb_channels != 0 else None

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        
        x = resample_3d(x, mode=self.resample_mode)
        if self.flavor == "enc":
            if self.conv_skip is not None:
                x = self.conv_skip(x)
            x = normalize(x, dim=1) # pixel norm

        y = self.conv_res0(mp_silu(x))

        if self.emb_channels > 0:
            c = self.emb_linear(emb, gain=self.emb_gain) + 1.
            y = mp_silu(y * c)
        else:
            y = mp_silu(y)

        y = self.conv_res1(y)

        if self.flavor == "dec" and self.conv_skip is not None:
            x = self.conv_skip(x)

        x = mp_sum(x, y, t=self.res_balance)

        if self.clip_act is not None:
            x = x.clip_(-self.clip_act, self.clip_act)
        return x

class DualDiffusionDAE_EDM2_B2(DualDiffusionDAE):

    supports_channels_last: Union[bool, Literal["3d"]] = "3d"

    def __init__(self, config: DualDiffusionDAE_EDM2_B2_Config) -> None:
        super().__init__()
        self.config = config

        block_kwargs = {"mlp_multiplier": config.mlp_multiplier,
                        "mlp_groups": config.mlp_groups,
                        "res_balance": config.res_balance} 
        
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
        cout = config.in_channels // 2 + 1
        for level, channels in enumerate(cblock):
            
            if level == 0:
                cin = cout
                cout = channels
                self.enc[f"conv_in"] = MPConv3D(cin, cout, kernel=(2,3,3))
            else:
                self.enc[f"block{level}_down"] = Block(level, cout, cout, cemb,
                                        flavor="enc", resample_mode="down", **block_kwargs)
            
            for idx in range(config.num_layers_per_block):
                cin = cout
                cout = channels
                self.enc[f"block{level}_layer{idx}"] = Block(level, cin, cout, cemb,
                                                             flavor="enc", **block_kwargs)

        self.conv_latents_out = MPConv3D(cout, config.latent_channels, kernel=(2,3,3))
        self.conv_latents_in = MPConv3D(config.latent_channels + 1, cout, kernel=(2,3,3))

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        for level, channels in reversed(list(enumerate(cblock))):
            
            if level == len(cblock) - 1:
                self.dec[f"block{level}_in0"] = Block(level, cout, cout, cemb, flavor="dec", **block_kwargs)
                self.dec[f"block{level}_in1"] = Block(level, cout, cout, cemb, flavor="dec", **block_kwargs)
            else:
                self.dec[f"block{level}_up"] = Block(level, cout, cout, cemb, flavor="dec",
                                                     resample_mode="up", **block_kwargs)
            for idx in range(config.num_layers_per_block + 1):
                cin = cout
                cout = channels
                self.dec[f"block{level}_layer{idx}"] = Block(level, cin, cout, cemb, flavor="dec", **block_kwargs)
        
        self.conv_out = MPConv3D(cout, config.out_channels // 2, kernel=(2,3,3))
            
    def get_embeddings(self, emb_in: torch.Tensor) -> torch.Tensor:
        if self.emb_label is not None:
            return self.emb_label(normalize(emb_in).to(device=self.device, dtype=self.dtype))
        else:
            return None
    
    def get_recon_loss_logvar(self) -> torch.Tensor:
        return self.recon_loss_logvar
    
    def get_latent_shape(self, sample_shape: Union[torch.Size, tuple[int, int, int, int]]) -> torch.Size:
        if len(sample_shape) == 4:
            return (sample_shape[0], self.config.latent_channels * self.config.in_channels,
                    sample_shape[2] // 2 ** (self.num_levels-1),
                    sample_shape[3] // 2 ** (self.num_levels-1))
        else:
            raise ValueError(f"Invalid sample shape: {sample_shape}")
        
    def get_sample_shape(self, latent_shape: Union[torch.Size, tuple[int, int, int, int]]) -> torch.Size:
        if len(latent_shape) == 4:
            return (latent_shape[0],
                    self.config.out_channels,
                    latent_shape[2] * 2 ** (self.num_levels-1),
                    latent_shape[3] * 2 ** (self.num_levels-1))
        else:
            raise ValueError(f"Invalid latent shape: {latent_shape}")
        
    def encode(self, x: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:

        x = tensor_4d_to_5d(x, num_channels=self.config.in_channels // 2)
        x = torch.cat((x, torch.ones_like(x[:, :1])), dim=1)

        if embeddings is not None:
            embeddings = embeddings.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        for name, block in self.enc.items():
            x = block(x) if "conv" in name else block(x, embeddings)
        
        latents = normalize(self.conv_latents_out(x))
        return tensor_5d_to_4d(latents)
    
    def decode(self, x: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:

        x = tensor_4d_to_5d(x, num_channels=self.config.latent_channels)
        x = torch.cat((x, torch.ones_like(x[:, :1])), dim=1)

        if embeddings is not None:
            embeddings = embeddings.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        for block in self.dec.values():
            x = block(x, embeddings)

        return tensor_5d_to_4d(self.conv_out(x, gain=self.out_gain))
    
    def forward(self, samples: torch.Tensor, embeddings: torch.Tensor, add_latents_noise: float = 0) -> tuple[torch.Tensor, torch.Tensor]:
        
        latents = self.encode(samples, embeddings)
        if add_latents_noise > 0:
            latents = normalize(latents + torch.randn_like(latents))
        
        reconstructed = self.decode(latents, embeddings)

        return latents, reconstructed