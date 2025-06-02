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
from modules.mp_tools import mp_silu, mp_sum, mp_cat, normalize, resample_3d
from utils.dual_diffusion_utils import tensor_4d_to_5d, tensor_5d_to_4d


@dataclass
class DAE_J2_Config(DualDiffusionDAEConfig):

    in_channels: int     = 1
    out_channels: int    = 1
    in_channels_emb: int = 0
    in_num_freqs: int    = 256
    latent_channels: int = 4

    model_channels: int         = 32         # Base multiplier for the number of channels.
    channel_mult_enc: int       = 1
    channel_mult_dec: list[int] = (1,2,3,4)
    channel_mult_emb: int       = 4          # Multiplier for final embedding dimensionality.
    num_enc_layers_per_block: int = 3        # Number of resnet blocks per resolution.
    num_dec_layers_per_block: int = 3        # Number of resnet blocks per resolution.
    res_balance: float     = 0.3             # Balance between main branch (0) and residual branch (1).
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
        flavor: Literal["enc", "dec", "enc_in", "dec_in"] = "enc",
        resample_mode: Literal["keep", "up", "down"] = "keep",
        dropout: float         = 0,        # Dropout probability.
        res_balance: float     = 0.3,      # Balance between main branch (0) and residual branch (1).
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
        self.dropout = dropout
        self.res_balance = res_balance
        self.clip_act = clip_act
        self.kernel = kernel

        self.conv_res0 = MPConv3D_E(out_channels if flavor.startswith("enc") else in_channels,
                        out_channels * mlp_multiplier, kernel=kernel, groups=mlp_groups)
        self.conv_res1 = MPConv3D_E(out_channels * mlp_multiplier,
                        out_channels, kernel=kernel, groups=mlp_groups)

        if in_channels != out_channels or mlp_groups > 1 or flavor.endswith("_in"):
            skip_kernel = (kernel[0],3,3) if flavor == "enc_in" else (kernel[0],1,1)
            self.conv_skip = MPConv3D_E(in_channels, out_channels, kernel=skip_kernel, groups=1)
        else:
            self.conv_skip = None

        self.out_scale = torch.nn.Parameter(torch.ones([]))
        self.out_shift = torch.nn.Parameter(torch.zeros([]))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        
        x = resample_3d(x, mode=self.resample_mode)

        if self.flavor.startswith("enc"):
            if self.conv_skip is not None:
                x = self.conv_skip(x)

        y = self.conv_res0(mp_silu(x))
        y = mp_silu(y)

        if self.dropout != 0 and self.training == True: # magnitude preserving fix for dropout
            y = torch.nn.functional.dropout(y, p=self.dropout) * (1. - self.dropout)**0.5

        y = self.conv_res1(y)

        if self.flavor.startswith("dec") and self.conv_skip is not None:
            x = self.conv_skip(x)
        
        x = mp_sum(x, y, t=self.res_balance)
        
        if self.clip_act is not None:
            x = x.clip_(-self.clip_act, self.clip_act)

        x = x * self.out_scale + self.out_shift

        kl_dims = (1,2,3,4) #if self.kernel[0] > 1 else (1,3,4)
        x_mean = x.mean(dim=kl_dims)
        x_var = x.var(dim=kl_dims).clip(min=1e-2)
        kld = x_mean.square() + x_var - 1 - x_var.log()

        return x, kld

class Encoder(torch.nn.Module):

    def __init__(self, in_channels: int, out_channels: int, num_layers: int,
            block_kwargs: dict, kernel: tuple[int, int] = (1,3,3)) -> None:
        super().__init__()

        self.enc = torch.nn.ModuleDict()
        for idx in range(num_layers):
            flavor = "enc_in" if idx == 0 else "enc"
            in_channels = out_channels if idx > 0 else in_channels + 1
            self.enc[f"layer{idx}"] = Block(0, in_channels, out_channels,
                                    flavor=flavor, kernel=kernel, **block_kwargs)

        self.dec = torch.nn.ModuleDict()
        for idx in range(num_layers):
            self.dec[f"layer{idx}"] = Block(0, out_channels * 2, out_channels,
                                    flavor="dec", kernel=kernel, **block_kwargs)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        
        x = torch.cat((x, torch.ones_like(x[:, :1])), dim=1)
        hidden_kld = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

        skips = []
        for block in self.enc.values():
            x, kld = block(x)
            skips.append(x)
            hidden_kld = hidden_kld + kld

        for block in self.dec.values():
            x, kld = block(mp_cat(x, skips.pop(), t=0.5))
            hidden_kld = hidden_kld + kld

        return x, hidden_kld
    
class DAE_J2(DualDiffusionDAE):

    supports_channels_last: Union[bool, Literal["3d"]] = "3d"

    def __init__(self, config: DAE_J2_Config) -> None:
        super().__init__()
        self.config = config

        block_kwargs = {"mlp_multiplier": config.mlp_multiplier,
                        "mlp_groups": config.mlp_groups,
                        "res_balance": config.res_balance}
        
        enc_channels = config.model_channels * config.channel_mult_enc
        dec_channels = [config.model_channels * m for m in config.channel_mult_dec]

        cemb = config.model_channels * config.channel_mult_emb if config.in_channels_emb > 0 else 0

        self.num_levels = len(config.channel_mult_dec)
        self.downsample_ratio = 2 ** (self.num_levels - 1)
        self.latents_out_gain = torch.nn.Parameter(torch.ones([]))
        self.out_gain = torch.nn.Parameter(torch.ones([]))
        self.recon_loss_logvar = torch.nn.Parameter(torch.zeros([]))

        # embedding
        if cemb > 0:
            self.emb_label = MPConv3D_E(config.in_channels_emb, cemb, kernel=())
            self.emb_gain = torch.nn.Parameter(torch.zeros([]))
            self.emb_dim = cemb
        else:
            self.emb_label = None
            self.emb_gain = None
            self.emb_dim = 0

        # encoders
        self.encoder = Encoder(config.in_channels, enc_channels,
            config.num_enc_layers_per_block, block_kwargs, kernel=(1,3,3))

        self.conv_latents_out = MPConv3D_E(enc_channels, config.latent_channels, kernel=(1,3,3))

        # decoder
        self.dec = torch.nn.ModuleDict()
        cin = config.latent_channels + 1

        for level in reversed(range(0, self.num_levels)):
            
            kernel = (1,1,1) if level == self.num_levels - 1 else (1,3,3)
            cout = dec_channels[level]

            if level == self.num_levels - 1:
                self.dec[f"block{level}_in"] = Block(level, cin,  cout, flavor="dec_in", **block_kwargs, kernel=kernel)
            else:
                self.dec[f"block{level}_up"] = Block(level, cin, cout, flavor="dec", resample_mode="up", **block_kwargs, kernel=kernel)
            
            for idx in range(config.num_dec_layers_per_block):
                self.dec[f"block{level}_layer{idx}"] = Block(level, cout, cout, flavor="dec", **block_kwargs, kernel=kernel)

            cin = cout

        self.conv_out = MPConv3D_E(cout, self.config.out_channels, kernel=(1,3,3))

    def get_embeddings(self, emb_in: torch.Tensor) -> torch.Tensor:
        if self.emb_label is not None:
            return self.emb_label(normalize(emb_in).to(device=self.device, dtype=self.dtype))
        else:
            return None
    
    def get_recon_loss_logvar(self) -> torch.Tensor:
        return self.recon_loss_logvar
    
    def get_latent_shape(self, mel_spec_shape: Union[torch.Size, tuple[int, int, int, int]]) -> torch.Size:
        if len(mel_spec_shape) == 4:
            return (mel_spec_shape[0], self.config.latent_channels*2,
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
        
        x, hidden_kld = self.encoder(tensor_4d_to_5d(x, num_channels=1))

        latents = tensor_5d_to_4d(self.conv_latents_out(x, gain=self.latents_out_gain))
        latents = torch.nn.functional.avg_pool2d(latents, self.downsample_ratio)

        if training == False:
            return latents
        else:
            return latents, hidden_kld

    def decode(self, x: torch.Tensor, embeddings: torch.Tensor, training: bool = False) -> torch.Tensor:
        
        x = tensor_4d_to_5d(x, num_channels=self.config.latent_channels)
        x = torch.cat((x, torch.ones_like(x[:, :1])), dim=1)

        hidden_kld = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        
        for block in self.dec.values():
            x, kld = block(x)
            hidden_kld = hidden_kld + kld

        decoded = tensor_5d_to_4d(self.conv_out(x, gain=self.out_gain))

        if training == False:
            return decoded
        else:
            return decoded, hidden_kld
    
    def forward(self, samples: torch.Tensor, dae_embeddings: torch.Tensor) -> tuple[torch.Tensor, ...]:
        
        latents, enc_hidden_kld = self.encode(samples, dae_embeddings, training=True)
        decoded, dec_hidden_kld = self.decode(latents, dae_embeddings, training=True)

        latents_mean = latents.mean(dim=(1,2,3))
        latents_var = latents.var(dim=(1,2,3)).clip(min=1e-2)
        latents_kld = latents_mean.square() + latents_var - 1 - latents_var.log()
    
        return latents, decoded, latents_kld, enc_hidden_kld + dec_hidden_kld

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