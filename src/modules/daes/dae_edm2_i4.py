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
from modules.mp_tools import mp_silu, normalize, mp_sum, mp_cat
from utils.resample import FilteredDownsample1D, FilteredUpsample1D


@dataclass
class DAE_I4_Config(DualDiffusionDAEConfig):

    in_channels: int      = 1
    out_channels: int     = 0 # unused
    in_channels_emb: int  = 0
    
    in_num_freqs: int    = 1
    latent_channels: int = 12
    
    resample_beta: float = 3.437
    resample_k_size: int = 23
    resample_factor: int = 2
    extra_downsamples: int = 4

    model_channels: int   = 32
    channel_mult_emb: int = 0
    channel_mult_enc: list[int] = (1,1,2,2,3,3,4,4)
    channel_mult_dec: list[int] = (1,1,2,2,3,3,4,4)
    num_enc_layers_per_block: list[int] = (1,1,1,1,1,1,1,1)
    num_dec_layers_per_block: list[int] = (1,1,1,1,1,1,1,1)
    kernel_enc: list[int] = (2,11)
    kernel_dec: list[int] = (2,11)
    mlp_multiplier: int = 1
    mlp_groups: int     = 1

    cat_balance: float  = 0.5
    res_balance: float  = 0.3

class MPConv1D(torch.nn.Module):

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
            assert kernel[0] <= 2
            pad_h, pad_w = kernel[0] // 2, kernel[1] // 2
            if pad_w != 0 or pad_h != 0:
                self.padding = torch.nn.ReflectionPad2d((kernel[1] // 2, kernel[1] // 2, 0, kernel[0] // 2))
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
        
        return torch.nn.functional.conv2d(self.padding(x), w, groups=self.groups)

    @torch.no_grad()
    def normalize_weights(self) -> None:
        if self.disable_weight_norm == False:
            self.weight.copy_(normalize(self.weight))

class MPConv2D_R(torch.nn.Module):

    def __init__(self, in_channels: int, out_channels: int,
                 kernel: tuple[int, int], groups: int = 1, stride: int = 1,
                 disable_weight_norm: bool = False) -> None:
        
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.stride = stride
        self.disable_weight_norm = disable_weight_norm
        
        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels // groups, *kernel))
        self.padding = torch.nn.ReflectionPad2d((kernel[1] // 2, kernel[1] // 2, 0, 0))

    def forward(self, x: torch.Tensor, gain: Union[float, torch.Tensor] = 1.) -> torch.Tensor:
        
        w = self.weight.float()
        if self.training == True and self.disable_weight_norm == False:
            w = normalize(w, dim=1) # traditional weight normalization
            
        w = w * (gain / w[0].numel()**0.5) # magnitude-preserving scaling
        w = w.to(x.dtype)

        if w.ndim == 2:
            return x @ w.t()
        
        return torch.nn.functional.conv2d(self.padding(x), w, 
            padding=(w.shape[-2]//2, 0), groups=self.groups, stride=self.stride)

    @torch.no_grad()
    def normalize_weights(self) -> None:
        if self.disable_weight_norm == False:
            self.weight.copy_(normalize(self.weight, dim=1))

class Block1D(torch.nn.Module):

    def __init__(self,
        level: int,                             # Resolution level.
        in_channels: int,                       # Number of input channels.
        out_channels: int,                      # Number of output channels.
        emb_channels: int,                      # Number of embedding channels.
        flavor: Literal["enc", "dec"] = "enc",
        resample: Optional[torch.nn.Module] = None,
        res_balance: float     = 0.3,
        clip_act: float        = 256,      # Clip output activations. None = do not clip.
        mlp_multiplier: int    = 1,        # Multiplier for the number of channels in the MLP.
        mlp_groups: int        = 1,        # Number of groups for the MLP.
        kernel: tuple[int, int] = (2,11),   # Kernel size for the convolutional layers.
    ) -> None:
        super().__init__()

        self.level = level
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.flavor = flavor
        self.res_balance = res_balance
        self.clip_act = clip_act

        self.resample = resample if resample is not None else torch.nn.Identity()
            
        self.conv_res0 = MPConv1D(in_channels,  out_channels * mlp_multiplier, kernel=kernel, groups=mlp_groups)
        self.conv_res1 = MPConv1D(out_channels * mlp_multiplier, out_channels, kernel=kernel, groups=mlp_groups)
        
        if in_channels != out_channels or mlp_groups > 1:
            self.conv_skip = MPConv1D(in_channels, out_channels, kernel=(1,1), groups=1)
        else:
            self.conv_skip = None

        self.emb_gain = torch.nn.Parameter(torch.zeros([])) if emb_channels != 0 else None
        self.emb_linear = MPConv1D(emb_channels, out_channels * mlp_multiplier,
            kernel=(1,1), groups=1) if emb_channels != 0 else None
    
    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        
        x = self.resample(x)
        if self.flavor == "enc":
            x = normalize(x, dim=1) # pixel norm

        y = self.conv_res0(mp_silu(x))

        if self.emb_linear is not None:
            c = self.emb_linear(emb, gain=self.emb_gain) + 1.
            y = mp_silu(y * c)
        else:
            y = mp_silu(y)

        y = self.conv_res1(y)
        
        if self.conv_skip is not None:
            x = self.conv_skip(x)
        
        x = mp_sum(x, y, self.res_balance)
        
        if self.clip_act is not None:
            x = x.clip_(-self.clip_act, self.clip_act)

        return x
    
class DAE_I4(DualDiffusionDAE):

    def __init__(self, config: DAE_I4_Config) -> None:
        super().__init__()
        self.config = config

        block_kwargs = {"mlp_multiplier": config.mlp_multiplier,
                        "mlp_groups": config.mlp_groups,
                        "res_balance": config.res_balance}
        
        enc_channels = [config.model_channels * m for m in config.channel_mult_enc]
        dec_channels = [config.model_channels * m for m in config.channel_mult_dec]

        cemb = config.model_channels * config.channel_mult_emb if config.in_channels_emb > 0 else 0

        self.num_levels = len(config.channel_mult_dec)
        self.total_downsample_ratio = config.resample_factor ** (self.num_levels - 1 + config.extra_downsamples)

        # embedding
        if cemb > 0:
            self.emb_label = MPConv1D(config.in_channels_emb, cemb, kernel=())
            self.emb_dim = cemb
        else:
            self.emb_label = None
            self.emb_dim = 0

        assert len(enc_channels) == len(config.num_enc_layers_per_block)
        assert len(dec_channels) == len(config.num_dec_layers_per_block)

        self.downsample = FilteredDownsample1D(k_size=config.resample_k_size,
                        beta=config.resample_beta, factor=config.resample_factor)
        self.upsample = FilteredUpsample1D(
            k_size=config.resample_k_size * config.resample_factor + config.resample_k_size % config.resample_factor,
            beta=config.resample_beta, factor=config.resample_factor)

        # encoder
        self.enc_skip_balance = torch.nn.Parameter(torch.zeros(self.num_levels))
        self.enc = torch.nn.ModuleDict()
        cout = 1 # 1 const channel

        for level, channels in enumerate(enc_channels):
            self.enc[f"block{level}_conv_in"] = MPConv1D(cout + config.in_channels, channels, kernel=config.kernel_enc)

            if level == 0:
                self.enc[f"block{level}_in"] = Block1D(
                    level, channels, channels, 0, flavor="enc", kernel=config.kernel_enc, **block_kwargs)
            else:
                self.enc[f"block{level}_down"] = Block1D(level, channels, channels, 0,
                    flavor="enc", kernel=config.kernel_enc, **block_kwargs)
            
            for idx in range(config.num_enc_layers_per_block[level]):
                self.enc[f"block{level}_layer{idx}"] = Block1D(
                    level, channels, channels, 0, flavor="enc", kernel=config.kernel_enc, **block_kwargs)
            
            self.enc[f"block{level}_conv_out"] = MPConv1D(channels, config.latent_channels, kernel=config.kernel_enc)
            cout = channels

        self.latents_out_gain = torch.nn.Parameter(torch.ones([]))

        # decoder
        self.conv_latents_reg = MPConv2D_R(config.latent_channels*2, config.latent_channels*2, kernel=(3,3))
        
        self.dec = torch.nn.ModuleDict()
        cout = 1 # 1 const channel

        for level in reversed(range(0, self.num_levels)):
            channels = dec_channels[level]

            self.dec[f"block{level}_conv_in"] = MPConv1D(cout + config.latent_channels, channels, kernel=config.kernel_dec)

            if level == self.num_levels - 1:
                self.dec[f"block{level}_in"] = Block1D(
                    level, channels, channels, cemb, flavor="dec", **block_kwargs, kernel=config.kernel_dec)
            else:
                self.dec[f"block{level}_up"] = Block1D(level, channels, channels,
                    cemb, flavor="dec", **block_kwargs, kernel=config.kernel_dec)
            
            for idx in range(config.num_dec_layers_per_block[level]):
                self.dec[f"block{level}_layer{idx}"] = Block1D(
                    level, channels, channels, cemb, flavor="dec", **block_kwargs, kernel=config.kernel_dec)

            self.dec[f"block{level}_conv_out"] = MPConv1D(channels, channels, kernel=config.kernel_dec)
            cout = channels

    def get_embeddings(self, emb_in: torch.Tensor) -> torch.Tensor:
        if self.emb_label is not None:
            return self.emb_label(normalize(emb_in).to(device=self.device, dtype=self.dtype))
        else:
            return None
    
    def get_recon_loss_logvar(self) -> torch.Tensor:
        return torch.ones(1, device=self.device, dtype=self.dtype)
    
    def get_latent_shape(self, mel_spec_shape: Union[torch.Size, tuple[int, int, int, int]]) -> torch.Size:
        raise NotImplementedError()
        if len(mel_spec_shape) == 4:
            return (mel_spec_shape[0], self.config.latent_channels*2,
                    mel_spec_shape[2] // 2 ** (self.num_levels-1),
                    mel_spec_shape[3] // 2 ** (self.num_levels-1))
        else:
            raise ValueError(f"Invalid sample shape: {mel_spec_shape}")
        
    def get_mel_spec_shape(self, latent_shape: Union[torch.Size, tuple[int, int, int, int]]) -> torch.Size:
        raise NotImplementedError()
        if len(latent_shape) == 4:
            return (latent_shape[0], 2,
                    latent_shape[2] * 2 ** (self.num_levels-1),
                    latent_shape[3] * 2 ** (self.num_levels-1))
        else:
            raise ValueError(f"Invalid latent shape: {latent_shape}")
        
    def encode(self, x: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        
        input_x = x
        x = torch.ones_like(x[:, :1])

        if embeddings is not None:
            embeddings = embeddings[:, :, None, None]

        latents = None

        for name, block in self.enc.items():
            
            if name.endswith("_conv_in"):                
                if not name.startswith("block0_"):
                    x = self.downsample(x)

                x = mp_cat(x, input_x, t=self.config.cat_balance)
                input_x = self.downsample(input_x)
                x = block(x)

            elif name.endswith("_conv_out"):
                latents_out = block(x).float()
                latents_out = latents_out.reshape(latents_out.shape[0], latents_out.shape[1] * 2, 1, latents_out.shape[3])
                if latents is None:
                    latents = latents_out
                else:
                    latents = torch.cat((normalize(latents_out), self.downsample(latents)), dim=2)
            else:
                x = block(x, embeddings)

        for _ in range(self.config.extra_downsamples):
            latents = self.downsample(latents)

        return latents * self.latents_out_gain

    def decode(self, x: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        
        latents = self.conv_latents_reg(x).to(dtype=torch.bfloat16)

        for _ in range(self.config.extra_downsamples):
            latents = self.upsample(latents)

        x = torch.ones_like(latents[:, :1, :2])

        if embeddings is not None:
            embeddings = embeddings[:, :, None, None]

        x_out: list[torch.Tensor] = []
        for name, block in self.dec.items():
            
            if name.endswith("_conv_in"):
                if not name.startswith(f"block{self.num_levels-1}_"):
                    x = self.upsample(x)

                latents_in = latents[:, :, 0:1, :].reshape(latents.shape[0], self.config.latent_channels, 2, latents.shape[3])
                x = mp_cat(x, latents_in, t=self.config.cat_balance)
                if not name.startswith(f"block0_"):
                    latents = self.upsample(latents[:, :, 1:, :])
                x = block(x)

            elif name.endswith("_conv_out"):
                x_out.append(normalize(block(x)))
            else:
                x = block(x, embeddings)

        x_out.reverse()
        return x_out
    
    def forward(self, samples: torch.Tensor, dae_embeddings: torch.Tensor,
            latents_sigma: torch.Tensor = None) -> tuple[torch.Tensor, ...]:
        
        latents = self.encode(samples, dae_embeddings)

        if latents_sigma is not None:
            latents = (latents + torch.randn_like(latents) * latents_sigma) / (1 + latents_sigma**2)**0.5

        decoded = self.decode(latents, dae_embeddings)

        latents_mean = latents.mean(dim=(1,2,3))
        latents_var = latents.var(dim=(1,2,3))
        latents_kld = latents_mean.square() + latents_var - 1 - latents_var.log()
        
        return latents, decoded, latents_kld

    def tiled_encode(self, x: torch.Tensor, embeddings: torch.Tensor, max_chunk: int = 6144, overlap: int = 256) -> torch.Tensor:
        raise NotImplementedError()
    
        x_w = x.shape[-1]
        ds = self.total_downsample_ratio
        
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
            latents_chunk = self.encode(chunk, embeddings)
            
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
        
        return latents
