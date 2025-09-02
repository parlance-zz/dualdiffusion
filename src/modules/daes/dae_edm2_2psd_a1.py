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
from modules.mp_tools import mp_silu, mp_sum, normalize, resample_2d


class MPConv2D(torch.nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel: tuple[int, int],
                 groups: int = 1, disable_weight_norm: bool = False, norm_dim: tuple[int] = None) -> None:
        
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.disable_weight_norm = disable_weight_norm
        self.norm_dim = norm_dim
        
        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels // groups, *kernel))
        if self.weight.numel() == 0:
            raise ValueError(f"Invalid weight shape: {self.weight.shape}")
        
        if self.weight.ndim == 4:
            pad_w = kernel[1] // 2
            if pad_w != 0:
                self.padding = torch.nn.ReflectionPad2d((kernel[1] // 2, kernel[1] // 2, 0, 0))
            else:
                self.padding = torch.nn.Identity()
        else:
            self.padding = None

    def forward(self, x: torch.Tensor, gain: Union[float, torch.Tensor] = 1) -> torch.Tensor:
        
        w = self.weight.float()
        if self.training == True and self.disable_weight_norm == False:
            if self.norm_dim is not None:
                w = normalize(w, dim=self.norm_dim) # traditional weight normalization
            else:
                w = normalize(w) # traditional weight normalization
            
        w = w * (gain / w[0].numel()**0.5) # magnitude-preserving scaling
        w = w.to(x.dtype)

        if w.ndim == 2:
            return x @ w.t()
        
        return torch.nn.functional.conv2d(self.padding(x), w,
            padding=(w.shape[-2]//2, 0), groups=self.groups)

    @torch.no_grad()
    def normalize_weights(self) -> None:
        if self.disable_weight_norm == False:
            if self.norm_dim is not None:
                self.weight.copy_(normalize(self.weight, dim=self.norm_dim))
            else:
                self.weight.copy_(normalize(self.weight))

@dataclass
class DAE_2PSD_A1_Config(DualDiffusionDAEConfig):

    in_channels: int     = 2
    in_channels_emb: int = 0
    out_channels: int    = 2
    latent_channels: int = 16
    
    in_num_freqs: tuple[int] = (64, 1024)
    downsample_ratio: int = 8

    model_channels: int  = 32

    channel_mult_enc0: list[int] = (1,1,2,2,4)
    channel_mult_enc1: list[int] = (1,1,2,2,4)
    channel_mult_dec0: list[int] = (1,1,2,2,4)
    channel_mult_dec1: list[int] = (1,1,2,2,4)
    num_enc0_layers_per_block: int = 2
    num_dec0_layers_per_block: int = 2
    num_enc1_layers_per_block: int = 2
    num_dec1_layers_per_block: int = 2
    
    channel_mult_enc: list[int] = (4, 8)
    channel_mult_dec: list[int] = (4, 8)
    num_enc_layers_per_block: int = 2
    num_dec_layers_per_block: int = 2

    channel_mult_emb: int  = 4

    res_balance: float     = 0.3  # Balance between main branch (0) and residual branch (1).
    mlp_multiplier: int    = 2    # Multiplier for the number of channels in the MLP.
    mlp_groups: int        = 1    # Number of groups for the MLPs.
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
        resample_mode: Literal["keep", "up", "up_h", "up_w", "down", "down_h", "down_w"] = "keep",
        dropout: float         = 0.,       # Dropout probability.
        res_balance: float     = 0.3,      # Balance between main branch (0) and residual branch (1).
        clip_act: float        = 256,      # Clip output activations. None = do not clip.
        mlp_multiplier: int    = 1,        # Multiplier for the number of channels in the MLP.
        mlp_groups: int        = 1,        # Number of groups for the MLP.
        emb_linear_groups: int = 1,
        use_pixel_norm: bool   = False,
    ) -> None:
        super().__init__()

        self.level = level
        self.use_pixel_norm = use_pixel_norm
        self.out_channels = out_channels
        self.flavor = flavor
        self.resample_mode = resample_mode
        self.dropout = dropout
        self.res_balance = res_balance
        self.clip_act = clip_act

        kernel = (3,3)

        self.conv_res0 = MPConv2D(out_channels if flavor == "enc" else in_channels,
                        out_channels * mlp_multiplier, kernel=kernel, groups=mlp_groups)
        self.conv_res1 = MPConv2D(out_channels * mlp_multiplier,
                    out_channels, kernel=kernel, groups=mlp_groups)

        if in_channels != out_channels or mlp_groups > 1:
            self.conv_skip = MPConv2D(in_channels, out_channels, kernel=(1,1), groups=1)
        else:
            self.conv_skip = None

        self.emb_gain = torch.nn.Parameter(torch.zeros([]))
        self.emb_linear = MPConv2D(emb_channels, out_channels * mlp_multiplier,
            kernel=(1,1), groups=emb_linear_groups) if emb_channels != 0 else None

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        
        x = resample_2d(x, self.resample_mode)

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

        if self.clip_act is not None:
            x = x.clip_(-self.clip_act, self.clip_act)

        return x

class DAE_2PSD_A1(DualDiffusionDAE):

    def __init__(self, config: DAE_2PSD_A1_Config) -> None:
        super().__init__()
        self.config = config

        block_kwargs = {"mlp_multiplier": config.mlp_multiplier,
                        "mlp_groups": config.mlp_groups,
                        "emb_linear_groups": config.emb_linear_groups,
                        "res_balance": config.res_balance,
                        "use_pixel_norm": config.add_pixel_norm}
        
        cemb = config.model_channels * config.channel_mult_emb * config.mlp_multiplier if config.in_channels_emb > 0 else 0

        self.out_gain = torch.nn.Parameter(torch.ones([]))
        self.recon_loss_logvar = torch.nn.Parameter(torch.zeros([]))

        # embedding
        if config.in_channels_emb > 0:
            self.emb_label = MPConv2D(config.in_channels_emb, cemb, kernel=())
            self.emb_dim = cemb
        else:
            cemb = 0
            self.emb_label = None
            self.emb_dim = 0

        in_channels = config.in_channels + int(config.add_constant_channel)
        out_channels = config.out_channels

        enc0_channels = [config.model_channels * m for m in config.channel_mult_enc0]
        dec0_channels = [config.model_channels * m for m in config.channel_mult_dec0]
        enc1_channels = [config.model_channels * m for m in config.channel_mult_enc1]
        dec1_channels = [config.model_channels * m for m in config.channel_mult_dec1]
        enc_channels = [config.model_channels * m for m in config.channel_mult_enc]
        dec_channels = [config.model_channels * m for m in config.channel_mult_dec]

        # encoder
        self.enc0 = self.make_enc(enc0_channels, in_channels, cemb,
            num_layers_per_block=config.num_enc0_layers_per_block, resample_mode="down_w", block_kwargs=block_kwargs)
        self.enc1 = self.make_enc(enc1_channels, in_channels, cemb,
            num_layers_per_block=config.num_enc1_layers_per_block, resample_mode="down_h", block_kwargs=block_kwargs)

        self.enc = self.make_enc(enc_channels, enc0_channels[-1] + enc1_channels[-1], cemb,
            num_layers_per_block=config.num_enc_layers_per_block, resample_mode="down", block_kwargs=block_kwargs)

        self.conv_latents_out = MPConv2D(enc_channels[-1], config.latent_channels, kernel=(3,3))
        self.conv_latents_out_gain = torch.nn.Parameter(torch.ones([]))
        self.conv_latents_in = MPConv2D(config.latent_channels + int(config.add_constant_channel), dec_channels[-1], kernel=(3,3))

        # decoder
        self.dec = self.make_dec(dec_channels, dec0_channels[-1] + dec1_channels[-1], cemb,
            num_layers_per_block=config.num_dec_layers_per_block, resample_mode="up", block_kwargs=block_kwargs)
        
        self.dec0 = self.make_dec(dec0_channels, out_channels, cemb,
            num_layers_per_block=config.num_dec0_layers_per_block, resample_mode="up_w", block_kwargs=block_kwargs)
        self.dec1 = self.make_dec(dec1_channels, out_channels, cemb,
            num_layers_per_block=config.num_dec1_layers_per_block, resample_mode="up_h", block_kwargs=block_kwargs)

        self.conv_out_gain0 = torch.nn.Parameter(torch.ones([]))
        self.conv_out_gain1 = torch.nn.Parameter(torch.ones([]))

    def make_enc(self, channels: list[int], in_channels: int, cemb: int,
            num_layers_per_block: int, resample_mode: str, block_kwargs: dict) -> torch.nn.ModuleDict:
        
        enc = torch.nn.ModuleDict()
        cout = in_channels

        for level, channels in enumerate(channels):
            
            if level == 0:
                
                if cout != channels:
                    enc["conv_in"] = MPConv2D(cout, channels, kernel=(3,3))
                    cout = channels
                                               
                enc[f"block{level}_in"] = Block(
                    level, cout, channels, cemb, flavor="enc", **block_kwargs)
            else:
                enc[f"block{level}_down"] = Block(level, cout, channels, cemb,
                    flavor="enc", resample_mode=resample_mode, **block_kwargs)
            
            for idx in range(num_layers_per_block):
                enc[f"block{level}_layer{idx}"] = Block(
                    level, channels, channels, cemb, flavor="enc", **block_kwargs)
            
            cout = channels

        return enc

    def make_dec(self, channels: list[int], out_channels: int, cemb: int,
            num_layers_per_block: int, resample_mode: str, block_kwargs: dict) -> torch.nn.ModuleDict:
        
        dec = torch.nn.ModuleDict()
        num_levels = len(channels)
        cin = channels[-1]

        for level in reversed(range(0, num_levels)):
            cout = channels[level]

            if level == num_levels - 1:
                dec[f"block{level}_in0"] = Block(
                    level, cin, cout, cemb, flavor="dec", **block_kwargs)
            else:
                dec[f"block{level}_up"] = Block(
                    level, cin, cout, cemb, flavor="dec", resample_mode=resample_mode, **block_kwargs)
                
            for idx in range(num_layers_per_block):
                dec[f"block{level}_layer{idx}"] = Block(
                    level, cout, cout, cemb, flavor="dec", **block_kwargs)

            cin = cout

        if cout != out_channels:
            dec["conv_out"] = MPConv2D(cout, out_channels, kernel=(3,3))

        return dec

    def run_codec(self, codec: torch.nn.ModuleDict, x: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:

        for name, block in codec.items():
            x = block(x) if "conv" in name else block(x, embeddings)
            
        return x

    def get_embeddings(self, emb_in: torch.Tensor) -> torch.Tensor:
        if self.emb_label is not None:
            return self.emb_label(normalize(emb_in).to(device=self.device, dtype=self.dtype))
        else:
            return None
    
    def get_recon_loss_logvar(self) -> torch.Tensor:
        return self.recon_loss_logvar
    
    def get_latent_shape(self, mel_spec_shape: Union[torch.Size, tuple[int, int, int, int]]) -> torch.Size:
        if len(mel_spec_shape) == 4:
            return (mel_spec_shape[0], self.config.latent_channels,
                    mel_spec_shape[2] // self.config.downsample_ratio,
                    mel_spec_shape[3] // self.config.downsample_ratio)
        else:
            raise ValueError(f"Invalid sample shape: {mel_spec_shape}")
        
    def get_mel_spec_shape(self, latent_shape: Union[torch.Size, tuple[int, int, int, int]]) -> torch.Size:
        if len(latent_shape) == 4:
            return (latent_shape[0], self.config.in_channels,
                    latent_shape[2] * self.config.downsample_ratio,
                    latent_shape[3] * self.config.downsample_ratio)
        else:
            raise ValueError(f"Invalid latent shape: {latent_shape}")
        
    def encode(self, x0: torch.Tensor, x1: torch.Tensor, embeddings: torch.Tensor, training: bool = False) -> torch.Tensor:

        if embeddings is not None:
            embeddings = embeddings.unsqueeze(-1).unsqueeze(-1)
        
        if self.config.add_constant_channel == True:
            x0 = torch.cat((x0, torch.ones_like(x0[:, :1])), dim=1)
            x1 = torch.cat((x1, torch.ones_like(x1[:, :1])), dim=1)

        x0 = self.run_codec(self.enc0, x0, embeddings)
        x1 = self.run_codec(self.enc1, x1, embeddings)

        x = torch.cat((x0, x1), dim=1)
        x = self.run_codec(self.enc, x, embeddings)

        latents = self.conv_latents_out(x, gain=self.conv_latents_out_gain)
        return latents

    def decode(self, x: torch.Tensor, embeddings: torch.Tensor, training: bool = False) -> tuple[torch.Tensor, torch.Tensor]:

        dec0_in_channels = self.config.model_channels * self.config.channel_mult_dec0[-1]
        if embeddings is not None:
            embeddings = embeddings.unsqueeze(-1).unsqueeze(-1)

        if self.config.add_constant_channel == True:
            x = torch.cat((x, torch.ones_like(x[:, :1])), dim=1)
        
        x = self.conv_latents_in(x)
        x = self.run_codec(self.dec, x, embeddings)

        x0 = x[:, :dec0_in_channels]
        x1 = x[:, dec0_in_channels:]

        x0 = self.run_codec(self.dec0, x0, embeddings) * self.conv_out_gain0
        x1 = self.run_codec(self.dec1, x1, embeddings) * self.conv_out_gain1

        return x0, x1
    
    def forward(self, x0: torch.Tensor, x1: torch.Tensor, dae_embeddings: torch.Tensor, latents_sigma: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, ...]:
        
        pre_norm_latents = self.encode(x0, x1, dae_embeddings, training=True)
        if latents_sigma is not None:
            #latents = pre_norm_latents + latents_sigma * torch.randn_like(pre_norm_latents) * pre_norm_latents.detach().std(dim=(1,2,3), keepdim=True).clamp(min=1e-2)
            total_var = pre_norm_latents.var(dim=(1,2,3), keepdim=True) + latents_sigma**2
            latents = (pre_norm_latents + latents_sigma * torch.randn_like(pre_norm_latents)) / total_var**0.5
        else:
            latents = pre_norm_latents
        
        recon0, recon1 = self.decode(latents, dae_embeddings, training=True)
        return recon0, recon1, latents, pre_norm_latents

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