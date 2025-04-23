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
from modules.mp_tools import mp_silu, mp_sum, mp_cat, normalize, resample_2d, wavelet_decompose2d


def wavelet_space_to_channel2d(x: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        (
            x[:, :, 0::2, 0::2],
            x[:, :, 0::2, 1::2],
            x[:, :, 1::2, 0::2],
            torch.ones_like(x[:, 0:1, 1::2, 1::2])
        ),
        dim=1
    )

@dataclass
class DAE_J1_Config(DualDiffusionDAEConfig):

    in_channels: int     = 2
    out_channels: int    = 2
    in_channels_emb: int = 1024
    in_num_freqs: int    = 256
    latent_channels: int = 8

    model_channels: int         = 128        # Base multiplier for the number of channels.
    channel_mult_enc: list[int] = (1,1,1)
    channel_mult_dec: list[int] = (1,2,3,4)
    channel_mult_emb: int       = 8          # Multiplier for final embedding dimensionality.
    channel_mult_fuser: int     = 3
    num_enc_layers_per_block: int = 2        # Number of resnet blocks per resolution.
    num_dec_layers_per_block: int = 3        # Number of resnet blocks per resolution.
    res_balance: float     = 0.3             # Balance between main branch (0) and residual branch (1).
    mlp_multiplier: int    = 2               # Multiplier for the number of channels in the MLP.
    mlp_groups: int        = 1               # Number of groups for the MLPs.

class MPConv2D_E(torch.nn.Module):

    def __init__(self, in_channels: int, out_channels: int,
                 kernel: tuple[int, int, int], groups: int = 1, stride: int = 1,
                 disable_weight_norm: bool = False, norm_dim: int = 1, out_gain_param: bool = False) -> None:
        
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.stride = stride
        self.disable_weight_norm = disable_weight_norm
        self.norm_dim = norm_dim
        
        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels // groups, *kernel))

        if self.weight.ndim == 4:
            pad_w = kernel[1] // 2
            if pad_w != 0:
                self.padding = torch.nn.ReflectionPad2d((kernel[1] // 2, kernel[1] // 2, 0, 0))
            else:
                self.padding = torch.nn.Identity()
        else:
            self.padding = None

        if out_gain_param == True:
            self.out_gain = torch.nn.Parameter(torch.ones([]))
        else:
            self.out_gain = None

    def forward(self, x: torch.Tensor, gain: Optional[Union[float, torch.Tensor]] = None) -> torch.Tensor:
        
        if self.out_gain is None:
            if gain is None:
                gain = 1.
        else:
            gain = self.out_gain
        
        w = self.weight.float()
        if self.training == True and self.disable_weight_norm == False:
            w = normalize(w, dim=self.norm_dim) # traditional weight normalization
            
        w = w * (gain / w[0].numel()**0.5) # magnitude-preserving scaling
        w = w.to(x.dtype)

        if w.ndim == 2:
            return x @ w.t()
        
        return torch.nn.functional.conv2d(self.padding(x), w, padding=(w.shape[-2]//2, 0), groups=self.groups, stride=self.stride)

    @torch.no_grad()
    def normalize_weights(self) -> None:
        if self.disable_weight_norm == False:
            self.weight.copy_(normalize(self.weight, dim=self.norm_dim))

class Block(torch.nn.Module):

    def __init__(self,
        level: int,                             # Resolution level.
        in_channels: int,                       # Number of input channels.
        out_channels: int,                      # Number of output channels.
        flavor: Literal["enc", "dec"] = "enc",
        resample_mode: Literal["keep", "up", "down"] = "keep",
        dropout: float         = 0,        # Dropout probability.
        res_balance: float     = 0.3,      # Balance between main branch (0) and residual branch (1).
        clip_act: float        = 256,      # Clip output activations. None = do not clip.
        mlp_multiplier: int    = 2,        # Multiplier for the number of channels in the MLP.
        mlp_groups: int        = 1,        # Number of groups for the MLP.
        kernel: tuple[int, int] = (3,3),   # Kernel size for the convolutional layers.
    ) -> None:
        super().__init__()

        self.level = level
        self.out_channels = out_channels
        self.flavor = flavor
        self.resample_mode = resample_mode
        self.dropout = dropout
        self.res_balance = res_balance
        self.clip_act = clip_act

        self.conv_res0 = MPConv2D_E(out_channels if flavor == "enc" else in_channels,
                        out_channels * mlp_multiplier, kernel=kernel, groups=mlp_groups)
        self.conv_res1 = MPConv2D_E(out_channels * mlp_multiplier,
                        out_channels, kernel=kernel, groups=mlp_groups)

        if in_channels != out_channels or mlp_groups > 1:
            self.conv_skip = MPConv2D_E(in_channels, out_channels, kernel=(1,1), groups=1)
        else:
            self.conv_skip = None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        
        x = resample_2d(x, mode=self.resample_mode)

        if self.flavor == "enc":
            if self.conv_skip is not None:
                x = self.conv_skip(x)

        y = self.conv_res0(mp_silu(x))
        y = mp_silu(y)

        if self.dropout != 0 and self.training == True: # magnitude preserving fix for dropout
            y = torch.nn.functional.dropout(y, p=self.dropout) * (1. - self.dropout)**0.5

        y = self.conv_res1(y)

        if self.flavor == "dec" and self.conv_skip is not None:
            x = self.conv_skip(x)
        
        x = mp_sum(x, y, t=self.res_balance)
        
        if self.clip_act is not None:
            x = x.clip_(-self.clip_act, self.clip_act)

        x_mean = x.mean(dim=(1,2,3))
        x_var = x.var(dim=(1,2,3)).clip(min=1e-2)
        kld = x_mean.square() + x_var - 1 - x_var.log()

        return x, kld

class WaveletEncoder(torch.nn.Module):

    def __init__(self, in_channels: int, out_channels: int, num_layers: int,
            block_kwargs: dict, kernel: tuple[int, int] = (3,3)) -> None:
        super().__init__()

        self.conv_in = MPConv2D_E(in_channels, out_channels, kernel=kernel)

        self.enc = torch.nn.ModuleDict()
        for idx in range(num_layers):
            self.enc[f"layer{idx}"] = Block(0, out_channels, out_channels, flavor="enc", kernel=kernel, **block_kwargs)

        self.dec = torch.nn.ModuleDict()
        for idx in range(num_layers):
            self.dec[f"layer{idx}"] = Block(0, out_channels * 2, out_channels, flavor="dec", kernel=kernel, **block_kwargs)
        
        self.out_gain = torch.nn.Parameter(torch.ones([]))
        self.conv_out = MPConv2D_E(out_channels, out_channels, kernel=(1,1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        
        x = self.conv_in(x)
        
        hidden_kld = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

        skips = []
        for block in self.enc.values():
            x, kld = block(x)
            skips.append(x)
            hidden_kld = hidden_kld + kld

        for block in self.dec.values():
            x, kld = block(mp_cat(x, skips.pop(), t=0.5))
            hidden_kld = hidden_kld + kld

        x = self.conv_out(x, gain=self.out_gain)

        return x, hidden_kld
    
class DAE_J1(DualDiffusionDAE):

    def __init__(self, config: DAE_J1_Config) -> None:
        super().__init__()
        self.config = config

        block_kwargs = {"mlp_multiplier": config.mlp_multiplier,
                        "mlp_groups": config.mlp_groups,
                        "res_balance": config.res_balance}
        
        assert len(config.channel_mult_enc) + 1 == len(config.channel_mult_dec)
        enc_channels = [config.model_channels * m for m in config.channel_mult_enc]
        dec_channels = [config.model_channels * m for m in config.channel_mult_dec]

        cemb = config.model_channels * config.channel_mult_emb if config.in_channels_emb > 0 else 0

        self.num_levels = len(config.channel_mult_dec)
        self.downsample_ratio = 2 ** (self.num_levels - 1)
        self.latents_out_gain = torch.nn.Parameter(torch.ones([]))
        self.out_gain = torch.nn.Parameter(torch.ones([]))
        self.recon_loss_logvar = torch.nn.Parameter(torch.zeros([]))

        assert cemb % (config.in_num_freqs // self.downsample_ratio) == 0 and cemb > 0

        # embedding
        self.emb_label = MPConv2D_E(config.in_channels_emb, cemb, kernel=())
        self.emb_gain = torch.nn.Parameter(torch.zeros([]))
        self.emb_dim = cemb

        # encoders
        self.wavelet_encoders = torch.nn.ModuleList()
        
        for i, channels in enumerate(enc_channels):
            in_channels = config.in_channels * 3 + 1 if i < len(enc_channels) - 1 else config.in_channels * 4 + 1
            wavelet_encoder = WaveletEncoder(in_channels, channels, config.num_enc_layers_per_block, block_kwargs, kernel=(3,3))
            self.wavelet_encoders.append(wavelet_encoder)

        total_feature_channels = sum(enc_channels)
        fuser_channels = config.model_channels * config.channel_mult_fuser
        self.fuser = Block(self.num_levels - 1, total_feature_channels,
                fuser_channels, flavor="enc", kernel=(1,1), **block_kwargs)
        
        self.conv_latents_out = MPConv2D_E(fuser_channels, config.latent_channels, kernel=(1,1))

        latents_num_freqs = config.in_num_freqs // self.downsample_ratio
        conditioning_latent_channels = cemb // latents_num_freqs
        self.conv_latents_in = MPConv2D_E(config.latent_channels + 1, conditioning_latent_channels, kernel=(1,1))

        # decoder
        self.dec = torch.nn.ModuleDict()
        cin = conditioning_latent_channels

        for level in reversed(range(0, self.num_levels)):
            
            cout = dec_channels[level]

            if level == self.num_levels - 1:
                self.dec[f"block{level}_in0"] = Block(level, cin,  cout, flavor="dec", **block_kwargs)
                self.dec[f"block{level}_in1"] = Block(level, cout, cout, flavor="dec", **block_kwargs)
            else:
                self.dec[f"block{level}_up"] = Block(level, cin, cout, flavor="dec", resample_mode="up", **block_kwargs)
            
            for idx in range(config.num_dec_layers_per_block):
                self.dec[f"block{level}_layer{idx}"] = Block(level, cout, cout, flavor="dec", **block_kwargs)

            cin = cout

        self.conv_out = MPConv2D_E(cout, self.config.out_channels, kernel=(3,3))

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
        
        hidden_kld = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        wavelets = wavelet_decompose2d(x, num_levels=self.num_levels)
        features = []; downsample = self.downsample_ratio // 2

        for i in range(len(wavelets) - 1):
            wx = wavelet_space_to_channel2d(wavelets[i])

            if i == len(wavelets) - 2:
                wx = torch.cat((wx, wavelets[-1]), dim=1)

            wx, kld = self.wavelet_encoders[i](wx)
            hidden_kld = hidden_kld + kld

            if downsample > 1:
                wx = torch.nn.functional.avg_pool2d(wx, downsample)
                downsample //= 2
            
            features.append(wx)

        wx_fused, kld = self.fuser(torch.cat(features, dim=1))
        hidden_kld = hidden_kld + kld

        latents = self.conv_latents_out(wx_fused, gain=self.latents_out_gain)

        if training == False:
            return latents
        else:
            return latents, hidden_kld

    def decode(self, x: torch.Tensor, embeddings: torch.Tensor, training: bool = False) -> torch.Tensor:
        
        x = self.conv_latents_in(torch.cat((x, torch.ones_like(x[:, :1])), dim=1))
        b, c, h, w = x.shape
        x = x.reshape(b, c * h, w) * (embeddings.unsqueeze(-1) * self.emb_gain + 1)
        x = x.reshape(b, c,  h, w)

        x_mean = x.mean(dim=(1,2,3))
        x_var = x.var(dim=(1,2,3)).clip(min=1e-2)
        hidden_kld = x_mean.square() + x_var - 1 - x_var.log()
        
        for block in self.dec.values():
            x, kld = block(x)
            hidden_kld = hidden_kld + kld

        decoded = self.conv_out(x, gain=self.out_gain)

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