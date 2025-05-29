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

from modules.daes.dae import DualDiffusionDAE, DualDiffusionDAEConfig
from modules.mp_tools import mp_silu, normalize, resample_3d, random_crop_2d, mp_sum
from utils.resample import FilteredDownsample2D, FilteredUpsample2D
from utils.dual_diffusion_utils import tensor_4d_to_5d, tensor_5d_to_4d


@dataclass
class DAE_J5_Config(DualDiffusionDAEConfig):

    in_channels: int     = 1
    out_channels: int    = 1
    in_channels_emb: int = 0
    in_num_freqs: int    = 256
    latent_channels: int = 4
    downsample_factor: int = 1
    res_balance: float   = 0.3

    model_channels: int   = 32
    channel_mult_emb: int = 4
    channel_mult_enc: list[int] = (1, 2, 4, 4)
    channel_mult_dec: list[int] = (1, 2, 4, 4)
    num_enc_layers_per_block: list[int] = (3, 3, 3, 3)
    num_dec_layers_per_block: list[int] = (3, 3, 3, 3)
    kernel_enc: list[int] = (2, 3, 3)
    kernel_dec: list[int] = (2, 3, 3)
    mlp_multiplier: int = 2
    mlp_groups: int     = 1

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
        emb_channels: int,                      # Number of embedding channels.
        flavor: Literal["enc", "dec"] = "enc",
        resample_mode: Literal["keep", "up", "down"] = "keep",
        res_balance: float     = 0.3,
        clip_act: float        = 256,      # Clip output activations. None = do not clip.
        mlp_multiplier: int    = 2,        # Multiplier for the number of channels in the MLP.
        mlp_groups: int        = 1,        # Number of groups for the MLP.
        kernel: tuple[int, int, int] = (2,3,3),   # Kernel size for the convolutional layers.
    ) -> None:
        super().__init__()

        self.level = level
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.flavor = flavor
        self.resample_mode = resample_mode
        self.res_balance = res_balance
        self.clip_act = clip_act
        self.kernel = kernel

        self.conv_res0 = MPConv3D_E(in_channels, out_channels * mlp_multiplier,
                                    kernel=kernel, groups=mlp_groups)
        self.conv_res1 = MPConv3D_E(out_channels * mlp_multiplier,
                        out_channels, kernel=kernel, groups=mlp_groups)
        
        if in_channels != out_channels or mlp_groups > 1:
            self.conv_skip = MPConv3D_E(in_channels, out_channels, kernel=(1,1,1), groups=1)
        else:
            self.conv_skip = None

        self.emb_gain = torch.nn.Parameter(torch.zeros([])) if emb_channels != 0 else None
        self.emb_linear = MPConv3D_E(emb_channels, out_channels * mlp_multiplier,
            kernel=(1,1,1), groups=1) if emb_channels != 0 else None
    
    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        
        x = resample_3d(x, mode=self.resample_mode)
        
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

        kl_dims = (1,2,3,4)
        x_mean = x.mean(dim=kl_dims)
        x_var = x.var(dim=kl_dims).clip(min=1e-2)
        kld = x_mean.square() + x_var - 1 - x_var.log()

        return x, kld

class Encoder(torch.nn.Module):

    def __init__(self, in_channels: int, enc_channels: list[int], latent_channels: int,
            num_layers: list[int], block_kwargs: dict, kernel: tuple[int, int] = (2,3,3)) -> None:
        super().__init__()

        self.input_gain = torch.nn.Parameter(torch.ones([]))
        self.input_shift = torch.nn.Parameter(torch.zeros([]))
        self.conv_in = MPConv3D_E(in_channels + 1, enc_channels[0], kernel=kernel)

        self.enc = torch.nn.ModuleDict()
        cout = enc_channels[0]
        for level, channels in enumerate(enc_channels):
            
            if level == 0:
                self.enc[f"block{level}_in"] = Block(
                    level, cout, channels, 0, flavor="enc", kernel=kernel, **block_kwargs)
            else:
                self.enc[f"block{level}_down"] = Block(
                    level, cout, channels, 0, flavor="enc", resample_mode="down", kernel=kernel, **block_kwargs)
            
            for idx in range(num_layers[level]):
                self.enc[f"block{level}_layer{idx}"] = Block(
                    level, channels, channels, 0, flavor="enc", kernel=kernel, **block_kwargs)
            
            cout = channels

        self.output_gain = torch.nn.Parameter(torch.ones([]))
        self.output_shift = torch.nn.Parameter(torch.zeros([]))
        self.conv_out = MPConv3D_E(enc_channels[-1], latent_channels, kernel=kernel)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        
        x = torch.cat((x, torch.ones_like(x[:, :1])), dim=1)
        x = self.conv_in(x, gain=self.input_gain) + self.input_shift

        hidden_kld = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

        for block in self.enc.values():
            x, kld = block(x, None)
            hidden_kld = hidden_kld + kld

        x = self.conv_out(x, gain=self.output_gain) + self.output_shift
        return x, hidden_kld
    
class DAE_J5(DualDiffusionDAE):

    supports_channels_last: Union[bool, Literal["3d"]] = "3d"

    def __init__(self, config: DAE_J5_Config) -> None:
        super().__init__()
        self.config = config

        block_kwargs = {"mlp_multiplier": config.mlp_multiplier,
                        "mlp_groups": config.mlp_groups,
                        "res_balance": config.res_balance}
        
        enc_channels = [config.model_channels * m for m in config.channel_mult_enc]
        dec_channels = [config.model_channels * m for m in config.channel_mult_dec]

        cemb = config.model_channels * config.channel_mult_emb if config.in_channels_emb > 0 else 0

        self.num_levels = len(config.channel_mult_dec)
        self.downsample_ratio = 2 ** (self.num_levels - 1)

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

        if isinstance(config.num_enc_layers_per_block, int):
            config.num_enc_layers_per_block = [config.num_enc_layers_per_block] * len(enc_channels)
        if isinstance(config.num_dec_layers_per_block, int):
            config.num_dec_layers_per_block = [config.num_dec_layers_per_block] * len(dec_channels)

        assert len(enc_channels) == len(config.num_enc_layers_per_block)
        assert len(dec_channels) == len(config.num_dec_layers_per_block)

        # encoder
        self.encoder = Encoder(config.in_channels, enc_channels, config.latent_channels,
            config.num_enc_layers_per_block, block_kwargs, kernel=config.kernel_enc)

        beta = 3.437; k_size = 23; factor = 2
        self.downsample = FilteredDownsample2D(k_size=k_size, beta=beta, factor=factor)
        self.upsample = FilteredUpsample2D(k_size=k_size*factor+k_size%factor, beta=beta, factor=factor)
        
        # decoder
        self.input_gain = torch.nn.Parameter(torch.ones([]))
        self.input_shift = torch.nn.Parameter(torch.zeros([]))
        self.latents_conv_in = MPConv3D_E(config.latent_channels + 1, dec_channels[-1], kernel=config.kernel_dec)

        self.dec = torch.nn.ModuleDict()
        cin = dec_channels[-1]

        for level in reversed(range(0, self.num_levels)):
            
            cout = dec_channels[level]

            if level == self.num_levels - 1:
                self.dec[f"block{level}_in"] = Block(level, cin,  cout, cemb, flavor="dec", **block_kwargs, kernel=config.kernel_dec)
            else:
                self.dec[f"block{level}_up"] = Block(level, cin, cout, cemb, flavor="dec", resample_mode="up", **block_kwargs, kernel=config.kernel_dec)
            
            for idx in range(config.num_dec_layers_per_block[level]):
                self.dec[f"block{level}_layer{idx}"] = Block(level, cout, cout, cemb, flavor="dec", **block_kwargs, kernel=config.kernel_dec)

            cin = cout

        self.output_gain = torch.nn.Parameter(torch.ones([]))
        self.output_shift = torch.nn.Parameter(torch.zeros([]))
        self.conv_out = MPConv3D_E(cout, self.config.out_channels, kernel=config.kernel_dec)

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
        latents = tensor_5d_to_4d(x)

        if self.config.downsample_factor > 1:
            latents = torch.nn.functional.avg_pool2d(
                latents, kernel_size=self.config.downsample_factor)

        if training == False:
            return latents
        else:
            return latents, hidden_kld

    def decode(self, x: torch.Tensor, embeddings: torch.Tensor, training: bool = False) -> torch.Tensor:
        
        if embeddings is not None:
            emb = embeddings[:, :, None, None, None]
        else:
            emb = None

        x = tensor_4d_to_5d(x, num_channels=self.config.latent_channels)
        x = torch.cat((x, torch.ones_like(x[:, :1])), dim=1)
        x = self.latents_conv_in(x, gain=self.input_gain) + self.input_shift

        hidden_kld = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        
        for block in self.dec.values():
            x, kld = block(x, emb)
            hidden_kld = hidden_kld + kld

        decoded = tensor_5d_to_4d(self.conv_out(x, gain=self.output_gain) + self.output_shift)

        if training == False:
            return decoded
        else:
            return decoded, hidden_kld
    
    def forward(self, samples: torch.Tensor, dae_embeddings: torch.Tensor,
            latents_sigma: torch.Tensor, equivariance_dropout: float = 0.5) -> tuple[torch.Tensor, ...]:
        
        latents, enc_hidden_kld = self.encode(samples, dae_embeddings, training=True)

        with torch.amp.autocast("cuda", enabled=False):
            samples = samples.float(); latents = latents.float()
            
            for i in range(3):
                latents = self.upsample(latents)
            #latents = torch.nn.functional.interpolate(latents, scale_factor=self.downsample_ratio, mode="nearest")
            #latents = torch.nn.functional.interpolate(latents, scale_factor=self.downsample_ratio, mode="bicubic")

            samples, latents = random_crop_2d(samples, latents,
                range_h=self.downsample_ratio, range_w=self.downsample_ratio, dropout=equivariance_dropout)
            
            for i in range(3):
                latents = self.downsample(latents)
            samples = samples.to(dtype=torch.bfloat16); latents = latents.to(dtype=torch.bfloat16)

        #latents = torch.nn.functional.avg_pool2d(latents, kernel_size=self.downsample_ratio)
        #latents = torch.nn.functional.interpolate(latents, scale_factor=1/self.downsample_ratio, mode="bicubic")

        #latents = (latents + torch.randn_like(latents) * latents_sigma) / (1 + latents_sigma**2)**0.5

        decoded, dec_hidden_kld = self.decode(latents, dae_embeddings, training=True)

        latents_mean = latents.mean(dim=(1,2,3))
        latents_var = latents.var(dim=(1,2,3)).clip(min=1e-2)
        latents_kld = latents_mean.square() + latents_var - 1 - latents_var.log()
    
        return latents, decoded, samples, latents_kld, enc_hidden_kld + dec_hidden_kld

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
        
        return latents