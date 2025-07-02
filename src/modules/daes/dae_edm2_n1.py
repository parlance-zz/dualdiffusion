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
from modules.mp_tools import mp_silu, normalize, mp_sum, resample_2d
from utils.resample import FilteredDownsample2D, FilteredUpsample2D


"""
def randn_like_hp(x: torch.Tensor) -> torch.Tensor:

    noise = torch.randn_like(x, dtype=torch.float32)
    rfft = torch.fft.rfft2(noise, norm="ortho")

    freq_h = torch.fft.fftfreq(x.shape[-2]).view(-1, 1)
    freq_w = torch.fft.rfftfreq(x.shape[-1]).view(1,-1)
    rfft *= ((freq_h < 0) * (freq_w >= 0.25)).float()
    rfft[..., rfft.shape[-2]//2:, :rfft.shape[-1]//2] = 0

    noise = torch.fft.irfft2(rfft, s=x.shape[-2:], norm="ortho")
    return noise.to(dtype=x.dtype)
"""
#"""
@torch.no_grad()
def randn_like_hp(x: torch.Tensor) -> torch.Tensor:

    b, c, h, w = x.shape
    device = x.device
    dtype = x.dtype

    # Create complex spectrum with normal real and imag parts
    noise_fft = torch.randn(b, c, h, w//2 + 1, 2, device=device, dtype=torch.float32)
    noise_fft = torch.view_as_complex(noise_fft)

    # Create highpass mask: True where both f_y and f_x >= 0.5 * Nyquist
    f_y = torch.fft.fftfreq(h, d=1, device=device)  # shape: (h,)
    f_x = torch.fft.rfftfreq(w, d=1, device=device) # shape: (w//2 + 1,)

    fy_mask = (f_y.abs() >= 0.25)  # half the Nyquist is 0.25 (Nyquist is 0.5)
    fx_mask = (f_x.abs() >= 0.25)

    highpass_mask = fy_mask[:, None] & fx_mask[None, :]  # shape: (h, w//2+1)

    # Broadcast mask to match noise_fft shape
    noise_fft *= highpass_mask[None, None, :, :]

    # Convert back to spatial domain
    noise = torch.fft.irfftn(noise_fft, s=(h, w), dim=(-2, -1), norm="ortho") * 2**0.5

    return noise.to(dtype=dtype).requires_grad_(False)

"""
from utils.dual_diffusion_utils import tensor_to_img, save_img

a = torch.randn( (1, 3, 256, 256))
noise = randn_like_hp(a)
print(a.std(), noise.std())

save_img(tensor_to_img(noise), "./noise.png")

noise_fft = torch.fft.rfft2(noise, norm="ortho")
save_img(tensor_to_img(noise_fft.abs()), "./noise_fft.png")
"""

@dataclass
class DAE_N1_Config(DualDiffusionDAEConfig):

    in_channels: int      = 2
    out_channels: int     = 2
    in_channels_emb: int  = 0
    
    in_num_freqs: int    = 256
    latent_channels: int = 4
    
    resample_beta: float = 3.437
    resample_k_size: int = 23
    use_filtered_resample: bool = True

    num_levels: int       = 3
    input_sigma: float    = 0.05

    model_channels: int   = 64
    channel_mult_enc: int = 1
    channel_mult_dec: int = 1
    channel_mult_emb: int = 1
    num_enc_layers_per_block: int = 4
    num_dec_layers_per_block: int = 4
    kernel_enc: list[int] = (5,5)
    kernel_dec: list[int] = (5,5)
    mlp_multiplier: int = 2
    mlp_groups: int     = 1

    res_balance: float  = 0.5

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

class Block2D(torch.nn.Module):

    def __init__(self,
        in_channels: int,                       # Number of input channels.
        out_channels: int,                      # Number of output channels.
        emb_channels: int,                      # Number of embedding channels.
        flavor: Literal["enc", "dec"] = "enc",
        res_balance: float     = 0.3,
        clip_act: float        = 256,      # Clip output activations. None = do not clip.
        mlp_multiplier: int    = 1,        # Multiplier for the number of channels in the MLP.
        mlp_groups: int        = 1,        # Number of groups for the MLP.
        kernel: tuple[int, int] = (3,3),   # Kernel size for the convolutional layers.
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.flavor = flavor
        self.res_balance = res_balance
        self.clip_act = clip_act
            
        self.conv_res0 = MPConv2D(in_channels,  out_channels * mlp_multiplier, kernel=kernel, groups=mlp_groups)
        self.conv_res1 = MPConv2D(out_channels * mlp_multiplier, out_channels, kernel=kernel, groups=mlp_groups)
        
        if in_channels != out_channels or mlp_groups > 1:
            self.conv_skip = MPConv2D(in_channels, out_channels, kernel=(1,1), groups=1)
        else:
            self.conv_skip = None

        self.emb_gain = torch.nn.Parameter(torch.zeros([])) if emb_channels != 0 else None
        self.emb_linear = MPConv2D(emb_channels, out_channels * mlp_multiplier,
            kernel=(1,1), groups=1) if emb_channels != 0 else None
    
    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        
        #if self.flavor == "enc":
        #    x = normalize(x, dim=1) # pixel norm

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

class DiffusionCodec(torch.nn.Module):

    def __init__(self,
        in_channels: int,                       # Number of input channels.
        out_channels: int,                      # Number of output channels.
        latents_channels: int,
        emb_channels: int,                      # Number of embedding channels.
        enc_channels: int,
        dec_channels: int,
        num_enc_layers_per_block: int,
        num_dec_layers_per_block: int,
        downsample: torch.nn.Module,
        upsample: torch.nn.Module,
        res_balance: float     = 0.3,
        mlp_multiplier: int    = 1,        # Multiplier for the number of channels in the MLP.
        mlp_groups: int        = 1,        # Number of groups for the MLP.
        kernel_enc: tuple[int, int] = (3,3),   # Kernel size for the convolutional layers.
        kernel_dec: tuple[int, int] = (3,3),   # Kernel size for the convolutional layers.
    ) -> None:
        super().__init__()

        self.downsample = downsample
        self.upsample = upsample

        self.conv_in = MPConv2D(in_channels + 1, enc_channels, kernel=kernel_enc)

        self.enc = torch.nn.ModuleList()
        for _ in range(num_enc_layers_per_block):
            self.enc.append(Block2D(enc_channels, enc_channels, emb_channels, flavor="enc", kernel=kernel_enc, 
                res_balance=res_balance, mlp_multiplier=mlp_multiplier, mlp_groups=mlp_groups))
        
        self.conv_latents_out = MPConv2D(enc_channels, latents_channels, kernel=kernel_enc)
        self.conv_latents_out_gain = torch.nn.Parameter(torch.ones([]))
        self.conv_latents_in = MPConv2D(latents_channels + 1, dec_channels, kernel=kernel_dec)

        self.dec = torch.nn.ModuleList()
        for _ in range(num_dec_layers_per_block):
            self.dec.append(Block2D(dec_channels, dec_channels, emb_channels, flavor="dec", kernel=kernel_dec,
                res_balance=res_balance, mlp_multiplier=mlp_multiplier, mlp_groups=mlp_groups))
        
        self.conv_out = MPConv2D(dec_channels, out_channels, kernel=kernel_dec)
        self.conv_out_gain = torch.nn.Parameter(torch.ones([]))
    
    def encode(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        
        if emb is not None:
            raise NotImplementedError()

        x = self.conv_in(torch.cat((x, torch.ones_like(x[:, :1])), dim=1))

        for block in self.enc:
            x = block(x, emb)

        latents = self.conv_latents_out(x, gain=self.conv_latents_out_gain)
        latents = self.downsample(latents)
        latents = self.downsample(latents)

        return latents

    def decode(self, x: torch.Tensor, emb: torch.Tensor, sigma: Optional[torch.Tensor] = None) -> torch.Tensor:

        if emb is not None:
            raise NotImplementedError()
        
        x = self.upsample(x)
        if sigma is not None:
            #x = (x + torch.randn_like(x) * sigma) / (1 + sigma**2)**0.5
            x = (x + randn_like_hp(x) * sigma) / (1 + sigma**2)**0.5
            
        x = self.upsample(x)
        if sigma is not None:
            #x = (x + torch.randn_like(x) * sigma) / (1 + sigma**2)**0.5
            x = (x + randn_like_hp(x) * sigma) / (1 + sigma**2)**0.5
            
        x = self.conv_latents_in(torch.cat((x, torch.ones_like(x[:, :1])), dim=1))

        for block in self.dec:
            x = block(x, emb)

        x = self.conv_out(x, gain=self.conv_out_gain)
        return x

class DAE_N1(DualDiffusionDAE):

    def __init__(self, config: DAE_N1_Config) -> None:
        super().__init__()
        self.config = config
        
        enc_channels = config.model_channels * config.channel_mult_enc
        dec_channels = config.model_channels * config.channel_mult_dec

        cemb = config.model_channels * config.channel_mult_emb if config.in_channels_emb > 0 else 0

        self.num_levels = config.num_levels
        self.total_downsample_ratio = 2 ** self.num_levels
        self.recon_loss_logvar = torch.nn.Parameter(torch.zeros([]))

        # embedding
        if cemb > 0:
            raise NotImplementedError()
            self.emb_label = MPConv2D(config.in_channels_emb, cemb, kernel=())
        else:
            self.emb_label = None

        if config.use_filtered_resample == True:
            self.downsample = FilteredDownsample2D(k_size=config.resample_k_size, beta=config.resample_beta, factor=2)
            self.upsample = FilteredUpsample2D(k_size=config.resample_k_size * 2 + config.resample_k_size % 2, beta=config.resample_beta, factor=2)
        else:
            self.downsample = lambda x: resample_2d(x, mode="down")
            self.upsample = lambda x: resample_2d(x, mode="up")

        # encoder
        self.codecs = torch.nn.ModuleList()

        for i in range(config.num_levels):
            codec = DiffusionCodec(
                in_channels=config.in_channels if i == 0 else config.latent_channels,
                out_channels=config.out_channels if i == 0 else config.latent_channels,
                latents_channels=config.latent_channels,
                emb_channels=cemb,
                enc_channels=enc_channels,
                dec_channels=dec_channels,
                num_enc_layers_per_block=config.num_enc_layers_per_block,
                num_dec_layers_per_block=config.num_dec_layers_per_block,
                downsample=self.downsample,
                upsample=self.upsample,
                res_balance=config.res_balance,
                mlp_multiplier=config.mlp_multiplier,
                mlp_groups=config.mlp_groups,
                kernel_enc=config.kernel_enc,
                kernel_dec=config.kernel_dec
            )
            self.codecs.append(codec)

    def get_embeddings(self, emb_in: torch.Tensor) -> torch.Tensor:
        return None
        #if self.emb_label is not None:
        #    return self.emb_label(normalize(emb_in).to(device=self.device, dtype=self.dtype))
        #else:
        #    return None
    
    def get_recon_loss_logvar(self) -> torch.Tensor:
        return self.recon_loss_logvar
    
    def get_latent_shape(self, mel_spec_shape: Union[torch.Size, tuple[int, int, int, int]]) -> torch.Size:
        if len(mel_spec_shape) == 4:
            return (mel_spec_shape[0], self.config.latent_channels,
                    mel_spec_shape[2] // 2 ** self.num_levels,
                    mel_spec_shape[3] // 2 ** self.num_levels)
        else:
            raise ValueError(f"Invalid sample shape: {mel_spec_shape}")
        
    def get_mel_spec_shape(self, latent_shape: Union[torch.Size, tuple[int, int, int, int]]) -> torch.Size:
        if len(latent_shape) == 4:
            return (latent_shape[0], 2,
                    latent_shape[2] * 2 ** self.num_levels,
                    latent_shape[3] * 2 ** self.num_levels)
        else:
            raise ValueError(f"Invalid latent shape: {latent_shape}")
        
    def encode(self, x: torch.Tensor, embeddings: torch.Tensor, level: Optional[int] = None) -> torch.Tensor:
        
        levels = range(self.num_levels) if level is None else range(level + 1)
        
        for level in levels:
            codec: DiffusionCodec = self.codecs[level]
            x = codec.encode(x, embeddings)

        return x

    def decode(self, x: torch.Tensor, embeddings: torch.Tensor, level: Optional[int] = None, sigma: float = 0) -> torch.Tensor:
        
        levels = range(self.num_levels) if level is None else range(level + 1)
        sigma = torch.Tensor([sigma]).to(device=x.device, dtype=x.dtype) if sigma > 0 else None

        for level in reversed(levels):
            codec: DiffusionCodec = self.codecs[level]
            x = codec.decode(x, embeddings, sigma)
        
        return x

    def forward(self, samples: torch.Tensor, dae_embeddings: torch.Tensor,
            sigma: Optional[torch.Tensor] = None, level: int = 0) -> tuple[torch.Tensor, ...]:
        
        target = samples.detach().to(dtype=torch.bfloat16).requires_grad_(False)
        for _level in range(level):
            codec: DiffusionCodec = self.codecs[_level]
            target = codec.encode(target, dae_embeddings)
        target = target.detach().to(dtype=torch.bfloat16).requires_grad_(False)

        codec: DiffusionCodec = self.codecs[level]
        latents = codec.encode(target, dae_embeddings)
        decoded = codec.decode(latents, dae_embeddings, sigma)

        latents_mean = latents.mean(dim=(1,2,3))
        latents_var = latents.var(dim=(1,2,3))
        latents_kld = latents_mean.square() + latents_var - 1 - latents_var.log()
        
        return latents, decoded, target, latents_kld

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
