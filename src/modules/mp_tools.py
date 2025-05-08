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

from typing import Optional, Union, Literal
import math

import torch
import numpy as np

from utils.dual_diffusion_utils import TF32_Disabled


def normalize(x: torch.Tensor, dim: Optional[Union[tuple, list]] = None,
              eps: float = 1e-4) -> torch.Tensor:
    
    with TF32_Disabled():
        norm = torch.linalg.vector_norm(x, dim=dim or list(range(1, x.ndim)),
                                        keepdim=True, dtype=torch.float32)
        norm = torch.add(eps, norm, alpha=(norm.numel() / x.numel())**0.5)
        return (x.float() / norm).to(x.dtype)

def resample_1d(x: torch.Tensor, mode: Literal["keep", "down", "up"] = "keep") -> torch.Tensor:

    if mode == "keep":
        return x
    elif mode == 'down':
        return torch.lerp(x[..., ::2], x[..., 1::2], 0.5) # should be multiplied by 2**0.5 to be magnitude preserving,
    elif mode == 'up':
        return torch.repeat_interleave(x, 2, dim=-1)
    
def resample_2d(x: torch.Tensor, mode: Literal["keep", "down", "up"] = "keep",
                ratio: int = 2, filtering: str = "nearest") -> torch.Tensor:
    
    if mode == "keep":
        return x
    elif mode == 'down':
        return torch.nn.functional.avg_pool2d(x, ratio) # should be multiplied by 2 to be magnitude preserving,
    elif mode == 'up':                              
        return torch.nn.functional.interpolate(x, scale_factor=ratio, mode=filtering)

def resample_3d(x: torch.Tensor, mode: Literal["keep", "down", "up"] = "keep") -> torch.Tensor:

    if mode == "keep":
        return x
    elif mode == 'down':

        original_shape = x.shape
        return torch.nn.functional.avg_pool2d(
            x.reshape(x.shape[0]*x.shape[1], x.shape[2], x.shape[3], x.shape[4]), 2).view(
                original_shape[0], original_shape[1], original_shape[2], original_shape[3]//2, original_shape[4]//2)
    
    elif mode == 'up': # torch.nn.functional.interpolate doesn't work properly with 5d tensors
        return x.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)

# applies a brick-wall low-pass filter preserving only the lowest 1/downsample frequencies
def lowpass_2d(x: torch.Tensor, blur_width: float = 16, use_circular_filter: bool = True) -> torch.Tensor:
    
    b, c, h, w = x.shape
    x_dtype = x.dtype
    
    # add padding to reduce boundary artifacts
    pad_h, pad_w = h // 2, w // 2
    x_padded = torch.nn.functional.pad(x, (pad_w, pad_w, pad_h, pad_h), mode="reflect")
    
    # compute 2d real fft on padded input
    x_f = torch.fft.rfft2(x_padded.float(), norm="ortho")
    
    # build proper circular frequency mask with absolute cutoff
    with torch.no_grad():
        padded_h, padded_w = h + 2 * pad_h, w + 2 * pad_w
        freq_h = torch.fft.fftfreq(padded_h, device=x.device)
        freq_w = torch.fft.rfftfreq(padded_w, device=x.device)
        
        # create 2d coordinate grid
        freq_grid_h, freq_grid_w = torch.meshgrid(freq_h, freq_w, indexing="ij")
        
        # calculate absolute distance from center (dc component) in cycles per pixel
        if use_circular_filter == True:
            dist_from_center = torch.sqrt(freq_grid_h**2 + freq_grid_w**2)
        else:
            dist_from_center = torch.maximum(torch.abs(freq_grid_h), torch.abs(freq_grid_w))
        
        # create mask using absolute cutoff frequency (cycles per pixel)
        mask = (dist_from_center <= (1/blur_width)).unsqueeze(0).unsqueeze(0)
    
    # apply mask
    x_f_filtered = x_f * mask
    
    # inverse real fft
    x_filtered = torch.fft.irfft2(x_f_filtered, s=(padded_h, padded_w), norm="ortho")
    
    # crop to original size
    x_filtered = x_filtered[:, :, pad_h:pad_h+h, pad_w:pad_w+w]
    
    return x_filtered.to(dtype=x_dtype)

def midside_transform(x: torch.Tensor) -> torch.Tensor:
    return torch.stack((x[:, 0] + x[:, 1], x[:, 0] - x[:, 1]), dim=1) * 0.5**0.5

def wavelet_decompose2d(x: torch.Tensor, num_levels: int = 4) -> list[torch.Tensor]:
    
    wavelets = []
    for i in range(num_levels):
        if i == num_levels - 1:
            wavelets.append(x)
        else:
            x_down = resample_2d(x, mode="down")
            wavelets.append(x - resample_2d(x_down, mode="up"))
            x = x_down #* 2
    
    return wavelets

def wavelet_recompose2d(wavelets: list[torch.Tensor], filtering: str = "nearest") -> list[torch.Tensor]:

    x = [w for w in wavelets]
    y = x.pop()

    while len(x) > 0:
        y = resample_2d(y, "up", filtering=filtering) + x.pop()
    
    return y

def space_to_channel3d(x: torch.Tensor) -> torch.Tensor:
    B, C, Z, H, W = x.shape
    
    x = x.view(B, C, Z, H // 2, 2, W // 2, 2)
    x = x.permute(0, 1, 4, 6, 3, 5).reshape(B, C * 4, Z, H // 2, W // 2)
    
    return x

def channel_to_space3d(x: torch.Tensor) -> torch.Tensor:
    B, C4, Z, H_half, W_half = x.shape
    
    C = C4 // 4 
    x = x.view(B, C, 2, 2, Z, H_half, W_half)
    x = x.permute(0, 1, 4, 5, 2, 6, 3)
    x = x.reshape(B, C, Z, H_half * 2, W_half * 2)

    return x

def space_to_channel2d(x: torch.Tensor) -> torch.Tensor:
    B, C, H, W = x.shape
    
    x = x.view(B, C, H // 2, 2, W // 2, 2)
    x = x.permute(0, 1, 3, 5, 2, 4).reshape(B, C * 4, H // 2, W // 2)
    
    return x

def channel_to_space2d(x: torch.Tensor) -> torch.Tensor:
    B, C4, H_half, W_half = x.shape
    
    C = C4 // 4
    x = x.view(B, C, 2, 2, H_half, W_half)  
    x = x.permute(0, 1, 4, 2, 5, 3)  
    x = x.reshape(B, C, H_half * 2, W_half * 2) 
    
    return x
    
#----------------------------------------------------------------------------
# Magnitude-preserving SiLU (Equation 81).

def mp_silu(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.silu(x) / 0.596

#----------------------------------------------------------------------------
# Magnitude-preserving sum (Equation 88).

def mp_sum(a: torch.Tensor, b: torch.Tensor, t: Union[torch.Tensor, float] = 0.5) -> torch.Tensor:
    return a.lerp(b, t) / ((1 - t) ** 2 + t ** 2) ** 0.5

#----------------------------------------------------------------------------
# Magnitude-preserving concatenation (Equation 103).

def mp_cat(a: torch.Tensor, b: torch.Tensor,
           dim: int = 1, t: float = 0.5) -> torch.Tensor:
    Na = a.shape[dim]
    Nb = b.shape[dim]
    C = ((Na + Nb) / ((1 - t) ** 2 + t ** 2)) ** 0.5
    wa = C / Na**0.5 * (1 - t)
    wb = C / Nb**0.5 * t
    return torch.cat([wa * a , wb * b], dim=dim)

def mp_cat_interleave(a: torch.Tensor, b: torch.Tensor,
                      dim: int = 1, t: float = 0.5) -> torch.Tensor:
    Na = a.shape[dim]
    Nb = b.shape[dim]
    C = ((Na + Nb) / ((1 - t) ** 2 + t ** 2)) ** 0.5
    wa = C / Na**0.5 * (1 - t)
    wb = C / Nb**0.5 * t
    return torch.stack([wa * a , wb * b], dim=dim+1).reshape(
        *a.shape[:dim], a.shape[dim]*2, *a.shape[dim+1:])

#----------------------------------------------------------------------------
# Magnitude-preserving Fourier features (Equation 75).

class MPFourier(torch.nn.Module):

    def __init__(self, num_channels: int, bandwidth: float = 1., eps: float = 1e-3) -> None:
        super().__init__()
        
        self.register_buffer('freqs', torch.pi * torch.linspace(0, 1-eps, num_channels).erfinv() * bandwidth)
        self.register_buffer('phases', torch.pi/2 * (torch.arange(num_channels) % 2 == 0).float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with TF32_Disabled():
            if x.ndim == 1:
                y = x.float().ger(self.freqs.float()) + self.phases.float()
            else:
                y = x.float() * self.freqs.float().view(1,-1, 1, 1) + self.phases.float().view(1,-1, 1, 1)
            return (y.cos() * 2**0.5).to(x.dtype)

class MPConv(torch.nn.Module):

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

    def forward(self, x: torch.Tensor, gain: Union[float, torch.Tensor] = 1.) -> torch.Tensor:
        
        w = self.weight.float()
        if self.training == True and self.disable_weight_norm == False:
            w = normalize(w) # traditional weight normalization
            
        w = w * (gain / w[0].numel()**0.5) # magnitude-preserving scaling
        w = w.to(x.dtype)

        if w.ndim == 2:
            return x @ w.t()
        
        return torch.nn.functional.conv2d(x, w, padding=(w.shape[-2]//2, w.shape[-1]//2), groups=self.groups, stride=self.stride)

    @torch.no_grad()
    def normalize_weights(self) -> None:
        if self.disable_weight_norm == False:
            self.weight.copy_(normalize(self.weight))

class MPConv3D(torch.nn.Module):

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

    def forward(self, x: torch.Tensor, gain: Union[float, torch.Tensor] = 1.) -> torch.Tensor:
        
        w = self.weight.float()
        if self.training == True and self.disable_weight_norm == False:
            w = normalize(w) # traditional weight normalization
            
        w = w * (gain / w[0].numel()**0.5) # magnitude-preserving scaling
        w = w.to(x.dtype)

        if w.ndim == 2:
            return x @ w.t()
        
        if w.ndim == 5:
            if w.shape[-3] == 2:
                x = torch.cat((x, x[:, :, 0:1]), dim=2)
            elif w.shape[-3] == 3:
                return torch.nn.functional.conv3d(x, w, padding=(w.shape[-3]//2, w.shape[-2]//2, w.shape[-1]//2), groups=self.groups, stride=self.stride)
            return torch.nn.functional.conv3d(x, w, padding=(0, w.shape[-2]//2, w.shape[-1]//2), groups=self.groups, stride=self.stride)
        else:
            return torch.nn.functional.conv2d(x, w, padding=(w.shape[-2]//2, w.shape[-1]//2), groups=self.groups, stride=self.stride)

    @torch.no_grad()
    def normalize_weights(self) -> None:
        if self.disable_weight_norm == False:
            self.weight.copy_(normalize(self.weight))

class FilteredDownsample2D(torch.nn.Module):

    def __init__(self, channels: int, stride: int = 8, lanczos_a: int = 3, use_3d_shape: bool = False) -> None:
        super(FilteredDownsample2D, self).__init__()

        self.use_3d_shape = use_3d_shape
        self.stride = stride

        k = self.downsampling_lanczos_kernel(downsample_factor=stride, a=lanczos_a)
        #kernel = 31
        #k = np.array(self._pascal_row(kernel - 1))
        
        filter = torch.Tensor(k[:, None] * k[None, :]).to(torch.float64)
        filter = (filter / filter.sum()).float()

        kernel = filter.shape[0]
        #padding_1, padding_2 = kernel//2, kernel//2 - (kernel//2+1)%2

        """
        kernel = 29
        filter = torch.zeros((1, 1, kernel, kernel))
        filter[:, :, kernel//2, kernel//2] = 1
        block_kernel = torch.ones((1, 1, 3, 3)) / 9
        for i in range(14):
            filter = torch.nn.functional.conv2d(filter, block_kernel, padding=1)
        filter = (filter / filter.sum()).view(kernel, kernel)
        """

        """
        kernel = 16
        x = torch.arange(16).float()
        filter = torch.max((7 - x.view(-1, 1)).abs(), (7 - x.view(1, -1)).abs())
        filter -= filter.amin()
        filter /= filter.amax()
        filter = 1 - filter
        filter /= filter.sum()
        """

        padding_1, padding_2 = kernel//2 - stride//2, kernel//2 - (kernel//2+1)%2 + stride//2
        self.pad = torch.nn.ReflectionPad2d([padding_1, padding_2, padding_1, padding_2])

        if use_3d_shape == True:
            self.register_buffer("filter", filter[None, None, None, :, :].expand((channels, 1, 1, kernel, kernel)), persistent=False)
        else:
            self.register_buffer("filter", filter[None, None, :, :].expand((channels, 1, kernel, kernel)), persistent=False)

    def _pascal_row(self, n: int) -> list[int]:
        return [math.comb(n, k) for k in range(n + 1)]

    def downsampling_lanczos_kernel(self, downsample_factor=8, a=3):
        """
        Create a low-pass Lanczos filter kernel for downsampling by `downsample_factor`.
        The cutoff is 1/(2 * downsample_factor), and the kernel is 2*a*downsample_factor + 1 in size.
        """

        cutoff = 0.5 / downsample_factor  # Nyquist after downsampling
        size = 2 * a * downsample_factor + 1  # ensure adequate sidelobe suppression
        center = size // 2
        x = np.arange(size) - center #+ ((size+1)%2)/2 # symmetric positions

        sinc_func = 2 * cutoff * np.sinc(2 * cutoff * x)  # ideal LPF
        lanczos_window = np.sinc(x / (a * downsample_factor))  # Lanczos window

        kernel = sinc_func * lanczos_window
        #kernel /= np.sum(kernel)  # Normalize to preserve DC gain

        return kernel
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_3d_shape == True:
            return torch.nn.functional.conv2d(self.pad(x), self.filter.squeeze(1), stride=self.stride, groups=x.shape[1])
        else:
            return torch.nn.functional.conv2d(self.pad(x), self.filter, stride=self.stride, groups=x.shape[1])