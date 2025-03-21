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

import torch

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
        return torch.lerp(x[..., ::2], x[..., 1::2], 0.5)
    elif mode == 'up':
        return torch.repeat_interleave(x, 2, dim=-1)
    
def resample_2d(x: torch.Tensor, mode: Literal["keep", "down", "up"] = "keep") -> torch.Tensor:
    if mode == "keep":
        return x
    elif mode == 'down':
        return torch.nn.functional.avg_pool2d(x, 2) # should be multiplied by 2 to be magnitude preserving,
    elif mode == 'up':                              # however, pixel norm is applied after downsampling so it doesn't matter
        return torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")

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

def midside_transform(x: torch.Tensor) -> torch.Tensor:
    return torch.stack((x[:, 0] + x[:, 1], x[:, 0] - x[:, 1]), dim=1) * 0.5**0.5

def wavelet_decompose2d(x: torch.Tensor, num_levels: int = 4) -> list[torch.Tensor]:
    
    wavelets = []
    for i in range(num_levels):
        x_down = resample_2d(x, mode="down")
        if i == num_levels - 1:
            wavelets.append(x)
        else:
            wavelets.append(x - resample_2d(x_down, mode="up"))
            x = x_down
    
    return wavelets

def wavelet_recompose2d(wavelets: list[torch.Tensor]) -> list[torch.Tensor]:

    x = [w for w in wavelets]
    y = x.pop()

    while len(x) > 0:
        y = resample_2d(y, "up")
        y = y + x.pop()
    
    return y

def _space_to_channel2d(x: torch.Tensor) -> torch.Tensor:
    B, C, H, W = x.shape
    
    x = x.view(B, C, H // 2, 2, W // 2, 2)  # reshape to split H and W into 2x2 blocks
    x = x.permute(0, 1, 3, 5, 2, 4).reshape(B, C * 4, H // 2, W // 2)  # rearrange and reshape
    
    return x

def _channel_to_space2d(x: torch.Tensor) -> torch.Tensor:
    B, C4, H_half, W_half = x.shape
    
    C = C4 // 4
    x = x.view(B, C, 4, H_half, W_half)  # reshape to extract 4 channels per original channel
    x = x.permute(0, 1, 3, 4, 2)  # move the split dimension to the last
    x = x.reshape(B, C, H_half * 2, W_half * 2)  # merge back
    
    return x

def space_to_channel2d(x: Union[torch.Tensor, list[torch.Tensor]]) -> Union[torch.Tensor, list[torch.Tensor]]:
    if isinstance(x, list):
        return [_space_to_channel2d(y) for y in x]
    else:
        return _space_to_channel2d(x)
    
def channel_to_space2d(x: Union[torch.Tensor, list[torch.Tensor]]) -> Union[torch.Tensor, list[torch.Tensor]]:
    if isinstance(x, list):
        return [_channel_to_space2d(y) for y in x]
    else:
        return _channel_to_space2d(x)
    
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
                 disable_weight_norm: bool = False, norm_dim: int = 1) -> None:
        
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.stride = stride
        self.disable_weight_norm = disable_weight_norm
        self.norm_dim = norm_dim
        
        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels // groups, *kernel))

    def forward(self, x: torch.Tensor, gain: Union[float, torch.Tensor] = 1.) -> torch.Tensor:
        
        w = self.weight.float()
        if self.training == True and self.disable_weight_norm == False:
            w = normalize(w, dim=self.norm_dim) # traditional weight normalization
            
        w = w * (gain / w[0].numel()**0.5) # magnitude-preserving scaling
        w = w.to(x.dtype)

        if w.ndim == 2:
            return x @ w.t()
        
        if w.ndim == 5:
            if w.shape[-3] == 2:
                x = torch.cat((x, x[:, :, 0:1]), dim=2)
            return torch.nn.functional.conv3d(x, w, padding=(0, w.shape[-2]//2, w.shape[-1]//2), groups=self.groups, stride=self.stride)
        else:
            return torch.nn.functional.conv2d(x, w, padding=(w.shape[-2]//2, w.shape[-1]//2), groups=self.groups, stride=self.stride)

    @torch.no_grad()
    def normalize_weights(self) -> None:
        if self.disable_weight_norm == False:
            self.weight.copy_(normalize(self.weight, dim=self.norm_dim))
