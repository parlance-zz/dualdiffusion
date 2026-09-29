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

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import numpy as np

#from modules.formats.frequency_scale import get_mel_density
from modules.mp_tools import patchify_2d, normalize
from training.loss.mss_2d import _is_prime


def sketch_2d(x: torch.Tensor, max_sketches: Optional[int] = None,
        normalize: bool = True, generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply random Gaussian CxC channel-mixing matrix to an image-like tensor.

    Args:
        x: Tensor of shape (B, C, H, W)
        normalize: If True, scales the random matrix by 1/sqrt(C)
        generator: Optional torch.Generator for reproducibility

    Returns:
        y: transformed tensor, shape: (B, C, H, W)
    """
    if x.ndim != 4:
        raise ValueError("x must have shape (B, C, H, W)")

    B, C, H, W = x.shape
    device = x.device
    dtype = x.dtype

    n_sketches = min(C, max_sketches) if max_sketches is not None else C
    G = torch.randn(n_sketches, C, device=device, dtype=dtype, generator=generator)
    if normalize:
        G = G / C ** 0.5

    # mix channels: for each pixel, new_channel_values = G @ old_channel_values
    return torch.einsum("ij,bjhw->bihw", G, x)

@dataclass
class LatentsReg2DConfig:

    block_low:  int = 11
    block_high: int = 511

    block_sampling_replace: bool = True
    block_sampling_scale: Literal["linear", "ln_linear", "natural"] = "natural"

    num_iterations: int = 1
    psd_eps: float = 1e-4
    loss_scale: float = 3

    sdpa_scale: float = 1

    #sample_rate: float = 32000
    max_sketches: Optional[int] = None

    disable_window_caching: bool = True
    increase_cuda_fft_plan_cache: bool = False

class LatentsReg2D:

    @torch.no_grad()
    def __init__(self, config: LatentsReg2DConfig, device: torch.device) -> None:

        self.config = config
        self.device = device

        primes = [i for i in range(self.config.block_low, self.config.block_high+1) if _is_prime(i)]

        n = 25000

        if self.config.block_sampling_scale == "ln_linear":
            targets = np.exp(np.linspace(np.log(self.config.block_low), np.log(self.config.block_high), n))
        elif self.config.block_sampling_scale == "linear":
            targets = np.linspace(self.config.block_low, self.config.block_high, n)
        else:
            if self.config.block_sampling_scale != "natural":
                raise ValueError(f"Invalid block_sampling_scale: {self.config.block_sampling_scale}")

        if self.config.block_sampling_scale == "natural":
            block_sizes = primes
            block_weights = [1.] * len(primes)
        else:
            spaced_primes = []
            for t in targets:
                closest = min(primes, key=lambda p: abs(p - t))
                spaced_primes.append(closest)

            block_sizes = []
            block_weights = []

            for b in sorted(set(spaced_primes)):
                count = spaced_primes.count(b)

                block_sizes.append(b)
                block_weights.append(float(count))

        self.block_sizes = np.array(block_sizes)
        self.block_weights = np.array(block_weights)
        self.block_weights /= self.block_weights.sum()

        for i in range(len(self.block_sizes)):
            print(f"Block size: {self.block_sizes[i]:3d} Weight: {(self.block_weights[i]*100):.3f}%")
        print(f"total unique block sizes: {len(block_sizes)}\n")

        if config.increase_cuda_fft_plan_cache == True:
            torch.backends.cuda.cufft_plan_cache.max_size = len(block_sizes)**2 * 2 + 250 # slight performance boost if fft plans are cached
            
        self.windows: dict[tuple[int, int], torch.Tensor] = {}
        self.loss_scale = config.loss_scale / self.config.num_iterations

    @torch.no_grad()
    def _flat_top_window(self, x: torch.Tensor) -> torch.Tensor:
        return (0.21557895 - 0.41663158 * torch.cos(x) + 0.277263158 * torch.cos(2*x)
                - 0.083578947 * torch.cos(3*x) + 0.006947368 * torch.cos(4*x))
    
    @torch.no_grad()
    def get_flat_top_window_2d(self, width: int, height: int, supersample: int = 9, supersample_threshold: int = 256) -> torch.Tensor:

        if (width, height) in self.windows:
            return self.windows[width, height]

        supersample_x = 1 if width  >= supersample_threshold else supersample
        supersample_y = 1 if height >= supersample_threshold else supersample

        block_width  = width  * supersample_x
        block_height = height * supersample_y

        hx = self._flat_top_window((torch.arange(block_height, device=self.device) + 0.5) / block_height * 2 * torch.pi)
        wx = self._flat_top_window((torch.arange(block_width,  device=self.device) + 0.5) / block_width  * 2 * torch.pi)

        window = hx.view(1, 1,-1, 1) * wx.view(1, 1, 1,-1)
        if supersample_x > 1 or supersample_y > 1:
            supersample = (supersample_y, supersample_x)
            window = torch.nn.functional.avg_pool2d(window, kernel_size=supersample, stride=supersample)
        window /= window.square().mean().sqrt()

        if self.config.disable_window_caching == False:
            self.windows[width, height] = window
        
        return window

    def stft2d(self, x: torch.Tensor, block_width: int, block_height: int, order: tuple[int],
               step_w: int, step_h: int, window: torch.Tensor, offset_h: int, offset_w: int, end_offset_h: int, end_offset_w: int) -> torch.Tensor:
        
        x = x[:, :, offset_h:end_offset_h, offset_w:end_offset_w]
        x = x.unfold(2, block_height, step_h).unfold(3, block_width, step_w)

        x = torch.fft.rfft2(x * window, norm="ortho", dim=order)

        return x
    
    def spectral_reg_loss(self, latents: torch.Tensor) -> torch.Tensor:
        
        loss = torch.zeros(latents.shape[0], device=self.device)

        block_widths  = self.block_sizes[:np.flatnonzero(self.block_sizes <= latents.shape[3])[-1] + 1]
        block_heights = self.block_sizes[:np.flatnonzero(self.block_sizes <= latents.shape[2])[-1] + 1]
        block_width_weights  = self.block_weights[:len(block_widths)];  block_width_weights  = block_width_weights / block_width_weights.sum()
        block_height_weights = self.block_weights[:len(block_heights)]; block_height_weights = block_height_weights / block_height_weights.sum()

        static_pad_width  = int(block_widths[-1])
        static_pad_height = int(block_heights[-1])
        latents = torch.nn.functional.pad(latents, (static_pad_width, static_pad_width, static_pad_height, static_pad_height), mode="reflect")

        _latents = latents

        block_widths  = np.random.choice(block_widths, size=self.config.num_iterations,
            replace=self.config.block_sampling_replace, p=block_width_weights)
        block_heights = np.random.choice(block_heights, size=self.config.num_iterations,
            replace=self.config.block_sampling_replace, p=block_height_weights)

        for i in range(self.config.num_iterations):
            
            latents = sketch_2d(_latents, max_sketches=self.config.max_sketches)

            block_width  = int(block_widths[i])
            block_height = int(block_heights[i])

            step_w = block_width
            step_h = block_height
            window = self.get_flat_top_window_2d(block_width, block_height)

            offset_min_h = int(max(0, static_pad_height - block_height))
            offset_max_h = int(max(offset_min_h, static_pad_height))
            offset_h = int(np.random.randint(offset_min_h, offset_max_h + 1))
            end_offset_h = -(static_pad_height - block_height) or None

            offset_min_w = int(max(0, static_pad_width - block_width))
            offset_max_w = int(max(offset_min_w, static_pad_width))
            offset_w = int(np.random.randint(offset_min_w, offset_max_w + 1))
            end_offset_w = -(static_pad_width - block_width) or None
            
            order = (-1, -2) if np.random.randint(0, 2) == 0 else (-2, -1)

            latents_fft_abs = self.stft2d(latents, block_width, block_height, order,
                step_w, step_h, window, offset_h, offset_w, end_offset_h, end_offset_w).abs()

            mse_loss = None #tbd

            loss = loss + mse_loss

        return loss * self.loss_scale
    
    def latents_reg_loss(self, latents: torch.Tensor) -> torch.Tensor:
        
        z = patchify_2d(latents, latents.shape[2], 1) if latents.shape[2] > 1 else latents
        z = z.permute(0, 2, 3, 1).contiguous()

        with torch.no_grad():
            z_scaled = z * self.config.sdpa_scale
            z_hat = torch.nn.functional.scaled_dot_product_attention(z_scaled, z_scaled, z_scaled)
            z_hat = normalize(z_hat, dim=-1)

        return torch.nn.functional.mse_loss(z.float(), z_hat.float(), reduction="none").mean(dim=(1,2,3))