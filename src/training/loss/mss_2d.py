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


def _is_prime(n: int) -> bool:
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

@dataclass
class MSSLoss2DConfig:

    block_low:  int = 5
    block_high: int = 254

    block_sampling_replace: bool = True
    block_sampling_scale: Literal["linear", "ln_linear"] = "ln_linear"

    num_iterations: int = 1
    midside_probability: float = 0.5
    psd_eps: float = 1e-4
    loss_scale: float = 3

class MSSLoss2D:

    @torch.no_grad()
    def __init__(self, config: MSSLoss2DConfig, device: torch.device) -> None:

        self.config = config
        self.device = device

        primes = [i for i in range(self.config.block_low, self.config.block_high+1) if _is_prime(i)]

        n = 25000

        if self.config.block_sampling_scale == "ln_linear":
            targets = np.exp(np.linspace(np.log(self.config.block_low), np.log(self.config.block_high), n))
        elif self.config.block_sampling_scale == "linear":
            targets = np.linspace(self.config.block_low, self.config.block_high, n)
        else:
            raise ValueError(f"Invalid block_sampling_scale: {self.config.block_sampling_scale}")

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

        self.windows[width, height] = window
        return window
    
    def stft2d(self, x: torch.Tensor, block_width: int, block_height: int, order: tuple[int],
               step_w: int, step_h: int, window: torch.Tensor, offset_h: int, offset_w: int, end_offset_h: int, end_offset_w: int, midside: bool) -> torch.Tensor:
        
        x = x[:, :, offset_h:end_offset_h, offset_w:end_offset_w]
        x = x.unfold(2, block_height, step_h).unfold(3, block_width, step_w)

        x = torch.fft.rfft2(x * window, norm="ortho", dim=order)
        if midside == True:
            x = torch.fft.fft(x, dim=1, norm="ortho")

        return x
    
    def mss_loss(self, sample: torch.Tensor, target: torch.Tensor,
            leak_pow: Optional[float] = None, leak_max: Optional[float] = None) -> torch.Tensor:

        if leak_pow is not None and leak_max is not None:  # useful at start of training for preventing polarity mismatch
            rnd_t = np.random.rand()**leak_pow * leak_max  # disable afterwards for better performance
            sample = torch.lerp(sample, target.detach(), rnd_t)
        
        loss = torch.zeros(target.shape[0], device=self.device)

        static_pad = int(self.block_sizes[-1])
        sample = torch.nn.functional.pad(sample, (static_pad, static_pad, static_pad, static_pad), mode="reflect")
        target = torch.nn.functional.pad(target, (static_pad, static_pad, static_pad, static_pad), mode="reflect")

        block_widths  = np.random.choice(self.block_sizes, size=self.config.num_iterations,
            replace=self.config.block_sampling_replace, p=self.block_weights)
        block_heights = np.random.choice(self.block_sizes, size=self.config.num_iterations,
            replace=self.config.block_sampling_replace, p=self.block_weights)

        for i in range(self.config.num_iterations):

            block_width = int(block_widths[i])
            block_height = int(block_heights[i])

            step_w = block_width
            step_h = block_height
            window = self.get_flat_top_window_2d(block_width, block_height)

            offset_min_h = int(max(0, static_pad - block_height))
            offset_max_h = int(max(offset_min_h, static_pad))
            offset_h = int(np.random.randint(offset_min_h, offset_max_h + 1))
            end_offset_h = -(static_pad - block_height) or None

            offset_min_w = int(max(0, static_pad - block_width))
            offset_max_w = int(max(offset_min_w, static_pad))
            offset_w = int(np.random.randint(offset_min_w, offset_max_w + 1))
            end_offset_w = -(static_pad - block_width) or None
            
            order = (-1, -2) if np.random.randint(0, 2) == 0 else (-2, -1)
            midside = np.random.rand() < self.config.midside_probability
            r_dims = (0, 2, 3) if midside == True else (0, 1, 2, 3)
            #r_dims = (0, 3) if midside == True else (0, 1, 3)

            with torch.no_grad():
                target_fft = self.stft2d(target, block_width, block_height, order,
                    step_w, step_h, window, offset_h, offset_w, end_offset_h, end_offset_w, midside)
                target_fft_abs = target_fft.abs().requires_grad_(False).detach()
                loss_weight = target_fft_abs.pow(2).mean(dim=r_dims, keepdim=True).clip(min=self.config.psd_eps).pow(0.5).requires_grad_(False).detach()

                """
                if order == (-2, -1):
                    blockfreq_y = torch.fft.fftfreq(block_height, 1/block_height, device=self.device)
                    blockfreq_x = torch.arange(block_width//2 + 1, device=self.device)
                else:
                    blockfreq_y = torch.arange(block_height//2 + 1, device=self.device)
                    blockfreq_x = torch.fft.fftfreq(block_width, 1/block_width, device=self.device)
                loss_weight = 1 / ((blockfreq_y.square().view(-1, 1) + blockfreq_x.square().view(1, -1)).sqrt() + 1)
                loss_weight = loss_weight[None, None, None, None, :, :] / 0.100311111
                if midside == True:
                    #loss_weight = loss_weight / (torch.arange(target_fft_abs.shape[1], device=self.device) + 1)[None, :, None, None, None, None]
                    loss_weight = loss_weight * target_fft_abs.pow(2).mean(dim=(0,2,3,4,5), keepdim=True).clip(min=self.config.psd_eps).pow(0.5) 
                """

            sample_fft = self.stft2d(sample, block_width, block_height, order,
                step_w, step_h, window, offset_h, offset_w, end_offset_h, end_offset_w, midside)

            sample_fft_abs = sample_fft.abs()
            
            mse_loss = torch.nn.functional.mse_loss(sample_fft_abs.float(), target_fft_abs.float(), reduction="none")
            loss = loss + (mse_loss / loss_weight).mean(dim=(1,2,3,4,5)) #** 2

        return loss * self.loss_scale


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = MSSLoss2DConfig()
    loss_fn = MSSLoss2D(config, device)

    batch_size = 4
    channels = 2
    height = 256
    width = 384

    sample = torch.randn(batch_size, channels, height, width, device=device)
    target = torch.randn(batch_size, channels, height, width, device=device)

    loss = loss_fn.mss_loss(sample, target)
    print("Loss:", loss)

    from utils.dual_diffusion_utils import tensor_to_img, save_img
    from utils import config
    import os

    output_path = os.path.join(config.DEBUG_PATH, "mss_2d_test")

    for blk_sz in loss_fn.block_sizes:
        window = loss_fn.get_flat_top_window_2d(blk_sz, blk_sz)
        save_img(tensor_to_img(window), os.path.join(output_path, f"wndw_{blk_sz}.png"))
        window.cpu().numpy().tofile(os.path.join(output_path, f"wndw_{blk_sz}.raw"))