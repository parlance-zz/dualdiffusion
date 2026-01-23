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
from typing import Literal

import torch
import numpy as np

import utils.custom_operators.rfft2_abs


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

    block_low:  int = 9
    block_high: int = 254

    block_sampling_replace: bool = True
    block_sampling_scale: Literal["linear", "ln_linear"] = "ln_linear"

    num_iterations: int = 100
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

        self.loss_scale = config.loss_scale / self.config.num_iterations

    @torch.no_grad()
    def _flat_top_window(self, x: torch.Tensor) -> torch.Tensor:
        return (0.21557895 - 0.41663158 * torch.cos(x) + 0.277263158 * torch.cos(2*x)
                - 0.083578947 * torch.cos(3*x) + 0.006947368 * torch.cos(4*x))

    @torch.no_grad()
    def get_flat_top_window_2d(self, block_width: int, block_height: int) -> torch.Tensor:

        hx = (torch.arange(block_height, device=self.device) + 0.5) / block_height * 2 * torch.pi
        wx = (torch.arange(block_width, device=self.device)  + 0.5) / block_width  * 2 * torch.pi

        window = self._flat_top_window(hx).view(-1, 1) * self._flat_top_window(wx).view(1,-1)
        window /= torch.linalg.vector_norm(window) / (block_width * block_height)**0.5
        return window
    
    def stft2d_abs(self, x: torch.Tensor, block_width: int, block_height: int,
            step_w: int, step_h: int, window: torch.Tensor, offset_h: int, offset_w: int, midside: bool) -> torch.Tensor:
        
        padding_h = block_height
        padding_w = block_width
        x = torch.nn.functional.pad(x, (padding_w, padding_w, padding_h, padding_h), mode="reflect")
        x = x[:, :, offset_h:, offset_w:]

        if midside == True:
            x = torch.stack((x[:, 0] + x[:, 1], x[:, 0] - x[:, 1]), dim=1) / 2**0.5

        x = x.unfold(2, block_height, step_h).unfold(3, block_width, step_w)
        #x = torch.fft.rfft2(x * window, norm="ortho", dim=order)
        x = torch.ops.custom.rfft2_abs(x * window)

        return x
    
    def mss_loss(self, sample: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        block_widths  = np.random.choice(self.block_sizes, size=self.config.num_iterations, replace=self.config.block_sampling_replace, p=self.block_weights)
        block_heights = np.random.choice(self.block_sizes, size=self.config.num_iterations, replace=self.config.block_sampling_replace, p=self.block_weights)

        offsets_h = []; offsets_w = []
        for i in range(self.config.num_iterations):
            offsets_h.append(int(np.random.randint(0, block_heights[i])))
            offsets_w.append(int(np.random.randint(0, block_widths[i])))

        return self._mss_loss(sample, target, list(block_widths), list(block_heights), offsets_h, offsets_w)

    def _mss_loss(self, sample: torch.Tensor, target: torch.Tensor, block_widths: list[int], block_heights: list[int], offsets_h: list[int], offsets_w: list[int]) -> tuple[torch.Tensor, torch.Tensor]:

        loss = torch.zeros(target.shape[0], device=self.device)

        for i in range(self.config.num_iterations):

            block_width = block_widths[i]; block_height = block_heights[i]
            window = self.get_flat_top_window_2d(block_width, block_height)

            step_w = block_width; step_h = block_height
            offset_h = offsets_h[i]; offset_w = offsets_w[i]

            midside = bool(i % 2)
            #r_dims = (0, 2, 3) if midside == True else (0, 1, 2, 3)
            r_dims = (0, 3) if midside == True else (0, 1, 3)

            with torch.no_grad():
                target_fft_abs = self.stft2d_abs(target, block_width, block_height, step_w, step_h, window, offset_h, offset_w, midside)
                loss_weight = target_fft_abs.pow(2).mean(dim=r_dims, keepdim=True).clip(min=self.config.psd_eps).pow(0.5)

            sample_fft_abs = self.stft2d_abs(sample, block_width, block_height, step_w, step_h, window, offset_h, offset_w, midside)
            
            mse_loss = torch.nn.functional.mse_loss(sample_fft_abs.float(), target_fft_abs.float(), reduction="none")
            loss = loss + (mse_loss / loss_weight).mean(dim=(1,2,3,4,5)) #** 2

        return loss * self.loss_scale

    def compile(self, **kwargs) -> None:
        self._mss_loss = torch.compile(self._mss_loss, fullgraph=True, dynamic=True)


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = MSSLoss2DConfig()
    loss_fn = MSSLoss2D(config, device)
    loss_fn.compile()

    batch_size = 4
    channels = 2
    height = 512
    width = 512

    sample = torch.randn(batch_size, channels, height, width, device=device)
    target = torch.randn(batch_size, channels, height, width, device=device)

    loss = loss_fn.mss_loss(sample, target)
    print("Loss:", loss)