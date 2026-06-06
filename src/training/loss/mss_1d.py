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

from modules.formats.frequency_scale import get_mel_density


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
class MSSLoss1DConfig:

    block_low:  int = 31
    block_high: int = 8200

    block_sampling_replace: bool = True
    block_sampling_scale: Literal["linear", "ln_linear"] = "linear"

    leak_pow: float = None
    leak_max: float = None

    sample_rate: float = 32000
    num_iterations: int = 20
    midside_probability: float = 0.5
    psd_eps: float = 1e-5
    
    cepstrum_pow: float = 0.25

    loss_scale: float = 100
    loss_scale_cepstrum: float = 120


class MSSLoss1D:

    @torch.no_grad()
    def __init__(self, config: MSSLoss1DConfig, device: torch.device) -> None:

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

        torch.backends.cuda.cufft_plan_cache.max_size = len(block_sizes) + 250 # slight performance boost if fft plans are cached
        self.windows: dict[int, torch.Tensor] = {}
        self.mel_densities: dict[int, torch.Tensor] = {}

        self.loss_scale = config.loss_scale / self.config.num_iterations
        self.cepstrum_loss_scale = config.loss_scale_cepstrum / self.config.num_iterations

    @torch.no_grad()
    def _flat_top_window(self, x: torch.Tensor) -> torch.Tensor:
        return (0.21557895 - 0.41663158 * torch.cos(x) + 0.277263158 * torch.cos(2*x)
                - 0.083578947 * torch.cos(3*x) + 0.006947368 * torch.cos(4*x))

    @torch.no_grad()
    def get_flat_top_window_1d(self, width: int) -> torch.Tensor:

        if width in self.windows:
            return self.windows[width]

        wx = self._flat_top_window((torch.arange(width,  device=self.device) + 0.5) / width  * 2 * torch.pi)
        window = wx.view(1, 1, 1,-1)
        window /= window.square().mean().sqrt()

        self.windows[width] = window
        return window
    
    @torch.no_grad()
    def get_mel_density(self, width: int) -> torch.Tensor:

        if width in self.mel_densities:
            return self.mel_densities[width]

        freqs = torch.fft.rfftfreq(width, device=self.device) * self.config.sample_rate
        mel_density = get_mel_density(freqs).view(1, 1, 1,-1)
        mel_density /= mel_density.mean()

        self.mel_densities[width] = mel_density
        return mel_density
    
    def stft1d(self, x: torch.Tensor, block_width: int, step: int, window: torch.Tensor, offset_w: int, end_offset_w: int, midside: bool) -> torch.Tensor:
        
        x = x[:, :, offset_w:end_offset_w]
        x = x.unfold(2, block_width, step)

        x = torch.fft.rfft(x * window, norm="ortho")
        if midside == True:
            x = torch.fft.fft(x, dim=1, norm="ortho")

        return x
    
    def mss_loss(self, sample1: torch.Tensor, sample2: torch.Tensor, target: torch.Tensor,
            leak_pow: Optional[float] = None, leak_max: Optional[float] = None, t: Optional[torch.Tensor] = None) -> dict[str, torch.Tensor]:

        sample = torch.cat((sample1, sample2), dim=0)

        if t is None:
            leak_pow = self.config.leak_pow or leak_pow
            leak_max = self.config.leak_max or leak_max

            if leak_pow is not None and leak_max is not None:  # useful at start of training for preventing polarity mismatch
                rnd_t = (torch.rand(sample1.shape[0], device=sample1.device).pow(leak_pow) * leak_max).repeat(2).view(-1, 1, 1)
                sample = torch.lerp(sample, target.repeat(2, 1, 1).detach(), rnd_t)
        else:
            sample = torch.lerp(sample, target.repeat(2, 1, 1).detach(), 1 - t.view(-1, 1, 1))

        loss = torch.zeros(sample.shape[0], device=self.device)
        loss_cepstrum = torch.zeros_like(loss)

        static_pad = int(self.block_sizes[-1])
        sample = torch.nn.functional.pad(sample, (static_pad, static_pad), mode="reflect")
        target = torch.nn.functional.pad(target, (static_pad, static_pad), mode="reflect")

        block_widths  = np.random.choice(self.block_sizes, size=self.config.num_iterations,
            replace=self.config.block_sampling_replace, p=self.block_weights)

        for i in range(self.config.num_iterations):

            block_width = int(block_widths[i])
            step_w = block_width
            window = self.get_flat_top_window_1d(block_width)

            offset_min_w = int(max(0, static_pad - block_width))
            offset_max_w = int(max(offset_min_w, static_pad))
            offset_w = int(np.random.randint(offset_min_w, offset_max_w + 1))
            end_offset_w = -(static_pad - block_width) or None
            
            midside = np.random.rand() < self.config.midside_probability
            r_dims = (0, 2) if midside == True else (0, 1, 2)

            with torch.no_grad():
                target_fft = self.stft1d(target, block_width, step_w, window, offset_w, end_offset_w, midside)
                target_fft_abs = target_fft.abs().requires_grad_(False).detach()

                if self.loss_scale > 0:
                    loss_weight = target_fft_abs.pow(2).mean(dim=r_dims, keepdim=True)
                    #print(block_width, loss_weight.amin(), loss_weight.mean())
                    loss_weight = loss_weight.clip(min=self.config.psd_eps).pow(0.5).requires_grad_(False).detach()

                if self.cepstrum_loss_scale > 0:
                    target_cepstrum: torch.Tensor = torch.fft.rfft((target_fft_abs + self.config.psd_eps).pow(self.config.cepstrum_pow), norm="ortho").abs()
                    target_cepstrum = target_cepstrum.repeat(2, 1, 1, 1)

                target_fft_abs = target_fft_abs.repeat(2, 1, 1, 1)

            sample_fft = self.stft1d(sample, block_width, step_w, window, offset_w, end_offset_w, midside)
            sample_fft_abs = sample_fft.abs()

            if self.loss_scale > 0:
                mse_loss = torch.nn.functional.mse_loss(sample_fft_abs.float(), target_fft_abs.float(), reduction="none")
                loss = loss + (mse_loss / loss_weight).mean(dim=(1,2,3)) #** 2

            if self.cepstrum_loss_scale > 0:
                sample_cepstrum: torch.Tensor = torch.fft.rfft((sample_fft_abs + self.config.psd_eps).pow(self.config.cepstrum_pow), norm="ortho").abs()

                mse_loss = torch.nn.functional.mse_loss(sample_cepstrum.float(), target_cepstrum.float(), reduction="none")
                loss_cepstrum = loss_cepstrum + mse_loss.mean(dim=(1,2,3))

        loss1, loss2 = loss.chunk(2, dim=0)
        loss_cepstrum1, loss_cepstrum2 = loss_cepstrum.chunk(2, dim=0)

        loss = (loss1 + loss2) / 2
        loss_cepstrum = (loss_cepstrum1 + loss_cepstrum2) / 2

        return {
            "loss/mss1d": loss * self.loss_scale,
            "loss/mss1d_cepstrum": loss_cepstrum * self.cepstrum_loss_scale,
        }


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = MSSLoss1DConfig(block_sampling_scale="ln_linear")
    loss_fn = MSSLoss1D(config, device)

    for i in range(len(loss_fn.block_sizes)):
        print(f"Block size: {loss_fn.block_sizes[i]:3d} Weight: {(loss_fn.block_weights[i]*100):.3f}%")
    print(f"total unique block sizes: {len(loss_fn.block_sizes)}\n")
    
    batch_size = 4
    channels = 2
    width = 128000

    sample = torch.randn(batch_size, channels, width, device=device)
    target = torch.randn(batch_size, channels, width, device=device)

    loss = loss_fn.mss_loss(sample, target)
    print("Loss:", loss)