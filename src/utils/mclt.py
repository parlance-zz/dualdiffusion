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

from typing import Optional

import torch

class WindowFunction:

    @staticmethod
    def hann(window_len: int, device: Optional[torch.device] = None) -> torch.Tensor:
        n = torch.arange(window_len, device=device) / window_len
        return (0.5 - 0.5 * torch.cos(2 * torch.pi * n))

    @staticmethod
    def kaiser(window_len: int, beta: float = 4*torch.pi, device: Optional[torch.device] = None) -> torch.Tensor:
        alpha = (window_len - 1) / 2
        n = torch.arange(window_len, device=device)
        return (torch.special.i0(beta * torch.sqrt(1 - ((n - alpha) / alpha).square()))
                / torch.special.i0(torch.tensor(beta)))

    @staticmethod
    def kaiser_derived(window_len:int , beta: float = 4*torch.pi, device: Optional[torch.device] = None) -> torch.Tensor:

        kaiserw = WindowFunction.kaiser(window_len // 2 + 1, beta, device)
        csum = torch.cumsum(kaiserw, dim=0)
        halfw = torch.sqrt(csum[:-1] / csum[-1])

        w = torch.zeros(window_len, device=device)
        w[:window_len//2] = halfw
        w[-window_len//2:] = halfw.flip(0)

        return w

    @staticmethod
    def hann_poisson(window_len: int, alpha: float = 2, device: Optional[torch.device] = None) -> torch.Tensor:
        x = torch.arange(window_len, device=device) / window_len
        return (torch.exp(-alpha * (1 - 2*x).abs()) * 0.5 * (1 - torch.cos(2*torch.pi*x)))

    @staticmethod
    def blackman_harris(window_len: int, device: Optional[torch.device] = None) -> torch.Tensor:
        x = torch.arange(window_len, device=device) / window_len * 2*torch.pi
        return (0.35875 - 0.48829 * torch.cos(x) + 0.14128 * torch.cos(2*x) - 0.01168 * torch.cos(3*x))

    @staticmethod
    def flat_top(window_len:int , device: Optional[torch.device] = None) -> torch.Tensor:
        x = torch.arange(window_len, device=device) / window_len * 2*torch.pi
        return (  0.21557895
                - 0.41663158 * torch.cos(x)
                + 0.277263158 * torch.cos(2*x)
                - 0.083578947 * torch.cos(3*x)
                + 0.006947368 * torch.cos(4*x))

    @staticmethod
    @torch.no_grad()
    def get_window_fn(window_fn: str):
        return getattr(WindowFunction, window_fn)

def mclt(x: torch.Tensor, block_width: int, window_fn: str ="hann",
         window_exponent: float = 1., **kwargs) -> torch.Tensor:

    padding_left = padding_right = block_width // 2
    remainder = x.shape[-1] % (block_width // 2)
    if remainder > 0:
        padding_right += block_width // 2 - remainder

    x = torch.nn.functional.pad(x, (padding_left, padding_right), mode="reflect").unfold(-1, block_width, block_width//2)

    N = x.shape[-1] // 2
    n = torch.arange(2*N, device=x.device)
    k = torch.arange(0.5, N + 0.5, device=x.device)

    window = 1 if window_exponent == 0 else WindowFunction.get_window_fn(window_fn)(
        2*N, device=x.device, **kwargs) ** window_exponent
    
    pre_shift = torch.exp(-1j * torch.pi / 2 / N * n)
    post_shift = torch.exp(-1j * torch.pi / 2 / N * (N + 1) * k)
    
    return torch.fft.fft(x * pre_shift * window, norm="forward")[..., :N] * post_shift * (2 * N ** 0.5)

def imclt(x: torch.Tensor, window_fn: str = "hann",
          window_degree: float = 1, **kwargs) -> torch.Tensor:
    
    N = x.shape[-1]
    n = torch.arange(2*N, device=x.device)
    k = torch.arange(0.5, N + 0.5, device=x.device)

    window = 1 if window_degree == 0 else WindowFunction.get_window_fn(window_fn)(2*N, device=x.device, **kwargs) ** window_degree

    pre_shift = torch.exp(-1j * torch.pi / 2 / N * n)
    post_shift = torch.exp(-1j * torch.pi / 2 / N * (N + 1) * k)

    y = (torch.fft.ifft(x / post_shift, norm="backward", n=2*N) / pre_shift) * window

    padded_sample_len = (y.shape[-2] + 1) * y.shape[-1] // 2
    raw_sample = torch.zeros(y.shape[:-2] + (padded_sample_len,), device=y.device, dtype=y.dtype)
    y_even = y[...,  ::2, :].reshape(*y[...,  ::2, :].shape[:-2], -1)
    y_odd  = y[..., 1::2, :].reshape(*y[..., 1::2, :].shape[:-2], -1)
    raw_sample[..., :y_even.shape[-1]] = y_even
    raw_sample[..., N:y_odd.shape[-1] + N] += y_odd

    return raw_sample[..., N:-N] * (2 * N ** 0.5)

def stft(x: torch.Tensor, block_width: int, window_fn: str = "hann", window_degree: float = 1.,
         step: Optional[int] = None, add_channelwise_fft: bool = False, **kwargs) -> torch.Tensor:

    step = step or block_width//2
    x = x.unfold(-1, block_width, step)

    window = 1 if window_degree == 0 else WindowFunction.get_window_fn(window_fn)(block_width, device=x.device, **kwargs) ** window_degree

    x = torch.fft.rfft(x * window, norm="ortho")
    x = torch.fft.fft(x, norm="ortho", dim=-3) if add_channelwise_fft else x

    return x
