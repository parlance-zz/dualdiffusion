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

from typing import Optional, Callable

import torch
from torch import Tensor

def _get_complex_dtype(real_dtype: torch.dtype):
    if real_dtype == torch.double:
        return torch.cdouble
    if real_dtype == torch.float:
        return torch.cfloat
    if real_dtype == torch.half:
        return torch.complex32
    raise ValueError(f"Unexpected dtype {real_dtype}")

@torch.no_grad()
def difference_map(
    specgram: Tensor,
    window: Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
    n_iter: int,
    beta: float,
    length: Optional[int],
    rand_init: bool,
    manual_init: Optional[Tensor] = None,
) -> Tensor:

    def pA(x):
        return x.div(x.abs().add(1e-16)) * specgram

    def pC(x):
        inverse = torch.istft(
            x, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, length=length
        )
        
        return torch.stft(
            input=inverse,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )

    def fA(x):
        pa = pA(x)
        return pa + (pa - x) / beta
    
    def fC(x):
        pc = pC(x)
        return pc - (pc - x) / beta
    
    shape = specgram.size()
    specgram = specgram.reshape([-1] + list(shape[-2:]))

    if manual_init is not None:
        angles = manual_init
    else:
        if rand_init:
            angles = torch.randn(specgram.size(), dtype=_get_complex_dtype(specgram.dtype), device=specgram.device)
        else:
            angles = torch.full(specgram.size(), 1, dtype=_get_complex_dtype(specgram.dtype), device=specgram.device)

    for _ in range(n_iter):
        angles = angles + beta * ( pC(fA(angles)) - pA(fC(angles)) )

    return angles.reshape(shape)

@torch.no_grad()
def griffinlim(
    specgram: Tensor,
    window: Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
    n_iter: int,
    momentum: float,
    length: Optional[int],
    rand_init: bool,
    stereo: bool,
    stereo_coherence: float,
    manual_init: Optional[Tensor] = None,
) -> Tensor:

    if not 0 <= momentum < 1:
        raise ValueError("momentum must be in range [0, 1). Found: {}".format(momentum))
    momentum = momentum / (1 + momentum)

    shape = specgram.size()
    specgram = specgram.reshape([-1] + list(shape[-2:]))

    if stereo:
        merged_specgram = ((specgram[0::2] + specgram[1::2]) / 2).repeat_interleave(2, dim=0)
    
    if manual_init is not None:
        angles = manual_init.reshape(specgram.shape)
    else:
        init_shape = (1,) + tuple(specgram.shape[1:])
        if rand_init:
            angles = torch.randn(init_shape, dtype=_get_complex_dtype(specgram.dtype), device=specgram.device)
        else:
            angles = torch.full(init_shape, 1, dtype=_get_complex_dtype(specgram.dtype), device=specgram.device)
        
    tprev = torch.tensor(0.0, dtype=specgram.dtype, device=specgram.device)

    for i in range(n_iter):
        
        if stereo:
            t = max(i / n_iter - stereo_coherence, 0)
            interp_specgram = specgram * t + merged_specgram * (1 - t)
        else:
            interp_specgram = specgram

        inverse = torch.istft(
            angles * interp_specgram, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, length=length
        )

        rebuilt = torch.stft(
            input=inverse,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )

        angles = rebuilt
        if momentum:
            angles = angles - tprev.mul_(momentum)
        angles = angles.div(angles.abs().add(1e-16))

        tprev = rebuilt

    waveform = torch.istft(
        angles * specgram, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, length=length
    )

    return waveform.reshape(shape[:-2] + waveform.shape[-1:])

class PhaseRecovery(torch.nn.Module):

    __constants__ = ["n_fft", "n_iter", "win_length", "hop_length", "length", "momentum", "rand_init", "beta", "stereo", "stereo_coherence"]

    @torch.no_grad()
    def __init__(
        self,
        n_fft: int = 400,
        n_fgla_iter: int = 400,
        n_dm_iter: int = 0,
        win_length: Optional[int] = None,
        hop_length: Optional[int] = None,
        window_fn: Callable[..., Tensor] = torch.hann_window,
        wkwargs: Optional[dict] = None,
        momentum: float = 0.99,
        length: Optional[int] = None,
        rand_init: bool = True,
        beta: float = 1.,
        stereo: bool = True,
        stereo_coherence: float = 0.67,
    ) -> None:
        super(PhaseRecovery, self).__init__()

        if not (0 <= momentum < 1):
            raise ValueError("momentum must be in the range [0, 1). Found: {}".format(momentum))

        self.n_fft = n_fft
        self.n_fgla_iter = n_fgla_iter
        self.n_dm_iter = n_dm_iter
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 2
        window = window_fn(self.win_length) if wkwargs is None else window_fn(self.win_length, **wkwargs)
        self.register_buffer("window", window, persistent=False)
        self.length = length
        self.momentum = momentum
        self.rand_init = rand_init
        self.beta = beta
        self.stereo = stereo
        self.stereo_coherence = stereo_coherence

    @torch.no_grad()
    def forward(self, specgram: Tensor, n_fgla_iter=None) -> Tensor:

        n_fgla_iter = n_fgla_iter or self.n_fgla_iter

        if self.n_dm_iter > 0:
            wave = difference_map(
                specgram,
                self.window,
                self.n_fft,
                self.hop_length,
                self.win_length,
                self.n_dm_iter,
                self.beta,
                self.length,
                self.rand_init,
            )
        else:
            wave = None
        
        if self.n_fgla_iter > 0:
            wave = griffinlim(
                specgram,
                self.window,
                self.n_fft,
                self.hop_length,
                self.win_length,
                n_fgla_iter,
                self.momentum,
                self.length,
                self.rand_init,
                self.stereo,
                self.stereo_coherence,
                manual_init=wave,
            )
        else:
            wave_shape = wave.size()
            wave = torch.istft(
                wave.reshape([-1] + list(wave_shape[-2:])),
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,
                length=self.length,
            ).reshape(wave_shape[:-2] + wave.shape[-1:])

        return wave
