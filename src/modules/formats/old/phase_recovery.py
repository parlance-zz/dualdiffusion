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
from tqdm.auto import tqdm


def _get_complex_dtype(real_dtype: torch.dtype):
    if real_dtype == torch.double:
        return torch.cdouble
    if real_dtype == torch.float:
        return torch.cfloat
    if real_dtype == torch.half:
        return torch.complex32
    raise ValueError(f"Unexpected dtype {real_dtype}")

@torch.inference_mode()
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
    show_tqdm: bool = True,
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

    progress_bar = tqdm(total=n_iter) if show_tqdm else None
    for i in range(n_iter):
        
        if stereo:
            #t = max(i / n_iter - stereo_coherence, 0)
            #interp_specgram = specgram * t + merged_specgram * (1 - t)

            t = i / n_iter - stereo_coherence
            if t > 0: # performance / precision improvement over above
                interp_specgram = torch.lerp(merged_specgram, specgram, t)
            else:
                interp_specgram = merged_specgram
        else:
            interp_specgram = specgram

        inverse = torch.istft(
            angles * interp_specgram, n_fft=n_fft, hop_length=hop_length,
            win_length=win_length, window=window, length=length
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
            #angles = angles - tprev.mul_(momentum)
            angles.sub_(tprev, alpha=momentum) # performance / precision improvement over above
        #angles = angles.div(angles.abs().add(1e-16))
        angles = angles.div(angles.abs().add_(1e-16))

        tprev = rebuilt
        if progress_bar is not None:
            progress_bar.update(1)

    waveform = torch.istft(
        angles * specgram, n_fft=n_fft, hop_length=hop_length,
        win_length=win_length, window=window, length=length
    )

    if progress_bar is not None:
        progress_bar.close()

    return waveform.reshape(shape[:-2] + waveform.shape[-1:])

class PhaseRecovery(torch.nn.Module):

    def __init__(
        self,
        n_fft: int,
        n_fgla_iter: int = 200,
        win_length: Optional[int] = None,
        hop_length: Optional[int] = None,
        window_fn: Callable[..., Tensor] = torch.hann_window,
        wkwargs: Optional[dict] = None,
        momentum: float = 0.99,
        length: Optional[int] = None,
        rand_init: bool = True,
        stereo: bool = True,
        stereo_coherence: float = 0.67,
    ) -> None:
        super().__init__()

        if not (0 <= momentum < 1):
            raise ValueError("momentum must be in the range [0, 1). Found: {}".format(momentum))

        self.n_fft = n_fft
        self.n_fgla_iter = n_fgla_iter
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 2
        self.length = length
        self.momentum = momentum
        self.rand_init = rand_init
        self.stereo = stereo
        self.stereo_coherence = stereo_coherence

        window = window_fn(self.win_length) if wkwargs is None else window_fn(self.win_length, **wkwargs)
        self.register_buffer("window", window, persistent=False)

    @torch.inference_mode()
    def forward(self, specgram: Tensor, n_fgla_iter: Optional[int] = None, quiet: bool = False) -> Tensor:

        n_fgla_iter = n_fgla_iter or self.n_fgla_iter

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
                manual_init=None,
                show_tqdm=not quiet,
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
