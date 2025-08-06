# MIT License
#
# Copyright (c) 2024 Kinyugo Maina
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

# Modifications under MIT License
#
# Copyright (c) 2025 Christopher Friesen
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

import math
from typing import Optional

import torch
from torch.nn import functional as F


def mdct(
    waveform: torch.Tensor, window: torch.Tensor, padding: bool = True, return_complex: bool = False
) -> torch.Tensor:
    """
    Compute the Modified Discrete Cosine Transform (MDCT) of a waveform.

    Parameters
    ----------
    waveform : torch.Tensor
        Input waveform tensor of shape (..., n_samples).
    window : torch.Tensor
        Window tensor of shape (win_length,).
    padding : bool, default=True
        If True, pad the waveform on both sides with half the window length, by default True.
    return_complex : bool
        If True, returns the complex-valued MCLT instead of the real-valued MDCT, by default False.

    Returns
    -------
    torch.Tensor
        MDCT spectrogram of the input waveform of shape (..., win_length // 2, n_frames).
    """
    n_samples = waveform.shape[-1]
    win_length = window.shape[-1]
    hop_length = win_length // 2

    # Flatten the input tensor
    shape = waveform.shape
    waveform = waveform.real.flatten(end_dim=-2)

    # Center the waveform by padding on both sides
    n_frames = int(math.ceil(n_samples / hop_length)) + 1
    if padding:
        waveform = F.pad(
            waveform,
            (hop_length, (n_frames + 1) * hop_length - n_samples),
            mode="reflect",
        )

    # Prepare pre&post processing arrays
    pre_twiddle = torch.exp(
        -1j
        * torch.pi
        / win_length
        * torch.arange(0, win_length, device=waveform.device)
    )
    post_twiddle = torch.exp(
        -1j
        * torch.pi
        / win_length
        * (win_length / 2 + 1)
        * torch.arange(0.5, win_length / 2 + 0.5, device=waveform.device)
    )

    # Convolve the waveform with the window and fft matrices
    spectrogram = waveform.unfold(dimension=-1, size=win_length, step=hop_length)
    spectrogram = torch.einsum("...kj,j->...kj", spectrogram, window)
    spectrogram = torch.einsum("...kj,j->...kj", spectrogram, pre_twiddle)
    spectrogram = torch.fft.fft(spectrogram, dim=-1)
    spectrogram = torch.einsum(
        "...jk,k->...kj", spectrogram[..., : win_length // 2], post_twiddle
    )
    if not return_complex:
        spectrogram = torch.real(spectrogram)

    # Unflatten the output
    spectrogram = spectrogram.reshape(shape[:-1] + spectrogram.shape[-2:])

    if padding:
        spectrogram = spectrogram[..., :-1]

    # Normalize the output to account for the FFT scaling (sqrt(win_length)) and the window contribution (sqrt(win_length // 2)).
    # The window's power, sum(window ** 2), equals (win_length // 2) due to energy conservation with 50% overlap.
    scaling_factor = 1.0 / math.sqrt(win_length * (win_length // 2))
    spectrogram = spectrogram * scaling_factor

    return spectrogram


def imdct(
    spectrogram: torch.Tensor,
    window: torch.Tensor,
    padding: bool = True
) -> torch.Tensor:
    """
    Compute the inverse Modified Discrete Cosine Transform (iMDCT) of a spectrogram.

    Parameters
    ----------
    spectrogram : torch.Tensor
        Input MDCT spectrogram tensor of shape (..., win_length // 2, n_frames).
    window : torch.Tensor
        Window tensor of shape (win_length,).
    padding : bool, default=True
        If True, remove the padding added during MDCT, by default True.

    Returns
    -------
    torch.Tensor
        Reconstructed waveform tensor.
    """
    win_length = window.shape[-1]
    hop_length = win_length // 2
    n_freqs, n_frames = spectrogram.shape[-2:]

    # Normalize the input to account for the FFT scaling (sqrt(win_length)) and the window contribution (sqrt(win_length // 2)).
    # The window's power, sum(window ** 2), equals (win_length // 2) due to energy conservation with 50% overlap.
    scaling_factor = 1.0 / math.sqrt(win_length * (win_length // 2))
    spectrogram = spectrogram / scaling_factor

    # Flatten the input tensor
    shape = spectrogram.shape
    spectrogram = spectrogram.real.flatten(end_dim=-3)

    # Prepare pre&post processing arrays
    pre_twiddle = torch.exp(
        -1j
        * torch.pi
        / (2 * n_freqs)
        * (n_freqs + 1)
        * torch.arange(n_freqs, device=spectrogram.device)
    )
    post_twiddle = (
        torch.exp(
            -1j
            * torch.pi
            / (2 * n_freqs)
            * torch.arange(
                0.5 + n_freqs / 2,
                2 * n_freqs + n_freqs / 2 + 0.5,
                device=spectrogram.device,
            )
        )
        / n_freqs
    )

    # Apply fft and the window
    spectrogram = torch.einsum("...jk,j->...jk", spectrogram, pre_twiddle)
    spectrogram = torch.fft.fft(spectrogram, n=2 * n_freqs, dim=1)
    spectrogram = torch.einsum("...jk,j->...jk", spectrogram, post_twiddle)
    spectrogram = torch.real(spectrogram)
    spectrogram = 2 * torch.einsum("...jk,j->...jk", spectrogram, window)

    # Recover the waveform with the time-domain aliasing cancelling principle
    waveform = F.fold(
        spectrogram,
        output_size=(1, hop_length * (n_frames + 1)),
        kernel_size=(1, win_length),
        stride=(1, hop_length),
    )

    # Remove padding
    if padding:
        waveform = waveform[..., hop_length:-hop_length]

    # Unflatten the output
    waveform = waveform.reshape((*shape[:-2], -1))

    return waveform

def mdct2(waveform: torch.Tensor, window: torch.Tensor, padding: bool = True, return_complex: bool = False) -> torch.Tensor:
    
    waveform = mdct(waveform, window, padding=padding, return_complex=False)
    waveform = mdct(waveform.permute(0, 1, 3, 4, 2), window, padding=padding, return_complex=return_complex)

    return waveform.permute(0, 1, 4, 2, 5, 3).contiguous()

def imdct2(waveform: torch.Tensor, window: torch.Tensor, padding: bool = True) -> torch.Tensor:
    
    waveform = imdct(waveform.permute(0, 1, 3, 5, 2, 4), window, padding=padding)
    waveform = imdct(waveform.permute(0, 1, 4, 2, 3), window, padding=padding)

    return waveform.contiguous()