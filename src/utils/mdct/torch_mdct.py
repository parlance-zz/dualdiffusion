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

from typing import Any, Callable, Dict, Optional

import torch
from torch import nn

from .functional import imdct, mdct, imdct2, mdct2
from .windows import vorbis


def create_window(
    win_length: int,
    window_fn: Callable[..., torch.Tensor],
    window_kwargs: Optional[Dict[str, Any]],
) -> torch.Tensor:
    """
    Utility function to create a window tensor.

    Parameters
    ----------
    win_length : int
        The length of the window.
    window_fn : callable
        Function to generate the window.
    window_kwargs : dict, optional
        Additional keyword arguments to pass to the window function.

    Returns
    -------
    torch.Tensor
        Precomputed window tensor.
    """
    return window_fn(win_length, **(window_kwargs or {}))


class MDCT(nn.Module):
    """
    Module to compute the Modified Discrete Cosine Transform (MDCT) of a waveform.

    Parameters
    ----------
    win_length : int
        The length of the window.
    window_fn : callable, default=vorbis
        A function to generate the window, by default vorbis. kaiser_bessel_derived is also available.
    window_kwargs : dict, optional
        Additional keyword arguments to pass to the window function, by default None.
    padding : bool, default=True
        If True, pad the waveform on both sides with half the window length, by default True.
    return_complex : bool
        If True, returns the complex-valued MCLT instead of the real-valued MDCT, by default False.

    Attributes
    ----------
    window : torch.Tensor
        The window tensor.

    Methods
    -------
    forward(waveform: torch.Tensor) -> torch.Tensor
        Compute the MDCT of the input waveform.

    Examples
    --------
    >>> waveform = torch.rand(2, 44100) # (channels, n_samples)
    >>> mdct = MDCT(win_length=1024)
    >>> spectrogram = mdct(waveform)
    >>> print(spectrogram.shape)  # (2, 512, 89)
    """

    def __init__(
        self,
        win_length: int,
        window_fn: Callable[..., torch.Tensor] = vorbis,
        window_kwargs: Optional[Dict[str, Any]] = None,
        padding: bool = True,
        return_complex: bool = False,
    ) -> None:
        super().__init__()

        self.win_length = win_length
        self.window_fn = window_fn
        self.window_kwargs = window_kwargs or {}
        self.padding = padding
        self.return_complex = return_complex
        self.fn = mdct

        self.register_buffer(
            "window", create_window(self.win_length, self.window_fn, self.window_kwargs), persistent=False
        )

    def forward(self, waveform: torch.Tensor, return_complex: Optional[bool] = None) -> torch.Tensor:
        """
        Compute the MDCT of the input waveform.

        Parameters
        ----------
        waveform : torch.Tensor
            Input waveform tensor of shape (..., n_samples).
        return_complex : bool, optional
            If specified and True, returns the complex-valued MCLT instead of the real-valued MDCT.

        Returns
        -------
        torch.Tensor
            MDCT spectrogram of shape (..., win_length // 2, n_frames).
        """
        return_complex = return_complex or self.return_complex
        return self.fn(waveform, self.window, padding=self.padding, return_complex=return_complex)


class IMDCT(nn.Module):
    """
    Module to compute the inverse Modified Discrete Cosine Transform (iMDCT) of a spectrogram.

    Parameters
    ----------
    win_length : int
        The length of the window.
    window_fn : callable, default=vorbis
        A function to generate the window, by default vorbis. kaiser_bessel_derived is also available.
    window_kwargs : dict, optional
        Additional keyword arguments to pass to the window function, by default None.
    padding : bool, default=True
        If True, pad the waveform on both sides with half the window length, by default True.

    Attributes
    ----------
    window : torch.Tensor
        The window tensor.

    Methods
    -------
    forward(spectrogram: torch.Tensor) -> torch.Tensor
        Compute the iMDCT of the input spectrogram.

    Examples
    --------
    >>> spectrogram = torch.rand(2, 512, 89) # (channels, win_length // 2, n_frames)
    >>> imdct = IMDCT(win_length=1024)
    >>> waveform = imdct(spectrogram) # (2, 44100)
    """

    def __init__(
        self,
        win_length: int,
        window_fn: Callable[..., torch.Tensor] = vorbis,
        window_kwargs: Optional[Dict[str, Any]] = None,
        padding: bool = True,
    ) -> None:
        super().__init__()

        # Save parameters for introspection or serialization
        self.win_length = win_length
        self.window_fn = window_fn
        self.window_kwargs = window_kwargs or {}
        self.padding = padding
        self.fn = imdct

        # Register window tensor
        self.register_buffer(
            "window", create_window(self.win_length, self.window_fn, self.window_kwargs)
        )

    def forward(
        self, spectrogram: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the iMDCT of the input spectrogram.

        Parameters
        ----------
        spectrogram : torch.Tensor
            Input MDCT spectrogram tensor of shape (..., win_length // 2, n_frames).

        Returns
        -------
        torch.Tensor
            Reconstructed waveform tensor.
        """
        return self.fn(spectrogram, self.window, padding=self.padding)
    
class MDCT2(MDCT):
    def __init__(
        self,
        win_length: int,
        window_fn: Callable[..., torch.Tensor] = vorbis,
        window_kwargs: Optional[Dict[str, Any]] = None,
        padding: bool = True,
        return_complex: bool = False,
    ) -> None:
        super().__init__(win_length, window_fn, window_kwargs, padding, return_complex)
        self.fn = mdct2

class IMDCT2(IMDCT):
    def __init__(
        self,
        win_length: int,
        window_fn: Callable[..., torch.Tensor] = vorbis,
        window_kwargs: Optional[Dict[str, Any]] = None,
        padding: bool = True,
    ) -> None:
        super().__init__(win_length, window_fn, window_kwargs, padding)
        self.fn = imdct2