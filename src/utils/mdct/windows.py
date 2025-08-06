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

from typing import Callable

import torch


def kaiser_bessel_derived(
    win_length: int,
    beta: float = 12.0,
    *,
    dtype: torch.dtype = None,
    device: torch.device = None
) -> torch.Tensor:
    """
    Generate a Kaiser-Bessel derived window.

    Parameters
    ----------
    win_length : int
        The length of the window.
    beta : float, optional
        The beta parameter for the Kaiser window, by default 12.0.
    dtype : torch.dtype, optional
        The desired data type of returned tensor, by default None.
    device : torch.device, optional
        The desired device of returned tensor, by default None.

    Returns
    -------
    torch.Tensor
        The generated Kaiser-Bessel derived window of shape (win_length,).
    """

    half_w_length = win_length // 2
    kaiser_w = torch.kaiser_window(
        half_w_length + 1, True, beta, dtype=dtype, device=device
    )
    kaiser_w_csum = torch.cumsum(kaiser_w, dim=-1)
    half_w = torch.sqrt(kaiser_w_csum[:-1] / kaiser_w_csum[-1])
    w = torch.cat((half_w, torch.flip(half_w, dims=(0,))), axis=0)

    return w


def vorbis(
    win_length: int, *, dtype: torch.dtype = None, device: torch.device = None
) -> torch.Tensor:
    """
    Generate a Vorbis window.

    Parameters
    ----------
    win_length : int
        The length of the window.
    dtype : torch.dtype, optional
        The desired data type of returned tensor, by default None.
    device : torch.device, optional
        The desired device of returned tensor, by default None.

    Returns
    -------
    torch.Tensor
        The generated Vorbis window of shape (win_length,).
    """

    arg = torch.arange(win_length, dtype=dtype, device=device) + 0.5
    w = torch.sin(
        torch.pi / 2.0 * torch.pow(torch.sin(torch.pi / win_length * arg), 2.0)
    )

    return w


def sin_window(
    win_length: int, *, dtype: torch.dtype = None, device: torch.device = None
) -> torch.Tensor:
    """
    Generate a standard sin window.

    Parameters
    ----------
    win_length : int
        The length of the window.
    dtype : torch.dtype, optional
        The desired data type of returned tensor, by default None.
    device : torch.device, optional
        The desired device of returned tensor, by default None.

    Returns
    -------
    torch.Tensor
        The generated sin window of shape (win_length,).
    """

    arg = torch.arange(win_length, dtype=dtype, device=device) + 0.5
    w = torch.sin(arg / win_length * torch.pi)

    return w


def get_window_fn(window_fn: str) -> Callable[..., torch.Tensor]:
    if window_fn == "sin":
        return sin_window
    elif window_fn == "kbd":
        return kaiser_bessel_derived
    elif window_fn == "vorbis":
        return vorbis
    else:
        raise ValueError(f"Unknown MDCT window function: {window_fn}. Supported functions are 'sin', 'kbd', and 'vorbis'.")