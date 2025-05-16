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

from typing import Optional, Literal
from logging import getLogger

import numpy as np
import torch


def _hz_to_mel(freq: float) -> float:
    return 2595.0 * np.log10(1.0 + (freq / 700.0))

def _mel_to_hz(mels: torch.Tensor) -> torch.Tensor:
    return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

def get_mel_density(hz: torch.Tensor) -> torch.Tensor:
    return 1127. / (700. + hz)

@torch.no_grad()
def _create_cos_filterbank(all_freqs: torch.Tensor, f_pts: torch.Tensor) -> torch.Tensor:

    filters = _create_triangular_filterbank(all_freqs, f_pts)
    return (torch.pi * filters / 2).sin()**2

@torch.no_grad()
def _create_triangular_filterbank(all_freqs: torch.Tensor, f_pts: torch.Tensor) -> torch.Tensor:
    # Adopted from Librosa
    # calculate the difference between each filter mid point and each stft freq point in hertz
    f_diff = f_pts[1:] - f_pts[:-1]  # (n_filter + 1)
    
    slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)  # (n_freqs, n_filter + 2)
    # create overlapping triangles
    zero = torch.zeros(1)
    down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1]  # (n_freqs, n_filter)
    up_slopes = slopes[:, 2:] / f_diff[1:]  # (n_freqs, n_filter)

    fb = torch.max(zero, torch.min(down_slopes, up_slopes))
    return fb

def regularized_filter_unmixing(filters: torch.Tensor, spectrogram: torch.Tensor, lam: float = 1e-2) -> torch.Tensor:
    """
    Args:
        filters: (w, f) filter matrix (each column is a filter)
        spectrogram: (batch, w, t) or (1, w, t)
        lam: regularization strength
    Returns:
        (batch, f, t) coefficients
    """
    w, f = filters.shape
    filters_T = filters.T  # (f, w)
    AtA = filters_T @ filters  # (f, f)
    AtA += lam * torch.eye(f, device=filters.device, dtype=filters.dtype)  # Regularization
    AtA_inv = torch.linalg.inv(AtA)  # (f, f)
    At = filters_T  # (f, w)

    # Apply to each time step
    batch, _, t = spectrogram.shape  # (b, w, t)
    coeffs = torch.zeros((batch, f, t), device=spectrogram.device, dtype=spectrogram.dtype)
    for i in range(t):
        b = spectrogram[:, :, i]  # (b, w)
        coeffs[:, :, i] = (AtA_inv @ (At @ b.transpose(0, 1))).transpose(0, 1)
    
    return coeffs

class FrequencyScale(torch.nn.Module):

    def __init__(
        self,
        freq_scale: Literal["mel", "log"] = "mel",
        freq_min: float = 0.0,
        freq_max: Optional[float] = None,
        sample_rate: int = 32000,
        num_stft_bins: int = 3201,
        num_filters: int = 256,
        filter_norm: Optional[Literal["slaney"]] = None,
        unscale_driver: Literal["gels", "gelsy", "gelsd", "gelss"] = "gels",
        filter_shape: Literal["triangular", "cos"] = "triangular",
    ) -> None:
        
        super().__init__()

        self.freq_scale = freq_scale
        self.freq_min = freq_min
        self.freq_max = freq_max or sample_rate / 2
        self.sample_rate = sample_rate
        self.num_stft_bins = num_stft_bins
        self.num_filters = num_filters
        self.filter_norm = filter_norm
        self.filter_shape = filter_shape
        self.unscale_driver = unscale_driver
        
        if freq_scale == "mel":
            self.scale_fn = _hz_to_mel
            self.unscale_fn = _mel_to_hz
            
        elif freq_scale == "log":
            self.scale_fn = np.log2
            self.unscale_fn = torch.exp2
        else:
            raise ValueError(f"Unknown frequency scale: {freq_scale}")
        
        self.register_buffer("filters", self.get_filters())

        if (self.filters.max(dim=0).values == 0.0).any():
            getLogger().warning("WARNING: At least one FrequencyScale filterbank has all zero values")

    def scale(self, specgram: torch.Tensor) -> torch.Tensor:
        return torch.matmul(specgram.transpose(-1, -2), self.filters).transpose(-1, -2)
    
    def unscale(self, spectrogram: torch.Tensor, rectify: bool = True) -> torch.Tensor:
        # pack batch
        original_shape = spectrogram.size()
        spectrogram = spectrogram.reshape(-1, original_shape[-2], original_shape[-1])

        #unscaled = regularized_filter_unmixing(self.filters.T, spectrogram, lam=5e-3)
        unscaled = torch.linalg.lstsq(self.filters.transpose(-1, -2)[None], spectrogram, driver=self.unscale_driver).solution
        
        if rectify == True:
            unscaled = torch.relu(unscaled)
        
        # unpack batch
        return unscaled.view(original_shape[:-2] + (self.num_stft_bins, original_shape[-1]))
    
    def get_unscaled(self, num_points: int, device: Optional[torch.device] = None) -> torch.Tensor:

        scaled_freqs = torch.linspace(
            self.scale_fn(self.freq_min), self.scale_fn(self.freq_max), num_points, device=device)
        
        return self.unscale_fn(scaled_freqs)
    
    @torch.no_grad()
    def get_filters(self) -> torch.Tensor:

        stft_freqs = torch.linspace(0, self.sample_rate / 2, self.num_stft_bins)
        unscaled_freqs = self.get_unscaled(self.num_filters + 2)

        if self.filter_shape == "triangular":
            filters = _create_triangular_filterbank(stft_freqs, unscaled_freqs)
        elif self.filter_shape == "cos":
            filters = _create_cos_filterbank(stft_freqs, unscaled_freqs)
        else:
            raise ValueError(f"Invalid filter shape: {self.filter_shape}")
        
        if self.filter_norm == "slaney":
            # Slaney-style mel is scaled to be approx constant energy per channel
            enorm = 2. / (unscaled_freqs[2:self.num_filters+2] - unscaled_freqs[:self.num_filters])
            filters *= enorm.unsqueeze(0)

        return filters