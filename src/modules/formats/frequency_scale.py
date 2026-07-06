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

    # calculate the difference between each filter mid point and each stft freq point in hertz
    f_diff = f_pts[1:] - f_pts[:-1]  # (n_filter + 1)
    
    slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)  # (n_freqs, n_filter + 2)
    # create overlapping triangles
    zero = torch.zeros(1)
    down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1]  # (n_freqs, n_filter)
    up_slopes = slopes[:, 2:] / f_diff[1:]  # (n_freqs, n_filter)

    fb = torch.max(zero, torch.min(down_slopes, up_slopes))
    return fb

@torch.no_grad()
def _create_triangular_filterbank_plus(all_freqs: torch.Tensor, f_pts: torch.Tensor) -> torch.Tensor:

    # calculate the difference between each filter mid point and each stft freq point in hertz
    f_diff = f_pts[1:] - f_pts[:-1]  # (n_filter + 1)

    slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)  # (n_freqs, n_filter + 2)

    # create overlapping triangles
    zero = torch.zeros(1, device=all_freqs.device, dtype=all_freqs.dtype)
    down_slopes = (-slopes[:, :-2]) / f_diff[:-1]  # (n_freqs, n_filter)
    up_slopes = slopes[:, 2:] / f_diff[1:]         # (n_freqs, n_filter)

    fb = torch.maximum(zero, torch.minimum(down_slopes, up_slopes))

    # First filter: truncate left half so its peak is at DC.
    fb[:, 0] = torch.clamp((f_pts[2] - all_freqs) / (f_pts[2] - f_pts[1]), min=0.0, max=1.0)
    # Last filter: truncate right half so its peak is at Nyquist.
    fb[:, -1] = torch.clamp((all_freqs - f_pts[-3]) / (f_pts[-2] - f_pts[-3]), min=0.0, max=1.0)

    return fb

@torch.no_grad()
def _create_triangular_filterbank_plus_plus(all_freqs: torch.Tensor, f_pts: torch.Tensor) -> torch.Tensor:

    f_diff = f_pts[1:] - f_pts[:-1]

    slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)

    zero = torch.zeros(1, device=all_freqs.device, dtype=all_freqs.dtype)
    down_slopes = (-slopes[:, :-2]) / f_diff[:-1]
    up_slopes = slopes[:, 2:] / f_diff[1:]

    fb = torch.maximum(zero, torch.minimum(down_slopes, up_slopes))

    # First filter: peak at bin 0.
    fb[:, 0] = torch.clamp(
        1.0 - all_freqs / (f_pts[2] - f_pts[1]),
        min=0.0,
        max=1.0,
    )

    # Last filter: peak at last frequency bin.
    fb[:, -1] = torch.clamp(
        1.0 - (all_freqs[-1] - all_freqs) / (f_pts[-2] - f_pts[-3]),
        min=0.0,
        max=1.0,
    )

    return fb

class FrequencyScale(torch.nn.Module):

    def __init__(
        self,
        freq_scale: Literal["mel", "log"] = "mel",
        freq_min: float = 0.0,
        freq_max: Optional[float] = None,
        sample_rate: int = 32000,
        num_stft_bins: int = 3201,
        num_filters: int = 256,
        filter_norm: Optional[Literal["slaney", "l2"]] = None,
        unscale_mode: Literal["lstsq", "grid_sample"] = "lstsq",
        unscale_lstsq_driver: Literal["gels", "gelsy", "gelsd", "gelss"] = "gels",
        filter_shape: Literal["triangular", "cos", "triangular+", "triangular++"] = "triangular",
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
        self.unscale_lstsq_driver = unscale_lstsq_driver
        self.unscale_mode = unscale_mode

        assert unscale_mode in ["lstsq", "grid_sample"], f"Invalid unscale_mode: {unscale_mode}"
        assert unscale_lstsq_driver in ["gels", "gelsy", "gelsd", "gelss"], f"Invalid unscale_lstsq_driver: {unscale_lstsq_driver}"
        assert filter_shape in ["triangular", "cos", "triangular+", "triangular++"]
        
        if freq_scale == "mel":
            self.scale_fn = _hz_to_mel
            self.unscale_fn = _mel_to_hz
            
        elif freq_scale == "log":
            self.scale_fn = np.log2
            self.unscale_fn = torch.exp2
        else:
            raise ValueError(f"Unknown frequency scale: {freq_scale}")
        
        self.filters: torch.Tensor
        self.register_buffer("filters", self.get_filters(), persistent=False)

        if (self.filters.max(dim=0).values == 0.0).any():
            getLogger().warning("WARNING: At least one FrequencyScale filterbank has all zero values")

    def scale(self, specgram: torch.Tensor) -> torch.Tensor:
        return torch.matmul(specgram.transpose(-1, -2), self.filters).transpose(-1, -2)
    
    def _unscale_lstsq(self, spectrogram: torch.Tensor) -> torch.Tensor:
        # pack batch
        original_shape = spectrogram.size()
        spectrogram = spectrogram.reshape(-1, original_shape[-2], original_shape[-1])

        unscaled = torch.linalg.lstsq(self.filters.transpose(-1, -2)[None], spectrogram, driver=self.unscale_lstsq_driver).solution
        
        # unpack batch
        return unscaled.view(original_shape[:-2] + (self.num_stft_bins, original_shape[-1]))
    
    def _unscale_grid_sample(self, spectrogram: torch.Tensor) -> torch.Tensor:
        # pack batch
        original_shape = spectrogram.size()
        spectrogram = spectrogram.reshape(-1, 1, original_shape[-2], original_shape[-1])

        # filter center freqs are evenly spaced in scaled domain (indices 1..num_filters of the +2 boundary points)
        scaled_freqs = np.linspace(self.scale_fn(self.freq_min), self.scale_fn(self.freq_max), self.num_filters + 2)
        stft_freqs = self.scale_fn(np.linspace(0, self.sample_rate / 2, self.num_stft_bins))

        grid_y = torch.from_numpy((stft_freqs - scaled_freqs[1]) / (scaled_freqs[-2] - scaled_freqs[1]) * 2 - 1)
        grid_y = grid_y.to(dtype=spectrogram.dtype, device=spectrogram.device)
        grid_x = torch.linspace(-1, 1, original_shape[-1], dtype=spectrogram.dtype, device=spectrogram.device)

        grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing="ij")
        grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).expand(spectrogram.shape[0], -1, -1, -1)

        unscaled = torch.nn.functional.grid_sample(spectrogram, grid, mode="bicubic", padding_mode="border", align_corners=True)

        # unpack batch
        return unscaled.squeeze(1).view(original_shape[:-2] + (self.num_stft_bins, original_shape[-1]))

    def unscale(self, spectrogram: torch.Tensor, rectify: bool = True) -> torch.Tensor:
        
        if self.unscale_mode == "lstsq":
            unscaled = self._unscale_lstsq(spectrogram)
        elif self.unscale_mode == "grid_sample":
            unscaled = self._unscale_grid_sample(spectrogram)

        if rectify == True:
            unscaled = torch.relu(unscaled)

        return unscaled
    
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
        elif self.filter_shape == "triangular+":
            filters = _create_triangular_filterbank_plus(stft_freqs, unscaled_freqs)
        elif self.filter_shape == "triangular++":
            filters = _create_triangular_filterbank_plus_plus(stft_freqs, unscaled_freqs)
        else:
            raise ValueError(f"Invalid filter shape: {self.filter_shape}")
        
        if self.filter_norm == "slaney":
            # slaney-style mel is scaled to be approx constant energy per channel
            enorm = 2. / (unscaled_freqs[2:self.num_filters+2] - unscaled_freqs[:self.num_filters])
            filters *= enorm.unsqueeze(0)

        elif self.filter_norm == "l2": # plays nicer with end-to-end training through the filterbank
            filters /= filters.pow(2).mean(dim=0, keepdim=True).pow(0.5)
        
        return filters