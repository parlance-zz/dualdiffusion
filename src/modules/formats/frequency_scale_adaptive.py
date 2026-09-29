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
from logging import getLogger

import numpy as np
import torch


def _hz_to_mel(freq: float | np.ndarray | torch.Tensor) -> float | np.ndarray | torch.Tensor:
    return 2595.0 * np.log10(1.0 + (freq / 700.0))

def _hz_to_adaptive(
    hz: torch.Tensor | np.ndarray | float,
    *,
    alpha: float,
    mel_min: float,
    mel_max: float,
    hz_min: float,
    hz_max: float,
):
    # normalized mel + normalized linear convex blend
    mel = _hz_to_mel(hz)
    mel_norm = (mel - mel_min) / (mel_max - mel_min)
    lin_norm = (hz - hz_min) / (hz_max - hz_min)
    return (1.0 - alpha) * mel_norm + alpha * lin_norm

def _adaptive_to_hz(
    s: torch.Tensor,
    *,
    alpha: float,
    mel_min: float,
    mel_max: float,
    hz_min: float,
    hz_max: float,
    max_iter: int = 40,
    tol: float = 1e-7,
) -> torch.Tensor:
    # monotone inverse via bisection
    lo = torch.full_like(s, float(hz_min))
    hi = torch.full_like(s, float(hz_max))

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = _hz_to_adaptive(
            mid,
            alpha=alpha,
            mel_min=mel_min,
            mel_max=mel_max,
            hz_min=hz_min,
            hz_max=hz_max,
        )
        go_right = f_mid < s
        lo = torch.where(go_right, mid, lo)
        hi = torch.where(go_right, hi, mid)
        if (hi - lo).max() <= tol:
            break

    return 0.5 * (lo + hi)

@torch.no_grad()
def _create_uniform_coverage_triangles(all_freqs: torch.Tensor, f_pts: torch.Tensor) -> torch.Tensor:
    """
    Standard triangular bank + row normalization.
    No special casing of first/last/any filter.
    Ensures per-bin total coverage is uniform where support exists.
    """
    f_diff = (f_pts[1:] - f_pts[:-1]).clamp_min(torch.finfo(all_freqs.dtype).eps)
    slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)

    zero = torch.zeros(1, device=all_freqs.device, dtype=all_freqs.dtype)
    down = (-slopes[:, :-2]) / f_diff[:-1]
    up = slopes[:, 2:] / f_diff[1:]
    fb = torch.maximum(zero, torch.minimum(down, up))  # (n_freqs, n_filters)

    fb[-1, -1] = 1

    # Uniform per-bin coverage
    row_sum = fb.sum(dim=1, keepdim=True)
    eps = torch.finfo(fb.dtype).eps
    fb = torch.where(row_sum > eps, fb / row_sum, fb)
    return fb

class FrequencyScale(torch.nn.Module):
    def __init__(
        self,
        alpha: Optional[float] = None,
        freq_min: float = 0.0,
        freq_max: Optional[float] = None,
        sample_rate: int = 32000,
        num_stft_bins: int = 3201,
        num_filters: int = 256,
        unscale_mode: str = "lstsq",
        unscale_lstsq_driver: str = "gels",
        adaptive_min_bins_per_filter: float = 1.0,
    ) -> None:
        super().__init__()

        self.freq_min = freq_min
        self.freq_max = freq_max or sample_rate / 2
        self.sample_rate = sample_rate
        self.num_stft_bins = num_stft_bins
        self.num_filters = num_filters
        self.unscale_mode = unscale_mode
        self.unscale_lstsq_driver = unscale_lstsq_driver
        self.adaptive_min_bins_per_filter = adaptive_min_bins_per_filter

        assert unscale_mode in ["lstsq", "grid_sample"], f"Invalid unscale_mode: {unscale_mode}"
        assert unscale_lstsq_driver in ["gels", "gelsy", "gelsd", "gelss"], f"Invalid unscale_lstsq_driver: {unscale_lstsq_driver}"

        self._adaptive_mel_min = float(_hz_to_mel(self.freq_min))
        self._adaptive_mel_max = float(_hz_to_mel(self.freq_max))
        self._adaptive_alpha = alpha if alpha is not None else self._solve_minimum_alpha()

        self.filters: torch.Tensor
        self.register_buffer("filters", self.get_filters(), persistent=False)

        if (self.filters.max(dim=0).values == 0.0).any():
            getLogger().warning("At least one filter is all-zero (unexpected).")

    def _scaled(self, hz):
        return _hz_to_adaptive(
            hz,
            alpha=self._adaptive_alpha,
            mel_min=self._adaptive_mel_min,
            mel_max=self._adaptive_mel_max,
            hz_min=self.freq_min,
            hz_max=self.freq_max,
        )

    def _unscaled(self, s: torch.Tensor):
        return _adaptive_to_hz(
            s,
            alpha=self._adaptive_alpha,
            mel_min=self._adaptive_mel_min,
            mel_max=self._adaptive_mel_max,
            hz_min=self.freq_min,
            hz_max=self.freq_max,
        )

    def _min_spacing_in_bins(self, alpha: float) -> float:
        u = torch.linspace(0.0, 1.0, self.num_filters + 2, dtype=torch.float64)
        f_pts = _adaptive_to_hz(
            u,
            alpha=alpha,
            mel_min=self._adaptive_mel_min,
            mel_max=self._adaptive_mel_max,
            hz_min=self.freq_min,
            hz_max=self.freq_max,
        )
        bin_hz = (self.sample_rate / 2) / (self.num_stft_bins - 1)
        return float(((f_pts[1:] - f_pts[:-1]) / bin_hz).min().item())

    def _solve_minimum_alpha(self) -> float:
        target = float(self.adaptive_min_bins_per_filter)

        if self._min_spacing_in_bins(0.0) >= target:
            return 0.0
        if self._min_spacing_in_bins(1.0) < target:
            getLogger().warning(
                f"Even linear spacing cannot satisfy min {target:.3f} bins/filter; using alpha=1.0."
            )
            return 1.0

        lo, hi = 0.0, 1.0
        for _ in range(36):
            mid = 0.5 * (lo + hi)
            if self._min_spacing_in_bins(mid) >= target:
                hi = mid
            else:
                lo = mid
        return hi

    def scale(self, specgram: torch.Tensor) -> torch.Tensor:
        return torch.matmul(specgram.transpose(-1, -2), self.filters).transpose(-1, -2)

    def _unscale_lstsq(self, spectrogram: torch.Tensor) -> torch.Tensor:
        original_shape = spectrogram.size()
        spectrogram = spectrogram.reshape(-1, original_shape[-2], original_shape[-1])

        unscaled = torch.linalg.lstsq(
            self.filters.transpose(-1, -2)[None],
            spectrogram,
            driver=self.unscale_lstsq_driver,
        ).solution

        return unscaled.view(original_shape[:-2] + (self.num_stft_bins, original_shape[-1]))

    def _unscale_grid_sample(self, spectrogram: torch.Tensor) -> torch.Tensor:
        original_shape = spectrogram.size()
        spectrogram = spectrogram.reshape(-1, 1, original_shape[-2], original_shape[-1])

        scaled_freqs = np.linspace(self._scaled(self.freq_min), self._scaled(self.freq_max), self.num_filters + 2)
        stft_freqs = self._scaled(np.linspace(0, self.sample_rate / 2, self.num_stft_bins))

        grid_y = torch.from_numpy((stft_freqs - scaled_freqs[1]) / (scaled_freqs[-2] - scaled_freqs[1]) * 2 - 1)
        grid_y = grid_y.to(dtype=spectrogram.dtype, device=spectrogram.device)
        grid_x = torch.linspace(-1, 1, original_shape[-1], dtype=spectrogram.dtype, device=spectrogram.device)

        grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing="ij")
        grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).expand(spectrogram.shape[0], -1, -1, -1)

        unscaled = torch.nn.functional.grid_sample(
            spectrogram, grid, mode="bicubic", padding_mode="border", align_corners=True
        )
        return unscaled.squeeze(1).view(original_shape[:-2] + (self.num_stft_bins, original_shape[-1]))

    def unscale(self, spectrogram: torch.Tensor, rectify: bool = True) -> torch.Tensor:
        if self.unscale_mode == "lstsq":
            unscaled = self._unscale_lstsq(spectrogram)
        else:
            unscaled = self._unscale_grid_sample(spectrogram)

        return torch.relu(unscaled) if rectify else unscaled

    def get_unscaled(self, num_points: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Return f_pts with virtual boundaries on BOTH ends:
        - left virtual point:  p0 = 2*p1 - p2   (typically < 0 Hz)
        - right virtual point: pN = 2*pN-1 - pN-2 (typically > Nyquist)
        Real interior points are adaptive-spaced on [freq_min, freq_max].

        This yields natural half-triangles at both ends without per-filter special casing.
        """
        assert num_points >= 4, "need at least 4 points (2 virtual + >=2 real)"

        # num_points includes 2 virtual endpoints
        n_real = num_points - 2

        s_min = float(self._scaled(self.freq_min))
        s_max = float(self._scaled(self.freq_max))
        s_real = torch.linspace(s_min, s_max, n_real, device=device)
        p_real = self._unscaled(s_real)  # length n_real

        # strict monotonicity guard on real points
        eps_hz = torch.tensor(1e-6, device=device, dtype=p_real.dtype)
        p_real = torch.cummax(p_real, dim=0).values
        p_real[1:] = torch.maximum(p_real[1:], p_real[:-1] + eps_hz)

        # virtual left boundary
        p_left = 2.0 * p_real[0] - p_real[1]
        p_left = torch.minimum(p_left, p_real[0] - eps_hz)

        # virtual right boundary
        p_right = 2.0 * p_real[-1] - p_real[-2]
        p_right = torch.maximum(p_right, p_real[-1] + eps_hz)

        return torch.cat([p_left.unsqueeze(0), p_real, p_right.unsqueeze(0)], dim=0)

    @torch.no_grad()
    def get_filters(self) -> torch.Tensor:
        stft_freqs = torch.linspace(0, self.sample_rate / 2, self.num_stft_bins)
        unscaled_freqs = self.get_unscaled(self.num_filters + 2)

        filters = _create_uniform_coverage_triangles(stft_freqs, unscaled_freqs)
        
        return filters