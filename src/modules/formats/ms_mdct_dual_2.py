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
from dataclasses import dataclass

import torch
import numpy as np

from modules.formats.format import DualDiffusionFormat, DualDiffusionFormatConfig
from modules.formats.frequency_scale import FrequencyScale, get_mel_density
from utils.dual_diffusion_utils import tensor_to_img
from utils.mdct import MDCT, IMDCT, sin_window, kaiser_bessel_derived, vorbis


@dataclass()
class MS_MDCT_DualFormatConfig(DualDiffusionFormatConfig):

    sample_rate: int = 32000
    num_raw_channels: int = 2
    default_raw_length: int = 1408768

    # mdct params
    raw_to_mdct_scale: float = 0.00395184212251821011433253029603

    mdct_psd_scale: float    = 0.07179056842448940381561506832112
    mdct_psd_offset: float   = -0.1806843343919556
    mdct_psd_exponent: float = 0.25
    mdct_phase_scale: float  = 1

    mdct_window_len: int = 512
    mdct_window_func: Literal["sin", "kaiser_bessel_derived", "vorbis"] = "sin"

    @property
    def mdct_num_frequencies(self) -> int:
        return self.mdct_window_len // 2
    
    @property
    def mdct_frame_hop_length(self) -> int:
        return self.mdct_window_len // 2

    # mel-spec params
    raw_to_mel_spec_scale: float  = 0.48693139085749312574067728443989
    raw_to_mel_spec_offset: float = -1.530891040808645

    mel_spec_to_linear_scale: float  = 15.11100987193986714324861053997
    mel_spec_to_linear_offset: float = 0

    ms_abs_exponent: float = 0.25
    ms_freq_min: float = 0 #25
    ms_num_filters: int = 256
    ms_ideal_num_filter_bins: float = 3
    ms_window_length: int = 4096
    ms_blend_sharpness: float = 30
    ms_window_exponents: list[float] = (9, 32, 112)
    
    @property
    def ms_num_stft_bins(self) -> int:
        return self.ms_window_length // 2 + 1
    
    @property
    def ms_hop_length(self) -> int:
        return self.mdct_frame_hop_length
    
    @property
    def ms_width_alignment(self) -> int:
        return self.mdct_frame_hop_length // 2
    
    @property
    def ms_freq_max(self) -> int:
        return self.sample_rate / 2

class MS_MDCT_DualFormat(DualDiffusionFormat):
    
    def __init__(self, config: MS_MDCT_DualFormatConfig) -> None:
        super().__init__()
        self.config = config

        # ***** mel-scale spectrogram setup *****

        ms_windows = []
        hann_window = torch.hann_window(self.config.ms_window_length, periodic=True, requires_grad=False)
        for i in range(len(self.config.ms_window_exponents)):
            ms_windows.append(hann_window ** self.config.ms_window_exponents[i])
            
        ms_windows = torch.stack(ms_windows, dim=0)
        ms_windows /= ms_windows.pow(2).mean(dim=1, keepdim=True).pow(0.5)
        self.ms_windows: torch.Tensor
        self.register_buffer("ms_windows", ms_windows)

        self.ms_freq_scale = FrequencyScale(
            freq_scale="mel",
            freq_min=self.config.ms_freq_min,
            freq_max=self.config.ms_freq_max,
            sample_rate=self.config.sample_rate,
            num_stft_bins=self.config.ms_num_stft_bins,
            num_filters=self.config.ms_num_filters,
            filter_norm="slaney",
            filter_shape="triangular"
        )

        mel_freqs = self.ms_freq_scale.get_unscaled(self.config.ms_num_filters + 2)

        ms_filter_center_hz = mel_freqs[1:-1]
        self.ms_filter_center_hz: torch.Tensor
        self.register_buffer("ms_filter_center_hz", ms_filter_center_hz)

        ms_filter_bandwidths = mel_freqs[2:] - mel_freqs[:-2]
        self.ms_filter_bandwidths: torch.Tensor
        self.register_buffer("ms_filter_bandwidths", ms_filter_bandwidths)

        ms_num_filter_bins = ms_filter_bandwidths / self.config.sample_rate * self.config.ms_num_stft_bins * 2
        ms_ideal_filter_widths = (self.config.ms_ideal_num_filter_bins / ms_num_filter_bins * self.config.ms_window_length).to(dtype=torch.float64)
        self.ms_ideal_filter_widths: torch.Tensor
        self.register_buffer("ms_ideal_filter_widths", ms_ideal_filter_widths)

        ms_filters = self.ms_freq_scale.get_filters()
        ms_filters /= ms_filters.pow(2).mean(dim=0, keepdim=True).pow(0.5)
        self.ms_filters: torch.Tensor
        self.register_buffer("ms_filters", ms_filters)

        ms_window_widths = torch.tensor([2 * np.arccos(2**(-1/exponent)) / np.pi * 2 * self.config.ms_window_length
                                         for exponent in self.config.ms_window_exponents], dtype=torch.float64)
        self.ms_window_widths: torch.Tensor
        self.register_buffer("ms_window_widths", ms_window_widths)
        
        ms_filter_window_weights = torch.zeros((self.config.ms_num_filters, ms_windows.shape[0]), dtype=torch.float32)
        for i in range(self.config.ms_num_filters):
            window_weights = (-self.config.ms_blend_sharpness * (ms_ideal_filter_widths[i] / ms_window_widths).log()**2).exp()
            ms_filter_window_weights[i] = (window_weights / window_weights.sum()).to(dtype=torch.float32)

        self.ms_filter_window_weights: torch.Tensor
        self.register_buffer("ms_filter_window_weights", ms_filter_window_weights)

        ms_stft_hz = torch.linspace(0, config.sample_rate / 2, config.ms_num_stft_bins)
        self.ms_stft_mel_density: torch.Tensor
        self.register_buffer("ms_stft_mel_density", get_mel_density(ms_stft_hz).view(1, 1,-1, 1))

        # ***** mdct setup *****
        self.mdct_hz: torch.Tensor
        self.mdct_mel_density: torch.Tensor

        mdct_hz = (torch.arange(config.mdct_num_frequencies) + 0.5) * config.sample_rate / config.mdct_window_len
        self.register_buffer("mdct_hz", mdct_hz)
        self.register_buffer("mdct_mel_density", get_mel_density(mdct_hz).view(1, 1,-1, 1))

        if config.mdct_window_func == "sin":
            mdct_window_fn = sin_window
        elif config.mdct_window_func == "kaiser_bessel_derived":
            mdct_window_fn = kaiser_bessel_derived
        elif config.mdct_window_func == "vorbis":
            mdct_window_fn = vorbis
        else:
            raise ValueError(f"Unsupported mdct window function: {config.mdct_window_func}. Supported functions are 'sin', 'kaiser_bessel_derived', and 'vorbis'.")
        
        self.mdct = MDCT(win_length=config.mdct_window_len, window_fn=mdct_window_fn, return_complex=True)
        self.imdct = IMDCT(win_length=config.mdct_window_len, window_fn=mdct_window_fn)

    # **************** mel-scale spectrogram methods ****************

    def _get_ms_shape(self, raw_shape: tuple[int, int, int]) -> tuple[int, int, int, int]:
        num_frames = 1 + raw_shape[-1] // self.config.ms_hop_length
        return raw_shape[:-1] + (self.config.ms_num_filters, num_frames)
    
    def _get_ms_raw_shape(self, mel_spec_shape: tuple[int, int, int, int]) -> tuple[int, int, int]:
        audio_len = (mel_spec_shape[-1] - 1) * self.config.ms_hop_length
        return mel_spec_shape[:-2] + (audio_len,)

    def get_raw_crop_width(self, raw_length: Optional[int] = None) -> int:
        raw_length = raw_length or self.config.default_raw_length
        mel_spec_len = self._get_ms_shape((1, raw_length))[-1]
        mel_spec_len = mel_spec_len // self.config.ms_width_alignment * self.config.ms_width_alignment
        return self._get_ms_raw_shape((1, mel_spec_len))[-1]

    def get_mel_spec_shape(self, bsz: int = 1, raw_length: Optional[int] = None) -> tuple[int, int, int, int]:
        raw_crop_width = self.get_raw_crop_width(raw_length)
        return self._get_ms_shape((bsz, self.config.num_raw_channels, raw_crop_width))
    
    @torch.no_grad()
    def raw_to_mel_spec(self, raw_samples: torch.Tensor) -> torch.Tensor:
        
        ms_blended = None

        for i in range(len(self.config.ms_window_exponents)):
            packed_raw = raw_samples.view(raw_samples.shape[0] * raw_samples.shape[1], raw_samples.shape[2])

            stft = torch.stft(packed_raw, n_fft=self.config.ms_window_length, hop_length=self.config.ms_hop_length,
                win_length=self.config.ms_window_length, window=self.ms_windows[i], center=True,
                pad_mode="reflect", normalized=True, onesided=True, return_complex=True).abs()
            
            stft = stft.view(raw_samples.shape[0], raw_samples.shape[1], stft.shape[1], stft.shape[2]) / self.ms_stft_mel_density
            mel_spec = torch.matmul(stft.transpose(-1,-2), self.ms_filters).transpose(-1, -2)
            mel_spec *= self.ms_filter_window_weights[:, i].view(1, 1,-1, 1)
            
            if ms_blended is None: ms_blended = mel_spec
            else: ms_blended += mel_spec
        
        return (ms_blended ** self.config.ms_abs_exponent + self.config.raw_to_mel_spec_offset) / self.config.raw_to_mel_spec_scale
    
    @torch.no_grad()
    def mel_spec_to_linear(self, mel_spec: torch.Tensor) -> torch.Tensor:
        ms_linear = (mel_spec * self.config.raw_to_mel_spec_scale - self.config.raw_to_mel_spec_offset).clip(min=0) ** (1 / self.config.ms_abs_exponent)
        return (ms_linear + self.config.mel_spec_to_linear_offset) / self.config.mel_spec_to_linear_scale

    @torch.inference_mode()
    def mel_spec_to_img(self, mel_spec: torch.Tensor, use_colormap: bool = False):
        if use_colormap == True:
            return tensor_to_img(mel_spec.mean(dim=(0,1)), flip_y=True, colormap=True)
        else:
            return tensor_to_img(mel_spec, flip_y=True)
    
    @torch.inference_mode()
    def mel_spec_linear_to_img(self, mel_spec_linear: torch.Tensor):
        return tensor_to_img(mel_spec_linear.clip(min=0), flip_y=True)

    # **************** mdct methods ****************

    def _get_mdct_raw_crop_width(self, raw_length: Optional[int] = None) -> int:
        block_width = self.config.mdct_window_len
        raw_length = raw_length or self.config.default_raw_length
        return raw_length // block_width // self.config.ms_width_alignment * self.config.ms_width_alignment * block_width + block_width
    
    def get_mdct_shape(self, bsz: int = 1, raw_length: Optional[int] = None):
        raw_crop_width = self.get_raw_crop_width(raw_length=raw_length)
        num_mdct_bins = self.config.mdct_num_frequencies
        num_mdct_frames = raw_crop_width // num_mdct_bins + 1
        return (bsz, self.config.num_raw_channels, num_mdct_bins, num_mdct_frames,)
    
    def raw_to_mdct(self, raw_samples: torch.Tensor, random_phase_augmentation: bool = False) -> torch.Tensor:

        _mclt: torch.Tensor = self.mdct(raw_samples.float())
        if random_phase_augmentation == True:
            phase_rotation = torch.exp(2j * torch.pi * torch.rand(_mclt.shape[0], device=_mclt.device)) 
            _mclt *= phase_rotation.view(-1, 1, 1, 1)

        return _mclt.real.contiguous() / self.mdct_mel_density / self.config.raw_to_mdct_scale
    
    def mdct_to_raw(self, mdct: torch.Tensor) -> torch.Tensor:

        mdct = mdct * self.mdct_mel_density * self.config.raw_to_mdct_scale
        raw_samples = self.imdct(mdct).real.contiguous()
        
        return raw_samples
    
    def normalize_psd(self, mdct_psd: torch.Tensor) -> torch.Tensor:
        return (mdct_psd + self.config.mdct_psd_offset) / self.config.mdct_psd_scale
    
    def unnormalize_psd(self, norm_mdct_psd: torch.Tensor) -> torch.Tensor:
        return norm_mdct_psd * self.config.mdct_psd_scale - self.config.mdct_psd_offset
    
    def normalize_phase(self, mdct_phase: torch.Tensor) -> torch.Tensor:
        return mdct_phase / self.config.mdct_phase_scale
    
    def unnormalize_phase(self, norm_mdct_phase: torch.Tensor) -> torch.Tensor:
        return norm_mdct_phase * self.config.mdct_phase_scale

    def raw_to_mdct_phase_psd(self, raw_samples: torch.Tensor, random_phase_augmentation: bool = False) -> tuple[torch.Tensor, torch.Tensor]:

        _mclt: torch.Tensor = self.mdct(raw_samples.float())
        if random_phase_augmentation == True:
            phase_rotation = torch.exp(2j * torch.pi * torch.rand(_mclt.shape[0], device=_mclt.device)) 
            _mclt *= phase_rotation.view(-1, 1, 1, 1)

        mdct_psd = _mclt.abs()
        mdct_phase = (_mclt.real / mdct_psd.clip(min=1e-20)).clip(min=-1, max=1)

        mdct_psd = (mdct_psd / self.mdct_mel_density).pow(self.config.mdct_psd_exponent)
        mdct_phase = mdct_phase * 2**0.5

        return self.normalize_phase(mdct_phase), self.normalize_psd(mdct_psd)
    
    def mdct_phase_psd_to_raw(self, mdct_phase: torch.Tensor, mdct_psd: torch.Tensor) -> torch.Tensor:

        mdct_psd = self.unnormalize_psd(mdct_psd)
        mdct_phase = self.unnormalize_phase(mdct_phase)

        raise NotImplementedError()

        raw_samples = self.imdct(mdct_phase * mdct_psd).real.contiguous()
        return raw_samples
    
    @torch.no_grad()
    def mdct_psd_to_img(self, mdct_psd: torch.Tensor):
        return tensor_to_img(mdct_psd, flip_y=True)


if __name__ == "__main__":

    from utils import config

    import os

    from utils.dual_diffusion_utils import load_audio, tensor_to_img, save_img


    format = MS_MDCT_DualFormat(MS_MDCT_DualFormatConfig())

    output_path = os.path.join(config.DEBUG_PATH, "ms_mdct_dual_2")
    os.makedirs(output_path, exist_ok=True)

    equispaced_window_exponents = torch.linspace(np.log(format.config.ms_window_exponents[0]),
        np.log(format.config.ms_window_exponents[-1]), steps=len(format.config.ms_window_exponents)).exp()
    
    print("window exponents:", format.config.ms_window_exponents)
    print("equi-spaced window exponents:", equispaced_window_exponents)
    print("window widths:", format.ms_window_widths)
    print("total window weights:", format.ms_filter_window_weights.sum(dim=0))
    
    _ms_windows = format.ms_windows / format.ms_windows.amax(dim=1, keepdim=True)
    _ms_windows.cpu().numpy().tofile(os.path.join(output_path, f"ms_windows.raw"))

    ms_filters = format.ms_filters
    ms_filters.T.cpu().numpy().tofile(os.path.join(output_path, f"ms_filters.raw"))

    for i in range(format.config.ms_num_filters):
        filter_center_hz = format.ms_filter_center_hz[i]
        ideal_filter_width = format.ms_ideal_filter_widths[i]
        filter_window_weights = format.ms_filter_window_weights[i]
        print(f"filter {i}: hz: {filter_center_hz:.1f} ideal width: {ideal_filter_width:.1f}, window weights: {filter_window_weights}")

    format.ms_stft_mel_density.cpu().numpy().tofile(os.path.join(output_path, f"ms_stft_mel_density.raw"))

    raw_sample = load_audio(os.path.join(config.DEBUG_PATH, "new_test3.flac")).unsqueeze(0)

    ms_blended = None

    for i in range(len(format.config.ms_window_exponents)):

        packed_raw = raw_sample.view(raw_sample.shape[0] * raw_sample.shape[1], raw_sample.shape[2])

        stft = torch.stft(packed_raw, n_fft=format.config.ms_window_length, hop_length=format.config.ms_hop_length,
            win_length=format.config.ms_window_length, window=format.ms_windows[i], center=True,
            pad_mode="reflect", normalized=True, onesided=True, return_complex=True).abs()
        
        stft = stft.view(raw_sample.shape[0], raw_sample.shape[1], stft.shape[1], stft.shape[2]) / format.ms_stft_mel_density
        mel_spec = torch.matmul(stft.transpose(-1,-2), format.ms_filters).transpose(-1, -2)

        melspec_img = tensor_to_img(mel_spec.clip(min=0)**0.25, flip_y=True)
        save_img(melspec_img, os.path.join(output_path, f"mel_spectrogram_{i}.png"))

        mel_spec *= format.ms_filter_window_weights[:, i].view(1, 1,-1, 1)

        if ms_blended is None: ms_blended = mel_spec
        else: ms_blended += mel_spec

    ms_blended = ms_blended ** format.config.ms_abs_exponent * format.config.raw_to_mel_spec_scale + format.config.raw_to_mel_spec_offset
    print("mel_spec_blended_norm:", torch.linalg.vector_norm(ms_blended) / ms_blended.numel()**0.5)
    print("mel_spec_blended_mean:", ms_blended.mean())

    melspec_img = format.mel_spec_to_img(ms_blended)
    save_img(melspec_img, os.path.join(output_path, f"mel_spectrogram_blended.png"))

    ms_blended_linear = format.mel_spec_to_linear(ms_blended)
    print("mel_spec_blended_linear_norm:", torch.linalg.vector_norm(ms_blended_linear) / ms_blended_linear.numel()**0.5)
    print("mel_spec_blended_linear_mean:", ms_blended_linear.mean())
    melspec_linear_img = format.mel_spec_linear_to_img(ms_blended_linear)
    save_img(melspec_linear_img, os.path.join(output_path, f"mel_spectrogram_blended_linear.png"))

    mdct_psd = format.raw_to_mdct_psd(raw_sample)
    mdct_psd_img = tensor_to_img(mdct_psd.clip(min=0), flip_y=True)
    save_img(mdct_psd_img, os.path.join(output_path, f"mdct_psd.png"))