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
from dataclasses import dataclass

import torch
import torchaudio

from modules.formats.format import DualDiffusionFormat, DualDiffusionFormatConfig
from modules.formats.frequency_scale import FrequencyScale, get_mel_density
from utils.dual_diffusion_utils import tensor_to_img
from utils.mclt import mclt, imclt


@dataclass()
class MS_MDCT_DualFormatConfig(DualDiffusionFormatConfig):

    sample_rate: int = 32000
    num_raw_channels: int = 2
    default_raw_length: int = 1408768

    # these values scale to unit norm for audio pre-normalized to -20 lufs
    raw_to_mel_spec_scale: float = 50.0
    mel_spec_to_mdct_psd_scale: float = 0.18
    mdct_to_raw_scale: float = 2.0
    raw_to_mdct_scale: float = 12.1

    mdct_window_len: int = 512
    mdct_psd_num_bins: int = 2048
    
    ms_width_alignment: int = 128
    ms_num_frequencies: int = 256
    ms_step_size_ms: int = 8
    ms_window_duration_ms: int = 128
    ms_padded_duration_ms: int = 128
    ms_window_exponent_low: float = 17
    ms_window_exponent_high: float = 58
    ms_window_periodic: bool = True
    
    @property
    def mdct_num_frequencies(self) -> int:
        return self.mdct_window_len // 2
    
    @property
    def ms_num_stft_bins(self) -> int:
        return self.ms_frame_padded_length // 2 + 1
    
    @property
    def ms_frame_padded_length(self) -> int:
        return int(self.ms_padded_duration_ms / 1000. * self.sample_rate)

    @property
    def ms_win_length(self) -> int:
        return int(self.ms_window_duration_ms / 1000. * self.sample_rate)

    @property
    def ms_frame_hop_length(self) -> int:
        return int(self.ms_step_size_ms / 1000. * self.sample_rate)

class MS_MDCT_DualFormat(DualDiffusionFormat):

    @staticmethod
    def _hann_power_window(window_length: int, periodic: bool = True, *, dtype: torch.dtype = None,
                          layout: torch.layout = torch.strided, device: torch.device = None,
                          requires_grad: bool = False, exponent: float = 1.) -> torch.Tensor:
        
        window = torch.hann_window(window_length*2, periodic=periodic, dtype=dtype,
                layout=layout, device=device, requires_grad=requires_grad) ** exponent
        return window[window_length//2:-window_length//2]
    
    def __init__(self, config: MS_MDCT_DualFormatConfig) -> None:
        super().__init__()
        self.config = config

        self.ms_spectrogram_func_low = torchaudio.transforms.Spectrogram(
            n_fft=config.ms_frame_padded_length,
            win_length=config.ms_win_length,
            hop_length=config.ms_frame_hop_length,
            window_fn=MS_MDCT_DualFormat._hann_power_window,
            power=1, normalized="window",
            wkwargs={
                "exponent": config.ms_window_exponent_low,
                "periodic": config.ms_window_periodic,
                "requires_grad": False
            },
            center=True, pad=0, pad_mode="reflect", onesided=True
        )

        self.ms_spectrogram_func_high = torchaudio.transforms.Spectrogram(
            n_fft=config.ms_frame_padded_length,
            win_length=config.ms_win_length,
            hop_length=config.ms_frame_hop_length,
            window_fn=MS_MDCT_DualFormat._hann_power_window,
            power=1, normalized="window",
            wkwargs={
                "exponent": config.ms_window_exponent_high,
                "periodic": config.ms_window_periodic,
                "requires_grad": False
            },
            center=True, pad=0, pad_mode="reflect", onesided=True
        )

        self.ms_freq_scale = FrequencyScale(
            freq_scale="mel",
            freq_min=0,
            freq_max=config.sample_rate / 2,
            sample_rate=config.sample_rate,
            num_stft_bins=config.ms_num_stft_bins,
            num_filters=config.ms_num_frequencies,
            filter_norm="slaney"
        )

        self.ms_freq_scale_mdct_psd = FrequencyScale(
            freq_scale="mel",
            freq_min=0,
            freq_max=config.sample_rate / 2,
            sample_rate=config.sample_rate,
            num_stft_bins=config.mdct_psd_num_bins,
            num_filters=config.ms_num_frequencies,
            filter_norm="slaney"
        )

        ms_stft_hz = torch.linspace(0, config.sample_rate / 2, config.ms_num_stft_bins)
        self.register_buffer("ms_stft_hz", ms_stft_hz)
        self.register_buffer("ms_stft_mel_density", get_mel_density(ms_stft_hz).view(1, 1,-1, 1))

        mdct_hz = (torch.arange(config.mdct_num_frequencies) + 0.5) * config.sample_rate / config.mdct_window_len
        self.register_buffer("mdct_hz", mdct_hz)
        self.register_buffer("mdct_mel_density", get_mel_density(mdct_hz).view(1, 1,-1, 1))
        
        spec_blend_hz = torch.linspace(0, config.sample_rate / 2, config.ms_num_stft_bins)
        spec_blend_mel_density = get_mel_density(spec_blend_hz)
        spec_blend_mel_density = (spec_blend_mel_density / spec_blend_mel_density.amax()) ** 2
        self.register_buffer("spec_blend_weight", spec_blend_mel_density.view(1, 1,-1, 1))
    
    # **************** mel-scale spectrogram methods ****************

    def _get_ms_shape(self, raw_shape: tuple[int, int, int]) -> tuple[int, int, int, int]:
        num_frames = 1 + (raw_shape[-1] + self.config.ms_frame_padded_length - self.config.ms_win_length) // self.config.ms_frame_hop_length
        return raw_shape[:-1] + (self.config.ms_num_frequencies, num_frames)
    
    def _get_ms_raw_shape(self, mel_spec_shape: tuple[int, int, int, int]) -> tuple[int, int, int]:
        audio_len = (mel_spec_shape[-1] - 1) * self.config.ms_frame_hop_length + self.config.ms_win_length - self.config.ms_frame_padded_length
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
        mel_spec = []
        for raw_sample in raw_samples.unbind(0):
            spec_low = self.ms_spectrogram_func_low(raw_sample.unsqueeze(0).float())
            spec_high = self.ms_spectrogram_func_high(raw_sample.unsqueeze(0).float())
            spec_blended = spec_low * self.spec_blend_weight + spec_high * (1 - self.spec_blend_weight)
            mel_spec.append(self.ms_freq_scale.scale(spec_blended / self.ms_stft_mel_density))

        return torch.cat(mel_spec, dim=0) * self.config.raw_to_mel_spec_scale

    def mel_spec_to_mdct_psd(self, mel_spec: torch.Tensor):
        return self.ms_freq_scale_mdct_psd.unscale(mel_spec.float(), rectify=False) * self.config.mel_spec_to_mdct_psd_scale
    
    @torch.inference_mode()
    def mel_spec_to_img(self, mel_spec: torch.Tensor):
        return tensor_to_img(mel_spec.clip(min=0)**0.5, flip_y=True)
    
    # **************** mdct methods ****************

    def _get_mdct_raw_crop_width(self, raw_length: Optional[int] = None) -> int:
        block_width = self.config.mdct_window_len
        raw_length = raw_length or self.config.default_raw_length
        return raw_length // block_width // self.config.ms_width_alignment * self.config.ms_width_alignment * block_width + block_width
    
    def get_mdct_shape(self, bsz: int = 1, raw_length: Optional[int] = None):
        raw_crop_width = self._get_mdct_raw_crop_width(raw_length=raw_length)
        num_mdct_bins = self.config.mdct_num_frequencies
        num_mdct_frames = raw_crop_width // num_mdct_bins - 2
        return (bsz, self.config.num_raw_channels, num_mdct_bins, num_mdct_frames,)

    @torch.no_grad()
    def raw_to_mdct(self, raw_samples: torch.Tensor, random_phase_augmentation: bool = False) -> torch.Tensor:

        _mclt = mclt(raw_samples.float(), self.config.mdct_window_len, "hann", 0.5).permute(0, 1, 3, 2)
        if random_phase_augmentation == True:
            phase_rotation = torch.exp(2j * torch.pi * torch.rand(_mclt.shape[0], device=_mclt.device)) 
            _mclt *= phase_rotation.view(-1, 1, 1, 1)

        return _mclt.real.contiguous() / self.mdct_mel_density * self.config.raw_to_mdct_scale
    
    @torch.no_grad()
    def raw_to_mdct_psd(self, raw_samples: torch.Tensor) -> torch.Tensor:
        _mclt = mclt(raw_samples.float(), self.config.mdct_window_len, "hann", 0.5).permute(0, 1, 3, 2)
        return _mclt.abs() / self.mdct_mel_density * self.config.raw_to_mdct_scale / 2**0.5
    
    @torch.no_grad()
    def mdct_to_raw(self, mdct: torch.Tensor) -> torch.Tensor:
        raw_samples = imclt((mdct * self.mdct_mel_density / self.config.raw_to_mdct_scale).permute(0, 1, 3, 2).contiguous(),
                            window_fn="hann", window_degree=0.5).real.contiguous()
        
        return raw_samples * self.config.mdct_to_raw_scale
    
    @torch.inference_mode()
    def mdct_psd_to_img(self, mdct: torch.Tensor):
        return tensor_to_img(mdct, flip_y=True)