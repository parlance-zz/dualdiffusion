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

from typing import Literal, Optional, Union
from dataclasses import dataclass

import torch
import torchaudio

from .format import DualDiffusionFormat, DualDiffusionFormatConfig
from .frequency_scale import FrequencyScale, get_mel_density
from .phase_recovery import PhaseRecovery

@dataclass()
class SpectrogramFormatConfig(DualDiffusionFormatConfig):

    # these values are for audio pre-normalized to -20 lufs
    raw_to_sample_scale: float = 2.247
    sample_to_raw_scale: float = 0.445
    sample_mean: float = 1.295
    abs_exp1_scale: float      = 0.008
    abs_exp1_mel_density: bool = False
    unscaled_psd_scale: float  = 0.625
    unscaled_psd_mel_density: bool = False
    unscaled_psd_num_fft_bins: int = 3328
    unscaled_psd_rectify: bool = True

    abs_exponent: float = 0.25

    # FFT parameters
    sample_rate: int = 32000
    step_size_ms: int = 8

    # hann ** 32 window settings
    window_duration_ms: int = 200
    padded_duration_ms: int = 200
    window_exponent: float = 32
    window_periodic: bool = True

    # hann ** 0.5 window settings
    #window_exponent: float = 0.5
    #window_duration_ms: int = 30
    #padded_duration_ms: int = 200

    # freq scale params
    freq_scale_type: Literal["mel", "log"] = "mel"
    num_frequencies: int = 256
    min_frequency: int = 20
    max_frequency: int = 16000
    freq_scale_norm: Optional[str] = None

    # phase recovery params
    num_fgla_iters: int = 200
    fgla_momentum: float = 0.99
    stereo_coherence: float = 0.67

    @property
    def stereo(self) -> bool:
        return self.sample_raw_channels == 2
    
    @property
    def num_stft_bins(self) -> int:
        return self.padded_length // 2 + 1
    
    @property
    def padded_length(self) -> int:
        return int(self.padded_duration_ms / 1000.0 * self.sample_rate)

    @property
    def win_length(self) -> int:
        return int(self.window_duration_ms / 1000.0 * self.sample_rate)

    @property
    def hop_length(self) -> int:
        return int(self.step_size_ms / 1000.0 * self.sample_rate)

class SpectrogramConverter(torch.nn.Module):

    @staticmethod
    def hann_power_window(window_length: int, periodic: bool = True, *, dtype: torch.dtype = None,
                          layout: torch.layout = torch.strided, device: torch.device = None,
                          requires_grad: bool = False, exponent: float = 1.) -> torch.Tensor:
        
        return torch.hann_window(window_length, periodic=periodic, dtype=dtype,
                                 layout=layout, device=device, requires_grad=requires_grad) ** exponent

    @torch.no_grad()
    def __init__(self, config: SpectrogramFormatConfig) -> None:
        super().__init__()
        self.config = config
        
        window_args = {
            "exponent": config.window_exponent,
            "periodic": config.window_periodic,
        }

        self.spectrogram_func = torchaudio.transforms.Spectrogram(
            n_fft=config.padded_length,
            win_length=config.win_length,
            hop_length=config.hop_length,
            pad=0,
            window_fn=SpectrogramConverter.hann_power_window,
            power=None,
            normalized=False,
            wkwargs=window_args,
            center=True,
            pad_mode="reflect",
            onesided=True,
        )

        self.inverse_spectrogram_func = PhaseRecovery(
            n_fft=config.padded_length,
            n_fgla_iter=config.num_fgla_iters,
            win_length=config.win_length,
            hop_length=config.hop_length,
            window_fn=SpectrogramConverter.hann_power_window,
            wkwargs=window_args,
            momentum=config.fgla_momentum,
            length=None,
            rand_init=False,#True,
            stereo=config.stereo,
            stereo_coherence=config.stereo_coherence
        )

        self.freq_scale = FrequencyScale(
            freq_scale=config.freq_scale_type,
            freq_min=config.min_frequency,
            freq_max=config.max_frequency,
            sample_rate=config.sample_rate,
            num_stft_bins=config.num_stft_bins,
            num_filters=config.num_frequencies,
            filter_norm=config.freq_scale_norm,
        )

        self.freq_scale_psd = FrequencyScale(
            freq_scale=config.freq_scale_type,
            freq_min=config.min_frequency,
            freq_max=config.max_frequency,
            sample_rate=config.sample_rate,
            num_stft_bins=config.unscaled_psd_num_fft_bins,
            num_filters=config.num_frequencies,
            filter_norm=config.freq_scale_norm,
        )

    def get_spectrogram_shape(self, audio_shape: torch.Size) -> torch.Size:
        num_frames = 1 + (audio_shape[-1] + self.config.padded_length - self.config.win_length) // self.config.hop_length
        return torch.Size(audio_shape[:-1] + (self.config.num_frequencies, num_frames))
    
    def get_audio_shape(self, spectrogram_shape: torch.Size) -> torch.Size:
        audio_len = (spectrogram_shape[-1] - 1) * self.config.hop_length + self.config.win_length - self.config.padded_length
        return torch.Size(spectrogram_shape[:-2] + (audio_len,))

    def sample_raw_crop_width(self, audio_len: int) -> int:
        spectrogram_len = self.get_spectrogram_shape(torch.Size((1, audio_len)))[-1] // 128 * 128
        return self.get_audio_shape(torch.Size((1, spectrogram_len)))[-1]

    @torch.inference_mode()
    def audio_to_spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        spectrogram_complex = self.spectrogram_func(audio)
        return self.freq_scale.scale(spectrogram_complex.abs()) ** self.config.abs_exponent

    @torch.inference_mode()
    def spectrogram_to_audio(self, spectrogram: torch.Tensor, n_fgla_iters: Optional[int] = None, quiet: bool = False) -> torch.Tensor:
        amplitudes_linear = self.freq_scale.unscale(spectrogram ** (1 / self.config.abs_exponent))
        return self.inverse_spectrogram_func(
            amplitudes_linear, n_fgla_iter=n_fgla_iters or self.config.num_fgla_iters, quiet=quiet)

    @torch.inference_mode()
    def audio_to_unscaled_spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        spectrogram_complex = self.spectrogram_func(audio)
        return spectrogram_complex.abs() ** self.config.abs_exponent

    @torch.inference_mode()
    def unscaled_spectrogram_to_audio(self, spectrogram: torch.Tensor, n_fgla_iters: Optional[int] = None, quiet: bool = False) -> torch.Tensor:
        amplitudes_linear = spectrogram ** (1 / self.config.abs_exponent)
        return self.inverse_spectrogram_func(
            amplitudes_linear, n_fgla_iter=n_fgla_iters or self.config.num_fgla_iters, quiet=quiet)
    
class SpectrogramFormat(DualDiffusionFormat):

    def __init__(self, config: SpectrogramFormatConfig) -> None:
        super().__init__()
        self.config = config
        self.spectrogram_converter = SpectrogramConverter(config)
    
    def sample_raw_crop_width(self, length: Optional[int] = None) -> int:
        return self.spectrogram_converter.sample_raw_crop_width(length or self.config.sample_raw_length)
    
    def get_sample_shape(self, bsz: int = 1, length: Optional[int] = None) -> tuple:

        num_output_channels = self.config.sample_raw_channels
        crop_width = self.sample_raw_crop_width(length=length)
        audio_shape = torch.Size((bsz, num_output_channels, crop_width))
        
        spectrogram_shape = self.spectrogram_converter.get_spectrogram_shape(audio_shape)
        return tuple(spectrogram_shape)

    @torch.inference_mode()
    def raw_to_sample(self, raw_samples: torch.Tensor, unscaled_spectrogram: bool = False) -> torch.Tensor:
        if unscaled_spectrogram == True:
            return (self.spectrogram_converter.audio_to_unscaled_spectrogram(raw_samples) - self.config.sample_mean) * self.config.raw_to_sample_scale
        else:
            #return self.spectrogram_converter.audio_to_spectrogram(raw_samples) * self.config.raw_to_sample_scale
            spec_samples = []
            for raw_sample in raw_samples.unbind(0):
                spec_samples.append(self.spectrogram_converter.audio_to_spectrogram(raw_sample.unsqueeze(0)) - self.config.sample_mean)
            return torch.cat(spec_samples, dim=0) * self.config.raw_to_sample_scale

    @torch.inference_mode()
    def sample_to_raw(self, samples: torch.Tensor, n_fgla_iters: Optional[int] = None,
                quiet: bool = False, unscaled_spectrogram: bool = False) -> torch.Tensor:
        
        samples = (samples / self.config.raw_to_sample_scale + self.config.sample_mean).clip(min=0)
        if unscaled_spectrogram == True:
            return self.spectrogram_converter.unscaled_spectrogram_to_audio(
                samples, n_fgla_iters=n_fgla_iters, quiet=quiet)
        else:
            return self.spectrogram_converter.spectrogram_to_audio(
                samples, n_fgla_iters=n_fgla_iters, quiet=quiet)
    
    @torch.no_grad()
    def get_ln_freqs(self, x: torch.Tensor) -> torch.Tensor:        
        ln_freqs = self.spectrogram_converter.freq_scale.get_unscaled(x.shape[2] + 2, device=x.device)[1:-1].log2()
        ln_freqs = ln_freqs.view(1, 1,-1, 1).repeat(x.shape[0], 1, 1, x.shape[3])
        return ((ln_freqs - ln_freqs.mean()) / ln_freqs.std()).to(x.dtype)
    
    @torch.inference_mode()
    def convert_to_abs_exp1(self, samples: torch.Tensor):
        
        samples = (samples / self.config.raw_to_sample_scale + self.config.sample_mean) * 2
        abs_exp1 = samples.clip(min=0) ** (1 / self.config.abs_exponent) * self.config.abs_exp1_scale

        if self.config.abs_exp1_mel_density == True:
            mel_freqs = self.spectrogram_converter.freq_scale.get_unscaled(
                self.config.num_frequencies + 2, device=samples.device)[1:-1]
            abs_exp1 /= get_mel_density(mel_freqs).view(1, 1,-1, 1)

        return abs_exp1

    def convert_to_unscaled_psd(self, samples: torch.Tensor):
        
        #samples = (samples / self.config.raw_to_sample_scale + self.config.sample_mean).clip(min=0)
        #unscaled_psd = self.spectrogram_converter.freq_scale.unscale(samples ** (1 / self.config.abs_exponent))
        #return unscaled_psd.clip(min=0) * self.config.unscaled_psd_scale

        if self.config.unscaled_psd_rectify == True:
            samples = (samples / self.config.raw_to_sample_scale + self.config.sample_mean).clip(min=0)
            unscaled_psd = self.spectrogram_converter.freq_scale_psd.unscale(samples ** (1 / self.config.abs_exponent))
        else:
            unscaled_psd = self.spectrogram_converter.freq_scale_psd.unscale(samples, rectify=False)

        if self.config.unscaled_psd_mel_density == True:
            fft_hz = torch.linspace(0, self.config.sample_rate / 2, self.config.unscaled_psd_num_fft_bins, device=unscaled_psd.device)
            unscaled_psd = unscaled_psd / get_mel_density(fft_hz).view(1, 1,-1, 1)

        return unscaled_psd * self.config.unscaled_psd_scale