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
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from .format import DualDiffusionFormat
from .frequency_scale import FrequencyScale
from .phase_recovery import PhaseRecovery

@dataclass()
class SpectrogramParams:

    stereo: bool = True
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
    freq_scale_type: Literal["mel", "log"] = "log"
    num_frequencies: int = 256
    min_frequency: int = 20
    max_frequency: int = 16000
    freq_scale_norm: Optional[str] = None

    # phase recovery params
    num_fgla_iters: int = 200
    fgla_momentum: float = 0.99
    stereo_coherence: float = 0.67

    @property
    def num_channels(self) -> int:
        return 2 if self.stereo else 1
    
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
    @torch.no_grad()
    def hann_power_window(window_length: int, periodic: bool = True, *, dtype: torch.dtype = None,
                          layout: torch.layout = torch.strided, device: torch.device = None,
                          requires_grad: bool = False, exponent: float = 1.) -> torch.Tensor:
        
        return torch.hann_window(window_length, periodic=periodic, dtype=dtype,
                                 layout=layout, device=device, requires_grad=requires_grad) ** exponent

    @torch.no_grad()
    def __init__(self, params: SpectrogramParams):
        super(SpectrogramConverter, self).__init__()
        self.p = params
        
        window_args = {
            "exponent": params.window_exponent,
            "periodic": params.window_periodic,
        }

        self.spectrogram_func = torchaudio.transforms.Spectrogram(
            n_fft=params.padded_length,
            win_length=params.win_length,
            hop_length=params.hop_length,
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
            n_fft=params.padded_length,
            n_fgla_iter=params.num_fgla_iters,
            win_length=params.win_length,
            hop_length=params.hop_length,
            window_fn=SpectrogramConverter.hann_power_window,
            wkwargs=window_args,
            momentum=params.fgla_momentum,
            length=None,
            rand_init=True,
            stereo=params.stereo,
            stereo_coherence=params.stereo_coherence
        )

        self.freq_scale = FrequencyScale(
            freq_scale=params.freq_scale_type,
            freq_min=params.min_frequency,
            freq_max=params.max_frequency,
            sample_rate=params.sample_rate,
            num_stft_bins=params.num_stft_bins,
            num_filters=params.num_frequencies,
            filter_norm=params.freq_scale_norm,
        )

    @torch.no_grad()
    def get_spectrogram_shape(self, audio_shape: torch.Size) -> torch.Size:
        num_frames = 1 + (audio_shape[-1] + self.p.padded_length - self.p.win_length) // self.p.hop_length
        return torch.Size(audio_shape[:-1] + (self.p.num_frequencies, num_frames))
    
    @torch.no_grad()
    def get_audio_shape(self, spectrogram_shape: torch.Size) -> torch.Size:
        audio_len = (spectrogram_shape[-1] - 1) * self.p.hop_length + self.p.win_length - self.p.padded_length
        return torch.Size(spectrogram_shape[:-2] + (audio_len,))

    @torch.no_grad()
    def get_raw_crop_width(self, audio_len: int) -> int:
        spectrogram_len = self.get_spectrogram_shape(torch.Size((1, audio_len)))[-1] // 64 * 64
        return self.get_audio_shape(torch.Size((1, spectrogram_len)))[-1]

    @torch.no_grad()    
    def audio_to_spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        spectrogram_complex = self.spectrogram_func(audio)
        return self.freq_scale.scale(spectrogram_complex.abs()) ** self.p.abs_exponent

    @torch.no_grad()
    def spectrogram_to_audio(self, spectrogram: torch.Tensor) -> torch.Tensor:
        amplitudes_linear = self.freq_scale.unscale(spectrogram ** (1 / self.p.abs_exponent))
        return self.inverse_spectrogram_func(amplitudes_linear, n_fgla_iter=self.p.num_fgla_iters)
    
    def half(self) -> "SpectrogramConverter": # prevent casting to fp16/bf16
        return self

class DualSpectrogramFormat(ModelMixin, ConfigMixin, DualDiffusionFormat):

    @register_to_config
    def __init__(self,
                 noise_floor: float = 2e-5,
                 sample_rate: int = 32000,
                 sample_raw_channels: int = 2,
                 sample_raw_length: int = 1057570,
                 t_scale: Optional[float] = None,
                 spectrogram_params: Optional[dict] = None
                 ) -> None:
        super(DualSpectrogramFormat, self).__init__()

        spectrogram_params = spectrogram_params or {}
        self.spectrogram_params = SpectrogramParams(**spectrogram_params)
        self.spectrogram_converter = SpectrogramConverter(self.spectrogram_params)
    
    @torch.no_grad()
    def get_raw_crop_width(self, length: Optional[int] = None) -> int:
        return self.spectrogram_converter.get_raw_crop_width(length or self.config["sample_raw_length"])
    
    @torch.no_grad()
    def get_num_channels(self) -> tuple[int, int]:
        in_channels = out_channels = self.spectrogram_params.num_channels
        return (in_channels, out_channels)
    
    @torch.no_grad()
    def get_sample_shape(self, bsz: int = 1, length: Optional[int] = None) -> tuple:

        _, num_output_channels = self.get_num_channels()
        crop_width = self.get_raw_crop_width(length=length)
        audio_shape = torch.Size((bsz, num_output_channels, crop_width))
        
        spectrogram_shape = self.spectrogram_converter.get_spectrogram_shape(audio_shape)
        return tuple(spectrogram_shape)

    @torch.no_grad()
    def raw_to_sample(self, raw_samples: torch.Tensor, return_dict: bool = False) -> Union[torch.Tensor, dict]:
        
        noise_floor = self.config["noise_floor"]
        samples = self.spectrogram_converter.audio_to_spectrogram(raw_samples)
        samples /= samples.std(dim=(1,2,3), keepdim=True).clip(min=noise_floor)

        if return_dict:
            return {"samples": samples, "raw_samples": raw_samples}
        else:
            return samples

    @torch.no_grad()
    def sample_to_raw(self, samples: torch.Tensor, return_dict: bool = False) -> Union[torch.Tensor, dict]:
        
        raw_samples = self.spectrogram_converter.spectrogram_to_audio(samples.clip(min=0))
        if return_dict:
            return {"raw_samples": raw_samples, "samples": samples}
        else:
            return raw_samples
    
    @torch.no_grad()
    def get_ln_freqs(self, x: torch.Tensor) -> torch.Tensor:

        ln_freqs = self.spectrogram_converter.freq_scale.get_unscaled(x.shape[2] + 2, device=x.device)[1:-1].log2()
        ln_freqs = ln_freqs.view(1, 1,-1, 1).repeat(x.shape[0], 1, 1, x.shape[3])

        return ((ln_freqs - ln_freqs.mean()) / ln_freqs.std()).to(x.dtype)