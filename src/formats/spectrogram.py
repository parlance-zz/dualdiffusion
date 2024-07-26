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

import typing as T
from dataclasses import dataclass

import numpy as np
import torch
import torchaudio

from .phase_recovery import PhaseRecovery
from training.loss import DualMultiscaleSpectralLoss2D

@dataclass()
class SpectrogramParams:

    stereo: bool = True
    abs_exponent: float = 0.25

    # FFT parameters
    sample_rate: int = 32000
    step_size_ms: int = 8

    # hann ** 40 window settings
    window_duration_ms: int = 200
    padded_duration_ms: int = 200
    window_exponent: float = 32
    window_periodic: bool = True

    # hann ** 0.5 window settings
    #window_exponent: float = 0.5
    #window_duration_ms: int = 30
    #padded_duration_ms: int = 200

    # mel scale params
    use_mel_scale: bool = True
    num_frequencies: int = 256
    min_frequency: int = 20
    max_frequency: int = 16000
    mel_scale_norm: T.Optional[str] = None
    mel_scale_type: str = "htk"

    # phase recovery params
    num_griffin_lim_iters: int = 400
    num_dm_iters: int = 0
    fgla_momentum: float = 0.99
    dm_beta: float = 1
    stereo_coherence: float = 0.67

    @property
    def n_fft(self) -> int:
        return int(self.padded_duration_ms / 1000.0 * self.sample_rate)

    @property
    def win_length(self) -> int:
        return int(self.window_duration_ms / 1000.0 * self.sample_rate)

    @property
    def hop_length(self) -> int:
        return int(self.step_size_ms / 1000.0 * self.sample_rate)

class SpectrogramConverter(torch.nn.Module):

    @staticmethod
    def hann_power_window(window_length, periodic=True, *, dtype=None, layout=torch.strided, device=None, requires_grad=False, exponent=1):
        return torch.hann_window(window_length, periodic=periodic, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad) ** exponent

    def __init__(self, params: SpectrogramParams):
        super(SpectrogramConverter, self).__init__()
        self.p = params
        
        window_args = {
            "exponent": params.window_exponent,
            "periodic": params.window_periodic,
        }

        self.spectrogram_func = torchaudio.transforms.Spectrogram(
            n_fft=params.n_fft,
            hop_length=params.hop_length,
            win_length=params.win_length,
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
            n_fft=params.n_fft,
            n_fgla_iter=params.num_griffin_lim_iters,
            n_dm_iter=params.num_dm_iters,
            win_length=params.win_length,
            hop_length=params.hop_length,
            window_fn=SpectrogramConverter.hann_power_window,
            wkwargs=window_args,
            momentum=params.fgla_momentum,
            beta=params.dm_beta,
            length=None,
            rand_init=True,
            stereo=params.stereo,
            stereo_coherence=params.stereo_coherence
        )

        if params.use_mel_scale:
            self.mel_scaler = torchaudio.transforms.MelScale(
                n_mels=params.num_frequencies,
                sample_rate=params.sample_rate,
                f_min=params.min_frequency,
                f_max=params.max_frequency,
                n_stft=params.n_fft // 2 + 1,
                norm=params.mel_scale_norm,
                mel_scale=params.mel_scale_type,
            )

            self.inverse_mel_scaler = torchaudio.transforms.InverseMelScale(
                n_stft=params.n_fft // 2 + 1,
                n_mels=params.num_frequencies,
                sample_rate=params.sample_rate,
                f_min=params.min_frequency,
                f_max=params.max_frequency,
                norm=params.mel_scale_norm,
                mel_scale=params.mel_scale_type,
            )
        else:
            self.mel_scaler = self.inverse_mel_scaler = torch.nn.Identity()

    def get_spectrogram_shape(self, audio_shape: torch.Size) -> torch.Size:
        num_frames = 1 + (audio_shape[-1] + self.p.n_fft - self.p.win_length) // self.p.hop_length
        return torch.Size(audio_shape[:-1] + (self.p.num_frequencies, num_frames))
    
    def get_audio_shape(self, spectrogram_shape: torch.Size) -> torch.Size:
        audio_len = (spectrogram_shape[-1] - 1) * self.p.hop_length + self.p.win_length - self.p.n_fft
        return torch.Size(spectrogram_shape[:-2] + (audio_len,))

    def get_crop_width(self, audio_len: int) -> int:
        spectrogram_len = self.get_spectrogram_shape(torch.Size((1, audio_len)))[-1] // 64 * 64
        return self.get_audio_shape(torch.Size((1, spectrogram_len)))[-1]

    @torch.no_grad()    
    def audio_to_spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        spectrogram_complex = self.spectrogram_func(audio)
        return self.mel_scaler(spectrogram_complex.abs()) ** self.p.abs_exponent

    @torch.no_grad()
    def spectrogram_to_audio(self, spectrogram: torch.Tensor) -> torch.Tensor:
        amplitudes_linear = self.inverse_mel_scaler(spectrogram ** (1 / self.p.abs_exponent))
        return self.inverse_spectrogram_func(amplitudes_linear, n_fgla_iter=self.p.num_griffin_lim_iters)
    
    def half(self): # prevent casting to fp16
        return self

class DualSpectrogramFormat(torch.nn.Module):

    __constants__ = ["mels_min", "mels_max"]

    def __init__(self, model_params):
        super(DualSpectrogramFormat, self).__init__()

        self.model_params = model_params
        self.spectrogram_params = SpectrogramParams(sample_rate=model_params["sample_rate"],
                                                    stereo=model_params["sample_raw_channels"] == 2,
                                                    **model_params["spectrogram_params"])
        
        self.spectrogram_converter = SpectrogramConverter(self.spectrogram_params)
        self.loss = DualMultiscaleSpectralLoss2D(model_params["vae_training_params"])

        self.mels_min = DualSpectrogramFormat._hz_to_mel(self.spectrogram_params.min_frequency)
        self.mels_max = DualSpectrogramFormat._hz_to_mel(self.spectrogram_params.max_frequency)

        if self.spectrogram_params.mel_scale_type != "htk":
            raise NotImplementedError("Only HTK mel scale is supported")
        
    def get_sample_crop_width(self, length=0):
        if length <= 0: length = self.model_params["sample_raw_length"]
        return self.spectrogram_converter.get_crop_width(length)
    
    def get_num_channels(self):
        in_channels = out_channels = self.model_params["sample_raw_channels"]
        return (in_channels, out_channels)

    @torch.no_grad()
    def raw_to_sample(self, raw_samples, return_dict=False):
        
        noise_floor = self.model_params["noise_floor"]
        samples = self.spectrogram_converter.audio_to_spectrogram(raw_samples)
        samples /= samples.std(dim=(1,2,3), keepdim=True).clip(min=noise_floor)

        if return_dict:
            samples_dict = {
                "samples": samples,
                "raw_samples": raw_samples,
            }
            return samples_dict
        else:
            return samples

    def sample_to_raw(self, samples, return_dict=False, decode=True):
        
        if decode:
            raw_samples = self.spectrogram_converter.spectrogram_to_audio(samples.clip(min=0))
        else:
            raw_samples = None

        if not return_dict:         
            return raw_samples
        else:
            samples_dict = {
                "samples": samples,
                "raw_samples": raw_samples,
            }
            return samples_dict

    def get_loss(self, sample, target):
        return self.loss(sample, target, self.model_params)
    
    def get_sample_shape(self, bsz=1, length=0):
        _, num_output_channels = self.get_num_channels()
        crop_width = self.get_sample_crop_width(length=length)
        audio_shape = torch.Size((bsz, num_output_channels, crop_width))
        
        spectrogram_shape = self.spectrogram_converter.get_spectrogram_shape(audio_shape)
        return tuple(spectrogram_shape)

    @staticmethod    
    def _hz_to_mel(freq):
        return 2595. * np.log10(1. + (freq / 700.))

    @staticmethod
    def _mel_to_hz(mels):
        return 700. * (10. ** (mels / 2595.) - 1.)

    @torch.no_grad()
    def get_positional_embedding(self, x, t_ranges, mode="linear", num_fourier_channels=0):

        mels = torch.linspace(self.mels_min, self.mels_max, x.shape[2] + 2, device=x.device)[1:-1]
        ln_freqs = DualSpectrogramFormat._mel_to_hz(mels).log2()

        if mode == "linear":
            emb_freq = ln_freqs.view(1, 1,-1, 1).repeat(x.shape[0], 1, 1, x.shape[3])
            emb_freq = (emb_freq - emb_freq.mean()) / emb_freq.std()

            if t_ranges is not None:
                t = torch.linspace(0, 1, x.shape[3], device=x.device).view(-1, 1)
                t = ((1 - t) * t_ranges[:, 0] + t * t_ranges[:, 1]).permute(1, 0).view(x.shape[0], 1, 1, x.shape[3])
                emb_time = t.repeat(1, 1, x.shape[2], 1)

                return torch.cat((emb_freq, emb_time), dim=1).to(x.dtype)
            else:
                return emb_freq.to(x.dtype)

        elif mode == "fourier":
            raise NotImplementedError("Fourier positional embeddings not implemented")
        else:
            raise ValueError(f"Unknown mode '{mode}'")