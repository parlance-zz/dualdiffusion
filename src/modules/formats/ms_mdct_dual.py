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
import torchaudio

from modules.formats.format import DualDiffusionFormat, DualDiffusionFormatConfig
from modules.formats.frequency_scale import FrequencyScale, get_mel_density
from utils.dual_diffusion_utils import tensor_to_img
from utils.mclt import mclt, imclt, WindowFunction


@dataclass()
class MS_MDCT_DualFormatConfig(DualDiffusionFormatConfig):

    sample_rate: int = 32000
    num_raw_channels: int = 2
    default_raw_length: int = 1408768

    # these values scale to unit norm for audio pre-normalized to -20 lufs
    raw_to_mel_spec_scale: float = 50
    raw_to_mel_spec_offset: float = 0
    mel_spec_to_mdct_psd_scale: float = 0.18
    mel_spec_to_mdct_psd_offset: float = 0
    mdct_to_raw_scale: float = 2
    raw_to_mdct_scale: float = 12.1

    mdct_window_len: int = 512
    mdct_window_func: Literal["sin", "kaiser_bessel_derived"] = "kaiser_bessel_derived"
    mdct_psd_num_bins: int = 2048
    mdct_dual_channel: bool = False

    ms_abs_exponent: float = 1
    ms_filter_shape: Literal["triangular", "cos"] = "triangular"    
    ms_freq_min: float = 0
    ms_width_alignment: int = 128
    ms_num_frequencies: int = 256
    ms_step_size_ms: int = 8
    ms_window_duration_ms: int = 128
    ms_padded_duration_ms: int = 128
    ms_window_exponent_low: float = 17
    ms_window_exponent_high: Optional[float] = 58
    ms_window_periodic: bool = True
    ms_window_func: Literal["hann", "blackman_harris"] = "blackman_harris"
    
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
    def _mel_spec_window(window_length: int, window_func: Literal["hann", "blackman_harris"], periodic: bool = True, *, dtype: torch.dtype = None,
                          layout: torch.layout = torch.strided, device: torch.device = None,
                          requires_grad: bool = False, exponent: float = 1) -> torch.Tensor:
        
        if window_func == "blackman_harris":
            return WindowFunction.blackman_harris(window_length, device=device) ** exponent
        elif window_func == "hann":
            return torch.hann_window(window_length, periodic=periodic, dtype=dtype,
                    layout=layout, device=device, requires_grad=requires_grad) ** exponent
        else:
            raise ValueError(f"Unsupported window function: {window_func}. Supported functions are 'hann' and 'blackman_harris'.")
    
    def __init__(self, config: MS_MDCT_DualFormatConfig) -> None:
        super().__init__()
        self.config = config

        # mel-scale spectrogram is built from 2 spectrograms with different windows
        # to maximize frequency resolution in lows and time resolution in highs

        self.ms_spectrogram_func_low = torchaudio.transforms.Spectrogram(
            n_fft=config.ms_frame_padded_length,
            win_length=config.ms_win_length,
            hop_length=config.ms_frame_hop_length,
            window_fn=MS_MDCT_DualFormat._mel_spec_window,
            power=1, normalized="window",
            wkwargs={
                "window_func": config.ms_window_func,
                "exponent": config.ms_window_exponent_low,
                "periodic": config.ms_window_periodic,
                "requires_grad": False
            },
            center=True, pad=0, pad_mode="reflect", onesided=True
        )

        if config.ms_window_exponent_high is not None:
            self.ms_spectrogram_func_high = torchaudio.transforms.Spectrogram(
                n_fft=config.ms_frame_padded_length,
                win_length=config.ms_win_length,
                hop_length=config.ms_frame_hop_length,
                window_fn=MS_MDCT_DualFormat._mel_spec_window,
                power=1, normalized="window",
                wkwargs={
                    "window_func": config.ms_window_func,
                    "exponent": config.ms_window_exponent_high,
                    "periodic": config.ms_window_periodic,
                    "requires_grad": False
                },
                center=True, pad=0, pad_mode="reflect", onesided=True
            )
        else:
            self.ms_spectrogram_func_high = None

        # this scaled is used for filtering to create the mel-scale spectrogram
        self.ms_freq_scale = FrequencyScale(
            freq_scale="mel",
            freq_min=config.ms_freq_min,
            freq_max=config.sample_rate / 2,
            sample_rate=config.sample_rate,
            num_stft_bins=config.ms_num_stft_bins,
            num_filters=config.ms_num_frequencies,
            filter_norm="slaney",
            filter_shape=config.ms_filter_shape,
        )

        # this scale is used for inverse filtering to create the conditioning for a mdct ddec model
        if config.mdct_psd_num_bins == self.config.ms_num_stft_bins - 1:
            self.ms_freq_scale_mdct_psd = None # if the size is close enough just crop the last bin out
        else:
            self.ms_freq_scale_mdct_psd = FrequencyScale(
                freq_scale="mel",
                freq_min=config.ms_freq_min,
                freq_max=config.sample_rate / 2,
                sample_rate=config.sample_rate,
                num_stft_bins=config.mdct_psd_num_bins,
                num_filters=config.ms_num_frequencies,
                filter_norm="slaney",
                filter_shape=config.ms_filter_shape,
            )

        ms_filter_freqs = self.ms_freq_scale.get_unscaled(config.ms_num_frequencies + 2)
        self.ms_lowest_filter_freq = ms_filter_freqs[1].item()
        self.register_buffer("ms_filter_freqs", ms_filter_freqs)

        ms_stft_hz = torch.linspace(0, config.sample_rate / 2, config.ms_num_stft_bins)
        self.register_buffer("ms_stft_hz", ms_stft_hz)
        self.register_buffer("ms_stft_mel_density", get_mel_density(ms_stft_hz).view(1, 1,-1, 1))

        mdct_hz = (torch.arange(config.mdct_num_frequencies) + 0.5) * config.sample_rate / config.mdct_window_len
        self.register_buffer("mdct_hz", mdct_hz)
        self.register_buffer("mdct_mel_density", get_mel_density(mdct_hz).view(1, 1,-1, 1))
        
        if config.ms_window_exponent_high is not None:
            spec_blend_hz = torch.linspace(0, config.sample_rate / 2, config.ms_num_stft_bins)
            spec_blend_mel_density = get_mel_density(spec_blend_hz)
            spec_blend_mel_density = (spec_blend_mel_density / spec_blend_mel_density.amax()) ** 2
            self.register_buffer("spec_blend_weight", spec_blend_mel_density.view(1, 1,-1, 1))
        else:
            self.spec_blend_weight = None
    
    def _high_pass(self, raw_samples: torch.Tensor) -> torch.Tensor:
        
        cutoff_freq = self.config.ms_freq_min

        if cutoff_freq <= 0 or (self.ms_lowest_filter_freq - cutoff_freq) <= 0:
            return raw_samples
        
        raw_len = raw_samples.shape[-1]

        raw_samples = torch.nn.functional.pad(raw_samples.float(), (raw_len // 2, raw_len // 2), mode="reflect")
        rfft: torch.Tensor = torch.fft.rfft(raw_samples, dim=-1, norm="ortho")
        rfft_freq = torch.fft.rfftfreq(raw_samples.shape[-1], d=1/self.config.sample_rate, device=raw_samples.device)

        filter: torch.Tensor = (rfft_freq - cutoff_freq) / (self.ms_lowest_filter_freq - cutoff_freq)
        filter = filter.clip(min=0., max=1.).view(1, 1,-1)

        raw_samples = torch.fft.irfft(rfft * filter, n=raw_samples.shape[-1], dim=-1, norm="ortho")
        return raw_samples[..., raw_len//2:-raw_len//2]
    
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
    def raw_to_mel_spec(self, raw_samples: torch.Tensor, use_slicing: bool = False) -> torch.Tensor:
        
        raw_samples = self._high_pass(raw_samples)

        if use_slicing == True:
            mel_spec = []
            for raw_sample in raw_samples.unbind(0):
                spec_low = self.ms_spectrogram_func_low(raw_sample.unsqueeze(0).float())
                if self.ms_spectrogram_func_high is not None:
                    spec_high = self.ms_spectrogram_func_high(raw_sample.unsqueeze(0).float())
                    spec_blended = spec_low * self.spec_blend_weight + spec_high * (1 - self.spec_blend_weight)
                else:
                    spec_blended = spec_low

                mel_spec.append(self.ms_freq_scale.scale(spec_blended / self.ms_stft_mel_density)
                                ** self.config.ms_abs_exponent + self.config.raw_to_mel_spec_offset)

            return torch.cat(mel_spec, dim=0) * self.config.raw_to_mel_spec_scale
        else:
            mel_spec_low = self.ms_spectrogram_func_low(raw_samples.float())
            if self.ms_spectrogram_func_high is not None:
                mel_spec_high = self.ms_spectrogram_func_high(raw_samples.float())
                spec_blended = mel_spec_low * self.spec_blend_weight + mel_spec_high * (1 - self.spec_blend_weight)
            else:
                spec_blended = mel_spec_low

            return (self.ms_freq_scale.scale(spec_blended / self.ms_stft_mel_density)
                ** self.config.ms_abs_exponent * self.config.raw_to_mel_spec_scale + self.config.raw_to_mel_spec_offset)

    def mel_spec_to_mdct_psd(self, mel_spec: torch.Tensor):

        mel_spec = mel_spec - self.config.raw_to_mel_spec_offset

        if self.ms_freq_scale_mdct_psd is None:
            mel_spec_mdct_psd = self.ms_freq_scale.unscale(
                mel_spec.float().clip(min=0) ** (1 / self.config.ms_abs_exponent), rectify=False)[:, :, :-1, :] * self.config.mel_spec_to_mdct_psd_scale
        else:
            mel_spec_mdct_psd = self.ms_freq_scale_mdct_psd.unscale(
                mel_spec.float().clip(min=0) ** (1 / self.config.ms_abs_exponent), rectify=False) * self.config.mel_spec_to_mdct_psd_scale
        
        return mel_spec_mdct_psd + self.config.mel_spec_to_mdct_psd_offset
    
    @torch.inference_mode()
    def mel_spec_to_img(self, mel_spec: torch.Tensor):
        mel_spec = mel_spec - self.config.raw_to_mel_spec_offset
        return tensor_to_img(mel_spec.clip(min=0)**(0.25 / self.config.ms_abs_exponent), flip_y=True)
    
    # **************** mdct methods ****************

    def _get_mdct_raw_crop_width(self, raw_length: Optional[int] = None) -> int:
        block_width = self.config.mdct_window_len
        raw_length = raw_length or self.config.default_raw_length
        return raw_length // block_width // self.config.ms_width_alignment * self.config.ms_width_alignment * block_width + block_width
    
    def get_mdct_shape(self, bsz: int = 1, raw_length: Optional[int] = None):
        raw_crop_width = self.get_raw_crop_width(raw_length=raw_length)
        num_mdct_bins = self.config.mdct_num_frequencies
        num_mdct_frames = raw_crop_width // num_mdct_bins + 1
        num_channels = self.config.num_raw_channels * (2 if self.config.mdct_dual_channel else 1)
        return (bsz, num_channels, num_mdct_bins, num_mdct_frames,)

    @torch.no_grad()
    def raw_to_mdct(self, raw_samples: torch.Tensor, random_phase_augmentation: bool = False) -> torch.Tensor:
        
        raw_samples = self._high_pass(raw_samples)

        _mclt = mclt(raw_samples.float(), self.config.mdct_window_len, self.config.mdct_window_func, 1).permute(0, 1, 3, 2)
        if random_phase_augmentation == True:
            phase_rotation = torch.exp(2j * torch.pi * torch.rand(_mclt.shape[0], device=_mclt.device)) 
            _mclt *= phase_rotation.view(-1, 1, 1, 1)

        if self.config.mdct_dual_channel == True:
            _mclt = torch.cat((_mclt.real, _mclt.imag), dim=1)
            return _mclt.contiguous(memory_format=self.memory_format) / self.mdct_mel_density * self.config.raw_to_mdct_scale
        else:
            return _mclt.real.contiguous(memory_format=self.memory_format) / self.mdct_mel_density * self.config.raw_to_mdct_scale
    
    @torch.no_grad()
    def raw_to_mdct_psd(self, raw_samples: torch.Tensor) -> torch.Tensor:

        raw_samples = self._high_pass(raw_samples)

        _mclt = mclt(raw_samples.float(), self.config.mdct_window_len, self.config.mdct_window_func, 1).permute(0, 1, 3, 2)
        return _mclt.abs().contiguous(memory_format=self.memory_format) / self.mdct_mel_density * self.config.raw_to_mdct_scale / 2**0.5
    
    @torch.no_grad()
    def mdct_to_raw(self, mdct: torch.Tensor) -> torch.Tensor:

        mdct = mdct * self.mdct_mel_density / self.config.raw_to_mdct_scale
        if self.config.mdct_dual_channel == True:
            mdct = torch.complex(*mdct.chunk(2, dim=1))

        raw_samples = imclt(mdct.permute(0, 1, 3, 2).contiguous(),
            window_fn=self.config.mdct_window_func, window_degree=1).real.contiguous()
        
        return raw_samples * self.config.mdct_to_raw_scale
    
    @torch.inference_mode()
    def mdct_psd_to_img(self, mdct_psd: torch.Tensor):
        return tensor_to_img(mdct_psd.clip(min=0)**0.25, flip_y=True)