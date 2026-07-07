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

# this is an implementation of the transform described in the paper:
# "Multi-Window STFT Phase Retrieval Lattice Uniqueness" (https://arxiv.org/pdf/2207.10620) by PHILIPP GROHS, LUKAS LIEHR, AND MARTIN RATHMAIR
# TLDR: 4 magnitude spectrograms on the same sampling lattice with windows that are complex linear combinations of the first 2 hermite functions
#  are sufficient to uniquely determine the phase of a signal up to a global phase factor with only 2x density (for real signals)
#  these 4 spectrograms are used as conditioning for a diffusion decoder to synthesize an MDCT/MCLT representation of the full signal

from typing import Optional, Literal
from dataclasses import dataclass

import torch

from modules.formats.format import DualDiffusionFormat, DualDiffusionFormatConfig
from modules.formats.frequency_scale import get_mel_density
from modules.formats.frequency_scale_adaptive import FrequencyScale
from modules.formats.mel_spec import MelSpecConfig, MelSpec
from utils.dual_diffusion_utils import tensor_to_img
from utils.mdct import MDCT, IMDCT, sin_window, kaiser_bessel_derived, vorbis


def _h0(window_len: int, t_scale: float) -> torch.Tensor:
    t = torch.linspace(-t_scale, t_scale, window_len)
    return 2 ** (1/4) * (-torch.pi * t.pow(2)).exp()

def _h1(window_len: int, t_scale: float) -> torch.Tensor:
    t = torch.linspace(-t_scale, t_scale, window_len)
    return 2 ** (5/4) * torch.pi * t * (-torch.pi * t.pow(2)).exp()

def _get_mdct_window_func(mdct_window_func: str) -> callable:
    if mdct_window_func == "sin":
        mdct_window_fn = sin_window
    elif mdct_window_func == "kaiser_bessel_derived":
        mdct_window_fn = kaiser_bessel_derived
    elif mdct_window_func == "vorbis":
        mdct_window_fn = vorbis
    else:
        raise ValueError(f"Unsupported mdct window function: {mdct_window_func}. Supported functions are 'sin', 'kaiser_bessel_derived', and 'vorbis'.")
    return mdct_window_fn

@dataclass()
class MS_MDCT_DualFormatConfig(DualDiffusionFormatConfig):

    # raw audio format params
    sample_rate: int = 32000
    num_raw_channels: int = 2
    default_raw_length: int = 131072
    width_alignment: int    = 8192

    # mdct params
    mdct_psd_exponent: float = 0.25
    window_len: int = 512
    window_func: Literal["sin", "kaiser_bessel_derived", "vorbis"] = "vorbis"

    @property
    def num_frequencies(self) -> int:
        return self.window_len // 2
    
    @property
    def hop_length(self) -> int:
        return self.window_len // 2
        
    # ms psd params
    ms_psd_t_scale: float = 2.3
    ms_psd_window_len: int = 1023
    ms_psd_num_filters: int = 128
    ms_psd_p_real: list[tuple[float, float]] = ( (2,0), (1,1), (-1,1), (0,1) )
    ms_psd_p_imag: list[tuple[float, float]] = ( (0,0), (0,0), ( 0,0), (1,0) )

    @property
    def ms_psd_num_frequencies(self) -> int:
        return self.ms_psd_window_len // 2 + 1
    
    # mel-spec params    
    mel_spec_config: Optional[MelSpecConfig] = None

class MS_MDCT_DualFormat(DualDiffusionFormat):
    
    has_trainable_parameters: bool = True

    @torch.no_grad()
    def __init__(self, config: MS_MDCT_DualFormatConfig) -> None:
        super().__init__()
        self.config = config

        assert int(1/self.config.mdct_psd_exponent) == 1/self.config.mdct_psd_exponent, "mdct_psd_exponent must be the reciprocal of an integer"

        # ***** mel_spec setup *****

        if config.mel_spec_config is not None:
            self.mel_spec = MelSpec(config.mel_spec_config)

        # ***** ms psd setup *****

        self.ms_psd_win_h0: torch.Tensor; self.ms_psd_win_h1: torch.Tensor
        self.register_buffer("ms_psd_win_h0", _h0(config.ms_psd_window_len, config.ms_psd_t_scale), persistent=False)
        self.register_buffer("ms_psd_win_h1", _h1(config.ms_psd_window_len, config.ms_psd_t_scale), persistent=False)

        self.ms_psd_offset: torch.Tensor; self.ms_psd_scale: torch.Tensor
        self.register_buffer(f"ms_psd_offset", torch.zeros(config.ms_psd_num_frequencies).view(1, 1,-1, 1), persistent=True)
        self.register_buffer(f"ms_psd_scale",  torch.ones(config.ms_psd_num_frequencies).view(1, 1,-1, 1),  persistent=True)
        
        assert config.ms_psd_num_frequencies % config.num_frequencies == 0
        self.ms_psd_freq_scale = FrequencyScale(
            freq_min=0,
            freq_max=config.sample_rate / 2,
            sample_rate=config.sample_rate,
            num_stft_bins=config.ms_psd_num_frequencies,
            num_filters=config.ms_psd_num_filters
        )

        self.ms_psd_mel_scale: torch.Tensor; self.ms_psd_mel_offset: torch.Tensor
        self.register_buffer("ms_psd_mel_scale", torch.ones(config.ms_psd_num_filters).view(1, 1,-1, 1), persistent=True)
        self.register_buffer("ms_psd_mel_offset", torch.zeros(config.ms_psd_num_filters).view(1, 1,-1, 1), persistent=True)
        self.ms_psd_mel_unscaled_scale: torch.Tensor; self.ms_psd_mel_unscaled_offset: torch.Tensor
        self.register_buffer("ms_psd_mel_unscaled_scale", torch.ones(config.ms_psd_num_frequencies).view(1, 1,-1, 1), persistent=True)
        self.register_buffer("ms_psd_mel_unscaled_offset", torch.zeros(config.ms_psd_num_frequencies).view(1, 1,-1, 1), persistent=True)

        # ***** mdct setup *****

        mdct_window_func = _get_mdct_window_func(config.window_func)
        self.mdct = MDCT(win_length=config.window_len, window_fn=mdct_window_func, return_complex=True)
        self.imdct = IMDCT(win_length=config.window_len, window_fn=mdct_window_func)

        self.mdct_phase_scale: torch.Tensor
        self.mdct_psd_offset: torch.Tensor; self.mdct_psd_scale: torch.Tensor
        self.register_buffer(f"mdct_phase_scale", torch.ones(config.num_frequencies).view(1, 1,-1, 1), persistent=True)
        self.register_buffer(f"mdct_psd_offset", torch.zeros(config.num_frequencies).view(1, 1,-1, 1), persistent=True)
        self.register_buffer(f"mdct_psd_scale",  torch.ones(config.num_frequencies).view(1, 1,-1, 1),  persistent=True)
        
        self.mdct_mel_density: torch.Tensor
        mdct_hz = (torch.arange(config.num_frequencies) + 0.5) * config.sample_rate / config.window_len
        self.register_buffer(f"mdct_mel_density", get_mel_density(mdct_hz).view(1, 1,-1, 1), persistent=False)

    # **************** mel-scale spectrogram methods ****************

    def get_raw_crop_width(self, raw_length: Optional[int] = None) -> int:
        raw_length = raw_length or self.config.default_raw_length
        return raw_length // self.config.width_alignment * self.config.width_alignment - self.config.num_frequencies

    @torch.no_grad()
    def raw_to_ms_psd(self, raw_samples: torch.Tensor, min_psd_eps: Optional[float] = None) -> torch.Tensor:

        raw_samples = torch.cat((raw_samples, raw_samples[..., -1:]), dim=-1) # fix stft shape with odd window length
        packed_raw = raw_samples.float().view(raw_samples.shape[0] * raw_samples.shape[1], raw_samples.shape[2])

        stft_h0 = torch.stft(packed_raw, n_fft=self.config.ms_psd_window_len, hop_length=self.config.hop_length,
            win_length=self.config.ms_psd_window_len, window=self.ms_psd_win_h0, center=True,
            pad_mode="reflect", normalized=True, onesided=True, return_complex=True)
        stft_h1 = torch.stft(packed_raw, n_fft=self.config.ms_psd_window_len, hop_length=self.config.hop_length,
            win_length=self.config.ms_psd_window_len, window=self.ms_psd_win_h1, center=True,
            pad_mode="reflect", normalized=True, onesided=True, return_complex=True)
        
        stft_h0 = stft_h0.view(raw_samples.shape[0], raw_samples.shape[1], stft_h0.shape[1], stft_h0.shape[2])
        stft_h1 = stft_h1.view(raw_samples.shape[0], raw_samples.shape[1], stft_h1.shape[1], stft_h1.shape[2])
        
        ms_psds: list[torch.Tensor] = []
        for i in range(4):
            ms_psds.append((
                 self.config.ms_psd_p_real[i][0] * stft_h0 + self.config.ms_psd_p_real[i][1] * stft_h1 +
                (self.config.ms_psd_p_imag[i][0] * stft_h0 + self.config.ms_psd_p_imag[i][1] * stft_h1) * 1j
            ))

        ms_psd = torch.cat(ms_psds, dim=1).abs()
        if min_psd_eps is not None:
            ms_psd = ms_psd.clip(min=min_psd_eps)
        ms_psd = ms_psd.pow(self.config.mdct_psd_exponent)

        return (ms_psd + self.ms_psd_offset) / self.ms_psd_scale
    
    @torch.no_grad()
    def scale_ms_psd(self, ms_psd: torch.Tensor) -> torch.Tensor:
        ms_psd = ms_psd.float() * self.ms_psd_scale - self.ms_psd_offset
        ms_psd_mel: torch.Tensor = self.ms_psd_freq_scale.scale(ms_psd)
        return (ms_psd_mel + self.ms_psd_mel_offset) / self.ms_psd_mel_scale

    def unscale_ms_psd(self, ms_psd_mel: torch.Tensor) -> torch.Tensor:
        ms_psd_mel = ms_psd_mel.float() * self.ms_psd_mel_scale - self.ms_psd_mel_offset
        ms_psd = self.ms_psd_freq_scale.unscale(ms_psd_mel, rectify=False)
        return (ms_psd + self.ms_psd_mel_unscaled_offset) / self.ms_psd_mel_unscaled_scale

    @torch.no_grad()
    def ms_psd_to_img(self, ms_psd: torch.Tensor, use_colormap: bool = False):
        
        if ms_psd.shape[1] == 4 * self.config.num_raw_channels:
            ms_psd = torch.cat(ms_psd.chunk(4, dim=1), dim=2)

        if use_colormap == True:
            return tensor_to_img(ms_psd.mean(dim=(0,1)), flip_y=True, colormap=True)
        else:
            return tensor_to_img(ms_psd, flip_y=True)

    # **************** mdct methods ****************

    def get_mdct_phase_psd_shape(self, bsz: int = 1, raw_length: Optional[int] = None):
        raw_crop_width = self.get_raw_crop_width(raw_length=raw_length)
        num_mdct_bins = self.config.num_frequencies
        num_mdct_frames = raw_crop_width // num_mdct_bins + 1
        return (bsz, self.config.num_raw_channels * 2, num_mdct_bins, num_mdct_frames,)

    def get_mdct_shape(self, bsz: int = 1, raw_length: Optional[int] = None):
        return self.get_mdct_phase_psd_shape(bsz=bsz, raw_length=raw_length)

    def raw_to_mdct_phase_psd(self, raw_samples: torch.Tensor,
            random_phase_augmentation: bool = False) -> torch.Tensor:
        
        _mclt: torch.Tensor = self.mdct(raw_samples.float())

        if random_phase_augmentation == True:
            phase_rotation = torch.exp(2j * torch.pi * torch.rand(_mclt.shape[0], device=_mclt.device)) 
            _mclt *= phase_rotation.view(-1, 1, 1, 1)

        mdct_psd = _mclt.abs()
        mdct_phase = (_mclt.real / mdct_psd.clip(min=1e-20)).clip(min=-1, max=1)

        mdct_psd = mdct_psd.pow(self.config.mdct_psd_exponent)
        mdct_phase = mdct_phase * mdct_psd

        mdct_phase = mdct_phase / self.mdct_phase_scale
        mdct_psd = (mdct_psd + self.mdct_psd_offset) / self.mdct_psd_scale

        mdct_phase_psd = torch.cat((mdct_phase, mdct_psd), dim=1)
        return mdct_phase_psd
    
    def mdct_phase_psd_to_raw(self, mdct_phase_psd: torch.Tensor) -> torch.Tensor:

        mdct_phase, mdct_psd = torch.chunk(mdct_phase_psd.float(), 2, dim=1)
        mdct_phase = mdct_phase * self.mdct_phase_scale
        mdct_psd = mdct_psd * self.mdct_psd_scale - self.mdct_psd_offset

        #mdct_psd = mdct_psd.clip(min=0).pow(1 / self.config.mdct_psd_exponent - 1)
        recon_exp = int((1 / self.config.mdct_psd_exponent - 1) / 2) * 2 + 1
        mdct_psd = mdct_psd.pow(recon_exp)
        raw_samples = self.imdct(mdct_phase * mdct_psd).real.contiguous()
        return raw_samples
        
    def raw_to_mdct_psd(self, raw_samples: torch.Tensor) -> torch.Tensor:

        mdct_phase_psd = self.raw_to_mdct_phase_psd(raw_samples.float())
        _, mdct_psd = torch.chunk(mdct_phase_psd, 2, dim=1)
        return mdct_psd