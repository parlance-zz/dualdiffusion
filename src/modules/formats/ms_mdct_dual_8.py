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

from typing import Optional, Literal, Union
from dataclasses import dataclass

import torch

from modules.formats.format import DualDiffusionFormat, DualDiffusionFormatConfig
from modules.formats.frequency_scale import FrequencyScale, get_mel_density
from utils.dual_diffusion_utils import tensor_to_img
from utils.mdct import MDCT, IMDCT, sin_window, kaiser_bessel_derived, vorbis


def _combine_fft_crossover(xs: list[torch.Tensor], sample_rate: float, crossovers: list[float], crossfade_hz: float = 100.0) -> torch.Tensor:
    n = xs[0].shape[-1]
    f = torch.fft.rfftfreq(n, d=1 / sample_rate).to(xs[0].device)
    X = torch.stack([torch.fft.rfft(x, dim=-1) for x in xs])  # (bands, b, c, F)

    edges = [0.0, *crossovers]
    masks = []
    for i in range(len(xs)):
        if i == 0:
            m = (f < edges[1]).to(X.real.dtype)
        elif i == len(xs) - 1:
            m = (f >= edges[i]).to(X.real.dtype)
        else:
            m = ((f >= edges[i]) & (f < edges[i + 1])).to(X.real.dtype)
        masks.append(m)

    masks = torch.stack(masks)

    for k, fc in enumerate(crossovers, start=1):
        lo, hi = fc - crossfade_hz, fc + crossfade_hz
        region = (f >= lo) & (f <= hi)
        t = (f[region] - lo) / (hi - lo)
        left = 0.5 * (1 + torch.cos(torch.pi * t))
        masks[k - 1, region] = left
        masks[k, region] = 1 - left

    Y = (X * masks[:, None, None, :]).sum(0)
    return torch.fft.irfft(Y, n=n, dim=-1)

def _flat_top_window(x: torch.Tensor) -> torch.Tensor:
    return (0.21557895 - 0.41663158 * torch.cos(x) + 0.277263158 * torch.cos(2*x)
            - 0.083578947 * torch.cos(3*x) + 0.006947368 * torch.cos(4*x))

def flat_top_window(width: int) -> torch.Tensor:
    return _flat_top_window((torch.arange(width) + 0.5) / width  * 2 * torch.pi)

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
class MDCT_Config:

    window_len: int = 128
    window_func: Literal["sin", "kaiser_bessel_derived", "vorbis"] = "vorbis"

    @property
    def num_frequencies(self) -> int:
        return self.window_len // 2
    
    @property
    def hop_length(self) -> int:
        return self.window_len // 2
    
@dataclass()
class MS_PSD_Config:

    ms_num_filters: int = 256
    ms_hop_length: int = 256
    ms_window_length: int = 4096

    @property
    def ms_num_stft_bins(self) -> int:
        return self.ms_window_length // 2 + 1
    
@dataclass()
class MS_MDCT_DualFormatConfig(DualDiffusionFormatConfig):

    # raw audio format params
    sample_rate: int = 32000
    num_raw_channels: int = 2
    default_raw_length: int = 1408768
    width_alignment: int    = 4096

    # mdct params
    mdcts: list[MDCT_Config] = ()
    mdct_out_crossover_freqs: list[float] = (300, 600, 1200)
    mdct_out_crossover_width_hz: float = 100
    mdct_psd_exponent: float = 0.25

    @property
    def num_mdcts(self) -> int:
        return len(self.mdcts)

    # mel-spec params
    ms_add_center_channel: bool = False
    ms_img_show_center_channel: bool = True
    ms_abs_exponent: float = 0.25
    ms_freq_min: float = 0
    
    ms_psds: list[MS_PSD_Config] = None

    @property
    def ms_freq_max(self) -> int:
        return self.sample_rate / 2
        
    @property
    def num_ms_psds(self) -> int:
        return len(self.ms_psds) if self.ms_psds is not None else 0

class MS_MDCT_DualFormat(DualDiffusionFormat):
    
    has_trainable_parameters: bool = True

    @torch.no_grad()
    def __init__(self, config: MS_MDCT_DualFormatConfig) -> None:
        super().__init__()
        self.config = config

        assert int(1/self.config.mdct_psd_exponent) == 1/self.config.mdct_psd_exponent, "mdct_psd_exponent must be the reciprocal of an integer"

        # ***** mel-scale spectrogram setup *****

        self.ms_psd_freq_scales: torch.nn.ModuleList = torch.nn.ModuleList()
        self.ms_psd_linear_freq_scales: torch.nn.ModuleList = torch.nn.ModuleList()

        for i in range(self.config.num_ms_psds):

            window = flat_top_window(config.ms_psds[i].ms_window_length)
            window /= window.pow(2).mean().pow(0.5)
            self.register_buffer(f"ms_psd_window_{i}", window, persistent=False)    

            ms_freq_scale = FrequencyScale(
                freq_scale="mel",
                freq_min=config.ms_freq_min,
                freq_max=config.ms_freq_max,
                sample_rate=config.sample_rate,
                num_stft_bins=config.ms_psds[i].ms_num_stft_bins,
                num_filters=config.ms_psds[i].ms_num_filters,
                filter_norm="slaney",
                filter_shape="triangular"
            )
            self.ms_psd_freq_scales.append(ms_freq_scale)

            ms_linear_freq_scale = FrequencyScale(
                freq_scale="mel",
                freq_min=config.ms_freq_min,
                freq_max=config.ms_freq_max,
                sample_rate=config.sample_rate,
                num_stft_bins=config.ms_psds[i].ms_window_length,
                num_filters=config.ms_psds[i].ms_num_filters,
                filter_norm="slaney",
                filter_shape="triangular"
            )
            self.ms_psd_linear_freq_scales.append(ms_linear_freq_scale)

            self.register_buffer(f"ms_psd_scale_{i}", torch.ones(config.ms_psds[i].ms_num_filters).view(1, 1,-1, 1), persistent=True)
            self.register_buffer(f"ms_psd_offset_{i}", torch.zeros(config.ms_psds[i].ms_num_filters).view(1, 1,-1, 1), persistent=True)
            self.register_buffer(f"ms_psd_linear_scale_{i}", torch.ones(config.ms_psds[i].ms_window_length // 2).view(1, 1,-1, 1), persistent=True)
            self.register_buffer(f"ms_psd_linear_offset_{i}", torch.zeros(config.ms_psds[i].ms_window_length // 2).view(1, 1,-1, 1), persistent=True)
            
        # ***** mdct setup *****

        self.mdcts = torch.nn.ModuleList(); self.imdcts = torch.nn.ModuleList()
        for i in range(self.config.num_mdcts):
            
            if config.mdcts[i].window_len == config.mdcts[0].window_len:
                last_frame = -1
            elif config.mdcts[i].window_len > config.mdcts[0].window_len:
                last_frame = -2
            else:
                last_frame = None

            mdct_window_func = _get_mdct_window_func(config.mdcts[i].window_func)

            mdct = MDCT(win_length=config.mdcts[i].window_len, window_fn=mdct_window_func, return_complex=True, last_frame=last_frame)
            self.mdcts.append(mdct)
            imdct = IMDCT(win_length=config.mdcts[i].window_len, window_fn=mdct_window_func)
            self.imdcts.append(imdct)

            self.register_buffer(f"mdct_phase_scale_{i}", torch.ones(config.mdcts[i].num_frequencies).view(1, 1,-1, 1), persistent=True)
            self.register_buffer(f"mdct_psd_offset_{i}", torch.zeros(config.mdcts[i].num_frequencies).view(1, 1,-1, 1), persistent=True)
            self.register_buffer(f"mdct_psd_scale_{i}",  torch.ones(config.mdcts[i].num_frequencies).view(1, 1,-1, 1),  persistent=True)
            
            mdct_hz = (torch.arange(config.mdcts[i].num_frequencies) + 0.5) * config.sample_rate / config.mdcts[i].window_len
            self.register_buffer(f"mdct_mel_density_{i}", get_mel_density(mdct_hz).view(1, 1,-1, 1), persistent=False)

    # **************** mel-scale spectrogram methods ****************

    def get_raw_crop_width(self, raw_length: Optional[int] = None) -> int:
        raw_length = raw_length or self.config.default_raw_length
        return raw_length // self.config.width_alignment * self.config.width_alignment - self.config.mdcts[0].num_frequencies

    @torch.no_grad()
    def raw_to_ms_psd(self, raw_samples: torch.Tensor, level: int = 2) -> Union[list[torch.Tensor], torch.Tensor]:

        raw_samples = raw_samples.float()
        levels = list(range(self.config.num_ms_psds)) if level < 0 else [level]

        ms_psds: list[torch.Tensor] = []
        for i in levels:

            ms_window_length = self.config.ms_psds[i].ms_window_length
            ms_hop_length = self.config.ms_psds[i].ms_hop_length
            ms_window = getattr(self, f"ms_psd_window_{i}")

            packed_raw = raw_samples.view(raw_samples.shape[0] * raw_samples.shape[1], raw_samples.shape[2])

            stft = torch.stft(packed_raw, n_fft=ms_window_length, hop_length=ms_hop_length,
                win_length=ms_window_length, window=ms_window, center=True,
                pad_mode="reflect", normalized=True, onesided=True, return_complex=True)
            
            stft = stft.view(raw_samples.shape[0], raw_samples.shape[1], stft.shape[1], stft.shape[2])
            
            if self.config.ms_add_center_channel == True:
                stft = torch.cat((stft, (stft[:, 0:1] + stft[:, 1:2]) / 2), dim=1).abs()
            else:
                stft = stft.abs()

            ms_psd: torch.Tensor = self.ms_psd_freq_scales[i].scale(stft.pow(self.config.ms_abs_exponent))

            ms_psd_offset: torch.Tensor = getattr(self, f"ms_psd_offset_{i}")
            ms_psd_scale: torch.Tensor = getattr(self, f"ms_psd_scale_{i}")
            ms_psd = (ms_psd + ms_psd_offset) / ms_psd_scale
            
            ms_psds.append(ms_psd)

        if len(ms_psds) == 1:
            return ms_psds[0]
        else:
            return ms_psds
    
    def ms_psd_to_psd_linear(self, ms_psds: Union[list[torch.Tensor], torch.Tensor]) -> Union[list[torch.Tensor], torch.Tensor]:

        if not isinstance(ms_psds, list):
            ms_psds = [ms_psds]

        linear_psds: list[torch.Tensor] = []
        for i in range(len(ms_psds)):

            ms_psd_offset: torch.Tensor = getattr(self, f"ms_psd_offset_{i}")
            ms_psd_scale: torch.Tensor = getattr(self, f"ms_psd_scale_{i}")
            ms_psd = (ms_psds[i].float() * ms_psd_scale - ms_psd_offset)

            linear_psd = self.ms_psd_linear_freq_scales[i].unscale(ms_psd, rectify=False)
            linear_psd = torch.nn.functional.avg_pool2d(linear_psd, (2, 1))
            
            ms_psd_linear_scale: torch.Tensor = getattr(self, f"ms_psd_linear_scale_{i}")
            ms_psd_linear_offset: torch.Tensor = getattr(self, f"ms_psd_linear_offset_{i}")

            linear_psd = (linear_psd + ms_psd_linear_offset) / ms_psd_linear_scale
            linear_psds.append(linear_psd)
            
        if len(linear_psds) == 1:
            return linear_psds[0]
        else:
            return linear_psds
    
    @torch.inference_mode()
    def ms_psd_to_img(self, mel_spec: torch.Tensor, use_colormap: bool = False):
        if use_colormap == True:
            return tensor_to_img(mel_spec.mean(dim=(0,1)), flip_y=True, colormap=True)
        else:
            if self.config.ms_add_center_channel == True:
                l, r, c = torch.chunk(mel_spec, 3, dim=1)
                if self.config.ms_img_show_center_channel == True:
                    mel_spec = torch.cat((l, c, r), dim=1)
                else:
                    mel_spec = torch.cat((l, r), dim=1)

            return tensor_to_img(mel_spec, flip_y=True)

    # **************** mdct methods ****************

    def get_mdct_phase_psd_shape(self, bsz: int = 1, raw_length: Optional[int] = None):
        raw_crop_width = self.get_raw_crop_width(raw_length=raw_length)
        num_mdct_bins = self.config.mdcts[0].num_frequencies
        num_mdct_frames = raw_crop_width // num_mdct_bins + 1
        return (bsz, self.config.num_raw_channels * 2, 1, num_mdct_bins * num_mdct_frames * self.config.num_mdcts)

    def get_mdct_shape(self, bsz: int = 1, raw_length: Optional[int] = None):
        return self.get_mdct_phase_psd_shape(bsz=bsz, raw_length=raw_length)

    def raw_to_mdct_phase_psd(self, raw_samples: torch.Tensor,
            random_phase_augmentation: Union[bool, torch.Tensor] = False, level: int = 0) -> Union[torch.Tensor, list[torch.Tensor]]:
        
        if level >= 0:
            _mclt: torch.Tensor = self.mdcts[level](raw_samples.float())

            if isinstance(random_phase_augmentation, bool):
                if random_phase_augmentation == True:
                    phase_rotation = torch.exp(2j * torch.pi * torch.rand(_mclt.shape[0], device=_mclt.device)) 
                    _mclt *= phase_rotation.view(-1, 1, 1, 1)
            else:
                _mclt *= random_phase_augmentation.view(-1, 1, 1, 1)

            mdct_psd = _mclt.abs()
            mdct_phase = (_mclt.real / mdct_psd.clip(min=1e-20)).clip(min=-1, max=1)

            mdct_psd = mdct_psd.pow(self.config.mdct_psd_exponent)
            mdct_phase = mdct_phase * mdct_psd

            mdct_phase = mdct_phase / getattr(self, f"mdct_phase_scale_{level}")
            mdct_psd = (mdct_psd + getattr(self, f"mdct_psd_offset_{level}")) / getattr(self, f"mdct_psd_scale_{level}")

            mdct_phase_psd = torch.cat((mdct_phase, mdct_psd), dim=1)
            return mdct_phase_psd
        else:
            if random_phase_augmentation == True:
                random_phase_augmentation = torch.exp(2j * torch.pi * torch.rand(raw_samples.shape[0], device=raw_samples.device))

            mdct_phase_psds = []
            for i in range(self.config.num_mdcts):
                mdct_phase_psd = self.raw_to_mdct_phase_psd(raw_samples, random_phase_augmentation=random_phase_augmentation, level=i)
                mdct_phase_psds.append(mdct_phase_psd)

            return mdct_phase_psds
    
    def mdct_phase_psd_to_raw(self, mdct_phase_psd: torch.Tensor, level: int = 0) -> torch.Tensor:

        if level >=0 :
            mdct_phase, mdct_psd = torch.chunk(mdct_phase_psd.float(), 2, dim=1)
            mdct_phase = mdct_phase * getattr(self, f"mdct_phase_scale_{level}")
            mdct_psd = mdct_psd * getattr(self, f"mdct_psd_scale_{level}") - getattr(self, f"mdct_psd_offset_{level}")

            #mdct_psd = mdct_psd.clip(min=0).pow(1 / self.config.mdct_psd_exponent - 1)
            recon_exp = int((1 / self.config.mdct_psd_exponent - 1) / 2) * 2 + 1
            mdct_psd = mdct_psd.pow(recon_exp)
            raw_samples = self.imdcts[level](mdct_phase * mdct_psd).real.contiguous()
            return raw_samples
        else:
            input_mdcts = self.unflatten_mdct_phase_psd(mdct_phase_psd)[:len(self.config.mdct_out_crossover_freqs) + 1]
            output_mdcts: list[torch.Tensor] = []
            for i in range(len(input_mdcts)):
                output_mdcts.append(self.mdct_phase_psd_to_raw(input_mdcts[i], level=i))

            crop_length = min(x.shape[-1] for x in output_mdcts)
            output_mdcts = [x[..., :crop_length] for x in output_mdcts]
            output_mdcts.reverse()

            return _combine_fft_crossover(output_mdcts, sample_rate=self.config.sample_rate,
                crossovers=self.config.mdct_out_crossover_freqs, crossfade_hz=self.config.mdct_out_crossover_width_hz)

    def flatten_mdct_phase_psd(self, mdct_phase_psds: list[torch.Tensor]) -> torch.Tensor:
        mdct_phase_psds = [x.flatten(2, 3)[:, :, None, :] for x in mdct_phase_psds]
        return torch.cat(mdct_phase_psds, dim=3)

    def unflatten_mdct_phase_psd(self, mdct_phase_psd: torch.Tensor) -> list[torch.Tensor]:
        assert mdct_phase_psd.shape[3] % self.config.num_mdcts == 0
        mdct_phase_psds: list[torch.Tensor] = []
        for i, chunk in enumerate(torch.chunk(mdct_phase_psd, chunks=self.config.num_mdcts, dim=3)):
            num_mdct_phase_psd_channels = self.config.num_raw_channels * 2
            mdct_phase_psds.append(chunk.reshape(chunk.shape[0], num_mdct_phase_psd_channels, self.config.mdcts[i].num_frequencies, -1))
        return mdct_phase_psds

    def get_mdct_mel_density(self, level: int = 0) -> Union[torch.Tensor:, list[torch.Tensor]]:
        if level >= 0:
            return getattr(self, f"mdct_mel_density_{level}")
        else:
            mdct_mel_densities = []
            for i in range(self.config.num_mdcts):
                mdct_mel_densities.append(getattr(self, f"mdct_mel_density_{i}"))
            return mdct_mel_densities

    def raw_to_mdct_psd(self, raw_samples: torch.Tensor, level: int = 0) -> torch.Tensor:

        mdct_phase_psd = self.raw_to_mdct_phase_psd(raw_samples.float(), level=level)
        _, mdct_psd = torch.chunk(mdct_phase_psd, 2, dim=1)
        return mdct_psd