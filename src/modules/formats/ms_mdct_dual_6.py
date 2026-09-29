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
from modules.formats.frequency_scale import get_mel_density
from modules.formats.mel_spec import MelSpecConfig, MelSpec
from utils.dual_diffusion_utils import tensor_to_img
from utils.mdct import MDCT, IMDCT, sin_window, kaiser_bessel_derived, vorbis


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
class MS_PSD_Config:

    window_len: int = 512
    window_func: Literal["sin", "kaiser_bessel_derived", "vorbis"] = "vorbis"
    add_center_channel: bool = True

    @property
    def num_frequencies(self) -> int:
        return self.window_len // 2
    
    @property
    def hop_length(self) -> int:
        return self.window_len // 2
    
@dataclass()
class MS_MDCT_DualFormatConfig(DualDiffusionFormatConfig):

    # raw audio format params
    sample_rate: int = 32000
    num_raw_channels: int = 2
    default_raw_length: int = 1408768
    width_alignment: int    = 4096

    use_per_freq_preconditioning: bool = True

    # mdct params
    mdct_window_len: int = 128
    mdct_window_func: Literal["sin", "kaiser_bessel_derived", "vorbis"] = "vorbis"
    mdct_psd_exponent: float = 0.25

    @property
    def mdct_num_frequencies(self) -> int:
        return self.mdct_window_len // 2
    
    @property
    def mdct_frame_hop_length(self) -> int:
        return self.mdct_window_len // 2
    
    # ms psd params
    ms_psd_img_show_center_channel: bool = True
    ms_psds: list[MS_PSD_Config] = ()

    @property
    def num_ms_psds(self) -> int:
        return len(self.ms_psds)
    
    # mel-spec params
    mel_spec_config: Optional[MelSpecConfig] = None


class MS_MDCT_DualFormat(DualDiffusionFormat):
    
    has_trainable_parameters: bool = True

    @torch.no_grad()
    def __init__(self, config: MS_MDCT_DualFormatConfig) -> None:
        super().__init__()
        self.config = config

        # ***** mel_spec setup *****

        if config.mel_spec_config is not None:
            self.mel_spec = MelSpec(config.mel_spec_config)

        # ***** ms psd setup *****

        self.ms_psd_mdcts = torch.nn.ModuleList()
        for i in range(self.config.num_ms_psds):
            
            if config.ms_psds[i].window_len == config.mdct_window_len:
                last_frame = -1
            elif config.ms_psds[i].window_len > config.mdct_window_len:
                last_frame = -2
            else:
                last_frame = None

            mdct_window_func = _get_mdct_window_func(config.ms_psds[i].window_func)
            mdct = MDCT(win_length=config.ms_psds[i].window_len, window_fn=mdct_window_func, return_complex=True, last_frame=last_frame)
            self.ms_psd_mdcts.append(mdct)

            self.register_buffer(f"ms_psd_offset_{i}", torch.zeros(config.ms_psds[i].num_frequencies).view(1, 1,-1, 1), persistent=True)
            self.register_buffer(f"ms_psd_scale_{i}",  torch.ones(config.ms_psds[i].num_frequencies).view(1, 1,-1, 1),  persistent=True)
            
            psd_hz = (torch.arange(config.ms_psds[i].num_frequencies) + 0.5) * config.sample_rate / config.ms_psds[i].window_len
            self.register_buffer(f"ms_psd_mel_density_{i}", get_mel_density(psd_hz).view(1, 1,-1, 1), persistent=False)
            
        # ***** mdct setup *****

        mdct_window_func = _get_mdct_window_func(config.mdct_window_func)
        self.mdct = MDCT(win_length=config.mdct_window_len, window_fn=mdct_window_func, return_complex=True)
        self.imdct = IMDCT(win_length=config.mdct_window_len, window_fn=mdct_window_func)

        self.mdct_phase_scale: torch.Tensor
        self.register_buffer("mdct_phase_scale", torch.ones(config.mdct_num_frequencies).view(1, 1,-1, 1), persistent=True)

        self.mdct_psd_scale: torch.Tensor; self.mdct_psd_offset: torch.Tensor
        self.register_buffer("mdct_psd_scale", torch.ones(config.mdct_num_frequencies).view(1, 1,-1, 1), persistent=True)
        self.register_buffer("mdct_psd_offset", torch.zeros(config.mdct_num_frequencies).view(1, 1,-1, 1), persistent=True)

        mdct_hz = (torch.arange(config.mdct_num_frequencies) + 0.5) * config.sample_rate / config.mdct_window_len
        self.mdct_hz: torch.Tensor; self.mdct_mel_density: torch.Tensor
        self.register_buffer("mdct_hz", mdct_hz, persistent=False)
        self.register_buffer("mdct_mel_density", get_mel_density(mdct_hz).view(1, 1,-1, 1), persistent=False)

    # **************** mel-scale spectrogram methods ****************

    def get_raw_crop_width(self, raw_length: Optional[int] = None) -> int:
        raw_length = raw_length or self.config.default_raw_length
        return raw_length // self.config.width_alignment * self.config.width_alignment - self.config.mdct_num_frequencies

    @torch.no_grad()
    def raw_to_ms_psd(self, raw_samples: torch.Tensor, level: int = 2) -> Union[list[torch.Tensor], torch.Tensor]:

        raw_samples = raw_samples.float()
        levels = list(range(self.config.num_ms_psds)) if level < 0 else [level]

        ms_psds: list[torch.Tensor] = []
        for i in levels:

            mclt: torch.Tensor = self.ms_psd_mdcts[i](raw_samples)
            if self.config.ms_psds[i].add_center_channel == True:
                ms_psd = torch.cat((mclt, (mclt[:, 0:1] + mclt[:, 1:2]) / 2), dim=1).abs()
            else:
                ms_psd = mclt.abs()

            if self.config.use_per_freq_preconditioning == False:
                ms_psd_mel_density: torch.Tensor = getattr(self, f"ms_psd_mel_density_{i}")
                ms_psd /= ms_psd_mel_density

            ms_psd_offset: torch.Tensor = getattr(self, f"ms_psd_offset_{i}")
            ms_psd_scale: torch.Tensor = getattr(self, f"ms_psd_scale_{i}")                      
            ms_psd = (ms_psd.pow(self.config.mdct_psd_exponent) + ms_psd_offset) / ms_psd_scale

            ms_psds.append(ms_psd)

        if len(ms_psds) == 1:
            return ms_psds[0]
        else:
            return ms_psds
    
    @torch.no_grad()
    def ms_psd_to_img(self, ms_psd: torch.Tensor, use_colormap: bool = False):
        if use_colormap == True:
            return tensor_to_img(ms_psd.mean(dim=(0,1)), flip_y=True, colormap=True)
        else:
            if ms_psd.shape[1] == 3:
                l, r, c = torch.chunk(ms_psd, 3, dim=1)
                if self.config.ms_psd_img_show_center_channel == True:
                    ms_psd = torch.cat((l, c, r), dim=1)
                else:
                    ms_psd = torch.cat((l, r), dim=1)

            return tensor_to_img(ms_psd, flip_y=True)

    # **************** mdct methods ****************

    def get_mdct_phase_psd_shape(self, bsz: int = 1, raw_length: Optional[int] = None):
        raw_crop_width = self.get_raw_crop_width(raw_length=raw_length)
        num_mdct_bins = self.config.mdct_num_frequencies
        num_mdct_frames = raw_crop_width // num_mdct_bins + 1
        return (bsz, self.config.num_raw_channels * 2, num_mdct_bins, num_mdct_frames,)

    def get_mdct_shape(self, bsz: int = 1, raw_length: Optional[int] = None):
        return self.get_mdct_phase_psd_shape(bsz=bsz, raw_length=raw_length)
    
    """
    def raw_to_mdct_phase(self, raw_samples: torch.Tensor, random_phase_augmentation: bool = False) -> torch.Tensor:

        _mclt: torch.Tensor = self.mdct(raw_samples.float())
        if random_phase_augmentation == True:
            phase_rotation = torch.exp(2j * torch.pi * torch.rand(_mclt.shape[0], device=_mclt.device)) 
            _mclt *= phase_rotation.view(-1, 1, 1, 1)

        mdct_psd = _mclt.real.abs()
        mdct_psd = mdct_psd.pow(self.config.ms_abs_exponent) / self.mdct_mel_density.pow(self.config.ms_abs_exponent)
        mdct_phase = _mclt.real.sign() * mdct_psd

        return mdct_phase / self.config.mdct_phase_scale
    
    def mdct_phase_to_raw(self, mdct_phase: torch.Tensor) -> torch.Tensor:

        mdct_phase = mdct_phase * self.config.mdct_phase_scale
        
        mdct_psd = mdct_phase.abs() * self.mdct_mel_density.pow(self.config.ms_abs_exponent)
        mdct_psd = mdct_psd.clip(min=0).pow(1 / self.config.ms_abs_exponent)
        mdct_phase = mdct_phase.sign()

        raw_samples = self.imdct(mdct_phase * mdct_psd).real.contiguous()
        return raw_samples
    """

    def raw_to_mdct_phase_psd(self, raw_samples: torch.Tensor, random_phase_augmentation: bool = False) -> torch.Tensor:

        _mclt: torch.Tensor = self.mdct(raw_samples.float())
        if random_phase_augmentation == True:
            phase_rotation = torch.exp(2j * torch.pi * torch.rand(_mclt.shape[0], device=_mclt.device)) 
            _mclt *= phase_rotation.view(-1, 1, 1, 1)

        mdct_psd = _mclt.abs()
        mdct_phase = (_mclt.real / mdct_psd.clip(min=1e-20)).clip(min=-1, max=1)

        mdct_psd = mdct_psd.pow(self.config.mdct_psd_exponent)
        mdct_phase = mdct_phase * mdct_psd

        if self.config.use_per_freq_preconditioning == False:
            mdct_psd /= self.mdct_mel_density.pow(self.config.mdct_psd_exponent)
            mdct_phase /= self.mdct_mel_density.pow(self.config.mdct_psd_exponent)

        mdct_phase = mdct_phase / self.mdct_phase_scale
        mdct_psd = (mdct_psd + self.mdct_psd_offset) / self.mdct_psd_scale

        mdct_phase_psd = torch.cat((mdct_phase, mdct_psd), dim=1)
        return mdct_phase_psd
    
    def mdct_phase_psd_to_raw(self, mdct_phase_psd: torch.Tensor) -> torch.Tensor:

        mdct_phase, mdct_psd = torch.chunk(mdct_phase_psd.float(), 2, dim=1)
        mdct_phase = mdct_phase * self.mdct_phase_scale
        mdct_psd = mdct_psd * self.mdct_psd_scale - self.mdct_psd_offset

        if self.config.use_per_freq_preconditioning == False:
            mdct_phase *= self.mdct_mel_density.pow(self.config.mdct_psd_exponent)
            mdct_psd *= self.mdct_mel_density.pow(self.config.mdct_psd_exponent)

        mdct_psd = mdct_psd.clip(min=0).pow(1 / self.config.mdct_psd_exponent - 1)
        raw_samples = self.imdct(mdct_phase * mdct_psd).real.contiguous()
        return raw_samples
    
    def raw_to_mdct_psd(self, raw_samples: torch.Tensor) -> torch.Tensor:

        mdct_phase_psd = self.raw_to_mdct_phase_psd(raw_samples.float())
        _, mdct_psd = torch.chunk(mdct_phase_psd, 2, dim=1)
        return mdct_psd