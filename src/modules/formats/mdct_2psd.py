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

from modules.formats.format import DualDiffusionFormat, DualDiffusionFormatConfig
from modules.formats.frequency_scale import get_mel_density
from utils.dual_diffusion_utils import tensor_to_img
from utils.mdct import MDCT, IMDCT, get_window_fn


@dataclass()
class MDCT_PSD_Config:

    raw_to_mdct_scale: float = 278.47124 # for stereo audio @ -20 lufs
    raw_to_mdct_psd_scale: float = 1

    psd_eps: float   = 1e-2
    psd_scale: float = 1.0
    psd_mean: float  = 0
    window_len: int = 2048
    window_func: Literal["sin", "kbd", "vorbis"] = "sin"
    
    @property
    def num_frequencies(self) -> int:
        return self.window_len // 2
    
    @property
    def frame_hop_length(self) -> int:
        return self.window_len // 2

@dataclass()
class MDCT_2PSD_FormatConfig(DualDiffusionFormatConfig):
    
    sample_rate: int = 32000
    num_raw_channels: int = 2
    default_raw_length: int = 1409024
    width_alignment: int    = 32768
    psd_pow: float = 3

    mdct0: MDCT_PSD_Config = None
    mdct1: MDCT_PSD_Config = None

class MDCT_2PSD_Format(DualDiffusionFormat):

    def __init__(self, config: MDCT_2PSD_FormatConfig) -> None:
        super().__init__()
        self.config = config

        assert config.mdct0 is not None and config.mdct1 is not None
        assert config.mdct1.window_len // config.mdct0.window_len > 1
        assert config.mdct1.window_len %  config.mdct0.window_len == 0
        assert (config.mdct1.frame_hop_length - config.mdct0.frame_hop_length) % 2 == 0

        for i, mdct_cfg in enumerate((config.mdct0, config.mdct1)):

            mdct_hz = (torch.arange(mdct_cfg.num_frequencies) + 0.5) * (config.sample_rate/2) / mdct_cfg.num_frequencies
            self.register_buffer(f"mdct{i}_hz", mdct_hz, persistent=False)
            self.register_buffer(f"mdct{i}_mel_density", get_mel_density(mdct_hz).view(1, 1,-1, 1), persistent=False)

            mdct_window_fn = get_window_fn(mdct_cfg.window_func)
            mdct = MDCT(win_length=mdct_cfg.window_len, window_fn=mdct_window_fn, return_complex=True)
            imdct = IMDCT(win_length=mdct_cfg.window_len, window_fn=mdct_window_fn)
            setattr(self, f"mdct{i}", mdct)
            setattr(self, f"imdct{i}", imdct)

        self.mdct1_edge_drop: int = (config.mdct1.frame_hop_length - config.mdct0.frame_hop_length) // 2
 
    def get_raw_crop_width(self, raw_length: Optional[int] = None) -> int:
        raw_length = raw_length or self.config.default_raw_length
        return raw_length // self.config.width_alignment * self.config.width_alignment - self.config.mdct0.num_frequencies

    def get_mdct_shape(self, bsz: int = 1, raw_length: Optional[int] = None, idx: int = 0) -> tuple[int, int, int, int]:
        raw_length = raw_length or self.config.default_raw_length
        num_frequencies = self.config.mdct0.num_frequencies if idx == 0 else self.config.mdct1.num_frequencies
        raw_crop_width = self.get_raw_crop_width(raw_length=raw_length + num_frequencies)
        num_mdct_frames = (raw_crop_width + num_frequencies) // num_frequencies
        num_channels = self.config.num_raw_channels
        return (bsz, num_channels, num_frequencies, num_mdct_frames,)

    @torch.no_grad()
    def raw_to_mdct(self, raw_samples: torch.Tensor, random_phase_augmentation: bool = False, idx: int = 0) -> torch.Tensor:

        if idx == 1:
            raw_samples = raw_samples[..., self.mdct1_edge_drop:-self.mdct1_edge_drop]
        mdct_tform: MDCT = self.mdct0 if idx == 0 else self.mdct1
        mdct_mel_density: torch.Tensor = self.mdct0_mel_density if idx == 0 else self.mdct1_mel_density
        raw_to_mdct_scale = self.config.mdct0.raw_to_mdct_scale if idx == 0 else self.config.mdct1.raw_to_mdct_scale

        mclt = mdct_tform(raw_samples.float())

        if random_phase_augmentation == True:
            phase_rotation = torch.exp(2j * torch.pi * torch.rand(mclt.shape[0], device=mclt.device)) 
            mclt *= phase_rotation.view(-1, 1, 1, 1)

        return mclt.real.contiguous(memory_format=self.memory_format) / mdct_mel_density * raw_to_mdct_scale
    
    @torch.no_grad()
    def raw_to_mdct_psd(self, raw_samples: torch.Tensor, idx: int = 0) -> torch.Tensor:

        if idx == 1:
            raw_samples = raw_samples[..., self.mdct1_edge_drop:-self.mdct1_edge_drop]
        mdct_mel_density: torch.Tensor = self.mdct0_mel_density if idx == 0 else self.mdct1_mel_density
        mdct_tform: MDCT = self.mdct0 if idx == 0 else self.mdct1
        raw_to_mdct_psd_scale = self.config.mdct0.raw_to_mdct_psd_scale if idx == 0 else self.config.mdct1.raw_to_mdct_psd_scale
        psd_mean = self.config.mdct0.psd_mean if idx == 0 else self.config.mdct1.psd_mean

        mdct_psd = (mdct_tform(raw_samples.float()).abs() / mdct_mel_density).pow(1 / self.config.psd_pow)
        return mdct_psd * raw_to_mdct_psd_scale - psd_mean
    
    @torch.no_grad()
    def mdct_to_raw(self, mdct: torch.Tensor, idx: int = 0) -> torch.Tensor:
        mdct_mel_density: torch.Tensor = self.mdct0_mel_density if idx == 0 else self.mdct1_mel_density
        imdct: IMDCT = self.imdct0 if idx == 0 else self.imdct1
        raw_to_mdct_scale = self.config.mdct0.raw_to_mdct_scale if idx == 0 else self.config.mdct1.raw_to_mdct_scale

        mdct = mdct * mdct_mel_density / raw_to_mdct_scale
        raw_samples = imdct(mdct).real.contiguous()
        
        return raw_samples

    @torch.no_grad()
    def scale_mdct_from_psd(self, mdct: torch.Tensor, mdct_psd: torch.Tensor, idx: int = 0) -> torch.Tensor:

        raw_to_mdct_psd_scale = self.config.mdct0.raw_to_mdct_psd_scale if idx == 0 else self.config.mdct1.raw_to_mdct_psd_scale
        psd_eps = self.config.mdct0.psd_eps if idx == 0 else self.config.mdct1.psd_eps
        psd_scale = self.config.mdct0.psd_scale if idx == 0 else self.config.mdct1.psd_scale
        psd_mean = self.config.mdct0.psd_mean if idx == 0 else self.config.mdct1.psd_mean

        mdct_psd = ((mdct_psd + psd_mean) / raw_to_mdct_psd_scale).pow(self.config.psd_pow)
        return mdct / (mdct_psd + psd_eps) * psd_scale

    @torch.no_grad()
    def unscale_mdct_from_psd(self, mdct: torch.Tensor, mdct_psd: torch.Tensor, idx: int = 0) -> torch.Tensor:

        raw_to_mdct_psd_scale = self.config.mdct0.raw_to_mdct_psd_scale if idx == 0 else self.config.mdct1.raw_to_mdct_psd_scale
        psd_eps = self.config.mdct0.psd_eps if idx == 0 else self.config.mdct1.psd_eps
        psd_scale = self.config.mdct0.psd_scale if idx == 0 else self.config.mdct1.psd_scale
        psd_mean = self.config.mdct0.psd_mean if idx == 0 else self.config.mdct1.psd_mean

        mdct_psd = ((mdct_psd + psd_mean) / raw_to_mdct_psd_scale).pow(self.config.psd_pow)
        return mdct * (mdct_psd + psd_eps) / psd_scale

    @torch.inference_mode()
    def psd_to_img(self, psd: torch.Tensor, colormap: bool = False, idx: int = 0) -> torch.Tensor:
        if colormap == True:
            psd = psd.mean(dim=1, keepdim=True)
        #if idx == 0:
        #    psd = (psd + self.config.mdct0.psd_mean)
        #    psd = psd.clip(min=0) ** self.config.psd_pow
        return tensor_to_img(psd, flip_y=True, colormap=colormap)