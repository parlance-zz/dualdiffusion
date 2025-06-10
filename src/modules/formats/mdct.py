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

from modules.formats.format import DualDiffusionFormat, DualDiffusionFormatConfig
from modules.formats.frequency_scale import get_mel_density
from utils.dual_diffusion_utils import tensor_to_img
from utils.mclt import mclt, imclt


@dataclass()
class MDCT_FormatConfig(DualDiffusionFormatConfig):

    sample_rate: int = 32000
    num_raw_channels: int = 2

    default_raw_length: int = 1409024
    width_alignment: int    = 32768

    # these values scale to unit norm for audio pre-normalized to -20 lufs (dual channel enabled)
    mdct_to_raw_scale: float = 1.
    raw_to_mdct_scale: float = 12.16
    
    mdct_window_len: int    = 64
    mdct_dual_channel: bool = True
    
    @property
    def mdct_num_frequencies(self) -> int:
        return self.mdct_window_len // 2
    
    @property
    def mdct_frame_hop_length(self) -> int:
        return self.mdct_window_len // 2

class MDCT_Format(DualDiffusionFormat):

    def __init__(self, config: MDCT_FormatConfig) -> None:
        super().__init__()
        self.config = config

        mdct_hz = (torch.arange(config.mdct_num_frequencies) + 0.5) * config.sample_rate / config.mdct_window_len
        self.register_buffer("mdct_hz", mdct_hz)
        self.register_buffer("mdct_mel_density", get_mel_density(mdct_hz).view(1, 1,-1, 1))
    
    # **************** mel-scale spectrogram methods ****************

    def get_raw_crop_width(self, raw_length: Optional[int] = None) -> int:
        raw_length = raw_length or self.config.default_raw_length
        return raw_length // self.config.width_alignment * self.config.width_alignment - self.config.mdct_num_frequencies

    def get_mdct_shape(self, bsz: int = 1, raw_length: Optional[int] = None):
        raw_crop_width = self.get_raw_crop_width(raw_length=raw_length)
        num_mdct_bins = self.config.mdct_num_frequencies
        num_mdct_frames = raw_crop_width // num_mdct_bins + 1
        num_channels = self.config.num_raw_channels * (2 if self.config.mdct_dual_channel else 1)
        return (bsz, num_channels, num_mdct_bins, num_mdct_frames,)

    @torch.no_grad()
    def raw_to_mdct(self, raw_samples: torch.Tensor, random_phase_augmentation: bool = False) -> torch.Tensor:

        _mclt = mclt(raw_samples.float(), self.config.mdct_window_len, "kaiser_bessel_derived", 1).permute(0, 1, 3, 2)
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
        _mclt = mclt(raw_samples.float(), self.config.mdct_window_len, "kaiser_bessel_derived", 1).permute(0, 1, 3, 2)
        return _mclt.abs().contiguous(memory_format=self.memory_format) / self.mdct_mel_density * self.config.raw_to_mdct_scale / 2**0.5
    
    @torch.no_grad()
    def mdct_to_raw(self, mdct: torch.Tensor) -> torch.Tensor:

        mdct = mdct * self.mdct_mel_density / self.config.raw_to_mdct_scale
        if self.config.mdct_dual_channel == True:
            mdct = torch.complex(*mdct.chunk(2, dim=1))

        raw_samples = imclt(mdct.permute(0, 1, 3, 2).contiguous(),
            window_fn="kaiser_bessel_derived", window_degree=1).real.contiguous()
        
        return raw_samples * self.config.mdct_to_raw_scale
    
    @torch.inference_mode()
    def mdct_psd_to_img(self, mdct_psd: torch.Tensor):
        return tensor_to_img(mdct_psd.clip(min=0)**0.25, flip_y=True)