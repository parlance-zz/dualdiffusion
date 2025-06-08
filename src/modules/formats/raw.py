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
from utils.mclt import mclt, imclt


@dataclass()
class RawFormatConfig(DualDiffusionFormatConfig):

    sample_rate: int = 32000
    num_raw_channels: int = 2
    default_raw_length: int = 1409024

    scale: float = 8.72164
    width_alignment: int = 2048

    mdct_window_len: int = 512

class RawFormat(DualDiffusionFormat):

    def __init__(self, config: RawFormatConfig) -> None:
        super().__init__()
        self.config = config

        mdct_hz = (torch.arange(config.mdct_window_len // 2) + 0.5) * config.sample_rate / config.mdct_window_len
        self.register_buffer("mdct_mel_density", get_mel_density(mdct_hz).view(1, 1, 1,-1))

    def get_raw_crop_width(self, raw_length: Optional[int] = None) -> int:
        raw_length = raw_length or self.config.default_raw_length
        return raw_length // self.config.width_alignment * self.config.width_alignment

    def get_raw_sample_shape(self, bsz: int = 1, raw_length: Optional[int] = None) -> tuple[int, int, int, int]:
        crop_width = self.get_raw_crop_width(raw_length)
        return (bsz, 1, self.config.num_raw_channels, crop_width)

    def scale(self, raw_samples: torch.Tensor, random_phase_augmentation: bool = False) -> torch.Tensor:
        _mclt = mclt(raw_samples.float(), self.config.mdct_window_len, "kaiser_bessel_derived", 1)

        if random_phase_augmentation == True:
            phase_rotation = torch.exp(2j * torch.pi * torch.rand(_mclt.shape[0], device=_mclt.device)) 
            _mclt *= phase_rotation.view(-1, 1, 1, 1)
        
        raw_samples = imclt(_mclt / self.mdct_mel_density, window_fn="kaiser_bessel_derived").real.unsqueeze(1)
        return raw_samples.contiguous() * self.config.scale

    def unscale(self, raw_samples: torch.Tensor):
        _mclt = mclt(raw_samples.float().squeeze(1), self.config.mdct_window_len, "kaiser_bessel_derived", 1)

        raw_samples = imclt(_mclt * self.mdct_mel_density, window_fn="kaiser_bessel_derived").real
        return raw_samples.contiguous() / self.config.scale