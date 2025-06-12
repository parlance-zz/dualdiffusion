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


@dataclass()
class RawFormatConfig(DualDiffusionFormatConfig):

    sample_rate: int = 32000
    num_raw_channels: int = 2
    default_raw_length: int = 1409024

    scale: float = 39.05
    width_alignment: int = 2048

class RawFormat(DualDiffusionFormat):

    def __init__(self, config: RawFormatConfig) -> None:
        super().__init__()
        self.config = config

    def get_raw_crop_width(self, raw_length: Optional[int] = None) -> int:
        raw_length = raw_length or self.config.default_raw_length
        return raw_length // self.config.width_alignment * self.config.width_alignment

    def get_raw_sample_shape(self, bsz: int = 1, raw_length: Optional[int] = None) -> tuple[int, int, int, int]:
        crop_width = self.get_raw_crop_width(raw_length)
        return (bsz, 1, self.config.num_raw_channels, crop_width)

    @torch.no_grad()
    def scale(self, raw_samples: torch.Tensor, random_phase_augmentation: bool = False) -> torch.Tensor:
        
        raw_len = raw_samples.shape[-1]
        raw_samples = torch.nn.functional.pad(raw_samples.float(), (raw_len // 2, raw_len // 2), mode="reflect")
        rfft: torch.Tensor = torch.fft.rfft(raw_samples, dim=-1, norm="ortho")

        if random_phase_augmentation == True:
            phase_rotation = torch.exp(2j * torch.pi * torch.rand(rfft.shape[0], device=rfft.device))
            rfft *= phase_rotation.view(-1, 1, 1)

        rfft_freq = torch.fft.rfftfreq(raw_samples.shape[-1], d=1/self.config.sample_rate, device=raw_samples.device)
        mel_density = get_mel_density(rfft_freq)
        mel_density /= mel_density.mean()
        raw_samples = torch.fft.irfft(rfft / mel_density.view(1, 1,-1), n=raw_samples.shape[-1], dim=-1, norm="ortho")

        return raw_samples[..., raw_len//2:-raw_len//2].unsqueeze(1).contiguous() * self.config.scale

    @torch.no_grad()
    def unscale(self, raw_samples: torch.Tensor):
        
        raw_samples = raw_samples.squeeze(1)
        
        raw_len = raw_samples.shape[-1]
        raw_samples = torch.nn.functional.pad(raw_samples.float(), (raw_len // 2, raw_len // 2), mode="reflect")
        rfft: torch.Tensor = torch.fft.rfft(raw_samples, dim=-1, norm="ortho")

        rfft_freq = torch.fft.rfftfreq(raw_samples.shape[-1], d=1/self.config.sample_rate, device=raw_samples.device)
        mel_density = get_mel_density(rfft_freq)
        mel_density /= mel_density.mean()
        raw_samples = torch.fft.irfft(rfft * mel_density.view(1, 1,-1), n=raw_samples.shape[-1], dim=-1, norm="ortho")

        return raw_samples[..., raw_len//2:-raw_len//2].contiguous() / self.config.scale