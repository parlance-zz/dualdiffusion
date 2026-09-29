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

from dataclasses import dataclass

import torch
import numpy as np

from modules.formats.frequency_scale import FrequencyScale
from utils.dual_diffusion_utils import tensor_to_img


@dataclass()
class MelSpecConfig:

    sample_rate: int = 32000

    ms_add_center_channel: bool = True
    ms_img_show_center_channel: bool = True
    ms_abs_exponent: float = 0.25
    ms_freq_min: float = 0
    ms_num_filters: int = 256
    ms_hop_length: int = 256
    ms_window_length: int = 4096
    ms_window_exponent: float = 15
    
    @property
    def ms_num_stft_bins(self) -> int:
        return self.ms_window_length // 2 + 1
    
    @property
    def ms_freq_max(self) -> int:
        return self.sample_rate / 2
    
class MelSpec(torch.nn.Module):

    def __init__(self, config: MelSpecConfig) -> None:
        super().__init__()
        self.config = config

        window = torch.hann_window(config.ms_window_length, periodic=True, requires_grad=False)
        window = window.pow(config.ms_window_exponent)
        window /= window.pow(2).mean().pow(0.5)
        self.ms_window: torch.Tensor
        self.register_buffer("ms_window", window, persistent=False)    

        self.ms_freq_scale = FrequencyScale(
            freq_scale="mel",
            freq_min=config.ms_freq_min,
            freq_max=config.ms_freq_max,
            sample_rate=config.sample_rate,
            num_stft_bins=config.ms_window_length // 2 + 1,
            num_filters=config.ms_num_filters,
            filter_norm="l2",
            filter_shape="triangular"
        )

    @torch.no_grad()
    def raw_to_mel_spec(self, raw_samples: torch.Tensor) -> torch.Tensor:

        packed_raw = raw_samples.float().view(raw_samples.shape[0] * raw_samples.shape[1], raw_samples.shape[2])

        stft = torch.stft(packed_raw, n_fft=self.config.ms_window_length, hop_length=self.config.ms_hop_length,
            win_length=self.config.ms_window_length, window=self.ms_window, center=True,
            pad_mode="reflect", normalized=True, onesided=True, return_complex=True)
        
        stft = stft.view(raw_samples.shape[0], raw_samples.shape[1], stft.shape[1], stft.shape[2])
        
        if self.config.ms_add_center_channel == True:
            stft = torch.cat((stft, (stft[:, 0:1] + stft[:, 1:2]) / 2), dim=1).abs()
        else:
            stft = stft.abs()

        mel_spec: torch.Tensor = self.ms_freq_scale.scale(stft).pow(self.config.ms_abs_exponent)

        return mel_spec
        
    @torch.no_grad()
    def mel_spec_to_img(self, mel_spec: torch.Tensor, use_colormap: bool = False) -> np.ndarray:
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
