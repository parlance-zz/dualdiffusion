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

from typing import Optional, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from modules.module import DualDiffusionModule, DualDiffusionModuleConfig
from utils.dual_diffusion_utils import tensor_to_img


@dataclass
class DualDiffusionFormatConfig(DualDiffusionModuleConfig):

    noise_floor: float = 2e-5
    sample_rate: int = 32000
    sample_raw_channels: int = 2
    sample_raw_length: int = 1057570
    t_scale: Optional[float] = None
    raw_to_sample_scale: float = 1
    sample_to_raw_scale: float = 1
    sample_mean: float = 0

class DualDiffusionFormat(DualDiffusionModule, ABC):

    module_name: str = "format"
    has_trainable_parameters: bool = False
    supports_half_precision: bool = False
    supports_compile: bool = False # format compilation disabled for now
                                   # as torch.compile does not support complex operators
    
    @abstractmethod
    def sample_raw_crop_width(self, length: Optional[int] = None) -> int:
        pass
    
    @abstractmethod
    def get_sample_shape(self, bsz: int = 1,
                         length: Optional[int] = None) -> tuple:
        pass

    @abstractmethod
    @torch.inference_mode()
    def raw_to_sample(self, raw_samples: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    @torch.inference_mode()
    def sample_to_raw(self, samples: torch.Tensor) -> torch.Tensor:
        pass
    
    @abstractmethod
    @torch.no_grad()
    def get_ln_freqs(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def compile(self, **kwargs) -> None:
        if type(self).supports_compile == True:
            super().compile(**kwargs)
            self.raw_to_sample = torch.compile(self.raw_to_sample, **kwargs)
            self.sample_to_raw = torch.compile(self.sample_to_raw, **kwargs)
    
    @torch.inference_mode()
    def sample_to_img(self, sample: torch.Tensor):
        return tensor_to_img(sample, flip_y=True)