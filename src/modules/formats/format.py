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

@dataclass
class DualDiffusionFormatConfig(DualDiffusionModuleConfig):

    noise_floor: float = 2e-5
    sample_rate: int = 32000
    sample_raw_channels: int = 2
    sample_raw_length: int = 1057570
    t_scale: Optional[float] = None

class DualDiffusionFormat(DualDiffusionModule, ABC):

    module_name: str = "format"
    has_trainable_parameters: bool = False
    
    @abstractmethod
    def __init__(self, config: DualDiffusionFormatConfig):
        super().__init__()
        self.config = config

    @torch.no_grad()
    def get_num_channels(self) -> tuple[int, int]:
        in_channels = out_channels = self.config.sample_raw_channels
        return (in_channels, out_channels)
    
    @abstractmethod
    def get_raw_crop_width(self, length: Optional[int] = None) -> int:
        pass
    
    @abstractmethod
    def get_num_channels(self) -> tuple[int, int]:
        pass
    
    @abstractmethod
    def get_sample_shape(self, bsz: int = 1,
                         length: Optional[int] = None) -> tuple:
        pass

    @abstractmethod
    def raw_to_sample(self, raw_samples: torch.Tensor,
                      return_dict: bool = False) -> Union[torch.Tensor, dict]:
        pass

    @abstractmethod
    def sample_to_raw(self, samples: torch.Tensor,
                      return_dict: bool = False) -> Union[torch.Tensor, dict]:
        pass
    
    @abstractmethod
    def get_ln_freqs(self, x: torch.Tensor) -> torch.Tensor:
        pass