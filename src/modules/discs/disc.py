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

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from modules.module import DualDiffusionModule, DualDiffusionModuleConfig


@dataclass
class DualDiffusionDiscriminatorConfig(DualDiffusionModuleConfig, ABC):

    in_channels: int     = 2
    in_channels_emb: int = 1024
    in_num_freqs: int    = 256

class DualDiffusionDiscriminator(DualDiffusionModule, ABC):

    module_name: str = "disc"
    
    @abstractmethod
    def get_embeddings(self, emb_in: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def forward(self, samples: torch.Tensor, target: torch.Tensor, embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    def compile(self, **kwargs) -> None:
        if type(self).supports_compile == True:
            super().compile(**kwargs)
            self.forward = torch.compile(self.forward, **kwargs)