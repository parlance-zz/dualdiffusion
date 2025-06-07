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

from typing import Union, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from modules.module import DualDiffusionModule, DualDiffusionModuleConfig
from modules.formats.format import DualDiffusionFormat

@dataclass
class DualDiffusionUNetConfig(DualDiffusionModuleConfig, ABC):

    in_channels:  int    = 4
    out_channels: int    = 4
    in_channels_emb: int = 512

    dropout:    float = 0.
    sigma_max:  float = 200.
    sigma_min:  float = 0.03
    sigma_data: float = 1.

class DualDiffusionUNet(DualDiffusionModule, ABC):

    module_name: str = "unet"

    @abstractmethod
    def get_embeddings(self, emb_in: torch.Tensor, conditioning_mask: torch.Tensor) -> torch.Tensor:
        pass
        
    @abstractmethod
    def get_sigma_loss_logvar(self, sigma: Optional[torch.Tensor] = None) -> torch.Tensor:
        pass
    
    @abstractmethod
    def get_latent_shape(self, latent_shape: Union[torch.Size, tuple[int, int, int, int]]) -> torch.Size:
        pass

    @abstractmethod
    def forward(self, x_in: torch.Tensor,
                sigma: torch.Tensor,
                format: DualDiffusionFormat,
                embeddings: Optional[torch.Tensor] = None,
                x_ref: Optional[torch.Tensor] = None) -> torch.Tensor:
        pass

    def compile(self, **kwargs) -> None:
        if type(self).supports_compile == True:
            super().compile(**kwargs)
            self.get_embeddings = torch.compile(self.get_embeddings, **kwargs)
            self.forward = torch.compile(self.forward, **kwargs)