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
class DualDiffusionDecoderConfig(DualDiffusionModuleConfig, ABC):

    in_channels:  int    = 4
    out_channels: int    = 4
    in_channels_emb: int = 0
    in_channels_x_ref: int = 2

    step_balance: list[int] = (0, 1/7, 2/7, 3/7, 4/7, 5/7, 6/7)

    @property
    def num_steps(self) -> int:
        return len(self.step_balance)

class DualDiffusionDecoder(DualDiffusionModule, ABC):

    module_name: str = "ddec"

    @abstractmethod
    def get_embeddings(self, emb_in: torch.Tensor) -> torch.Tensor:
        pass

    def get_recon_loss_logvar(self) -> torch.Tensor:
        return getattr(self, "error_logvar", torch.zeros(1, device=self.device))

    @abstractmethod
    def get_latent_shape(self, latent_shape: Union[torch.Size, tuple[int, int, int, int]]) -> torch.Size:
        pass

    @abstractmethod
    def get_state_shape(self, format: DualDiffusionFormat, x_ref: Union[torch.Tensor, list[torch.Tensor]]) -> torch.Size:
        pass

    @abstractmethod
    def step(self, x: torch.Tensor, step: int, format: DualDiffusionFormat,
        x_ref: Union[torch.Tensor, list[torch.Tensor]], embeddings: Optional[torch.Tensor] = None) -> torch.Tensor:
        pass

    def forward(self, format: DualDiffusionFormat, x_ref: Union[torch.Tensor, list[torch.Tensor]],
        audio_embeddings: Optional[torch.Tensor] = None, use_gradient_checkpointing: bool = False, return_output_states: bool = False) -> torch.Tensor:

        embeddings = self.get_embeddings(audio_embeddings)
        noise = torch.randn(self.get_state_shape(format, x_ref), device=self.device)
        state_curr = noise
        output_states: list[torch.Tensor] = []

        self.config: DualDiffusionDecoderConfig
        for i in range(self.config.num_steps):

            if use_gradient_checkpointing == True:
                state_new: torch.Tensor = torch.utils.checkpoint.checkpoint(
                    self.step, state_curr, i, format, x_ref, embeddings, use_reentrant=False)
            else:
                state_new: torch.Tensor = self.step(state_curr, i, format, x_ref, embeddings)

            if return_output_states:
                output_states.append(state_new)
                
            state_curr = torch.lerp(state_new, state_curr, self.config.step_balance[i])

        output = torch.lerp(state_curr, noise, 0.5)

        if return_output_states:
            return output, output_states
        else:
            return output

    def compile(self, **kwargs) -> None:
        if type(self).supports_compile == True:
            self.step = torch.compile(self.step, **kwargs)