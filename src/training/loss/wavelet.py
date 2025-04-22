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
from typing import Literal

import torch


@dataclass
class WaveletLoss_Config:

    levels: int = 7
    level_weight_exponent: float = 1
    use_midside_transform: Literal["stack", "cat", "none"] = "cat"

class WaveletLoss:

    def __init__(self, config: WaveletLoss_Config) -> None:
        super().__init__()

        self.config = config
    
    def wavelet_loss(self, sample: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:

        x = sample
        y = target

        if self.config.use_midside_transform not in ["none", None]:
            if self.config.use_midside_transform == "stack":
                x = torch.stack((x[:, 0] + x[:, 1], x[:, 0] - x[:, 1]), dim=1) * 0.5**0.5
                with torch.no_grad():
                    y = torch.stack((y[:, 0] + y[:, 1], y[:, 0] - y[:, 1]), dim=1) * 0.5**0.5
            elif self.config.use_midside_transform == "cat":
                x = torch.cat((x, (x[:, 0:1] + x[:, 1:2])*0.5**0.5, (x[:, 0:1] - x[:, 1:2])*0.5**0.5), dim=1)
                with torch.no_grad():
                    y = torch.cat((y, (y[:, 0:1] + y[:, 1:2])*0.5**0.5, (y[:, 0:1] - y[:, 1:2])*0.5**0.5), dim=1)
            else:
                raise ValueError(f"Invalid midside transform type: {self.config.use_midside_transform}")
            
        level_losses = []
        total_loss = torch.zeros(sample.shape[0], device=sample.device)

        for i in range(self.config.levels):

            dx = x[:, :, 1::2, ::2] - x[:, :, :-1:2, ::2]
            with torch.no_grad():
                dy = y[:, :, 1::2, ::2] - y[:, :, :-1:2, ::2]
            #level_loss = torch.nn.functional.l1_loss(dx, dy, reduction="none").mean(dim=(1,2,3))
            level_loss = torch.nn.functional.l1_loss(dx.square(), dy.square(), reduction="none").mean(dim=(1,2,3))

            dx = x[:, :, ::2, 1::2] - x[:, :, ::2, :-1:2]
            with torch.no_grad():
                dy = y[:, :, ::2, 1::2] - y[:, :, ::2, :-1:2]
            #level_loss = level_loss + torch.nn.functional.l1_loss(dx, dy, reduction="none").mean(dim=(1,2,3))
            level_loss = level_loss + torch.nn.functional.l1_loss(dx.square(), dy.square(), reduction="none").mean(dim=(1,2,3))

            dx = x[:, :, 1::2, 1::2] - x[:, :, :-1:2, :-1:2]
            with torch.no_grad():
                dy = y[:, :, 1::2, 1::2] - y[:, :, :-1:2, :-1:2]
            #level_loss = level_loss + torch.nn.functional.l1_loss(dx, dy, reduction="none").mean(dim=(1,2,3))
            level_loss = level_loss + torch.nn.functional.l1_loss(dx.square(), dy.square(), reduction="none").mean(dim=(1,2,3))

            dx = x[:, :, 1::2, :-1:2] - x[:, :, :-1:2, 1::2]
            with torch.no_grad():
                dy = y[:, :, 1::2, :-1:2] - y[:, :, :-1:2, 1::2]
            #level_loss = level_loss + torch.nn.functional.l1_loss(dx, dy, reduction="none").mean(dim=(1,2,3))
            level_loss = level_loss + torch.nn.functional.l1_loss(dx.square(), dy.square(), reduction="none").mean(dim=(1,2,3))

            level_losses.append(level_loss.detach())
            level_weight = 4**(-i * self.config.level_weight_exponent)
            total_loss = total_loss + level_loss * level_weight

            if i < self.config.levels - 1:
                x = torch.nn.functional.avg_pool2d(x, kernel_size=2)
                with torch.no_grad():
                    y = torch.nn.functional.avg_pool2d(y, kernel_size=2)

        return total_loss, level_losses

    def compile(self, **kwargs) -> None:
        self.wavelet_loss = torch.compile(self.wavelet_loss, **kwargs)
