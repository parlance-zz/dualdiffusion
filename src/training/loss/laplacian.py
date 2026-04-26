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
class LaplacianLoss_Config:

    levels: int = 8
    num_channels: int = 3
    pad_mode: Literal["constant", "reflect", "replicate"] = "replicate"
    eps: float = 1e-16

class LaplacianLoss(torch.nn.Module):

    def __init__(self, config: LaplacianLoss_Config) -> None:
        super().__init__()

        self.config = config
    
        kernel = torch.tensor(
            [[1.0, 1.0, 1.0],
             [1.0,-8.0, 1.0],
             [1.0, 1.0, 1.0]],
        ).view(1, 1, 3, 3).expand(config.num_channels, 1, 3, 3)

        self.kernel: torch.Tensor
        self.register_buffer("kernel", -kernel, persistent=False)
        
    def convolve(self, x: torch.Tensor) -> torch.Tensor:

        x_padded = torch.nn.functional.pad(x, (1, 1, 1, 1), mode="reflect")
        return torch.nn.functional.conv2d(x_padded, self.kernel, padding=0, groups=self.config.num_channels)

    def forward(self, sample: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        sample = sample.float()
        target = target.float()
        
        loss = torch.nn.functional.mse_loss(sample, target, reduction="none").mean(dim=(1,2,3))
        loss = loss / target.square().mean(dim=(1,2,3)).clamp(min=self.config.eps)

        for _ in range(self.config.levels):
            
            sample = self.convolve(sample)
            with torch.no_grad():
                target = self.convolve(target)

            _loss = torch.nn.functional.mse_loss(sample, target, reduction="none").mean(dim=(1,2,3))
            _loss = _loss / target.square().mean(dim=(1,2,3)).clamp(min=self.config.eps)
            loss = loss + _loss

        return loss

    def compile(self, **kwargs) -> None:
        self.forward = torch.compile(self.forward, **kwargs)


if __name__ == "__main__":
    
    config = LaplacianLoss_Config()
    loss_fn = LaplacianLoss(config)
