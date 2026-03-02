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

from typing import Optional, Literal

import torch


class LatentStatsTracker(torch.nn.Module):

    def __init__(self, num_channels: int, momentum: float = 0.99, eps: float = 1e-6,
            static_mean: Optional[float] = None, static_scale: Optional[float] = None) -> None:
        
        super().__init__()

        self.num_channels = num_channels
        self.momentum = momentum
        self.eps = eps

        self.static_mean = static_mean
        self.static_scale = static_scale
        
        self.mean: torch.Tensor
        self.register_buffer("mean", torch.zeros(num_channels))
        self.var: torch.Tensor
        self.register_buffer("var", torch.ones(num_channels))

        self.global_mean: torch.Tensor
        self.register_buffer("global_mean", torch.zeros(1))
        self.global_var: torch.Tensor
        self.register_buffer("global_var", torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self.training == True:
            dx = x.detach().to(dtype=self.mean.dtype)

            per_channel_mean = dx.mean(dim=(0,2,3))
            self.mean.lerp_(per_channel_mean, 1. - self.momentum)
            per_channel_var = dx.var(dim=(0,2,3))
            self.var.lerp_(per_channel_var, 1. - self.momentum)

            global_mean = dx.mean()
            self.global_mean.lerp_(global_mean, 1. - self.momentum)
            global_var = dx.var()
            self.global_var.lerp_(global_var, 1. - self.momentum)

        return x
    
    def remove_mean(self, x: torch.Tensor, mode: Literal["per_channel", "global", "static", "none"] = "per_channel") -> torch.Tensor:

        if mode == "per_channel":
            return (x - self.mean[None, :, None, None].detach()).to(dtype=x.dtype)
        elif mode == "global":
            return (x - self.global_mean.detach()).to(dtype=x.dtype)
        elif mode == "static":
            if self.static_mean is not None:
                return (x - self.static_mean).to(dtype=x.dtype)

        return x
    
    def add_mean(self, x: torch.Tensor, mode: Literal["per_channel", "global", "static", "none"] = "per_channel") -> torch.Tensor:

        if mode == "per_channel":
            return (x + self.mean[None, :, None, None].detach()).to(dtype=x.dtype)
        elif mode == "global":
            return (x + self.global_mean.detach()).to(dtype=x.dtype)
        elif mode == "static":
            if self.static_mean is not None:
                return (x + self.static_mean).to(dtype=x.dtype)
            
        return x
    
    def unscale(self, x: torch.Tensor, mode: Literal["per_channel", "global", "static", "none"] = "per_channel") -> torch.Tensor:

        if mode == "per_channel":
            std = (self.var[None, :, None, None] + self.eps).pow(0.5)
            return (x / std.detach()).to(dtype=x.dtype)
        elif mode == "global":
            std = (self.global_var + self.eps).pow(0.5)
            return (x / std.detach()).to(dtype=x.dtype)
        elif mode == "static":
            if self.static_scale is not None:
                return (x / self.static_scale).to(dtype=x.dtype)
            
        return x
    
    def rescale(self, x: torch.Tensor, mode: Literal["per_channel", "global", "static", "none"] = "per_channel") -> torch.Tensor:
        
        if mode == "per_channel":
            std = (self.var[None, :, None, None] + self.eps).pow(0.5)
            return (x * std.detach()).to(dtype=x.dtype)
        elif mode == "global":
            std = (self.global_var + self.eps).pow(0.5)
            return (x * std.detach()).to(dtype=x.dtype)
        elif mode == "static":
            if self.static_scale is not None:
                return (x * self.static_scale).to(dtype=x.dtype)
            
        return x
  