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
from typing import Literal, Optional, Union

import torch

from modules.module import DualDiffusionModule, DualDiffusionModuleConfig
from utils.mclt import WindowFunction


@dataclass
class MSSLoss2DConfig(DualDiffusionModuleConfig):

    block_widths: tuple[int] = (8, 16, 32, 64)
    block_steps: tuple[int] = (1, 2, 4, 8)
    block_window_fn: Literal["none", "flat_top", "flat_top_circular", "hann", "kaiser", "blackman_harris"] = "flat_top"

    frequency_weighting_init: Literal["product", "f^2", "dynamic"] = "product"
    use_midside_transform: Literal["stack", "cat", "none"] = "cat"
    use_mse_loss: bool = True

class MSSLoss2D(DualDiffusionModule):

    module_name: str = "mss"
    supports_channels_last: Union[bool, Literal["3d"]] = False

    def __init__(self, config: MSSLoss2DConfig) -> None:
        super().__init__()
        self.config = config

        self.steps: list[int] = []
        self.loss_weights = torch.nn.ParameterList()
        self.paddings = torch.nn.ModuleList()

        for block_width, block_step in zip(config.block_widths, config.block_steps):
            self.steps.append(max(block_step, 1))

            if config.block_window_fn == "hann":
                window = self.get_sin_power_window_2d(block_width)
            elif config.block_window_fn == "flat_top":
                window = self.get_flat_top_window_2d(block_width)
            elif config.block_window_fn == "flat_top_circular":
                window = self.get_flat_top_window_2d_circular(block_width)
            elif config.block_window_fn == "kaiser":
                window = torch.kaiser_window(block_width, beta=12, periodic=False)
                window = torch.outer(window, window)
            elif config.block_window_fn == "blackman_harris":
                window = WindowFunction.blackman_harris(block_width)
                window = torch.outer(window, window)
            elif config.block_window_fn == "none":
                window = torch.ones((block_width, block_width))
            else:
                raise ValueError(f"Invalid block window function: {config.block_window_fn}")
            
            window /= window.square().mean().sqrt()
            self.register_buffer(f"window_{block_width}", window, persistent=False)

            freq_h: torch.Tensor = torch.fft.fftfreq(block_width, d=1/block_width)
            freq_w: torch.Tensor = torch.fft.rfftfreq(block_width, d=1/block_width)
            if config.frequency_weighting_init == "product":
                loss_weight = (freq_h.view(-1, 1).abs() + 1) * (freq_w.view(1, -1).abs() + 1)
            elif config.frequency_weighting_init == "f^2":
                loss_weight = freq_h.view(-1, 1)**2 + freq_w.view(1, -1)**2 + 1
            elif config.frequency_weighting_init != "dynamic":
                raise ValueError(f"Invalid frequency weighting: {config.frequency_weighting_init}")
            
            channel_dim = 4 if config.use_midside_transform == "cat" else 2
            loss_weight = loss_weight[None, None, None, None, :, :].repeat(1, channel_dim, 1, 1, 1, 1)
            loss_weight = torch.nn.Parameter(-loss_weight.float().log())
            self.loss_weights.append(loss_weight)

            padding = torch.nn.ReflectionPad2d((block_width // 2, block_width // 2,
                                                block_width // 2, block_width // 2))
            self.paddings.append(padding)

    @torch.no_grad()
    def _flat_top_window(self, x: torch.Tensor) -> torch.Tensor:
        return (0.21557895 - 0.41663158 * torch.cos(x) + 0.277263158 * torch.cos(2*x)
                - 0.083578947 * torch.cos(3*x) + 0.006947368 * torch.cos(4*x))

    @torch.no_grad()
    def get_flat_top_window_2d(self, block_width: int) -> torch.Tensor:
        #wx = (torch.arange(block_width) + 0.5) / block_width * 2 * torch.pi
        wx = torch.arange(block_width) / block_width * 2 * torch.pi
        return self._flat_top_window(wx.view(1, 1,-1, 1)) * self._flat_top_window(wx.view(1, 1, 1,-1))
    
    @torch.no_grad()
    def _sin_power_window(self, x: torch.Tensor, e: float = 2) -> torch.Tensor:
        return x.sin()**e
    
    @torch.no_grad()
    def get_sin_power_window_2d(self, block_width: int, e: float = 2) -> torch.Tensor:
        #wx = (torch.arange(block_width) + 0.5) / block_width * torch.pi
        wx = torch.arange(block_width) / block_width * torch.pi
        return self._sin_power_window(wx.view(1, 1,-1, 1), e) * self._sin_power_window(wx.view(1, 1, 1,-1), e)
    
    @torch.no_grad()
    def create_distance_tensor(self, block_width: int) -> torch.Tensor:
        x_coords = (torch.arange(block_width) + 0.5).view(1, -1)
        y_coords = (torch.arange(block_width) + 0.5).view(-1, 1)
        return torch.sqrt((x_coords - block_width/2) ** 2 + (y_coords - block_width/2) ** 2)

    @torch.no_grad()
    def get_flat_top_window_2d_circular(self, block_width: int) -> torch.Tensor:
        dist = self.create_distance_tensor(block_width) / (block_width // 2)
        return (self._flat_top_window(dist * torch.pi + torch.pi) * (dist <= 1))[None, None, :, :]
    
    def stft2d(self, x: torch.Tensor, block_width: int,
               step: int, window: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        x = x.unfold(2, block_width, step).unfold(3, block_width, step) * window
        x = torch.fft.rfft2(x, norm="ortho")

        if self.config.use_midside_transform not in ["none", None]:
            if self.config.use_midside_transform == "stack":
                x = torch.stack((x[:, 0] + x[:, 1], x[:, 0] - x[:, 1]), dim=1)
            elif self.config.use_midside_transform == "cat":
                x = torch.cat((x, x[:, 0:1] + x[:, 1:2], x[:, 0:1] - x[:, 1:2]), dim=1)
            else:
                raise ValueError(f"Invalid midside transform type: {self.config.use_midside_transform}")

        return x
    
    def forward(self, sample: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        loss = torch.zeros(target.shape[0], device=self.device)
        for i, block_width in enumerate(self.config.block_widths):
            
            if block_width > target.shape[-1]:
                continue

            step = self.steps[i]
            padding = self.paddings[i]
            window = getattr(self, f"window_{block_width}")
            loss_weight = self.loss_weights[i]
            
            with torch.no_grad():
                target_fft = self.stft2d(padding(target), block_width, step, window)
                target_fft_abs = target_fft.abs().requires_grad_(False).detach()
                target_fft_abs[:, :, :, :, 0, 0] = target_fft[:, :, :, :, 0, 0].real

            sample_fft = self.stft2d(padding(sample), block_width, step, window)
            sample_fft_abs = sample_fft.abs()
            sample_fft_abs[:, :, :, :, 0, 0] = sample_fft[:, :, :, :, 0, 0].real

            if self.config.use_mse_loss == True:
                block_loss = torch.nn.functional.mse_loss(sample_fft_abs, target_fft_abs, reduction="none")
            else:
                block_loss = torch.nn.functional.l1_loss(sample_fft_abs, target_fft_abs, reduction="none")
            loss = loss + (block_loss / loss_weight.exp() + loss_weight).mean(dim=(1,2,3,4,5))
        
        return loss