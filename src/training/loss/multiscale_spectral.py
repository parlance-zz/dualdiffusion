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
from typing import Literal, Optional

import torch
import torchaudio

from modules.formats.frequency_scale import get_mel_density
from modules.formats.ms_mdct_dual import MS_MDCT_DualFormat


@dataclass
class MSSLoss1DConfig:

    block_widths: tuple[int] = (64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768)
    block_overlap: int = 2
    sample_rate: float = 32000
    loss_scale: float = 1

class MSSLoss1D:

    @staticmethod
    def _hann_power_window(window_length: int, periodic: bool = True, *, dtype: torch.dtype = None,
                          layout: torch.layout = torch.strided, device: torch.device = None,
                          requires_grad: bool = False, exponent: float = 1.) -> torch.Tensor:
        
        return torch.hann_window(window_length, periodic=periodic, dtype=dtype,
                layout=layout, device=device, requires_grad=requires_grad) ** exponent
    
    @torch.no_grad()
    def __init__(self, config: MSSLoss1DConfig, device: torch.device) -> None:

        self.config = config
        self.device = device

        self.block_specs: list[torchaudio.transforms.Spectrogram] = []
        self.loss_weights: list[torch.Tensor]= []
        
        for block_width in self.config.block_widths:
            
            block_spec = torchaudio.transforms.Spectrogram(
                n_fft=block_width,
                win_length=block_width,
                hop_length=max(block_width // self.config.block_overlap, 1),
                window_fn=MS_MDCT_DualFormat._hann_power_window,
                power=None, normalized="window",
                wkwargs={
                    "exponent": 1,
                    "periodic": True,
                    "requires_grad": False
                },
                center=True, pad=0, pad_mode="reflect", onesided=True
            )
            self.block_specs.append(block_spec.to(device=device))

            blockfreq = torch.fft.rfftfreq(block_width) * self.config.sample_rate
            loss_weight = get_mel_density(blockfreq).view(1, 1,-1, 1).to(device=device)
            self.loss_weights.append((loss_weight / loss_weight.amax() / torch.pi).requires_grad_(False).detach())

    @torch.no_grad()
    def _flat_top_window(self, x: torch.Tensor) -> torch.Tensor:
        return (0.21557895 - 0.41663158 * torch.cos(x) + 0.277263158 * torch.cos(2*x)
                - 0.083578947 * torch.cos(3*x) + 0.006947368 * torch.cos(4*x))

    @torch.no_grad()
    def get_flat_top_window_2d(self, block_width: int) -> torch.Tensor:
        wx = (torch.arange(block_width) + 0.5) / block_width * 2 * torch.pi
        return self._flat_top_window(wx.view(1, 1,-1, 1)) * self._flat_top_window(wx.view(1, 1, 1,-1))
    
    def mss_loss(self, sample: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        loss = torch.zeros(target.shape[0], device=self.device)
        phase_loss = torch.zeros_like(loss)

        for spec, loss_weight in zip(self.block_specs, self.loss_weights):

            with torch.no_grad():
                target_fft: torch.Tensor = spec(target)
                target_fft_abs = target_fft.abs().requires_grad_(False).detach()
                target_fft_angle = target_fft.angle().requires_grad_(False).detach()
                phase_loss_weight = ((target_fft_abs - target_fft_abs.amin(dim=2, keepdim=True)) * loss_weight).requires_grad_(False)

            sample_fft: torch.Tensor = spec(sample)
            sample_fft_abs = sample_fft.abs()
            sample_fft_angle = sample_fft.angle()
            
            l1_loss = torch.nn.functional.l1_loss(sample_fft_abs.float(), target_fft_abs.float(), reduction="none")
            loss = loss + l1_loss.mean(dim=(1,2,3))
  
            phase_error = (sample_fft_angle - target_fft_angle).abs()
            phase_error_wrap_mask = (phase_error > torch.pi).detach().requires_grad_(False)
            phase_error[phase_error_wrap_mask] = 2*torch.pi - phase_error[phase_error_wrap_mask]
            phase_loss = phase_loss + (phase_error * phase_loss_weight).mean(dim=(1,2,3))
        
        return loss * self.config.loss_scale, phase_loss * self.config.loss_scale

    def compile(self, **kwargs) -> None:
        self.mss_loss = torch.compile(self.mss_loss, **kwargs)

@dataclass
class MSSLoss2DConfig:

    block_widths: tuple[int] = (8, 16, 32, 64)
    block_overlap: int = 2
    block_width_weight_exponent: float = 0
    block_window_fn: Literal["none", "flat_top", "hann", "kaiser"] = "hann"

    frequency_weight_exponent: float = 0.5
    use_midside_transform: Literal["stack", "cat", "none"] = "stack"
    use_mse_loss: bool = False
    phase_loss_scale: float = 1
    loss_scale: float = 2

class MSSLoss2D:

    @torch.no_grad()
    def __init__(self, config: MSSLoss2DConfig, device: torch.device) -> None:

        self.config = config
        self.device = device

        self.steps = []
        self.windows = []
        
        for block_width in self.config.block_widths:
            self.steps.append(max(block_width // self.config.block_overlap, 1))

            if self.config.block_window_fn == "hann":
                window = self.get_sin_power_window_2d(block_width)
            elif self.config.block_window_fn == "flat_top":
                window = self.get_flat_top_window_2d(block_width)
            elif self.config.block_window_fn == "kaiser":
                window = torch.kaiser_window(block_width, beta=12, periodic=False)
                window = torch.outer(window, window)
            elif self.config.block_window_fn != "none":
                raise ValueError(f"Invalid block window function: {self.config.block_window_fn}")
            
            if self.config.block_window_fn != "none":
                window /= window.square().mean().sqrt()
                self.windows.append(window.to(device=device).requires_grad_(False).detach())

    @torch.no_grad()
    def _flat_top_window(self, x: torch.Tensor) -> torch.Tensor:
        return (0.21557895 - 0.41663158 * torch.cos(x) + 0.277263158 * torch.cos(2*x)
                - 0.083578947 * torch.cos(3*x) + 0.006947368 * torch.cos(4*x))

    @torch.no_grad()
    def get_flat_top_window_2d(self, block_width: int) -> torch.Tensor:
        wx = (torch.arange(block_width) + 0.5) / block_width * 2 * torch.pi
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
        x_coords = (torch.arange(block_width) + 0.5).repeat(block_width, 1)
        y_coords = (torch.arange(block_width) + 0.5).view(-1, 1).repeat(1, block_width)
        return torch.sqrt((x_coords - block_width/2) ** 2 + (y_coords - block_width/2) ** 2)

    @torch.no_grad()
    def get_flat_top_window_2d_circular(self, block_width: int) -> torch.Tensor:
        dist = self.create_distance_tensor(block_width)
        wx = (dist / (block_width/2 + 0.5)).clip(max=1) * torch.pi + torch.pi
        return self._flat_top_window(wx)
    
    def stft2d(self, x: torch.Tensor, block_width: int,
               step: int, window: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        padding = block_width // 2
        x = torch.nn.functional.pad(x, (0, 0, padding, padding), mode="reflect")
        x = x.unfold(2, block_width, step).unfold(3, block_width, step)

        if window is not None:
            x = x * window

        x = torch.fft.rfft2(x, norm="ortho")

        if self.config.use_midside_transform not in ["none", None]:
            if self.config.use_midside_transform == "stack":
                x = torch.stack((x[:, 0] + x[:, 1], x[:, 0] - x[:, 1]), dim=1)
            elif self.config.use_midside_transform == "cat":
                x = torch.cat((x, (x[:, 0:1] + x[:, 1:2])*0.5**0.5, (x[:, 0:1] - x[:, 1:2])*0.5**0.5), dim=1)
            else:
                raise ValueError(f"Invalid midside transform type: {self.config.use_midside_transform}")

        return x
    
    def mss_loss(self, sample: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        loss = torch.zeros(target.shape[0], device=self.device)

        for i, block_width in enumerate(self.config.block_widths):
            
            if block_width > target.shape[-1]:
                continue

            step = self.steps[i]
            if self.config.block_window_fn != "none":
                window = self.windows[i]
            else:
                window = None

            with torch.no_grad():
                target_fft = self.stft2d(target, block_width, step, window)
                target_fft_abs = target_fft.abs().requires_grad_(False).detach()
                
                if self.config.frequency_weight_exponent != 0:

                    loss_weight = (1 / target_fft_abs.mean(dim=(0,2,3), keepdim=True).clip(min=1e-2)).requires_grad_(False).detach()

                    if self.config.frequency_weight_exponent != 1:
                        loss_weight = loss_weight.pow(self.config.frequency_weight_exponent)
                
                else:
                    loss_weight = 1

                if self.config.block_width_weight_exponent != 0:
                    loss_weight = loss_weight * (block_width ** self.config.block_width_weight_exponent)

            sample_fft = self.stft2d(sample, block_width, step, window)
            sample_fft_abs = sample_fft.abs()

            if self.config.use_mse_loss == True:

                block_loss = torch.nn.functional.mse_loss(sample_fft_abs.float(), target_fft_abs.float(), reduction="none")

                if self.config.phase_loss_scale > 0:
                    block_loss = block_loss + (torch.nn.functional.mse_loss(sample_fft.real, target_fft.real, reduction="none") \
                                            +  torch.nn.functional.mse_loss(sample_fft.imag, target_fft.imag, reduction="none"))
                
            else:
                block_loss = torch.nn.functional.l1_loss(sample_fft_abs.float(), target_fft_abs.float(), reduction="none")
                
                if self.config.phase_loss_scale > 0:
                    block_loss = block_loss + (sample_fft - target_fft).abs() * self.config.phase_loss_scale

            abs_loss = (block_loss * loss_weight).mean(dim=(1,2,3,4,5))
            loss = loss + abs_loss
             
        return loss * self.config.loss_scale

    def compile(self, **kwargs) -> None:
        self.mss_loss = torch.compile(self.mss_loss, **kwargs)