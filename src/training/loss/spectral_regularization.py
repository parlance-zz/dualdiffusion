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

from typing import Literal
from dataclasses import dataclass

import torch

from modules.formats.frequency_scale import get_mel_density


@dataclass
class SpecRegLossConfig:

    mel_density_sample_rate: float = 32000
    match_mel_spec_spectral_profile: bool = True
    falloff_exponent: float = 1
    use_mse_loss: bool = False

    loss_type: Literal["l1", "mse", "kl"] = "l1"
    loss_scale: float = 1

class SpecRegLoss:

    @torch.no_grad()
    def __init__(self, config: SpecRegLossConfig, latents_shape: tuple[int], mel_spec_shape: tuple[int], device: torch.device) -> None:

        self.config = config
        self.device = device

        if config.loss_type not in ["l1", "mse", "kl"]:
            raise ValueError(f"Invalid loss type: {config.loss_type}. Must be one of ['l1', 'mse', 'kl']")
        
        self.latents_padding = torch.nn.ReflectionPad2d(
            (latents_shape[3] // 2, latents_shape[3] // 2, latents_shape[2] // 2, latents_shape[2] // 2)
        )

        if config.match_mel_spec_spectral_profile == True:
            self.mel_spec_padding = torch.nn.ReflectionPad2d(
                (mel_spec_shape[3] // 2, mel_spec_shape[3] // 2, mel_spec_shape[2] // 2, mel_spec_shape[2] // 2)
            )

            self.target_density = None
        else:
            h_freq = torch.fft.fftfreq(latents_shape[2] * 2).abs() * config.mel_density_sample_rate
            w_freq = torch.fft.rfftfreq(latents_shape[3] * 2).abs() * config.mel_density_sample_rate

            target_density = torch.outer(get_mel_density(h_freq), get_mel_density(w_freq)) ** config.falloff_exponent
            target_density[0, 0] = 0
            target_density = target_density / target_density.square().mean().sqrt()
            target_density = target_density[None, None, :, :].expand(latents_shape[0], latents_shape[1], -1, -1)

            self.target_density = target_density.requires_grad_(False).detach().to(device)

    def spec_reg_loss(self, latents: torch.Tensor, mel_spec: torch.Tensor) -> torch.Tensor:

        padded_latents = self.latents_padding(latents)
        latents_fft_abs = torch.fft.rfft2(padded_latents, norm="ortho").abs()
        latents_fft_abs = (latents_fft_abs / latents_fft_abs.square().mean(dim=(1,2,3), keepdim=True).sqrt())

        if self.config.match_mel_spec_spectral_profile == True:
            with torch.no_grad():
                padded_mel_spec = self.latents_padding(mel_spec)
                mel_spec_fft_abs = torch.fft.rfft2(padded_mel_spec, norm="ortho").abs()
                mel_spec_fft_abs[:, :, 0, 0] = 0
                mel_spec_fft_abs = torch.nn.functional.interpolate(
                    mel_spec_fft_abs, size=latents_fft_abs.shape[2:], mode="area"
                )
                mel_spec_fft_abs[:, :, 0, 0] = 0
                mel_spec_fft_abs = (mel_spec_fft_abs / mel_spec_fft_abs.square().mean(dim=(1,2,3), keepdim=True).sqrt())
                mel_spec_fft_abs = mel_spec_fft_abs.repeat(1, latents.shape[1] // mel_spec_fft_abs.shape[1], 1, 1)
            
            target_density = mel_spec_fft_abs
        else:
            target_density = self.target_density
        
        if self.config.loss_type == "mse":
            loss = torch.nn.functional.mse_loss(latents_fft_abs, target_density, reduction="none").mean(dim=(1,2,3))
        elif self.config.loss_type == "l1":
            loss = torch.nn.functional.l1_loss(latents_fft_abs, target_density, reduction="none").mean(dim=(1,2,3))
        elif self.config.loss_type == "kl":
            ratio = target_density / latents_fft_abs
            loss = (ratio - 1 - ratio.log()).mean(dim=(1,2,3))
   
        return loss * self.config.loss_scale

    def compile(self, **kwargs) -> None:
        self.spec_reg_loss = torch.compile(self.spec_reg_loss, **kwargs)