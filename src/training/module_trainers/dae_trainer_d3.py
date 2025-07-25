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
from typing import Union

import torch
import numpy as np

from training.trainer import DualDiffusionTrainer
from .module_trainer import ModuleTrainerConfig, ModuleTrainer
from modules.daes.dae_edm2_d3 import DAE_D3
from modules.formats.ms_mdct_dual import MS_MDCT_DualFormat
from modules.mp_tools import normalize


@dataclass
class MSSLoss2DConfig:

    block_widths: tuple[int] = (11, 13, 17, 19, 23, 29, 31)
    block_steps: tuple[int] =  ( 2,  3,  5,  7, 11, 13, 17)
    #block_steps: tuple[int] = ( 1,  2,  3,  5,  7, 11, 13)
    
    #block_widths: tuple[int] = (5, 7, 11, 19, 37, 71)
    #block_steps:  tuple[int] = (2, 3,  5,  7, 17, 31)

class MSSLoss2D:

    @torch.no_grad()
    def __init__(self, config: MSSLoss2DConfig, device: torch.device) -> None:

        self.config = config
        self.device = device

        self.steps = []
        self.windows = []
        
        for block_width, block_step in zip(config.block_widths, config.block_steps):

            self.steps.append(block_step)
            window = self.get_flat_top_window_2d(block_width)
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
    
    def stft2d(self, x: torch.Tensor, block_width: int,
               step: int, window: torch.Tensor, offset_h: int, offset_w: int) -> torch.Tensor:
        
        padding = block_width // 2
        x = torch.nn.functional.pad(x, (padding+1+step, padding, padding+1+step, padding), mode="reflect")
        x = x[:, :, offset_h:, offset_w:]
        x = x.unfold(2, block_width, step).unfold(3, block_width, step)

        x = torch.fft.rfft2(x * window, norm="ortho")
        #if x.shape[1] == 2:
            #x = torch.stack((x[:, 0] + x[:, 1], x[:, 0] - x[:, 1]), dim=1)
            #x = torch.cat((x, (x[:, 0:1] + x[:, 1:2])*0.5**0.5, (x[:, 0:1] - x[:, 1:2])*0.5**0.5), dim=1)

        return x
    
    def mss_loss(self, sample: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        loss = torch.zeros(target.shape[0], device=self.device)

        for i, block_width in enumerate(self.config.block_widths):
            
            step = self.steps[i]
            window = self.windows[i]

            offset_h = np.random.randint(0, step)
            offset_w = np.random.randint(0, step)
            
            with torch.no_grad():
                target_fft = self.stft2d(target, block_width, step, window, offset_h, offset_w)
                target_fft_abs = target_fft.abs().requires_grad_(False).detach()
                loss_weight = (block_width / target_fft_abs.square().mean(dim=(0,1,2,3), keepdim=True).clip(min=1e-4).sqrt()).requires_grad_(False).detach()
                #loss_weight = (block_width / target_fft_abs.mean(dim=(0,1,2,3), keepdim=True).clip(min=1e-2)).requires_grad_(False).detach()

            sample_fft = self.stft2d(sample, block_width, step, window, offset_h, offset_w)
            sample_fft_abs = sample_fft.abs()
            
            mse_loss = torch.nn.functional.mse_loss(sample_fft_abs.float(), target_fft_abs.float(), reduction="none")
            loss = loss + (mse_loss * loss_weight).mean(dim=(1,2,3,4,5))

        return loss, torch.zeros_like(loss)

    def compile(self, **kwargs) -> None:
        self.mss_loss = torch.compile(self.mss_loss, **kwargs)

@dataclass
class DAETrainer_D3_Config(ModuleTrainerConfig):

    kl_loss_weight: float = 2e-2
    kl_warmup_steps: int  = 1000
    phase_loss_weight: float = 0.1
    add_latents_noise: float = 0
    latents_noise_warmup_steps: int = 10000

class DAETrainer_D3(ModuleTrainer):
    
    @torch.no_grad()
    def __init__(self, config: DAETrainer_D3_Config, trainer: DualDiffusionTrainer) -> None:

        self.config = config
        self.trainer = trainer
        self.logger = trainer.logger
        self.module: DAE_D3 = trainer.get_train_module("dae")

        self.is_validation_batch = False
        self.device_generator = None
        self.cpu_generator = None

        self.mss_loss = MSSLoss2D(MSSLoss2DConfig(), device=trainer.accelerator.device)
        self.format: MS_MDCT_DualFormat = trainer.pipeline.format.to(self.trainer.accelerator.device)

        if trainer.config.enable_model_compilation:
            self.module.compile(**trainer.config.compile_params)
            self.format.compile(**trainer.config.compile_params)
            #self.mss_loss.compile(**trainer.config.compile_params)

        self.logger.info("Training DAE model:")
        self.logger.info(f"KL loss weight: {self.config.kl_loss_weight} KL warmup steps: {self.config.kl_warmup_steps}")
        self.logger.info(f"Add latents noise: {self.config.add_latents_noise}")
        self.logger.info(f"Phase loss weight: {self.config.phase_loss_weight}")

    def train_batch(self, batch: dict) -> dict[str, Union[torch.Tensor, float]]:

        if "audio_embeddings" in batch:
            sample_audio_embeddings = normalize(batch["audio_embeddings"])
            dae_embeddings = self.module.get_embeddings(sample_audio_embeddings)
        else:
            dae_embeddings = None
        
        if self.config.add_latents_noise > 0:
            if self.trainer.global_step < self.config.latents_noise_warmup_steps:
                latents_sigma = self.config.add_latents_noise * (self.trainer.global_step / self.config.latents_noise_warmup_steps)
            else:
                latents_sigma = self.config.add_latents_noise
        else:
            latents_sigma = 0

        mel_spec = self.format.raw_to_mel_spec(batch["audio"]).clone().detach()
        latents, reconstructed, pre_norm_latents = self.module(mel_spec, dae_embeddings,
            torch.Tensor([latents_sigma]).to(device=self.trainer.accelerator.device) if latents_sigma > 0 else None)

        pre_norm_latents_var = pre_norm_latents.var(dim=(1,2,3))
        kl_loss = pre_norm_latents.mean(dim=(1,2,3)).square() + pre_norm_latents_var - 1 - pre_norm_latents_var.log()

        abs_loss, phase_loss = self.mss_loss.mss_loss(reconstructed, mel_spec)
        recon_loss = abs_loss + phase_loss * self.config.phase_loss_weight

        recon_loss_logvar = self.module.get_recon_loss_logvar()
        recon_loss_nll = (recon_loss / 2) / recon_loss_logvar.exp() + recon_loss_logvar

        point_loss = torch.nn.functional.l1_loss(reconstructed, mel_spec, reduction="none").mean(dim=(1,2,3))

        kl_loss_weight = self.config.kl_loss_weight
        if self.trainer.global_step < self.config.kl_warmup_steps:
            kl_loss_weight *= self.trainer.global_step / self.config.kl_warmup_steps

        if self.trainer.config.enable_debug_mode == True:
            print("mel_spec.shape:", mel_spec.shape)
            print("reconstructed.shape:", reconstructed.shape)
            print("latents.shape:", latents.shape)

        return {
            "loss": recon_loss_nll + kl_loss * kl_loss_weight,
            "loss/recon": recon_loss,
            "loss/mss_abs": abs_loss,
            "loss/mss_phase": phase_loss,
            "loss/point": point_loss,
            "loss/kl_latents": kl_loss,
            "loss_weight/kl_latents": kl_loss_weight,
            "loss_weight/phase": self.config.phase_loss_weight,
            "io_stats/mel_spec_std": mel_spec.std(dim=(1,2,3)),
            "io_stats/mel_spec_mean": mel_spec.mean(dim=(1,2,3)),
            "io_stats/recon_mel_std": reconstructed.std(dim=(1,2,3)),
            "io_stats/recon_mel_mean": reconstructed.mean(dim=(1,2,3)),
            "io_stats/latents_std": latents.std(dim=(1,2,3)),
            "io_stats/latents_mean": latents.mean(dim=(1,2,3)),
            "io_stats/latents_pre_norm_std": pre_norm_latents_var.sqrt(),
            "io_stats/latents_sigma": latents_sigma
        }