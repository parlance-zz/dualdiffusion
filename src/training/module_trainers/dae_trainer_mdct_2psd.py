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
from modules.daes.dae_edm2_2psd_a1 import DAE_2PSD_A1
from modules.formats.mdct_2psd import MDCT_2PSD_Format
from modules.mp_tools import normalize


@dataclass
class MSSLoss2DConfig:

    block_widths: tuple[int] = (7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 47, 53)
    block_steps: tuple[int]  = (1,  2,  3,  5,  7, 11, 13, 17, 19, 23, 29, 31)
    
    #block_widths: tuple[int] = (11, 13, 17, 19, 23, 29, 31)
    #block_steps: tuple[int] = (1,  2,  3,  5,  7, 11, 13)
    #block_steps: tuple[int] = ( 2,  3,  5,  7, 11, 13, 17)

    #block_widths: tuple[int] = (5, 7, 11, 19, 37, 71)
    #block_steps:  tuple[int] = (2, 3,  5,  7, 17, 31)
    loss_scale: float = 0.123

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
        if x.shape[1] == 2:
            #x = torch.stack((x[:, 0] + x[:, 1], x[:, 0] - x[:, 1]), dim=1)
            x = torch.cat((x, (x[:, 0:1] + x[:, 1:2])*0.5**0.5, (x[:, 0:1] - x[:, 1:2])*0.5**0.5), dim=1)

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

                #loss_weight = (block_width / target_fft_abs.square().mean(dim=(0,1,2,3), keepdim=True).clip(min=1e-4).sqrt()).requires_grad_(False).detach()
                #loss_weight = (1 / target_fft_abs.square().mean(dim=(0,1,2,3), keepdim=True).clip(min=1e-2).sqrt()).requires_grad_(False).detach()
                loss_weight = (1 / target_fft_abs.square().mean(dim=(0,2,3), keepdim=True).clip(min=1e-2).sqrt()).requires_grad_(False).detach()

                #print(loss_weight.amin().item(), loss_weight.amax().item())

            sample_fft = self.stft2d(sample, block_width, step, window, offset_h, offset_w)
            sample_fft_abs = sample_fft.abs()
            
            mse_loss = torch.nn.functional.mse_loss(sample_fft_abs.float(), target_fft_abs.float(), reduction="none")
            loss = loss + (mse_loss * loss_weight).mean(dim=(1,2,3,4,5))

        return loss * self.config.loss_scale

    def compile(self, **kwargs) -> None:
        self.mss_loss = torch.compile(self.mss_loss, **kwargs)

@dataclass
class DAETrainer_MDCT_2PSD_Config(ModuleTrainerConfig):

    kl_loss_weight: float = 2e-2
    kl_warmup_steps: int  = 1000

    point_loss_weight: float = 3
    point_loss_warmup_steps: int = 2000
    point_loss_min_weight: float = 0

    add_latents_noise: float = 1
    add_latents_noise_pow: float = 7
    latents_noise_warmup_steps: int = 2000
    loss_weight_min: float = 0.1

class DAETrainer_MDCT_2PSD(ModuleTrainer):
    
    @torch.no_grad()
    def __init__(self, config: DAETrainer_MDCT_2PSD_Config, trainer: DualDiffusionTrainer) -> None:

        self.config = config
        self.trainer = trainer
        self.logger = trainer.logger
        self.module: DAE_2PSD_A1 = trainer.get_train_module("dae")

        self.is_validation_batch = False
        self.device_generator = None
        self.cpu_generator = None

        self.mss_loss = MSSLoss2D(MSSLoss2DConfig(), device=trainer.accelerator.device)
        self.format: MDCT_2PSD_Format = trainer.pipeline.format.to(self.trainer.accelerator.device)

        if trainer.config.enable_model_compilation:
            self.module.compile(**trainer.config.compile_params)
            self.format.compile(**trainer.config.compile_params)
            #self.mss_loss.compile(**trainer.config.compile_params)

        self.logger.info("Training DAE model:")
        self.logger.info(f"Point loss weight: {self.config.point_loss_weight} Point warmup steps: {self.config.point_loss_warmup_steps}")
        self.logger.info(f"Point loss min weight: {self.config.point_loss_min_weight}")
        self.logger.info(f"KL loss weight: {self.config.kl_loss_weight} KL warmup steps: {self.config.kl_warmup_steps}")
        self.logger.info(f"Add latents noise: {self.config.add_latents_noise}")

    def train_batch(self, batch: dict) -> dict[str, Union[torch.Tensor, float]]:

        if "audio_embeddings" in batch:
            sample_audio_embeddings = normalize(batch["audio_embeddings"])
            dae_embeddings = self.module.get_embeddings(sample_audio_embeddings)
        else:
            dae_embeddings = None
        
        latents_sigma = self.config.add_latents_noise
        if self.trainer.global_step < self.config.latents_noise_warmup_steps:
            latents_sigma *= ((self.trainer.global_step + 1) / self.config.latents_noise_warmup_steps)
        latents_sigma *= np.random.rand() ** self.config.add_latents_noise_pow

        raw_samples: torch.Tensor = batch["audio"].detach().clone()
        mdct_psd0 = self.format.raw_to_mdct_psd(raw_samples, idx=0)
        mdct_psd1 = self.format.raw_to_mdct_psd(raw_samples, idx=1)

        recon0, recon1, latents, pre_norm_latents = self.module(mdct_psd0, mdct_psd1, dae_embeddings,
            torch.Tensor([latents_sigma]).to(device=self.trainer.accelerator.device) if latents_sigma > 0 else None)

        pre_norm_latents_var = pre_norm_latents.var(dim=(1,2,3))
        kl_loss = pre_norm_latents.mean(dim=(1,2,3)).square() + pre_norm_latents_var - 1 - pre_norm_latents_var.log()

        point_loss_weight = self.config.point_loss_weight
        if self.trainer.global_step < self.config.point_loss_warmup_steps:
            point_loss_weight *= 1 - self.trainer.global_step / self.config.point_loss_warmup_steps
        else:
            point_loss_weight = 0
        point_loss_weight = max(point_loss_weight, self.config.point_loss_min_weight)

        point_loss0 = torch.nn.functional.l1_loss(recon0, mdct_psd0, reduction="none").mean(dim=(1,2,3))
        point_loss1 = torch.nn.functional.l1_loss(recon1, mdct_psd1, reduction="none").mean(dim=(1,2,3))
        point_loss = (point_loss0 + point_loss1) / 2

        abs_loss0 = self.mss_loss.mss_loss(recon0 + self.format.config.mdct0.psd_mean, mdct_psd0 + self.format.config.mdct0.psd_mean)
        abs_loss1 = self.mss_loss.mss_loss(recon1 + self.format.config.mdct1.psd_mean, mdct_psd1 + self.format.config.mdct1.psd_mean)

        recon_loss = (abs_loss0 + abs_loss1) / 2
        if point_loss_weight > 0:
            recon_loss = recon_loss + point_loss * point_loss_weight

        recon_loss_logvar = self.module.get_recon_loss_logvar()
        recon_loss_nll = recon_loss / recon_loss_logvar.exp() + recon_loss_logvar
        
        kl_loss_weight = self.config.kl_loss_weight
        if self.trainer.global_step < self.config.kl_warmup_steps:
            kl_loss_weight *= self.trainer.global_step / self.config.kl_warmup_steps

        if self.trainer.config.enable_debug_mode == True:
            print("mdct_psd0.shape:", mdct_psd0.shape)
            print("mdct_psd1.shape:", mdct_psd1.shape)
            print("recon0.shape:", recon0.shape)
            print("recon1.shape:", recon1.shape)
            print("latents.shape:", latents.shape)

        return {
            "loss": recon_loss_nll + kl_loss * kl_loss_weight,
            "loss/recon": recon_loss,
            "loss/point": point_loss,
            "loss/abs_mss0": abs_loss0,
            "loss/abs_mss1": abs_loss1,
            "loss/point0": point_loss0,
            "loss/point1": point_loss1,
            "loss/kl_latents": kl_loss,
            "loss_weight/kl_latents": kl_loss_weight,
            "loss_weight/point": point_loss_weight,
            "io_stats/mdct_psd0_std": mdct_psd0.std(dim=(1,2,3)),
            "io_stats/mdct_psd1_std": mdct_psd1.std(dim=(1,2,3)),
            "io_stats/mdct_psd0_mean": mdct_psd0.mean(dim=(1,2,3)),
            "io_stats/mdct_psd1_mean": mdct_psd1.mean(dim=(1,2,3)),
            "io_stats/recon_mdct_psd0_std": recon0.std(dim=(1,2,3)),
            "io_stats/recon_mdct_psd1_std": recon1.std(dim=(1,2,3)),
            "io_stats/recon_mdct_psd0_mean": recon0.mean(dim=(1,2,3)),
            "io_stats/recon_mdct_psd1_mean": recon1.mean(dim=(1,2,3)),
            "io_stats/latents_std": latents.std(dim=(1,2,3)),
            "io_stats/latents_mean": latents.mean(dim=(1,2,3)),
            "io_stats/latents_pre_norm_std": pre_norm_latents_var.sqrt(),
            "io_stats/latents_sigma": latents_sigma
        }