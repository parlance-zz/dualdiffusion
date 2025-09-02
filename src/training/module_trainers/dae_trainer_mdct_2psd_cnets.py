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

from training.trainer import DualDiffusionTrainer
from .module_trainer import ModuleTrainerConfig, ModuleTrainer
from modules.daes.dae_edm2_2psd_a1 import DAE_2PSD_A1
from modules.cnets.cnet_edm2_2psd_a1 import CNet_2PSD_A1
from modules.formats.mdct_2psd import MDCT_2PSD_Format
from modules.mp_tools import normalize


@dataclass
class DAETrainer_MDCT_2PSD_Config(ModuleTrainerConfig):

    kl_loss_weight: float = 3e-2
    kl_warmup_steps: int  = 0

    add_latents_noise: float = 0.1
    add_latents_noise_pow: float = 0
    latents_noise_warmup_steps: int = 2000
    
    loss_weight_pow: float = 0.3333
    loss_weight_min: float = 0.1

class DAETrainer_MDCT_2PSD(ModuleTrainer):
    
    @torch.no_grad()
    def __init__(self, config: DAETrainer_MDCT_2PSD_Config, trainer: DualDiffusionTrainer) -> None:

        self.config = config
        self.trainer = trainer
        self.logger = trainer.logger
        self.dae: DAE_2PSD_A1 = trainer.get_train_module("dae")
        self.cnet0: CNet_2PSD_A1 = trainer.get_train_module("cnet0")
        self.cnet1: CNet_2PSD_A1 = trainer.get_train_module("cnet1")

        assert self.dae is not None, "DAE model not found!"
        assert self.cnet0 is not None, "CNet0 model not found!"
        assert self.cnet1 is not None, "CNet1 model not found!"

        self.is_validation_batch = False
        self.device_generator = None
        self.cpu_generator = None

        self.format: MDCT_2PSD_Format = trainer.pipeline.format.to(self.trainer.accelerator.device)

        if trainer.config.enable_model_compilation:
            self.dae.compile(**trainer.config.compile_params)
            self.cnet0.compile(**trainer.config.compile_params)
            self.cnet1.compile(**trainer.config.compile_params)
            self.format.compile(**trainer.config.compile_params)

        self.logger.info("Training DAE and CNet models:")
        self.logger.info(f"KL loss weight: {self.config.kl_loss_weight} KL warmup steps: {self.config.kl_warmup_steps}")
        self.logger.info(f"Add latents noise: {self.config.add_latents_noise}")

    def train_batch(self, batch: dict) -> dict[str, Union[torch.Tensor, float]]:

        if "audio_embeddings" in batch:
            sample_audio_embeddings = normalize(batch["audio_embeddings"])
            dae_embeddings = self.dae.get_embeddings(sample_audio_embeddings)
        else:
            dae_embeddings = None
        
        latents_sigma = self.config.add_latents_noise
        if self.config.latents_noise_warmup_steps and self.trainer.global_step < self.config.latents_noise_warmup_steps:
            latents_sigma *= ((self.trainer.global_step + 1) / self.config.latents_noise_warmup_steps)
        
        if latents_sigma > 0:
            latents_noise = torch.Tensor([latents_sigma]).to(device=self.trainer.accelerator.device)
            #latents_noise = latents_noise * torch.rand(batch["audio"].shape[0],
            #    device=self.trainer.accelerator.device).view(-1, 1, 1, 1) ** self.config.add_latents_noise_pow
        else:
            latents_noise = None

        raw_samples: torch.Tensor = batch["audio"].detach().clone()
        mdct0 = self.format.raw_to_mdct(raw_samples, random_phase_augmentation=True, idx=0)
        mdct1 = self.format.raw_to_mdct(raw_samples, random_phase_augmentation=True, idx=1)
        mdct_psd0 = self.format.raw_to_mdct_psd(raw_samples, idx=0)
        mdct_psd1 = self.format.raw_to_mdct_psd(raw_samples, idx=1)
        mdct0 = self.format.scale_mdct_from_psd(mdct0, mdct_psd0, idx=0)
        mdct1 = self.format.scale_mdct_from_psd(mdct1, mdct_psd1, idx=1)

        x_ref0, x_ref1, latents, pre_norm_latents = self.dae(mdct_psd0, mdct_psd1, dae_embeddings,
            latents_noise.detach().requires_grad_(False) if latents_noise is not None else None)

        pre_norm_latents_var = pre_norm_latents.var(dim=(1,2,3))
        kl_latents = pre_norm_latents.mean(dim=(1,2,3)).square() + pre_norm_latents_var - 1 - pre_norm_latents_var.log()

        kl_xref0 = x_ref0.mean(dim=(1,2,3)).square() + x_ref0.var(dim=(1,2,3)) - 1 - x_ref0.var(dim=(1,2,3)).log()
        kl_xref1 = x_ref1.mean(dim=(1,2,3)).square() + x_ref1.var(dim=(1,2,3)) - 1 - x_ref1.var(dim=(1,2,3)).log()
        x_ref_kl_weight = 1#max(1 - self.trainer.global_step / 100, 0)

        kl_loss = kl_latents + (kl_xref0 + kl_xref1) * x_ref_kl_weight

        cnet_input0 = torch.cat([mdct0, mdct_psd0], dim=1)[..., 4:-4]
        cnet_input1 = torch.cat([mdct1, mdct_psd1], dim=1)[..., 4:-4]
        cnet_output0 = self.cnet0(cnet_input0, self.format, x_ref0[..., 4:-4])
        cnet_output1 = self.cnet1(cnet_input1, self.format, x_ref1[..., 4:-4])

        mdct0_mel_density = 1#1 / self.format.mdct0_mel_density ** (1/self.format.config.psd_pow)
        mdct1_mel_density = 1#self.format.mdct1_mel_density ** (1/self.format.config.psd_pow)

        loss_weight0 = (mdct_psd0 + self.format.config.mdct0.psd_mean).clip(min=0) * mdct0_mel_density
        loss_weight0 = (loss_weight0 / loss_weight0.mean(dim=(1,2,3), keepdim=True).clip(min=self.config.loss_weight_min))[..., 4:-4]
        loss_weight0 = torch.cat((loss_weight0, torch.ones_like(loss_weight0)), dim=1).requires_grad_(False).detach()
        loss_weight1 = (mdct_psd1 + self.format.config.mdct1.psd_mean).clip(min=0) * mdct1_mel_density
        loss_weight1 = (loss_weight1 / loss_weight1.mean(dim=(1,2,3), keepdim=True).clip(min=self.config.loss_weight_min))[..., 4:-4]
        loss_weight1 = torch.cat((loss_weight1, torch.ones_like(loss_weight1)), dim=1).requires_grad_(False).detach()

        mse_loss0 = 2 * torch.nn.functional.mse_loss(cnet_output0, cnet_input0, reduction="none") * loss_weight0
        mse_loss1 = 2 * torch.nn.functional.mse_loss(cnet_output1, cnet_input1, reduction="none") * loss_weight1

        mdct_loss0 = mse_loss0[:, :2, :, :].mean(dim=(1,2,3))
        mdct_loss1 = mse_loss1[:, :2, :, :].mean(dim=(1,2,3))
        psd_loss0 = mse_loss0[:, 2:, :, :].mean(dim=(1,2,3))
        psd_loss1 = mse_loss1[:, 2:, :, :].mean(dim=(1,2,3))

        loss_logvar0 = self.cnet0.get_loss_logvar()
        loss_logvar1 = self.cnet1.get_loss_logvar()
        loss_nll0 = (mse_loss0 / loss_logvar0.exp() + loss_logvar0).mean(dim=(1,2,3))
        loss_nll1 = (mse_loss1 / loss_logvar1.exp() + loss_logvar1).mean(dim=(1,2,3))

        kl_loss_weight = self.config.kl_loss_weight
        if self.config.kl_warmup_steps and self.trainer.global_step < self.config.kl_warmup_steps:
            kl_loss_weight *= (self.trainer.global_step + 1) / self.config.kl_warmup_steps

        if self.trainer.config.enable_debug_mode == True:
            print("mdct0.shape:", mdct0.shape)
            print("mdct1.shape:", mdct1.shape)
            print("mdct_psd0.shape:", mdct_psd0.shape)
            print("mdct_psd1.shape:", mdct_psd1.shape)
            print("x_ref0.shape:", x_ref0.shape)
            print("x_ref1.shape:", x_ref1.shape)
            print("latents.shape:", latents.shape)

        return {
            "loss": loss_nll0 + loss_nll1 + kl_loss * kl_loss_weight,
            "loss/mdct_loss0": mdct_loss0.detach(),
            "loss/mdct_loss1": mdct_loss1.detach(),
            "loss/psd_loss0": psd_loss0.detach(),
            "loss/psd_loss1": psd_loss1.detach(),
            "loss/kl_latents": kl_latents.detach(),
            "loss_weight/kl_latents": kl_loss_weight,
            "io_stats/cnet_input0_std": cnet_input0.std(dim=(1,2,3)),
            "io_stats/cnet_input1_std": cnet_input1.std(dim=(1,2,3)),
            "io_stats/cnet_output0_std": cnet_output0.std(dim=(1,2,3)),
            "io_stats/cnet_output1_std": cnet_output1.std(dim=(1,2,3)),
            "io_stats/mdct0_std": mdct0.std(dim=(1,2,3)),
            "io_stats/mdct1_std": mdct1.std(dim=(1,2,3)),
            "io_stats/mdct_psd0_std": mdct_psd0.std(dim=(1,2,3)),
            "io_stats/mdct_psd1_std": mdct_psd1.std(dim=(1,2,3)),
            "io_stats/mdct_psd0_mean": mdct_psd0.mean(dim=(1,2,3)),
            "io_stats/mdct_psd1_mean": mdct_psd1.mean(dim=(1,2,3)),
            "io_stats/x_ref0_std": x_ref0.std(dim=(1,2,3)),
            "io_stats/x_ref1_std": x_ref1.std(dim=(1,2,3)),
            "io_stats/x_ref0_mean": x_ref0.mean(dim=(1,2,3)),
            "io_stats/x_ref1_mean": x_ref1.mean(dim=(1,2,3)),
            "io_stats/latents_std": latents.std(dim=(1,2,3)),
            "io_stats/latents_mean": latents.mean(dim=(1,2,3)),
            "io_stats/latents_pre_norm_std": pre_norm_latents_var.sqrt(),
            "io_stats/latents_sigma": latents_noise if latents_noise is not None else 0
        }