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
from typing import Union, Optional, Any

import torch
import numpy as np

from training.trainer import DualDiffusionTrainer
from training.module_trainers.module_trainer import ModuleTrainer, ModuleTrainerConfig
from training.module_trainers.unet_trainer_p4 import UNetTrainerConfig, UNetTrainer
from modules.unets.unet_edm2_q1_ddec import UNet
from modules.daes.dae_edm2_q1 import DAE
from modules.formats.ms_mdct_dual_2 import MS_MDCT_DualFormat
from modules.mp_tools import normalize


@torch.no_grad()
def random_stereo_augmentation(x: torch.Tensor, p: float = 0.5) -> torch.Tensor:
    
    output = x.clone()
    flip_mask = (torch.rand(x.shape[0]) > p).to(x.device)
    output[flip_mask] = output[flip_mask].flip(dims=(1,))
    
    return output

@dataclass
class DiffusionDecoder_Trainer_Config(ModuleTrainerConfig):

    ddec: dict[str, Any]

    kl_loss_weight: float = 1e-2
    kl_warmup_steps: int  = 2000
    
    phase_invariance_loss_weight: float = 1e-2
    phase_invariance_warmup_steps: int  = 2000

    random_stereo_augmentation: bool = True
    random_phase_augmentation: bool  = True

    crop_edges: int = 4 # used to avoid artifacts due to mdct lapped blocks at beginning and end of sample

class DiffusionDecoder_Trainer(ModuleTrainer):
    
    @torch.no_grad()
    def __init__(self, config: DiffusionDecoder_Trainer_Config, trainer: DualDiffusionTrainer) -> None:

        self.config = config
        self.trainer = trainer
        self.logger = trainer.logger

        self.dae: DAE = trainer.get_train_module("dae")
        self.ddec: UNet = trainer.get_train_module("ddec")
        self.format: MS_MDCT_DualFormat = trainer.pipeline.format.to(self.trainer.accelerator.device)

        if trainer.config.enable_model_compilation:
            self.dae.compile(**trainer.config.compile_params)
            self.ddec.compile(**trainer.config.compile_params)
            self.format.compile(**trainer.config.compile_params)

        if self.config.random_stereo_augmentation == True:
            self.logger.info("Using random stereo augmentation")
        else: self.logger.info("Random stereo augmentation is disabled")
        if self.config.random_phase_augmentation == True:
            self.logger.info("Using random phase augmentation")
        else: self.logger.info("Random phase augmentation is disabled")

        self.logger.info(f"Crop edges: {self.config.crop_edges}")
        self.logger.info(f"KL Loss Weight: {self.config.kl_loss_weight} Warmup Steps: {self.config.kl_warmup_steps}")
        self.logger.info(f"Phase Invariance Loss Weight: {self.config.phase_invariance_loss_weight} Warmup Steps: {self.config.phase_invariance_warmup_steps}")

        self.logger.info("DDEC Trainer:")
        self.ddec_trainer = UNetTrainer(UNetTrainerConfig(**config.ddec), trainer, self.ddec, "ddec")
    
    def phase_invariance_loss(self, raw_samples: torch.Tensor, dae_embeddings: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:

        mdct_phase, _ = self.format.raw_to_mdct_phase_psd(raw_samples, random_phase_augmentation=self.config.random_phase_augmentation)
        mdct_phase = mdct_phase[..., self.config.crop_edges:-self.config.crop_edges]

        with torch.autocast(device_type="cuda", dtype=self.trainer.mixed_precision_dtype, enabled=self.trainer.mixed_precision_enabled):
            latents2 = self.dae.encode(mdct_phase, dae_embeddings)

        return (latents - latents2.float())[..., 2:-2].pow(2).mean().expand(latents.shape[0])
    
    @torch.no_grad()
    def init_batch(self, validation: bool = False) -> Optional[dict[str, Union[torch.Tensor, float]]]:
        return self.ddec_trainer.init_batch(validation)

    def train_batch(self, batch: dict) -> Optional[dict[str, Union[torch.Tensor, float]]]:

        # prepare model inputs
        if "audio_embeddings" in batch:
            audio_embeddings = normalize(batch["audio_embeddings"]).detach()
            dae_embeddings = self.dae.get_embeddings(audio_embeddings)
        else:
            audio_embeddings = dae_embeddings = None

        if self.config.random_stereo_augmentation == True:
            raw_samples = random_stereo_augmentation(batch["audio"])
        else: raw_samples = batch["audio"]

        mdct_phase, _ = self.format.raw_to_mdct_phase_psd(raw_samples, random_phase_augmentation=self.config.random_phase_augmentation)
        mdct_phase = mdct_phase[..., self.config.crop_edges:-self.config.crop_edges]
        
        latents, ddec_cond, pre_norm_latents = self.dae(mdct_phase, dae_embeddings)
        latents: torch.Tensor = latents.float()
        pre_norm_latents: torch.Tensor = pre_norm_latents.float()

        if self.config.phase_invariance_loss_weight > 0:
            phase_invariance_loss = self.phase_invariance_loss(raw_samples, dae_embeddings, latents)
        else: phase_invariance_loss = None

        pre_norm_latents_var = pre_norm_latents.pow(2).mean() + 1e-20
        var_kl = pre_norm_latents_var - 1 - pre_norm_latents_var.log()
        kl_loss = var_kl.mean() + 0.5 * pre_norm_latents.mean().square().mean()
        kl_loss = kl_loss.expand(latents.shape[0]) # needed for per-sample logging

        phase_invariance_loss_weight = self.config.phase_invariance_loss_weight
        if self.trainer.global_step < self.config.phase_invariance_warmup_steps:
            warmup_factor = self.trainer.global_step / self.config.phase_invariance_warmup_steps
            phase_invariance_loss_weight *= warmup_factor

        kl_loss_weight = self.config.kl_loss_weight
        if self.trainer.global_step < self.config.kl_warmup_steps:
            kl_loss_weight *= self.trainer.global_step / self.config.kl_warmup_steps

        logs = {
            "loss": kl_loss * kl_loss_weight,
            "io_stats/mdct_phase_var": mdct_phase.var(dim=(1,2,3)),
            "io_stats/ddec_cond_var": ddec_cond.var(dim=(1,2,3)),
            "io_stats/ddec_cond_mean": ddec_cond.mean(dim=(1,2,3)),
            "io_stats/latents_var": latents.var(dim=(1,2,3)).detach(),
            "io_stats/latents_mean": latents.mean(dim=(1,2,3)).detach(),
            "loss/kl_latents": kl_loss.detach(),
            "loss_weight/kl_latents": kl_loss_weight,
            "loss_weight/phase_invariance": phase_invariance_loss_weight,
        }

        logs.update(self.ddec_trainer.train_batch(mdct_phase, audio_embeddings, ddec_cond))
        logs["loss"] = logs["loss"] + logs["loss/ddec"]

        if self.config.phase_invariance_loss_weight > 0:
            logs["loss"] = logs["loss"] + phase_invariance_loss * phase_invariance_loss_weight
            logs["loss/phase_invariance"] = phase_invariance_loss.detach()

        if self.trainer.config.enable_debug_mode == True:
            print("mdct_phase.shape:", mdct_phase.shape)
            print("ddec_cond.shape:", ddec_cond.shape)
            print("latents.shape:", latents.shape)
            
        return logs
      
    @torch.no_grad()
    def finish_batch(self) -> Optional[dict[str, Union[torch.Tensor, float]]]:
        return self.ddec_trainer.finish_batch()