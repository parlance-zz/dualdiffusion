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
from typing import Union, Optional

import torch
import numpy as np

from training.trainer import DualDiffusionTrainer
from training.module_trainers.unet_trainer_p3 import UNetTrainer, UNetTrainerConfig
from modules.daes.dae_edm2_p3 import DAE
from modules.unets.unet_edm2_p3_ddec import UNet
from modules.formats.ms_mdct_dual_2 import MS_MDCT_DualFormat
from modules.mp_tools import normalize


def get_cos_angle(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    dot = torch.einsum('bchw,bchw->bhw', x, y)
    return dot / x.shape[1]

@torch.no_grad()
def random_stereo_augmentation(x: torch.Tensor) -> torch.Tensor:
    
    output = x.clone()
    flip_mask = (torch.rand(x.shape[0]) > 0.5).to(x.device)
    output[flip_mask] = output[flip_mask].flip(dims=(1,))
    
    return output

@dataclass
class DiffusionDecoder_Trainer_Config(UNetTrainerConfig):

    kl_loss_weight: float = 1e-2
    kl_warmup_steps: int  = 2000

    phase_invariance_loss_weight: float = 1
    phase_invariance_loss_bsz: int = -1
    latents_dispersion_loss_weight: float = 0
    latents_dispersion_loss_bsz: int = -1
    latents_dispersion_num_iterations: int = 1
    latents_regularization_warmup_steps: int = 20000

    loss_buckets_sigma_min: float = 0.01
    loss_buckets_sigma_max: float = 100

    random_stereo_augmentation: bool = False
    random_phase_augmentation: bool  = False

    crop_edges: int = 4 # used to avoid artifacts due to mdct lapped blocks at beginning and end of sample

class DiffusionDecoder_Trainer(UNetTrainer):
    
    @torch.no_grad()
    def __init__(self, config: DiffusionDecoder_Trainer_Config, trainer: DualDiffusionTrainer) -> None:

        self.config = config
        self.trainer = trainer
        self.logger = trainer.logger

        self.ddec: UNet = trainer.get_train_module("ddec")
        self.dae: DAE = trainer.get_train_module("dae")

        if self.ddec is None:
            self.freeze_ddec = True
            self.ddec: UNet = trainer.pipeline.ddec.to(device=trainer.accelerator.device, dtype=torch.bfloat16).requires_grad_(False).eval()
            self.logger.info(f"DDEC module loaded from pipeline and frozen (step: {self.ddec.config.last_global_step})")
        else:
            self.freeze_ddec = False
                                                                                   
        if self.dae is None:
            self.freeze_dae = True
            self.dae: DAE = trainer.pipeline.dae.to(device=trainer.accelerator.device, dtype=torch.bfloat16).requires_grad_(False).eval()
            self.logger.info(f"DAE module loaded from pipeline and frozen (step: {self.dae.config.last_global_step})")
            assert self.dae.config.last_global_step > 0, "DAE module loaded from pipeline is untrained"
        else:
            self.freeze_dae = False

        if self.freeze_dae == False:
            if self.config.phase_invariance_loss_weight > 0:
                assert self.config.phase_invariance_loss_bsz != 0, "phase_invariance_loss_weight > 0 but phase_invariance_loss_bsz is 0"
            if self.config.phase_invariance_loss_bsz == -1:
                self.config.phase_invariance_loss_bsz = self.trainer.config.device_batch_size
            
            if self.config.latents_dispersion_loss_weight > 0:
                assert self.config.latents_dispersion_loss_bsz != 0, "latents_dispersion_loss_weight > 0 but latents_dispersion_loss_bsz is 0"
            if self.config.latents_dispersion_loss_bsz == -1:
                self.config.latents_dispersion_loss_bsz = self.trainer.config.device_batch_size
            else:
                assert self.config.latents_dispersion_loss_bsz <= self.trainer.config.device_batch_size, "latents_dispersion_loss_bsz cannot be larger than device_batch_size"
        
        self.format: MS_MDCT_DualFormat = trainer.pipeline.format.to(self.trainer.accelerator.device)

        if trainer.config.enable_model_compilation:
            self.ddec.compile(**trainer.config.compile_params)
            self.dae.compile(**trainer.config.compile_params)
            self.format.compile(**trainer.config.compile_params)

        self.logger.info(f"Training modules: {trainer.config.train_modules}")
        if self.freeze_dae == False:
            self.logger.info(f"KL loss weight: {self.config.kl_loss_weight} KL warmup steps: {self.config.kl_warmup_steps}")
            self.logger.info(f"Latents phase-invariance loss weight: {self.config.phase_invariance_loss_weight} Batch size: {self.config.phase_invariance_loss_bsz}")
            self.logger.info(f"Latents dispersion loss weight: {self.config.latents_dispersion_loss_weight} Batch size: {self.config.latents_dispersion_loss_bsz}")
            self.logger.info(f"Latents dispersion loss num iterations: {self.config.latents_dispersion_num_iterations}")
            self.logger.info(f"Latents regularization loss warmup steps: {self.config.latents_regularization_warmup_steps}")
        self.logger.info(f"Crop edges: {self.config.crop_edges}")

        if self.config.random_stereo_augmentation == True:
            self.logger.info("Using random stereo augmentation")
        else: self.logger.info("Random stereo augmentation is disabled")

        self.loss_weight = self.format.mdct_mel_density.clone().requires_grad_(False).repeat(1,4,1,1)
        self.loss_weight /= self.loss_weight.mean()
        
        self.unet = self.ddec
        self.unet_trainer_init()

    def train_batch(self, batch: dict) -> Optional[dict[str, Union[torch.Tensor, float]]]:

        # prepare model inputs
        if "audio_embeddings" in batch:
            audio_embeddings = normalize(batch["audio_embeddings"]).detach()
            dae_embeddings = self.dae.get_embeddings(audio_embeddings)
        else:
            audio_embeddings = dae_embeddings = None

        if self.config.random_stereo_augmentation == True:
            raw_samples = random_stereo_augmentation(batch["audio"])
        else:
            raw_samples = batch["audio"]

        mdct_samples: torch.Tensor = self.format.raw_to_mdct(raw_samples, random_phase_augmentation=self.config.random_phase_augmentation)
        mdct_samples = mdct_samples[:, :, :, self.config.crop_edges:-self.config.crop_edges].detach()
        mdct_psd: torch.Tensor = (self.format.raw_to_mdct_psd(raw_samples) / 2**0.5).clip(min=1e-3)
        mdct_psd = mdct_psd[:, :, :, self.config.crop_edges:-self.config.crop_edges].detach()
        
        #mel_spec = self.format.raw_to_mel_spec(raw_samples)
        #mel_spec = mel_spec[:, :, :, self.config.crop_edges:-self.config.crop_edges].detach()

        latents, ddec_cond, pre_norm_latents = self.dae(mdct_samples, dae_embeddings)
        latents: torch.Tensor = latents.float()
        pre_norm_latents: torch.Tensor = pre_norm_latents.float()
        mod: torch.Tensor = self.dae.get_mod(ddec_cond).float()#.abs().clip(min=1e-3)

        if self.freeze_dae == False and self.config.phase_invariance_loss_bsz > 0:

            mdct_samples2: torch.Tensor = self.format.raw_to_mdct(raw_samples[:self.config.phase_invariance_loss_bsz], random_phase_augmentation=True)
            mdct_samples2 = mdct_samples2[:, :, :, self.config.crop_edges:-self.config.crop_edges].detach()
            dae_embeddings2 = dae_embeddings[:self.config.phase_invariance_loss_bsz] if dae_embeddings is not None else None

            with torch.autocast(device_type="cuda", dtype=self.trainer.mixed_precision_dtype, enabled=self.trainer.mixed_precision_enabled):
                latents2 = self.dae.encode(mdct_samples2, dae_embeddings2)

            #cos_angle = get_cos_angle(latents[:self.config.phase_invariance_loss_bsz], latents2.float())
            #phase_invariance_loss = (1 - cos_angle).mean() / 2
            phase_invariance_loss = (latents[:self.config.phase_invariance_loss_bsz] - latents2.float()).pow(2).mean()

            phase_invariance_loss = phase_invariance_loss.expand(latents.shape[0]) # needed for per-sample logging
        else:
            phase_invariance_loss = None

        if self.freeze_dae == False and self.config.latents_dispersion_loss_bsz > 0:
            dispersion_loss = torch.zeros(1, device=latents.device)
            total_dispersion_iterations = 0

            for i in range(self.config.latents_dispersion_loss_bsz - 1):
                repulse_latents = latents.roll(shifts=i+1, dims=0)

                for j in range(self.config.latents_dispersion_num_iterations):

                    repulse_latents = repulse_latents.roll(shifts=np.random.randint(1, repulse_latents.shape[3]), dims=3)
                    if repulse_latents.shape[2] > 1:
                        repulse_latents = repulse_latents.roll(shifts=np.random.randint(1, repulse_latents.shape[2]), dims=2)

                    #cos_angle = get_cos_angle(latents, repulse_latents)
                    #dispersion_loss = dispersion_loss + (cos_angle**2).mean()
                    dispersion_loss = dispersion_loss + (latents - repulse_latents).pow(2).mean()

                total_dispersion_iterations += self.config.latents_dispersion_num_iterations

            if total_dispersion_iterations > 0:
                dispersion_loss = dispersion_loss / total_dispersion_iterations

            dispersion_loss = 1 / (dispersion_loss + 1)
            dispersion_loss = ((dispersion_loss - 1/3) * 3/2).clip(min=0)  # scale to [0,1]

            dispersion_loss = dispersion_loss.expand(latents.shape[0]) # needed for per-sample logging
        else:
            dispersion_loss = None

        if self.freeze_dae == False:
            #pre_norm_latents_var = pre_norm_latents.pow(2).mean(dim=(0,2,3)) + 1e-20
            pre_norm_latents_var = pre_norm_latents.pow(2).mean() + 1e-20
            var_kl = pre_norm_latents_var - 1 - pre_norm_latents_var.log()
            kl_loss = var_kl.mean() + 0.5 * pre_norm_latents.mean().square().mean()

            per_row_pre_norm_latents_var = pre_norm_latents.pow(2).mean(dim=(0,2,3))
            per_row_pre_norm_latents_mean = pre_norm_latents.mean(dim=(0,2,3)).abs().mean()
            #pre_norm_latents_var = pre_norm_latents.pow(2).mean(dim=1) + 1e-20
            #var_kl = pre_norm_latents_var - 1 - pre_norm_latents_var.log()
            #kl_loss = var_kl.mean() + pre_norm_latents.mean().square() * self.config.kl_mean_weight
            kl_loss = kl_loss.expand(latents.shape[0]) # needed for per-sample logging

            phase_invariance_loss_weight = self.config.phase_invariance_loss_weight
            dispersion_loss_weight = self.config.latents_dispersion_loss_weight
            if self.trainer.global_step < self.config.latents_regularization_warmup_steps:
                warmup_factor = self.trainer.global_step / self.config.latents_regularization_warmup_steps
                phase_invariance_loss_weight *= warmup_factor
                dispersion_loss_weight *= warmup_factor

            kl_loss_weight = self.config.kl_loss_weight
            if self.trainer.global_step < self.config.kl_warmup_steps:
                kl_loss_weight *= self.trainer.global_step / self.config.kl_warmup_steps
        else:
            ddec_cond = ddec_cond.detach()

        normalized_mdct_samples = mdct_samples / mdct_psd
        #normalized_mdct_samples = mdct_samples / mod.detach()
        loss_weight = 1 / mdct_psd.pow(0.75)
        target = mdct_samples
        logs = self.unet_train_batch(normalized_mdct_samples, audio_embeddings, ddec_cond, loss_weight=loss_weight, target=target, mod=mod)
        
        logs.update({
            "io_stats/ddec_cond_std": ddec_cond.std(dim=(1,2,3)),
            "io_stats/ddec_cond_mean": ddec_cond.mean(dim=(1,2,3)),
            "io_stats/mdct_std": mdct_samples.std(dim=(1,2,3)),
            "io_stats/normalized_mdct_std": normalized_mdct_samples.std(dim=(1,2,3)),
        })
        if self.freeze_dae == False:

            logs.update({
                "io_stats/latents_std": latents.std(dim=1).detach(),
                "io_stats/latents_mean": latents.mean(dim=1).detach(),
                "io_stats/latents_pre_norm_std": pre_norm_latents_var.sqrt(),
                "io_stats/per_row_latents_pre_norm_std": per_row_pre_norm_latents_var.sqrt(),
                "io_stats/per_row_latents_mean": per_row_pre_norm_latents_mean,
                "loss/kl_latents": kl_loss.detach(),
                "loss_weight/kl_latents": kl_loss_weight,
                "loss_weight/phase_invariance": phase_invariance_loss_weight,
                "loss_weight/dispersion": dispersion_loss_weight,
            })

            logs["loss"] = logs["loss"] + kl_loss * kl_loss_weight

            if self.config.phase_invariance_loss_weight > 0:
                logs["loss"] = logs["loss"] + phase_invariance_loss * phase_invariance_loss_weight
            if phase_invariance_loss is not None:
                logs["loss/phase_invariance"] = phase_invariance_loss.detach()

            if self.config.latents_dispersion_loss_weight > 0:
                logs["loss"] = logs["loss"] + dispersion_loss * dispersion_loss_weight
            if dispersion_loss is not None:
                logs["loss/latents_dispersion"] = dispersion_loss.detach()

        if self.trainer.config.enable_debug_mode == True:
            print("mdct_samples.shape:", mdct_samples.shape)
            print("ddec_cond.shape:", ddec_cond.shape)
            print("latents.shape:", latents.shape)

        return logs