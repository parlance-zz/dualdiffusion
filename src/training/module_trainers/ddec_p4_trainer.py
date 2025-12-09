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
from modules.daes.dae_edm2_p4 import DAE
from modules.unets.unet_edm2_p4_ddec import UNet
from modules.formats.ms_mdct_dual_2 import MS_MDCT_DualFormat
from modules.mp_tools import normalize


@torch.no_grad()
def random_stereo_augmentation(x: torch.Tensor) -> torch.Tensor:
    
    output = x.clone()
    flip_mask = (torch.rand(x.shape[0]) > 0.5).to(x.device)
    output[flip_mask] = output[flip_mask].flip(dims=(1,))
    
    return output

@dataclass
class DiffusionDecoder_Trainer_Config(ModuleTrainerConfig):

    ddecm: dict[str, Any]
    ddecp: dict[str, Any]

    kl_loss_weight: float = 3e-2
    kl_warmup_steps: int  = 2000

    ddecp_loss_weight_multiplier: float = 1.3
    ddecm_loss_weight_multiplier: float = 0.75

    ddecp_loss_weight_mel_density_exponent: float = 0.5
    ddecm_loss_weight_mel_density_exponent: float = 0.25

    phase_invariance_loss_weight: float = 1
    phase_invariance_loss_bsz: int = -1
    latents_dispersion_loss_weight: float = 0
    latents_dispersion_loss_bsz: int = 2
    latents_dispersion_num_iterations: int = 1
    latents_regularization_warmup_steps: int = 25000

    random_stereo_augmentation: bool = False
    random_phase_augmentation: bool  = False

    crop_edges: int = 4 # used to avoid artifacts due to mdct lapped blocks at beginning and end of sample

class DiffusionDecoder_Trainer(ModuleTrainer):
    
    @torch.no_grad()
    def __init__(self, config: DiffusionDecoder_Trainer_Config, trainer: DualDiffusionTrainer) -> None:

        self.config = config
        self.trainer = trainer
        self.logger = trainer.logger

        self.ddecp: UNet = trainer.get_train_module("ddecp")
        self.ddecm: UNet = trainer.get_train_module("ddecm")
        self.dae: DAE = trainer.get_train_module("dae")

        assert self.ddecp is not None
        assert self.ddecm is not None
        assert self.dae is not None

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
            self.ddecp.compile(**trainer.config.compile_params)
            self.ddecm.compile(**trainer.config.compile_params)
            self.dae.compile(**trainer.config.compile_params)
            self.format.compile(**trainer.config.compile_params)

        self.logger.info(f"Training modules: {trainer.config.train_modules}")
        self.logger.info(f"KL loss weight: {self.config.kl_loss_weight} KL warmup steps: {self.config.kl_warmup_steps}")
        self.logger.info(f"Latents phase-invariance loss weight: {self.config.phase_invariance_loss_weight} Batch size: {self.config.phase_invariance_loss_bsz}")
        self.logger.info(f"Latents dispersion loss weight: {self.config.latents_dispersion_loss_weight} Batch size: {self.config.latents_dispersion_loss_bsz}")
        self.logger.info(f"Latents dispersion loss num iterations: {self.config.latents_dispersion_num_iterations}")
        self.logger.info(f"Latents regularization loss warmup steps: {self.config.latents_regularization_warmup_steps}")
        self.logger.info(f"Crop edges: {self.config.crop_edges}")

        if self.config.random_stereo_augmentation == True:
            self.logger.info("Using random stereo augmentation")
        else: self.logger.info("Random stereo augmentation is disabled")

        self.loss_weight_p = self.format.mdct_mel_density.clone().pow(
            self.config.ddecp_loss_weight_mel_density_exponent).requires_grad_(False)
        self.loss_weight_p /= self.loss_weight_p.mean()
        self.loss_weight_p *= self.config.ddecp_loss_weight_multiplier

        self.loss_weight_m = self.format.mdct_mel_density.clone().pow(
            self.config.ddecm_loss_weight_mel_density_exponent).requires_grad_(False)
        self.loss_weight_m /= self.loss_weight_m.mean()
        self.loss_weight_m *= self.config.ddecm_loss_weight_multiplier

        self.logger.info("DDEC-P trainer:")
        self.ddecp_trainer = UNetTrainer(UNetTrainerConfig(**config.ddecp), trainer, self.ddecp, "ddecp")
        self.logger.info("DDEC-M trainer:")
        self.ddecm_trainer = UNetTrainer(UNetTrainerConfig(**config.ddecm), trainer, self.ddecm, "ddecm")

    @torch.no_grad()
    def init_batch(self, validation: bool = False) -> Optional[dict[str, Union[torch.Tensor, float]]]:
        
        self.ddecp_trainer.init_batch(validation)
        self.ddecm_trainer.init_batch(validation)

        return None
    
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
        mdct_psd: torch.Tensor = self.format.raw_to_mdct_psd(raw_samples)
        mdct_psd = mdct_psd[:, :, :, self.config.crop_edges:-self.config.crop_edges].detach()
        mdct_psd_clipped = mdct_psd.clip(min=1e-4).detach()
        mdct_psd_normalized = self.format.normalize_psd(mdct_psd).detach()
        mdct_phase = (mdct_samples / (mdct_psd_clipped / 2**0.5)).detach()
        #mel_spec = self.format.raw_to_mel_spec(raw_samples)[:, :, :, self.config.crop_edges:-self.config.crop_edges].detach()

        #dae_input = torch.cat((mdct_samples, mdct_psd_normalized), dim=1)
        dae_input = torch.cat((mdct_phase, mdct_psd_normalized), dim=1)
        latents, ddec_cond, pre_norm_latents = self.dae(dae_input, dae_embeddings)
        latents: torch.Tensor = latents.float()
        pre_norm_latents: torch.Tensor = pre_norm_latents.float()

        if self.config.phase_invariance_loss_bsz > 0:

            mdct_samples2: torch.Tensor = self.format.raw_to_mdct(raw_samples[:self.config.phase_invariance_loss_bsz], random_phase_augmentation=True)
            mdct_samples2 = mdct_samples2[:, :, :, self.config.crop_edges:-self.config.crop_edges].detach()
            #dae_input2 = torch.cat((mdct_samples2, mdct_psd_normalized[:self.config.phase_invariance_loss_bsz]), dim=1)
            mdct_phase2 = (mdct_samples2 / (mdct_psd_clipped[:self.config.phase_invariance_loss_bsz] / 2**0.5)).detach()
            dae_input2 = torch.cat((mdct_phase2, mdct_psd_normalized[:self.config.phase_invariance_loss_bsz]), dim=1)
            dae_embeddings2 = dae_embeddings[:self.config.phase_invariance_loss_bsz] if dae_embeddings is not None else None
            
            with torch.autocast(device_type="cuda", dtype=self.trainer.mixed_precision_dtype, enabled=self.trainer.mixed_precision_enabled):
                latents2 = self.dae.encode(dae_input2, dae_embeddings2)

            phase_invariance_loss = (latents[:self.config.phase_invariance_loss_bsz] - latents2.float()).pow(2).mean().expand(latents.shape[0])
        else:
            phase_invariance_loss = None

        if self.config.latents_dispersion_loss_bsz > 0:
            dispersion_loss = torch.zeros(1, device=latents.device)
            total_dispersion_iterations = 0

            for i in range(self.config.latents_dispersion_loss_bsz - 1):
                repulse_latents = latents.roll(shifts=i+1, dims=0)

                for j in range(self.config.latents_dispersion_num_iterations):

                    repulse_latents = repulse_latents.roll(shifts=np.random.randint(1, repulse_latents.shape[3]), dims=3)
                    if repulse_latents.shape[2] > 1:
                        repulse_latents = repulse_latents.roll(shifts=np.random.randint(1, repulse_latents.shape[2]), dims=2)

                    dispersion_loss = dispersion_loss + (latents - repulse_latents).pow(2).mean()

                total_dispersion_iterations += self.config.latents_dispersion_num_iterations

            if total_dispersion_iterations > 0:
                dispersion_loss = dispersion_loss / total_dispersion_iterations

            dispersion_loss = 1 / (dispersion_loss + 1)
            dispersion_loss = ((dispersion_loss - 1/3) * 3/2).clip(min=0).expand(latents.shape[0])
        else:
            dispersion_loss = None

        pre_norm_latents_var = pre_norm_latents.pow(2).mean() + 1e-20
        var_kl = pre_norm_latents_var - 1 - pre_norm_latents_var.log()
        kl_loss = var_kl.mean() + 0.5 * pre_norm_latents.mean().square().mean()
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
        
        logs = {
            "loss": kl_loss * kl_loss_weight,
            "io_stats/ddec_cond_var": ddec_cond.var(dim=(1,2,3)),
            "io_stats/ddec_cond_mean": ddec_cond.mean(dim=(1,2,3)),
            "io_stats/mdct_var": mdct_samples.var(dim=(1,2,3)),
            #"io_stats/mel_spec_var": mel_spec.var(dim=(1,2,3)),
            #"io_stats/mel_spec_mean": mel_spec.mean(dim=(1,2,3)),
            "io_stats/latents_var": latents.var(dim=(1,2,3)).detach(),
            "io_stats/latents_mean": latents.mean(dim=(1,2,3)).detach(),

            "io_stats_ddecp/mdct_phase_var": mdct_phase.var(dim=(1,2,3)),
            "io_stats_ddecm/mdct_psd_normalized_var": mdct_psd_normalized.var(dim=(1,2,3)),
            "io_stats_ddecm/mdct_psd_normalized_mean": mdct_psd_normalized.mean(dim=(1,2,3)),

            "loss/kl_latents": kl_loss.detach(),
            "loss_weight/kl_latents": kl_loss_weight,
            "loss_weight/phase_invariance": phase_invariance_loss_weight,
            "loss_weight/dispersion": dispersion_loss_weight,
        }

        if self.config.phase_invariance_loss_weight > 0:
            logs["loss"] = logs["loss"] + phase_invariance_loss * phase_invariance_loss_weight
            logs["loss/phase_invariance"] = phase_invariance_loss.detach()

        if self.config.latents_dispersion_loss_weight > 0:
            logs["loss"] = logs["loss"] + dispersion_loss * dispersion_loss_weight
            logs["loss/latents_dispersion"] = dispersion_loss.detach()

        noise = torch.randn_like(mdct_samples)
        loss_weight_ddecp = self.loss_weight_p * mdct_psd.pow(0.25)
        logs.update(self.ddecp_trainer.train_batch(mdct_phase, audio_embeddings, ddec_cond, loss_weight=loss_weight_ddecp, noise=noise))
        loss_weight_ddecm = self.loss_weight_m
        logs.update(self.ddecm_trainer.train_batch(mdct_psd_normalized, audio_embeddings, ddec_cond, loss_weight=loss_weight_ddecm, noise=noise))

        logs["loss"] = logs["loss"] + logs["loss/ddecp"] + logs["loss/ddecm"]

        dynamic_range_ddecm = mdct_psd_normalized.amax(dim=(1,2,3)) - mdct_psd_normalized.amin(dim=(1,2,3))
        logs["io_stats_ddecm/dynamic_range"] = dynamic_range_ddecm

        if self.trainer.config.enable_debug_mode == True:
            print("mdct_samples.shape:", mdct_samples.shape)
            #print("mel_spec.shape:", mel_spec.shape)
            print("ddec_cond.shape:", ddec_cond.shape)
            print("latents.shape:", latents.shape)

        return logs
      
    @torch.no_grad()
    def finish_batch(self) -> Optional[dict[str, Union[torch.Tensor, float]]]:

        logs = {}
        logs.update(self.ddecp_trainer.finish_batch())
        logs.update(self.ddecm_trainer.finish_batch())

        return logs