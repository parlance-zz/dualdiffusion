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

from training.trainer import DualDiffusionTrainer
from training.module_trainers.unet_trainer import UNetTrainer, UNetTrainerConfig
from modules.daes.dae_edm2_p1 import DAE
from modules.unets.unet_edm2_p1_ddec import UNet
from modules.formats.mdct import MDCT_Format
from modules.mp_tools import normalize

def get_cos_angle(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

    dot = torch.einsum('bchw,bchw->bhw', x, y)
    return dot / (x.norm(dim=1) * y.norm(dim=1) + 1e-10)

@torch.no_grad()
def random_stereo_augmentation(x: torch.Tensor) -> torch.Tensor:
    
    output = x.clone()
    flip_mask = (torch.rand(x.shape[0]) > 0.5).to(x.device)
    output[flip_mask] = output[flip_mask].flip(dims=(1,))
    
    return output

@dataclass
class DiffusionDecoder_Trainer_Config(UNetTrainerConfig):

    kl_loss_weight: float = 1.5e-3
    kl_warmup_steps: int  = 1000

    phase_invariance_loss_weight: float = 1e-3
    phase_invariance_loss_bsz: int = -1
    latents_dispersion_loss_weight: float = 1e-3
    latents_dispersion_loss_bsz: int = -1
    latents_dispersion_num_iterations: int = 4
    latents_regularization_warmup_steps: int = 1000

    loss_buckets_sigma_min: float = 0.0002
    loss_buckets_sigma_max: float = 11

    random_stereo_augmentation: bool = True

    crop_edges: int = 8 # used to avoid artifacts due to mdct lapped blocks at beginning and end of sample

class DiffusionDecoder_Trainer(UNetTrainer):
    
    @torch.no_grad()
    def __init__(self, config: DiffusionDecoder_Trainer_Config, trainer: DualDiffusionTrainer) -> None:

        self.config = config
        self.trainer = trainer
        self.logger = trainer.logger

        self.ddec: UNet = trainer.get_train_module("ddec")
        self.dae: DAE = trainer.get_train_module("dae")

        assert self.ddec is not None, "DDEC module not found in train modules"
        assert self.dae is not None, "DAE module not found in train modules"

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
        
        self.format: MDCT_Format = trainer.pipeline.format.to(self.trainer.accelerator.device)

        if trainer.config.enable_model_compilation:
            self.ddec.compile(**trainer.config.compile_params)
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

        #self.loss_weight = self.format.mdct_mel_density**0.5
        #self.loss_weight /= self.format.mdct_mel_density.mean()
        self.loss_weight = None

        self.unet = self.ddec
        self.unet_trainer_init(crop_edges=0)#config.crop_edges)

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

        #mdct_samples: torch.Tensor     = self.format.raw_to_mdct(raw_samples, random_phase_augmentation=True)[:, :, :, self.config.crop_edges:-self.config.crop_edges].detach()
        #mdct_samples_dae: torch.Tensor = self.format.raw_to_mdct(raw_samples, random_phase_augmentation=True)[:, :, :, self.config.crop_edges:-self.config.crop_edges].detach()
        mdct_samples: torch.Tensor = self.format.raw_to_mdct(raw_samples, random_phase_augmentation=True)
        mdct_samples = mdct_samples[:, :, :, self.config.crop_edges:-self.config.crop_edges].detach()
        mdct_samples_dae = mdct_samples
        
        latents, ddec_cond, pre_norm_latents = self.dae(mdct_samples_dae, dae_embeddings)
        latents: torch.Tensor = latents.float()
        pre_norm_latents: torch.Tensor = pre_norm_latents.float()

        if self.config.phase_invariance_loss_bsz > 0:

            mdct_samples_dae2: torch.Tensor = self.format.raw_to_mdct(raw_samples[:self.config.phase_invariance_loss_bsz], random_phase_augmentation=True)
            mdct_samples_dae2 = mdct_samples_dae2[:, :, :, self.config.crop_edges:-self.config.crop_edges].detach()
            with torch.autocast(device_type="cuda", dtype=self.trainer.mixed_precision_dtype, enabled=self.trainer.mixed_precision_enabled):
                latents2 = self.dae.encode(mdct_samples_dae2, dae_embeddings[:self.config.phase_invariance_loss_bsz])
            
            #phase_invariance_loss = torch.nn.functional.mse_loss(latents[:self.config.phase_invariance_loss_bsz], latents2.float(), reduction="none").mean()
            cos_angle = get_cos_angle(latents[:self.config.phase_invariance_loss_bsz], latents2.float())
            phase_invariance_loss = (1 - cos_angle).mean() / 2

            phase_invariance_loss = phase_invariance_loss.expand(latents.shape[0]) # needed for per-sample logging
            phase_invariance_loss_nll = phase_invariance_loss / self.dae.phase_invariance_error_logvar.exp() + self.dae.phase_invariance_error_logvar
        else:
            phase_invariance_loss = phase_invariance_loss_nll = None

        if self.config.latents_dispersion_loss_bsz > 0:
            dispersion_loss = torch.zeros(1, device=latents.device)
            total_dispersion_iterations = 0

            for i in range(self.config.latents_dispersion_loss_bsz - 1):

                repulse_latents = latents.roll(shifts=i+1, dims=0)

                for j in range(self.config.latents_dispersion_num_iterations):
                    rnd_indices_w = torch.randperm(repulse_latents.shape[3], device=latents.device)
                    rnd_indices_h = torch.randperm(repulse_latents.shape[2], device=latents.device)
                    repulse_latents = repulse_latents[:, :, :, rnd_indices_w]
                    repulse_latents = repulse_latents[:, :, rnd_indices_h, :]

                    #repulse_latents = repulse_latents.roll(shifts=np.random.randint(1, repulse_latents.shape[3]), dims=3)
                    #repulse_latents = repulse_latents.roll(shifts=np.random.randint(1, repulse_latents.shape[2]), dims=2)

                    cos_angle = get_cos_angle(latents, repulse_latents)
                    dispersion_loss = dispersion_loss + (cos_angle**2).mean()

                total_dispersion_iterations += self.config.latents_dispersion_num_iterations

            dispersion_loss = dispersion_loss / total_dispersion_iterations
            dispersion_loss = dispersion_loss.expand(latents.shape[0]) # needed for per-sample logging
            dispersion_loss_nll = dispersion_loss / self.dae.dispersion_error_logvar.exp() + self.dae.dispersion_error_logvar
        else:
            dispersion_loss = dispersion_loss_nll = None

        pre_norm_latents_var = pre_norm_latents.var(dim=1)
        var_kl = pre_norm_latents_var - 1 - pre_norm_latents_var.log()
        kl_loss = var_kl.mean(dim=(1,2)) + pre_norm_latents.mean(dim=(1,2,3)).square()

        phase_invariance_loss_weight = self.config.phase_invariance_loss_weight
        dispersion_loss_weight = self.config.latents_dispersion_loss_weight
        if self.trainer.global_step < self.config.latents_regularization_warmup_steps:
            warmup_factor = self.trainer.global_step / self.config.latents_regularization_warmup_steps
            phase_invariance_loss_weight *= warmup_factor
            dispersion_loss_weight *= warmup_factor

        kl_loss_weight = self.config.kl_loss_weight
        if self.trainer.global_step < self.config.kl_warmup_steps:
            kl_loss_weight *= self.trainer.global_step / self.config.kl_warmup_steps

        logs = self.unet_train_batch(mdct_samples, audio_embeddings, ddec_cond, self.loss_weight)
        
        logs.update({
            "io_stats/ddec_cond_std": ddec_cond.std(dim=(1,2,3)),
            "io_stats/ddec_cond_mean": ddec_cond.mean(dim=(1,2,3)),
            "io_stats/mdct_std": mdct_samples.std(dim=(1,2,3)),
            "io_stats/mdct_mean": mdct_samples.mean(dim=(1,2,3)),
            "io_stats/latents_std": latents.std(dim=1).detach(),
            "io_stats/latents_mean": latents.mean(dim=1).detach(),
            "io_stats/latents_pre_norm_std": pre_norm_latents_var.sqrt(),
            "loss/kl_latents": kl_loss.detach(),
            "loss_weight/kl_latents": kl_loss_weight,
            "loss_weight/phase_invariance": phase_invariance_loss_weight,
            "loss_weight/dispersion": dispersion_loss_weight
        })

        logs["loss"] = logs["loss"] + kl_loss * kl_loss_weight

        if self.config.phase_invariance_loss_weight > 0:
            logs["loss"] = logs["loss"] + phase_invariance_loss_nll * phase_invariance_loss_weight
        if phase_invariance_loss is not None:
            logs["loss/phase_invariance"] = phase_invariance_loss.detach()
            logs["loss/phase_invariance_nll"] = phase_invariance_loss_nll.detach()

        if self.config.latents_dispersion_loss_weight > 0:
            logs["loss"] = logs["loss"] + dispersion_loss_nll * dispersion_loss_weight
        if dispersion_loss is not None:
            logs["loss/latents_dispersion"] = dispersion_loss.detach()
            logs["loss/latents_dispersion_nll"] = dispersion_loss_nll.detach()

        if self.trainer.config.enable_debug_mode == True:
            print("mdct_samples.shape:", mdct_samples.shape)
            print("mdct_samples_dae.shape:", mdct_samples_dae.shape)
            print("ddec_cond.shape:", ddec_cond.shape)
            print("latents.shape:", latents.shape)

        return logs