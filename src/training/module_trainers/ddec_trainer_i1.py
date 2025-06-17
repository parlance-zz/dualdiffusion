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
from modules.daes.dae_edm2_i1 import DAE_I1
from modules.unets.unet_edm2_ddec_i1 import DDec_UNet_I1
from modules.formats.raw import RawFormat
from modules.mp_tools import normalize


def random_stereo_augmentation(x: torch.Tensor) -> torch.Tensor:
    
    output = x.clone()
    flip_mask = (torch.rand(x.shape[0]) > 0.5).to(x.device)
    output[flip_mask] = output[flip_mask].flip(dims=(1,))
    
    return output

@dataclass
class DiffusionDecoder_Trainer_Config(UNetTrainerConfig):

    add_latents_noise: float = 0
    latents_noise_warmup_steps: int = 10000

    latents_kl_loss_weight: float = 1e-2
    kl_warmup_steps: int = 250

    loss_buckets_sigma_max: float = 14
    loss_buckets_sigma_min: float = 0.00008

    random_stereo_augmentation: bool = False
    random_phase_augmentation: bool = True

    crop_edges: int = 0 # used to avoid artifacts due to mdct lapped blocks at beginning and end of sample

class DiffusionDecoder_Trainer(UNetTrainer):
    
    @torch.no_grad()
    def __init__(self, config: DiffusionDecoder_Trainer_Config, trainer: DualDiffusionTrainer) -> None:

        self.config = config
        self.trainer = trainer
        self.logger = trainer.logger

        self.ddec: DDec_UNet_I1 = trainer.get_train_module("ddec_raw")
        self.dae: DAE_I1 = trainer.get_train_module("dae")
        assert self.ddec is not None and self.dae is not None, "DiffusionDecoder_Trainer train modules ddec_raw and dae"

        self.format: RawFormat = trainer.pipeline.format.to(self.trainer.accelerator.device)

        if trainer.config.enable_model_compilation:
            self.ddec.compile(**trainer.config.compile_params)
            self.dae.compile(**trainer.config.compile_params)
            self.format.compile(**trainer.config.compile_params)

        self.logger.info(f"Training modules: {trainer.config.train_modules}")
        
        if self.config.random_stereo_augmentation == True:
            self.logger.info("Using random stereo augmentation")
        else: self.logger.info("Random stereo augmentation is disabled")

        if self.config.random_phase_augmentation == True:
            self.logger.info("Using random phase augmentation")
        else: self.logger.info("Random phase augmentation is disabled")

        self.logger.info(f"Crop edges: {self.config.crop_edges}")
        self.logger.info(f"Add latents noise: {self.config.add_latents_noise}")
        self.logger.info(f"Latents KL loss weight: {self.config.latents_kl_loss_weight}")
        self.logger.info(f"KL warmup steps: {self.config.kl_warmup_steps}")
        
        self.unet = self.ddec
        self.unet_trainer_init(crop_edges=config.crop_edges)

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

        if self.config.add_latents_noise > 0:
            if self.trainer.global_step < self.config.latents_noise_warmup_steps:
                latents_sigma = self.config.add_latents_noise * (self.trainer.global_step / self.config.latents_noise_warmup_steps)
            else:
                latents_sigma = self.config.add_latents_noise
            latents_sigma = torch.Tensor([latents_sigma]).to(device=self.trainer.accelerator.device)
        else:
            latents_sigma = None
            
        latents_kl_loss_weight = self.config.latents_kl_loss_weight
        if self.trainer.global_step < self.config.kl_warmup_steps:
            warmup_scale = self.trainer.global_step / self.config.kl_warmup_steps
            latents_kl_loss_weight *= warmup_scale

        raw_samples = self.format.scale(raw_samples,
            random_phase_augmentation=self.config.random_phase_augmentation).detach()
        latents, ddec_embeddings, latents_kld = self.dae(raw_samples, dae_embeddings, latents_sigma)

        logs = self.unet_train_batch(raw_samples, ddec_embeddings, None)
        logs["loss"] = logs["loss"] + latents_kl_loss_weight * latents_kld

        logs.update({
            "io_stats/raw_samples_std": raw_samples.std(dim=(1,2,3)),
            "io_stats/raw_samples_mean": raw_samples.mean(dim=(1,2,3)),
            "io_stats/latents_std": latents.std(dim=(1,2,3)).detach(),
            "io_stats/latents_mean": latents.mean(dim=(1,2,3)).detach(),
            "io_stats/latents_sigma": latents_sigma if latents_sigma is not None else 0,
            "loss_weight/kl_latents": latents_kl_loss_weight,
            "loss/kl_latents": latents_kld.detach(),
        })
        
        for i in range(latents.shape[2]):
            logs[f"io_stats/latents_f_std_{i}"] = latents[:, :, i, :].std(dim=(1,2)).detach()
            logs[f"io_stats/latents_f_mean_{i}"] = latents[:, :, i, :].mean(dim=2).abs().mean(dim=1).detach()

        if self.trainer.config.enable_debug_mode == True:
            print(f"Latents shape: {latents.shape}")

        return logs