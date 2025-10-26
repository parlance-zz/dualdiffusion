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


def random_stereo_augmentation(x: torch.Tensor) -> torch.Tensor:
    
    output = x.clone()
    flip_mask = (torch.rand(x.shape[0]) > 0.5).to(x.device)
    output[flip_mask] = output[flip_mask].flip(dims=(1,))
    
    return output

@dataclass
class DiffusionDecoder_Trainer_Config(UNetTrainerConfig):

    kl_loss_weight: float = 4e-2
    kl_warmup_steps: int  = 1000

    loss_buckets_sigma_min: float = 0.0002
    loss_buckets_sigma_max: float = 11

    random_stereo_augmentation: bool = False

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

        self.format: MDCT_Format = trainer.pipeline.format.to(self.trainer.accelerator.device)

        if trainer.config.enable_model_compilation:
            self.ddec.compile(**trainer.config.compile_params)
            self.dae.compile(**trainer.config.compile_params)
            self.format.compile(**trainer.config.compile_params)

        self.logger.info(f"Training modules: {trainer.config.train_modules}")
        self.logger.info(f"KL loss weight: {self.config.kl_loss_weight} KL warmup steps: {self.config.kl_warmup_steps}")
        self.logger.info(f"Crop edges: {self.config.crop_edges}")

        if self.config.random_stereo_augmentation == True:
            self.logger.info("Using random stereo augmentation")
        else: self.logger.info("Random stereo augmentation is disabled")

        #self.loss_weight = self.format.mdct_mel_density / self.format.mdct_mel_density.mean()
        self.loss_weight = None

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

        mdct_samples: torch.Tensor = self.format.raw_to_mdct(raw_samples, random_phase_augmentation=True).detach()
        mdct_samples2: torch.Tensor = self.format.raw_to_mdct(raw_samples, random_phase_augmentation=True, dual_channel=True).detach()
        
        latents, ddec_cond, pre_norm_latents = self.dae(mdct_samples2, dae_embeddings)

        pre_norm_latents_var = pre_norm_latents.var(dim=1)
        kl_loss = pre_norm_latents.mean(dim=1).square() + pre_norm_latents_var - 1 - pre_norm_latents_var.log()
        kl_loss = kl_loss.mean(dim=(1,2))

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
            "loss/kl_latents": kl_loss,
            "loss_weight/kl_latents": kl_loss_weight,
        })

        logs["loss"] = logs["loss"] + kl_loss * kl_loss_weight

        if self.trainer.config.enable_debug_mode == True:
            print("mdct_samples.shape:", mdct_samples.shape)
            print("mdct_samples2.shape:", mdct_samples2.shape)
            print("ddec_cond.shape:", ddec_cond.shape)
            print("latents.shape:", latents.shape)

        return logs