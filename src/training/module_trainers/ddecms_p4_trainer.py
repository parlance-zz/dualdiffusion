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

    ddecms: dict[str, Any]

    kl_loss_weight: float = 1e-2
    kl_warmup_steps: int  = 2000
    
    shift_equivariance_loss_weight: float = 1e-2
    shift_equivariance_warmup_steps: int  = 2000

    random_stereo_augmentation: bool = False
    random_phase_augmentation: bool  = False

    crop_edges: int = 4 # used to avoid artifacts due to mdct lapped blocks at beginning and end of sample

class DiffusionDecoder_Trainer(ModuleTrainer):
    
    @torch.no_grad()
    def __init__(self, config: DiffusionDecoder_Trainer_Config, trainer: DualDiffusionTrainer) -> None:

        self.config = config
        self.trainer = trainer
        self.logger = trainer.logger

        #self.ddecmp: UNet = trainer.get_train_module("ddecmp")
        self.ddecms: UNet = trainer.get_train_module("ddecms")
        self.dae: DAE = trainer.get_train_module("dae")
        
        self.format: MS_MDCT_DualFormat = trainer.pipeline.format.to(self.trainer.accelerator.device)

        if trainer.config.enable_model_compilation:
            #self.ddecmp.compile(**trainer.config.compile_params)
            self.ddecms.compile(**trainer.config.compile_params)
            self.dae.compile(**trainer.config.compile_params)
            self.format.compile(**trainer.config.compile_params)

        self.logger.info(f"Training modules: {trainer.config.train_modules}")
        self.logger.info(f"KL loss weight: {self.config.kl_loss_weight} KL warmup steps: {self.config.kl_warmup_steps}")
        self.logger.info(f"Shift equivariance loss weight: {self.config.shift_equivariance_loss_weight}")

        assert self.config.crop_edges * 2 == self.dae.downsample_ratio
        self.logger.info(f"Crop edges: {self.config.crop_edges}")

        if self.config.random_stereo_augmentation == True:
            self.logger.info("Using random stereo augmentation")
        else: self.logger.info("Random stereo augmentation is disabled")

        #self.logger.info("DDECMP Trainer:")
        #self.ddecmp_trainer = UNetTrainer(UNetTrainerConfig(**config.ddecmp), trainer, self.ddecmp, "ddecmp")
        self.logger.info("DDECMS Trainer:")
        self.ddecms_trainer = UNetTrainer(UNetTrainerConfig(**config.ddecms), trainer, self.ddecms, "ddecms")

    def shift_equivariance_loss(self, mel_spec: torch.Tensor, dae_embeddings: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:

        crop_left = np.random.randint(1, self.config.crop_edges * 2)
        crop_right = self.config.crop_edges * 2 - crop_left
        mel_spec = mel_spec[..., crop_left:-crop_right]

        with torch.autocast(device_type="cuda", dtype=self.trainer.mixed_precision_dtype, enabled=self.trainer.mixed_precision_enabled):
            latents2 = self.dae.encode(mel_spec, dae_embeddings)

        latents_up: torch.Tensor = torch.repeat_interleave(latents, self.dae.downsample_ratio, dim=-1)
        latents_up_cropped = latents_up[..., crop_left:-crop_right]
        latents_down: torch.Tensor = torch.nn.functional.avg_pool2d(latents_up_cropped, kernel_size=(1,self.dae.downsample_ratio))

        return (latents_down - latents2.float())[..., 2:-2].pow(2).mean().expand(latents.shape[0])
            
    @torch.no_grad()
    def init_batch(self, validation: bool = False) -> Optional[dict[str, Union[torch.Tensor, float]]]:
        #return self.ddecmp_trainer.init_batch(validation)
        return self.ddecms_trainer.init_batch(validation)
    
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

        mel_spec = self.format.raw_to_mel_spec(raw_samples)
        mel_spec = mel_spec[..., self.config.crop_edges:-self.config.crop_edges]
        #mel_spec_linear = self.format.mel_spec_to_linear(mel_spec)

        #mdct = self.format.raw_to_mdct(raw_samples, random_phase_augmentation=self.config.random_phase_augmentation)
        #mdct = mdct[..., self.config.crop_edges:-self.config.crop_edges]

        latents, ddec_cond, pre_norm_latents = self.dae(mel_spec, dae_embeddings)
        latents: torch.Tensor = latents.float()
        pre_norm_latents: torch.Tensor = pre_norm_latents.float()

        if self.config.shift_equivariance_loss_weight > 0:
            shift_equivariance_loss = self.shift_equivariance_loss(mel_spec, dae_embeddings, latents)
        else:
            shift_equivariance_loss = None

        pre_norm_latents_var = pre_norm_latents.pow(2).mean() + 1e-20
        var_kl = pre_norm_latents_var - 1 - pre_norm_latents_var.log()
        kl_loss = var_kl.mean() + 0.5 * pre_norm_latents.mean().square().mean()
        kl_loss = kl_loss.expand(latents.shape[0]) # needed for per-sample logging

        shift_equivariance_loss_weight = self.config.shift_equivariance_loss_weight
        if self.trainer.global_step < self.config.shift_equivariance_warmup_steps:
            warmup_factor = self.trainer.global_step / self.config.shift_equivariance_warmup_steps
            shift_equivariance_loss_weight *= warmup_factor

        kl_loss_weight = self.config.kl_loss_weight
        if self.trainer.global_step < self.config.kl_warmup_steps:
            kl_loss_weight *= self.trainer.global_step / self.config.kl_warmup_steps
        
        logs = {
            "loss": kl_loss * kl_loss_weight,
            "io_stats/ddec_cond_var": ddec_cond.var(dim=(1,2,3)),
            "io_stats/ddec_cond_mean": ddec_cond.mean(dim=(1,2,3)),
            "io_stats/latents_var": latents.var(dim=(1,2,3)).detach(),
            "io_stats/latents_mean": latents.mean(dim=(1,2,3)).detach(),

            "io_stats/mel_spec_var": mel_spec.var(dim=(1,2,3)),
            "io_stats/mel_spec_mean": mel_spec.mean(dim=(1,2,3)),
            #"io_stats/mel_spec_linear_var": mel_spec_linear.var(dim=(1,2,3)),
            #"io_stats/mel_spec_linear_mean": mel_spec_linear.mean(dim=(1,2,3)),
            #"io_stats/mel_spec_linear_mean_square": mel_spec_linear.pow(2).mean(dim=(1,2,3)),
            #"io_stats/mdct_var": mdct.var(dim=(1,2,3)),

            "loss/kl_latents": kl_loss.detach(),
            "loss_weight/kl_latents": kl_loss_weight,
            "loss_weight/shift_equivariance": shift_equivariance_loss_weight,
        }

        if self.config.shift_equivariance_loss_weight > 0:
            logs["loss"] = logs["loss"] + shift_equivariance_loss * shift_equivariance_loss_weight
            logs["loss/shift_equivariance"] = shift_equivariance_loss.detach()

        #logs.update(self.ddecmp_trainer.train_batch(mdct, audio_embeddings, mel_spec_linear))
        #logs["loss"] = logs["loss/ddecmp"]
        logs.update(self.ddecms_trainer.train_batch(mel_spec, audio_embeddings, ddec_cond))
        logs["loss"] = logs["loss"] + logs["loss/ddecms"]

        if self.trainer.config.enable_debug_mode == True:
            #print("mdct.shape:", mdct.shape)
            #print("mel_spec_linear.shape:", mel_spec_linear.shape)
            print("mel_spec.shape:", mel_spec.shape)

        return logs
      
    @torch.no_grad()
    def finish_batch(self) -> Optional[dict[str, Union[torch.Tensor, float]]]:
        #return self.ddecmp_trainer.finish_batch()
        return self.ddecms_trainer.finish_batch()