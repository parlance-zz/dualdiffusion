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

from training.trainer import DualDiffusionTrainer
from training.module_trainers.module_trainer import ModuleTrainer, ModuleTrainerConfig
from training.module_trainers.unet_trainer_q4 import UNetTrainerConfig, UNetTrainer
from modules.unets.unet_edm2_q4_ddec import UNet
from modules.daes.dae_edm2_q4 import DAE
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

    ddec: dict[str, Any]

    random_stereo_augmentation: bool = True
    random_phase_augmentation: bool  = True

    phase_invariance_loss_weight: float = 0
    kl_loss_weight: float = 0

    crop_edges: int = 4 # used to avoid artifacts due to mdct lapped blocks at beginning and end of sample

class DiffusionDecoder_Trainer(ModuleTrainer):
    
    @torch.no_grad()
    def __init__(self, config: DiffusionDecoder_Trainer_Config, trainer: DualDiffusionTrainer) -> None:

        self.config = config
        self.trainer = trainer
        self.logger = trainer.logger

        self.ddec: UNet = trainer.get_train_module("ddec")
        self.dae: DAE = trainer.get_train_module("dae")
        self.format: MS_MDCT_DualFormat = trainer.pipeline.format.to(self.trainer.accelerator.device)

        if trainer.config.enable_model_compilation:
            self.ddec.compile(**trainer.config.compile_params)
            self.dae.compile(**trainer.config.compile_params)
            self.format.compile(**trainer.config.compile_params)

        if self.config.random_stereo_augmentation == True:
            self.logger.info("Using random stereo augmentation")
        else: self.logger.info("Random stereo augmentation is disabled")

        if self.config.random_phase_augmentation == True:
            self.logger.info("Using random phase augmentation")
        else: self.logger.info("Random phase augmentation is disabled")

        self.logger.info(f"Crop edges: {self.config.crop_edges}")

        self.logger.info("DDEC Trainer:")
        self.ddec_trainer = UNetTrainer(UNetTrainerConfig(**config.ddec), trainer, self.ddec, "ddec")
           
    @torch.no_grad()
    def init_batch(self, validation: bool = False) -> Optional[dict[str, Union[torch.Tensor, float]]]:
        return self.ddec_trainer.init_batch(validation)
    
    def train_batch(self, batch: dict) -> Optional[dict[str, Union[torch.Tensor, float]]]:

        with torch.no_grad(): # prepare model inputs
            
            if "audio_embeddings" in batch:
                audio_embeddings = normalize(batch["audio_embeddings"]).detach()
                dae_embeddings = self.dae.get_embeddings(audio_embeddings)
            else:
                audio_embeddings = dae_embeddings = None

            if self.config.random_stereo_augmentation == True:
                raw_samples = random_stereo_augmentation(batch["audio"])
            else:
                raw_samples = batch["audio"]

            #mdct = self.format.raw_to_mdct(raw_samples, random_phase_augmentation=self.config.random_phase_augmentation)
            #mdct = mdct[..., self.config.crop_edges:-self.config.crop_edges]
            #mdct2 = self.format.raw_to_mdct(raw_samples, random_phase_augmentation=self.config.random_phase_augmentation)
            #mdct2 = mdct2[..., self.config.crop_edges:-self.config.crop_edges]
            mdct_phase, mdct_psd = self.format.raw_to_mdct_phase_psd(raw_samples, random_phase_augmentation=self.config.random_phase_augmentation)
            mdct_phase = mdct_phase[..., self.config.crop_edges:-self.config.crop_edges]
            mdct_psd = mdct_psd[..., self.config.crop_edges:-self.config.crop_edges]
            mdct_phase_psd = torch.cat((mdct_phase, mdct_psd), dim=1)

            #spec = self.format.raw_to_spec(raw_samples)
            #spec = spec[..., self.config.crop_edges:-self.config.crop_edges]

            #mel_spec = self.format.raw_to_mel_spec(raw_samples)
            #mel_spec = mel_spec[..., self.config.crop_edges:-self.config.crop_edges]

        #latents, ddec_cond, _ = self.dae(mdct, dae_embeddings)
        latents, ddec_cond, _ = self.dae(mdct_phase_psd, dae_embeddings)
        #latents, ddec_cond, _ = self.dae(mel_spec, dae_embeddings)
        #latents, ddec_cond, _ = self.dae(spec, dae_embeddings)
        latents: torch.Tensor = latents.float()
        
        if self.config.phase_invariance_loss_weight > 0:
            raise NotImplementedError()
            with torch.autocast(device_type="cuda", dtype=self.trainer.mixed_precision_dtype, enabled=self.trainer.mixed_precision_enabled):
                latents2 = self.dae.encode(mdct2, dae_embeddings).float()

            phase_invariance_loss = torch.nn.functional.mse_loss(latents, latents2, reduction="none").mean(dim=(1,2,3))
        else:
            phase_invariance_loss = None

        #kl_loss = latents.mean().pow(2).expand(latents.shape[0])
        latents_var = latents.pow(2).mean() + 1e-20
        var_kl = latents_var - 1 - latents_var.log()
        kl_loss = var_kl.mean() + latents.mean().square()
        kl_loss = kl_loss.expand(latents.shape[0])
    
        logs = {
            "io_stats/ddec_cond_var": ddec_cond.var(dim=(1,2,3)),
            "io_stats/ddec_cond_mean": ddec_cond.mean(dim=(1,2,3)),
            #"io_stats/mdct_var": mdct.var(dim=(1,2,3)),
            #"io_stats/spec_var": spec.var(dim=(1,2,3)),
            #"io_stats/spec_mean": spec.mean(dim=(1,2,3)),
            #"io_stats/mel_spec_var": mel_spec.var(dim=(1,2,3)),
            #"io_stats/mel_spec_mean": mel_spec.mean(dim=(1,2,3)),
            "io_stats/mdct_phase_var": mdct_phase.var(dim=(1,2,3)),
            "io_stats/mdct_psd_var": mdct_psd.var(dim=(1,2,3)),
            "io_stats/mdct_psd_mean": mdct_psd.mean(dim=(1,2,3)),
            "io_stats/mdct_phase_psd_var": mdct_phase_psd.var(dim=(1,2,3)),
            "io_stats/mdct_phase_psd_mean": mdct_phase_psd.mean(dim=(1,2,3)),
            "io_stats/latents_var": latents.var(dim=(1,2,3)),
            "io_stats/latents_mean": latents.mean(dim=(1,2,3)),
            "loss_weight/phase_invariance": self.config.phase_invariance_loss_weight,
            "loss_weight/kl": self.config.kl_loss_weight,
            "loss/phase_invariance": phase_invariance_loss,
            "loss/kl": kl_loss
        }

        #logs.update(self.ddec_trainer.train_batch(mdct, audio_embeddings, ddec_cond))
        logs.update(self.ddec_trainer.train_batch(mdct_phase_psd, audio_embeddings, ddec_cond))
        logs["loss"] = logs["loss/ddec"] + self.config.kl_loss_weight * kl_loss

        if phase_invariance_loss is not None:
            logs["loss"] = logs["loss"] + self.config.phase_invariance_loss_weight * phase_invariance_loss

        if self.trainer.config.enable_debug_mode == True:
            #print("mdct.shape:", mdct.shape)
            #print("spec.shape:", spec.shape)
            #print("mel_spec.shape:", mel_spec.shape)
            print("mdct_phase.shape:", mdct_phase.shape)
            print("mdct_psd.shape:", mdct_psd.shape)
            print("mdct_phase_psd.shape:", mdct_phase_psd.shape)
            print("ddec_cond.shape:", ddec_cond.shape)
            print("latents.shape:", latents.shape)

        return logs
      
    @torch.no_grad()
    def finish_batch(self) -> Optional[dict[str, Union[torch.Tensor, float]]]:
        return self.ddec_trainer.finish_batch()