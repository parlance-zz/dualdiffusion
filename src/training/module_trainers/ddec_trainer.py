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
from modules.daes.dae import DualDiffusionDAE
from modules.unets.unet_edm2_ddec_d1 import DDec_UNet_D1
from modules.formats.ms_mdct_dual import MS_MDCT_DualFormat
from modules.mp_tools import normalize


def random_stereo_augmentation(x: torch.Tensor) -> torch.Tensor:
    
    output = x.clone()
    flip_mask = (torch.rand(x.shape[0]) > 0.5).to(x.device)
    output[flip_mask] = output[flip_mask].flip(dims=(1,))
    
    return output

@dataclass
class DiffusionDecoder_Trainer_Config(UNetTrainerConfig):

    loss_buckets_sigma_min: float = 0.00008
    loss_buckets_sigma_max: float = 14

    random_stereo_augmentation: bool = True
    random_phase_augmentation: bool = True

    crop_edges: int = 8 # used to avoid artifacts due to mdct lapped blocks at beginning and end of sample

class DiffusionDecoder_Trainer(UNetTrainer):
    
    @torch.no_grad()
    def __init__(self, config: DiffusionDecoder_Trainer_Config, trainer: DualDiffusionTrainer) -> None:

        self.config = config
        self.trainer = trainer
        self.logger = trainer.logger

        if "ddec_mdct" in trainer.config.train_modules:
            self.train_mode = "ddec_mdct"
            self.ddec: DDec_UNet_D1 = trainer.get_train_module("ddec_mdct")
        elif "ddec_ms" in trainer.config.train_modules:
            self.train_mode = "ddec_ms"
            self.ddec: DDec_UNet_D1 = trainer.get_train_module("ddec_ms")
        else:
            raise ValueError("Neither ddec_mdct or ddec_ms found in train_modules")

        if len(trainer.config.train_modules) > 1:
            raise ValueError("DiffusionDecoder_Trainer can only be used with a single train module")
                             
        if self.train_mode == "ddec_ms":
            self.dae: DualDiffusionDAE = trainer.pipeline.dae.to(
                dtype=torch.bfloat16, device=trainer.accelerator.device).requires_grad_(False)
        else:
            self.dae = None

        self.format: MS_MDCT_DualFormat = trainer.pipeline.format.to(self.trainer.accelerator.device)

        if trainer.config.enable_model_compilation:
            self.ddec.compile(**trainer.config.compile_params)
            if self.dae is not None:
                self.dae.compile(**trainer.config.compile_params)
            self.format.compile(**trainer.config.compile_params)

        self.logger.info(f"Training module: {trainer.config.train_modules}")
        
        if self.config.random_stereo_augmentation == True:
            self.logger.info("Using random stereo augmentation")
        else: self.logger.info("Random stereo augmentation is disabled")

        if self.config.random_phase_augmentation == True:
            self.logger.info("Using random phase augmentation")
        else: self.logger.info("Random phase augmentation is disabled")

        self.unet = self.ddec
        self.unet_trainer_init(crop_edges=config.crop_edges)

    def train_batch(self, batch: dict) -> Optional[dict[str, Union[torch.Tensor, float]]]:

        # prepare model inputs
        if "audio_embeddings" in batch:
            audio_embeddings = normalize(batch["audio_embeddings"]).detach()
            if self.dae is not None:
                dae_embeddings = self.dae.get_embeddings(audio_embeddings.to(dtype=self.dae.dtype))
            else:
                dae_embeddings = None
        else:
            audio_embeddings = dae_embeddings = None

        if self.config.random_stereo_augmentation == True:
            raw_samples = random_stereo_augmentation(batch["audio"])
        else:
            raw_samples = batch["audio"]

        mel_spec = self.format.raw_to_mel_spec(raw_samples).detach()

        if self.train_mode == "ddec_mdct":
            recon_mel_spec = latents = None
            ref_samples = self.format.mel_spec_to_mdct_psd(mel_spec).detach()
            mdct_samples: torch.Tensor = self.format.raw_to_mdct(raw_samples,
                random_phase_augmentation=self.config.random_phase_augmentation).detach()
            
            logs = self.unet_train_batch(mdct_samples, audio_embeddings, ref_samples)
        else:
            latents_sigma = torch.zeros(1, device=self.dae.device, dtype=self.dae.dtype)
            latents, recon_mel_spec, _, _, _ = self.dae(
                mel_spec.to(dtype=self.dae.dtype), dae_embeddings, latents_sigma, equivariance_dropout=1)
            
            mel_spec, recon_mel_spec = self.ddec.scale_input(mel_spec), self.ddec.scale_input(recon_mel_spec)
            ref_samples = recon_mel_spec.detach()

            logs = self.unet_train_batch(mel_spec, audio_embeddings, ref_samples)
        
        logs.update({
            "io_stats/mel_spec_std": mel_spec.std(dim=(1,2,3)),
            "io_stats/mel_spec_mean": mel_spec.mean(dim=(1,2,3)),
            "io_stats/x_ref_std": ref_samples.std(dim=(1,2,3)),
            "io_stats/x_ref_mean": ref_samples.mean(dim=(1,2,3)),
        })
        
        if self.train_mode == "ddec_mdct":
            logs.update({
                "io_stats/mdct_std": mdct_samples.std(dim=(1,2,3)),
                "io_stats/mdct_mean": mdct_samples.mean(dim=(1,2,3)),
            })
        else:
            logs.update({
                "io_stats/latents_std": latents.std(dim=(1,2,3)).detach(),
                "io_stats/latents_mean": latents.mean(dim=(1,2,3)).detach(),
            })

        return logs