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
from modules.daes.dae_edm2_j3 import DAE_J3
from modules.unets.unet_edm2_ddec_mdct_b2 import DDec_MDCT_UNet_B2
from modules.formats.ms_mdct_dual import MS_MDCT_DualFormat
from modules.mp_tools import normalize


def random_stereo_augmentation(x: torch.Tensor) -> torch.Tensor:
    
    output = x.clone()
    flip_mask = (torch.rand(x.shape[0]) > 0.5).to(x.device)
    output[flip_mask] = output[flip_mask].flip(dims=(1,))
    
    return output

@dataclass
class DiffusionDecoder_MDCT_Trainer_B2_Config(UNetTrainerConfig):

    loss_buckets_sigma_min: float = 0.00003
    loss_buckets_sigma_max: float = 20

    latents_perturbation: float = 0.01

class DiffusionDecoder_MDCT_Trainer_B2(UNetTrainer):
    
    @torch.no_grad()
    def __init__(self, config: DiffusionDecoder_MDCT_Trainer_B2_Config, trainer: DualDiffusionTrainer) -> None:

        self.config = config
        self.trainer = trainer
        self.logger = trainer.logger

        self.ddec: DDec_MDCT_UNet_B2 = trainer.get_train_module("ddec")
        self.dae: DAE_J3 = trainer.pipeline.dae.to(dtype=torch.bfloat16, device=trainer.accelerator.device)
        self.format: MS_MDCT_DualFormat = trainer.pipeline.format.to(self.trainer.accelerator.device)

        if trainer.config.enable_model_compilation:
            self.ddec.compile(**trainer.config.compile_params)
            self.dae.compile(**trainer.config.compile_params)
            self.format.compile(**trainer.config.compile_params)

        if self.config.latents_perturbation > 0:
            self.logger.info(f"Using latents perturbation: {self.config.latents_perturbation}")
        else: self.logger.info("Latents perturbation is disabled")

        self.unet_trainer_init()

    def train_batch(self, batch: dict) -> Optional[dict[str, Union[torch.Tensor, float]]]:

        # prepare model inputs
        if "audio_embeddings" in batch:
            audio_embeddings = normalize(batch["audio_embeddings"]).detach()
            dae_class_embeddings = self.dae.get_embeddings(audio_embeddings.to(dtype=self.dae.dtype))
        else:
            audio_embeddings = dae_class_embeddings = None

        raw_samples = random_stereo_augmentation(batch["audio"])
        mdct_samples: torch.Tensor = self.format.raw_to_mdct(raw_samples, random_phase_augmentation=True).detach()
        mel_spec = self.format.raw_to_mel_spec(raw_samples).detach()

        latents, recon_mel_spec, _, _ = self.dae(mel_spec, dae_class_embeddings)#, add_latents_noise=self.config.latents_perturbation)
        ref_samples = self.format.mel_spec_to_mdct_psd(recon_mel_spec).detach()
        
        logs = self.unet_train_batch(mdct_samples, audio_embeddings, ref_samples)
        logs.update({
            "io_stats/mel_spec_std": mel_spec.std(dim=(1,2,3)),
            "io_stats/mel_spec_mean": mel_spec.mean(dim=(1,2,3)),
            "io_stats/mdct_std": mdct_samples.std(dim=(1,2,3)),
            "io_stats/mdct_mean": mdct_samples.mean(dim=(1,2,3)),
            "io_stats/recon_mel_spec_std": recon_mel_spec.std(dim=(1,2,3)),
            "io_stats/recon_mel_spec_mean": recon_mel_spec.mean(dim=(1,2,3)),
            "io_stats/x_ref_std": ref_samples.std(dim=(1,2,3)),
            "io_stats/x_ref_mean": ref_samples.mean(dim=(1,2,3)),
            "io_stats/latents_std": latents.std(dim=(1,2,3)),
            "io_stats/latents_mean": latents.mean(dim=(1,2,3))
        })
        return logs