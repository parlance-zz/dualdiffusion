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

import torch
import numpy as np

from training.trainer import DualDiffusionTrainer
from .module_trainer import ModuleTrainerConfig, ModuleTrainer
from modules.daes.dae_edm2_d1 import DAE_D1
from modules.formats.spectrogram import SpectrogramFormat
from modules.mp_tools import normalize, wavelet_decompose2d


@dataclass
class DAETrainer_D1_Config(ModuleTrainerConfig):

    add_latents_noise: float = 0
    kl_loss_weight: float = 0.1
    kl_warmup_steps: int  = 1000
    freq_crop: int = 0

class DAETrainer_D1(ModuleTrainer):
    
    @torch.no_grad()
    def __init__(self, config: DAETrainer_D1_Config, trainer: DualDiffusionTrainer) -> None:

        self.config = config
        self.trainer = trainer
        self.logger = trainer.logger
        self.module: DAE_D1 = trainer.module

        self.is_validation_batch = False
        self.device_generator = None
        self.cpu_generator = None

        self.format: SpectrogramFormat = trainer.pipeline.format.to(self.trainer.accelerator.device)

        if trainer.config.enable_model_compilation:
            self.module.compile(**trainer.config.compile_params)
            self.format.compile(**trainer.config.compile_params)

        self.logger.info("Training DAE model:")
        self.logger.info(f"Add latents noise: {self.config.add_latents_noise}")
        self.logger.info(f"KL loss weight: {self.config.kl_loss_weight} KL warmup steps: {self.config.kl_warmup_steps}")
        self.logger.info(f"Frequencies cropped: {self.config.freq_crop}")

    @torch.no_grad()
    def init_batch(self, validation: bool = False) -> None:
        pass

    def train_batch(self, batch: dict) -> dict[str, torch.Tensor]:

        if self.is_validation_batch == False:
            device_batch_size = self.trainer.config.device_batch_size
            #assert latents.shape == self.trainer.latent_shape, f"Expected shape {self.trainer.latent_shape}, got {latents.shape}"
            #assert samples.shape == self.trainer.sample_shape, f"Expected shape {self.trainer.sample_shape}, got {samples.shape}"
        else:
            device_batch_size = self.trainer.config.validation_device_batch_size
            #assert latents.shape == self.trainer.validation_latent_shape, f"Expected shape {self.trainer.validation_latent_shape}, got {latents.shape}"
            #assert samples.shape == self.trainer.validation_sample_shape, f"Expected shape {self.trainer.validation_sample_shape}, got {samples.shape}"

        if "audio_embeddings" in batch:
            sample_audio_embeddings = normalize(batch["audio_embeddings"])
            dae_embeddings = self.module.get_embeddings(sample_audio_embeddings)
        else:
            dae_embeddings = None
        
        spec_samples = self.format.raw_to_sample(batch["audio"]).clone().detach()

        if self.config.freq_crop > 0:
            freq_crop_offset = np.random.randint(0, spec_samples.shape[2] - self.config.freq_crop + 1)
            spec_samples = spec_samples[:, :, freq_crop_offset:freq_crop_offset + self.config.freq_crop, :]

        latents, reconstructed, latents_pre_norm_std = self.module(
            spec_samples, dae_embeddings, self.config.add_latents_noise)

        spec_samples_wavelet = wavelet_decompose2d(spec_samples, self.module.num_levels)
        reconstructed_wavelet = wavelet_decompose2d(reconstructed, self.module.num_levels)
        recon_loss = torch.zeros(spec_samples.shape[0], device=spec_samples.device)

        logs = {}

        for spec_wavelet, recon_wavelet in zip(spec_samples_wavelet, reconstructed_wavelet):
            loss_weight = 1 / spec_wavelet.square().mean(dim=(1,2,3)).clip(min=1e-8)
            level_loss = torch.nn.functional.mse_loss(recon_wavelet, spec_wavelet, reduction="none").mean(dim=(1,2,3)) * loss_weight
            recon_loss = recon_loss + level_loss * spec_wavelet[0].numel() / spec_samples_wavelet[0][0].numel()

            logs[f"loss/level{len(logs)}"] = level_loss
            logs[f"loss_weight/level{len(logs)}"] = loss_weight

        #recon_loss = (reconstructed - spec_samples).abs().mean(dim=(1,2,3))
        recon_loss_logvar = self.module.get_recon_loss_logvar()
        recon_loss_nll = recon_loss / recon_loss_logvar.exp() + recon_loss_logvar

        kl_loss = latents.mean(dim=(1,2,3)).square()
        kl_loss_weight = self.config.kl_loss_weight
        if self.trainer.global_step < self.config.kl_warmup_steps:
            kl_loss_weight *= self.trainer.global_step / self.config.kl_warmup_steps

        logs.update({
            "loss": recon_loss_nll + kl_loss * kl_loss_weight,
            "loss/recon": recon_loss,
            "loss/kl": kl_loss,
            "loss_weight/kl": kl_loss_weight,
            "io_stats/input_std": spec_samples.std(dim=(1,2,3)),
            "io_stats/input_mean": spec_samples.mean(dim=(1,2,3)),
            "io_stats/output_std": reconstructed.std(dim=(1,2,3)),
            "io_stats/output_mean": reconstructed.mean(dim=(1,2,3)),
            "io_stats/latents_std": latents.std(dim=(1,2,3)),
            "io_stats/latents_mean": latents.mean(dim=(1,2,3)),
            "io_stats/latents_pre_norm_std": latents_pre_norm_std
        })

        return logs

    @torch.no_grad()
    def finish_batch(self) -> dict[str, torch.Tensor]:
        return {}