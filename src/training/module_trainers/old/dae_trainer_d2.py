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

from training.trainer import DualDiffusionTrainer
from .module_trainer import ModuleTrainerConfig, ModuleTrainer
from modules.daes.dae_edm2_d2 import DAE_D2
from modules.formats.spectrogram import SpectrogramFormat
from modules.mp_tools import normalize, wavelet_decompose2d, midside_transform


@dataclass
class DAETrainer_D2_Config(ModuleTrainerConfig):

    add_latents_noise: float = 0
    kl_loss_weight: float = 2e-2
    kl_warmup_steps: int  = 1000
    num_wavelet_loss_levels: int = 7

class DAETrainer_D2(ModuleTrainer):
    
    @torch.no_grad()
    def __init__(self, config: DAETrainer_D2_Config, trainer: DualDiffusionTrainer) -> None:

        self.config = config
        self.trainer = trainer
        self.logger = trainer.logger
        self.module: DAE_D2 = trainer.module

        self.is_validation_batch = False
        self.device_generator = None
        self.cpu_generator = None

        self.format: SpectrogramFormat = trainer.pipeline.format.to(self.trainer.accelerator.device)

        if trainer.config.enable_model_compilation:
            self.module.compile(**trainer.config.compile_params)
            self.format.compile(**trainer.config.compile_params)

        self.logger.info("Training DAE model:")
        self.logger.info(f"Wavelet loss levels: {self.config.num_wavelet_loss_levels}")
        self.logger.info(f"Add latents noise: {self.config.add_latents_noise}")
        self.logger.info(f"KL loss weight: {self.config.kl_loss_weight} KL warmup steps: {self.config.kl_warmup_steps}")

    @torch.no_grad()
    def init_batch(self, validation: bool = False) -> None:
        pass

    def train_batch(self, batch: dict) -> dict[str, torch.Tensor]:

        if "audio_embeddings" in batch:
            sample_audio_embeddings = normalize(batch["audio_embeddings"])
            dae_embeddings = self.module.get_embeddings(sample_audio_embeddings)
        else:
            dae_embeddings = None
        
        spec_samples = self.format.raw_to_sample(batch["audio"]).clone().detach()

        latents, reconstructed, latents_pre_norm_std = self.module(
            spec_samples, dae_embeddings, self.config.add_latents_noise)

        kl_loss = latents.mean(dim=(1,2,3)).square() + latents_pre_norm_std.square() - 1 - latents_pre_norm_std.square().log()

        ms_recon = midside_transform(reconstructed)
        ms_spec = midside_transform(spec_samples)

        recon_loss = torch.zeros(spec_samples.shape[0], device=spec_samples.device)

        logs = {}

        spec_samples_wavelets = wavelet_decompose2d(spec_samples, self.config.num_wavelet_loss_levels)
        reconstructed_wavelets = wavelet_decompose2d(reconstructed, self.config.num_wavelet_loss_levels)

        for i, (spec_wavelet, recon_wavelet) in enumerate(zip(spec_samples_wavelets, reconstructed_wavelets)):
            level_weight = (spec_wavelet[0].numel() / spec_samples_wavelets[0][0].numel())**0.5
            #recon_wavelet = recon_wavelet / (spec_wavelet.std(dim=(1,2,3), keepdim=True) / recon_wavelet.std(dim=(1,2,3), keepdim=True))
            level_loss = torch.nn.functional.mse_loss(recon_wavelet, spec_wavelet, reduction="none").mean(dim=(1,2,3))
            recon_loss = recon_loss + (level_loss * level_weight)#.clip(min=1e-8).sqrt()

            logs[f"loss/level{i}"] = level_loss

            relative_var = (recon_wavelet.var(dim=(1,2,3)) / spec_wavelet.var(dim=(1,2,3))).clip(min=0.1, max=10)
            #level_rvar_kl_loss = relative_var - 1 - relative_var.log()
            #kl_loss = kl_loss + level_rvar_kl_loss * level_weight
            logs[f"io_stats/rvar_level{i}"] = relative_var

        ms_spec_samples_wavelets = wavelet_decompose2d(ms_spec, self.config.num_wavelet_loss_levels)
        ms_reconstructed_wavelets = wavelet_decompose2d(ms_recon, self.config.num_wavelet_loss_levels)

        for i, (spec_wavelet, recon_wavelet) in enumerate(zip(ms_spec_samples_wavelets, ms_reconstructed_wavelets)):
            level_weight = (spec_wavelet[0].numel() / ms_spec_samples_wavelets[0][0].numel())**0.5
            #recon_wavelet = recon_wavelet / (spec_wavelet.std(dim=(1,2,3), keepdim=True) / recon_wavelet.std(dim=(1,2,3), keepdim=True))
            level_loss = torch.nn.functional.mse_loss(recon_wavelet, spec_wavelet, reduction="none").mean(dim=(1,2,3))
            recon_loss = recon_loss + (level_loss * level_weight)#.clip(min=1e-8).sqrt()

            logs[f"loss/ms_level{i}"] = level_loss

            relative_var = (recon_wavelet.var(dim=(1,2,3)) / spec_wavelet.var(dim=(1,2,3))).clip(min=0.1, max=10)
            #level_rvar_kl_loss = relative_var - 1 - relative_var.log()
            #kl_loss = kl_loss + level_rvar_kl_loss * level_weight
            logs[f"io_stats/ms_rvar_level{i}"] = relative_var

        recon_loss_logvar = self.module.get_recon_loss_logvar()
        recon_loss_nll = (recon_loss / 2) / recon_loss_logvar.exp() + recon_loss_logvar

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