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
from modules.daes.dae_edm2_f1 import DAE_F1
from modules.formats.spectrogram import SpectrogramFormat
from modules.mp_tools import normalize, wavelet_decompose2d, wavelet_recompose2d


@dataclass
class DAETrainer_F1_Config(ModuleTrainerConfig):

    kl_loss_weight: float = 2e-2
    kl_warmup_steps: int  = 1000

class DAETrainer_F1(ModuleTrainer):
    
    @torch.no_grad()
    def __init__(self, config: DAETrainer_F1_Config, trainer: DualDiffusionTrainer) -> None:

        self.config = config
        self.trainer = trainer
        self.logger = trainer.logger
        self.module: DAE_F1 = trainer.module

        self.is_validation_batch = False
        self.device_generator = None
        self.cpu_generator = None

        self.format: SpectrogramFormat = trainer.pipeline.format.to(self.trainer.accelerator.device)

        if trainer.config.enable_model_compilation:
            self.module.compile(**trainer.config.compile_params)
            self.format.compile(**trainer.config.compile_params)

        self.logger.info("Training DAE model:")
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
        spec_wavelets = wavelet_decompose2d(spec_samples, self.module.num_levels)

        latents, pre_norm_latents, dec_outputs = self.module(spec_samples, dae_embeddings)
        reconstructed = wavelet_recompose2d(dec_outputs)

        dec_loss = torch.zeros(spec_samples.shape[0], device=spec_samples.device)

        pre_norm_latents_var = pre_norm_latents.var(dim=(1,2,3))
        pre_norm_latents_mean = pre_norm_latents.mean(dim=(1,2,3))
        kl_loss = pre_norm_latents_mean.square() + pre_norm_latents_var - 1 - pre_norm_latents_var.log()

        logs = {}

        for i, (spec, dec) in enumerate(zip(spec_wavelets, dec_outputs)):
            
            #level_weight = (spec[0].numel() / spec_wavelets[0][0].numel())**0.5
            level_weight = spec[0].numel() / spec_wavelets[0][0].numel()
            level_dec_loss = torch.nn.functional.mse_loss(dec, spec, reduction="none").mean(dim=(1,2,3))
            #dec_loss = dec_loss + level_dec_loss * level_weight
            dec_loss = dec_loss + (level_dec_loss * level_weight).sqrt()
            kl_loss = kl_loss + level_dec_loss.detach() / self.module.level_recon_loss_logvar[i].exp() + self.module.level_recon_loss_logvar[i]

            logs[f"loss/level{i}_dec"] = level_dec_loss.sqrt()
            logs[f"io_stats/level{i}_std_spec"] = spec.std(dim=(1,2,3))
            logs[f"io_stats/level{i}_std_dec"] = dec.std(dim=(1,2,3))

        dec_loss_nll = dec_loss / self.module.total_recon_loss_logvar.exp() + self.module.total_recon_loss_logvar

        kl_loss_weight = self.config.kl_loss_weight
        if self.trainer.global_step < self.config.kl_warmup_steps:
            kl_loss_weight *= self.trainer.global_step / self.config.kl_warmup_steps

        logs.update({
            "loss": dec_loss_nll + kl_loss * kl_loss_weight,
            "loss/dec": dec_loss,
            "loss/kl": kl_loss,
            "loss_weight/kl": kl_loss_weight,
            "io_stats/std_input": spec_samples.std(dim=(1,2,3)),
            "io_stats/mean_input": spec_samples.mean(dim=(1,2,3)),
            "io_stats/std_output": reconstructed.std(dim=(1,2,3)),
            "io_stats/mean_output": reconstructed.mean(dim=(1,2,3)),
            "io_stats/latents_std": latents.std(dim=(1,2,3)),
            "io_stats/latents_mean": latents.mean(dim=(1,2,3)),
            "io_stats/latents_mean_pre-norm": pre_norm_latents_mean,
            "io_stats/latents_std_pre-norm": pre_norm_latents_var.sqrt()
        })

        return logs

    @torch.no_grad()
    def finish_batch(self) -> dict[str, torch.Tensor]:
        return {}