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
from modules.daes.dae_edm2_e1 import DAE_E1
from modules.formats.spectrogram import SpectrogramFormat
from modules.mp_tools import normalize, wavelet_decompose2d, wavelet_recompose2d


@dataclass
class DAETrainer_E1_Config(ModuleTrainerConfig):

    add_latents_noise: float = 0
    kl_loss_weight: float = 2e-2
    kl_warmup_steps: int  = 1000

class DAETrainer_E1(ModuleTrainer):
    
    @torch.no_grad()
    def __init__(self, config: DAETrainer_E1_Config, trainer: DualDiffusionTrainer) -> None:

        self.config = config
        self.trainer = trainer
        self.logger = trainer.logger
        self.module: DAE_E1 = trainer.module

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

        latents, latents_pre_norm_std, dec_outputs, dif_outputs, dif_noise = self.module(spec_samples, dae_embeddings)

        reconstructed = wavelet_recompose2d(dec_outputs) + wavelet_recompose2d(dif_noise) - wavelet_recompose2d(dif_outputs)
        spec_wavelets = wavelet_decompose2d(spec_samples, self.module.num_levels)
    
        dec_loss = torch.zeros(spec_samples.shape[0], device=spec_samples.device)
        dec_loss_nll = torch.zeros_like(dec_loss)
        dif_loss = torch.zeros_like(dec_loss)
        dif_loss_nll = torch.zeros_like(dec_loss)

        kl_loss = latents.mean(dim=(1,2,3)).square() + latents_pre_norm_std.square() - 1 - latents_pre_norm_std.square().log()

        logs = {}

        for i, (spec, dec, dif, noise) in enumerate(zip(spec_wavelets, dec_outputs, dif_outputs, dif_noise)):

            _dec_loss = torch.nn.functional.mse_loss(dec, spec, reduction="none").mean(dim=(1,2,3))
            dec_loss = dec_loss + _dec_loss
            dec_loss_nll = dec_loss_nll + _dec_loss / self.module.recon_loss_logvar[i].exp() + self.module.recon_loss_logvar[i]
            logs[f"loss/level{i}_dec"] = _dec_loss

            _dif_loss = torch.nn.functional.mse_loss(dif, noise, reduction="none").mean(dim=(1,2,3))
            dif_loss = dif_loss + _dif_loss
            dif_loss_nll = dif_loss_nll + _dif_loss / self.module.recon_loss_logvar_dif[i].exp() + self.module.recon_loss_logvar_dif[i]
            logs[f"loss/level_dif{i}"] = _dec_loss

            logs[f"io_stats/level{i}_std_spec"] = spec.std(dim=(1,2,3))
            logs[f"io_stats/level{i}_std_dec"] = dec.std(dim=(1,2,3))
            logs[f"io_stats/level{i}_std_noise"] = noise.std(dim=(1,2,3))
            logs[f"io_stats/level{i}_std_dif"] = dif.std(dim=(1,2,3))

        kl_loss_weight = self.config.kl_loss_weight
        if self.trainer.global_step < self.config.kl_warmup_steps:
            kl_loss_weight *= self.trainer.global_step / self.config.kl_warmup_steps

        logs.update({
            "loss": dec_loss_nll + dif_loss_nll + kl_loss * kl_loss_weight,
            "loss/dec": dec_loss,
            "loss/dif": dif_loss,
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