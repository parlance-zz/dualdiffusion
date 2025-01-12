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

from modules.formats.format import DualDiffusionFormat
from modules.vaes.vae_edm2_c1 import AutoencoderKL_EDM2_C1
from modules.mp_tools import normalize
from training.trainer import DualDiffusionTrainer
from .module_trainer import ModuleTrainerConfig, ModuleTrainer
from utils.dual_diffusion_utils import dict_str


@dataclass
class VAETrainerConfig(ModuleTrainerConfig):

    kl_loss_weight: float = 0.1

class VAETrainer(ModuleTrainer):
    
    @torch.no_grad()
    def __init__(self, config: VAETrainerConfig, trainer: DualDiffusionTrainer) -> None:

        self.config = config
        self.trainer = trainer
        self.logger = trainer.logger
        self.vae: AutoencoderKL_EDM2_C1 = trainer.module
        self.format: DualDiffusionFormat = trainer.pipeline.format.to(trainer.accelerator.device)

        if trainer.config.enable_model_compilation:
            self.vae.compile(**trainer.config.compile_params)
            self.format.compile(**trainer.config.compile_params)

        self.logger.info("Training VAE model:")
        self.logger.info(f"VAE Training params: {dict_str(config.__dict__)}")
        self.logger.info(f"Dropout: {self.vae.config.dropout}")
    
    @torch.no_grad()
    def init_batch(self, validation: bool = False) -> None:
        pass

    def train_batch(self, batch: dict) -> dict[str, torch.Tensor]:

        samples = self.format.raw_to_sample(batch["input"])
        if self.trainer.config.enable_channels_last == True:
            samples = samples.to(memory_format=torch.channels_last)

        sample_audio_embeddings = normalize(batch["audio_embeddings"])
        vae_emb = self.vae.get_embeddings(sample_audio_embeddings)

        with torch.amp.autocast(enabled=self.trainer.mixed_precision_enabled, dtype=self.trainer.mixed_precision_dtype):
            posterior, enc_states = self.vae.encode(samples, vae_emb, self.format, return_hidden_states=True)

        latents: torch.Tensor = posterior.mode()
        if self.trainer.config.enable_channels_last == True:
            latents = latents.to(memory_format=torch.contiguous_format)

        with torch.amp.autocast(enabled=self.trainer.mixed_precision_enabled, dtype=self.trainer.mixed_precision_dtype):
            _, dec_states = self.vae.decode(latents, vae_emb, self.format, return_hidden_states=True)

        recon_loss = torch.zeros((samples.shape[0]), device=self.trainer.accelerator.device)
        for enc_state, dec_state in zip(enc_states, reversed(dec_states)):
            recon_loss = recon_loss + torch.nn.functional.mse_loss(enc_state, dec_state, reduce=False).sum(dim=(1,2,3))
        
        recon_loss_logvar = self.vae.get_recon_loss_logvar()
        recon_loss = recon_loss / recon_loss_logvar.exp() + recon_loss_logvar

        latents_pixel_var = latents.var(dim=1).clip(min=0.1)
        latents_pixel_mean = latents.mean(dim=1)
        kl_loss = (latents_pixel_mean.square() + latents_pixel_var - 1 - latents_pixel_var.log()).sum(dim=(1, 2))
        
        loss = recon_loss + self.config.kl_loss_weight * kl_loss
        
        return {
            "loss": loss,
            "recon_loss": recon_loss.detach().mean(),
            "kl_loss": kl_loss.detach().mean(),
            "latents_mean": latents.mean(),
            "latents_std": latents.std(),
        }

    @torch.no_grad()
    def finish_batch(self) -> dict[str, torch.Tensor]:
        return {}