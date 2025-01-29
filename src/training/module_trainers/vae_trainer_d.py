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
from modules.vaes.vae_edm2_d1 import AutoencoderKL_EDM2_D1
from modules.mp_tools import normalize
from training.trainer import DualDiffusionTrainer
from .module_trainer import ModuleTrainerConfig, ModuleTrainer
from utils.dual_diffusion_utils import dict_str


@dataclass
class VAETrainer_D_Config(ModuleTrainerConfig):
    kl_loss_weight: float = 0.1
    add_latents_noise: float = 0
    
class VAETrainer_D(ModuleTrainer):
    
    @torch.no_grad()
    def __init__(self, config: VAETrainer_D_Config, trainer: DualDiffusionTrainer) -> None:

        self.config = config
        self.trainer = trainer
        self.logger = trainer.logger
        self.vae: AutoencoderKL_EDM2_D1 = trainer.module
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

        samples = self.format.raw_to_sample(batch["audio"]).detach().clone()
        if self.trainer.config.enable_channels_last == True:
            samples = samples.to(memory_format=torch.channels_last)

        sample_audio_embeddings = normalize(batch["audio_embeddings"])
        vae_emb = self.vae.get_embeddings(sample_audio_embeddings)
        enc_states, dec_states, sigma = self.vae(samples, vae_emb, add_latents_noise=self.config.add_latents_noise)

        # latents kl loss
        latents: torch.Tensor = enc_states[-1][1]

        # output state kl loss
        output_states = [state[1] for state in enc_states + dec_states[:-1]]
        kl_loss = torch.zeros(samples.shape[0], device=self.trainer.accelerator.device)
        for state in output_states:
            state_var = state.var(dim=1).clip(min=0.1)
            state_mean = state.mean(dim=1)
            loss_weight = 1 if state is latents else 1/len(output_states)
            kl_loss = kl_loss + (state_mean.square() + state_var - 1 - state_var.log()).mean(dim=(1,2,3)) * loss_weight

        # input/output sample kl loss
        output_samples = dec_states[-1][1].float().squeeze(1)
        relative_var = (output_samples.var(dim=(1,2,3)) / samples.var(dim=(1,2,3))).clip(min=0.1, max=10)
        relative_mean = samples.mean(dim=(1,2,3)) - output_samples.mean(dim=(1,2,3))
        kl_loss = kl_loss + (relative_mean.square() + relative_var - 1 - relative_var.log())

        # recon loss
        recon_loss = (samples - output_samples).abs().mean(dim=(1,2,3))
        #recon_loss = torch.nn.functional.mse_loss(output_samples, samples, reduction="none").mean(dim=(1,2,3))
        recon_loss_logvar = self.vae.get_recon_loss_logvar()
        recon_loss_nll = recon_loss / recon_loss_logvar.exp() + recon_loss_logvar

        return {
            "loss": kl_loss * self.config.kl_loss_weight + recon_loss_nll, # / sigma,
            "loss/recon_nll": recon_loss_nll.mean().detach(),
            "loss/recon": recon_loss.mean().detach(),
            "loss/kl": kl_loss.mean().detach(),
            "latents/mean": latents.mean().detach(),
            "latents/std": latents.std().detach(),
        }

    @torch.no_grad()
    def finish_batch(self) -> dict[str, torch.Tensor]:
        return {}