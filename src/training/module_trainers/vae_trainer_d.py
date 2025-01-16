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

        samples = self.format.raw_to_sample(batch["audio"]).detach()
        if self.trainer.config.enable_channels_last == True:
            samples = samples.to(memory_format=torch.channels_last)

        sample_audio_embeddings = normalize(batch["audio_embeddings"])
        vae_emb = self.vae.get_embeddings(sample_audio_embeddings)
        
        latents, output_samples, enc_states, dec_states = self.vae(samples, vae_emb, self.format)

        # recon input/output sample loss
        #output_samples: torch.Tensor = output_samples.float()
        #recon_loss = torch.nn.functional.mse_loss(
        #    samples.clone().detach(), output_samples, reduction="none").mean(dim=(1,2,3))
        #recon_loss_logvar = self.vae.get_recon_loss_logvar()
        #recon_loss_nll = recon_loss / recon_loss_logvar.exp() + recon_loss_logvar

        # diffusion loss
        #noise = noise.float()
        #noise_pred = noise_pred.float()
        #noise_std = (recon_loss_logvar/2).exp().detach() * self.vae.config.noise_multiplier
        #diff_loss = torch.nn.functional.mse_loss(
        #    noise, noise_pred, reduction="none").mean(dim=(1,2,3,4))
        #diff_loss_logvar = self.vae.get_diff_loss_logvar()
        #diff_loss_nll = diff_loss / diff_loss_logvar.exp() + diff_loss_logvar

        # latents kl loss
        latents: torch.Tensor = latents.float()
        latents_var = latents.var(dim=(1,2,3,4)).clip(min=0.1)
        latents_mean = latents.mean(dim=(1,2,3,4))
        latents_kl_loss = (latents_mean.square() + latents_var - 1 - latents_var.log())

        # input/output sample kl loss
        relative_var = (output_samples.var(dim=(1,2,3)) / samples.var(dim=(1,2,3))).clip(min=0.1, max=10)
        relative_mean = samples.mean(dim=(1,2,3)) - output_samples.mean(dim=(1,2,3))
        samples_kl_loss = relative_mean.square() + relative_var - 1 - relative_var.log()

        # hidden state losses
        hidden_state_kl_loss = torch.zeros_like(samples_kl_loss)
        hidden_state_recon_loss = torch.zeros_like(samples_kl_loss)
        hidden_state_recon_loss_nll = torch.zeros_like(samples_kl_loss)
        hidden_state_logs = {}

        for i, (enc_state, (dec_state, error_logvar)) in enumerate(zip(enc_states, reversed(dec_states))):

            enc_state = enc_state.float()
            dec_state = dec_state.float()

            if i == 0: enc_state = enc_state.clone()

            hidden_state_logs[f"enc_state_std/{i}"] = enc_state.std().detach()
            hidden_state_logs[f"enc_state_mean/{i}"] = enc_state.mean().detach()
            hidden_state_logs[f"dec_state_std/{i}"] = dec_state.std().detach()
            hidden_state_logs[f"dec_state_mean/{i}"] = dec_state.mean().detach()

            if i > 0:
                for state in (enc_state, dec_state):
                    state_var = state.var(dim=(1,2,3,4)).clip(min=0.1)
                    state_mean = state.mean(dim=(1,2,3,4))
                    state_kl_loss = state_mean.square() + state_var - 1 - state_var.log()
                    hidden_state_kl_loss = hidden_state_kl_loss + state_kl_loss

            recon_loss = torch.nn.functional.mse_loss(enc_state, dec_state, reduction="none").mean(dim=(1,2,3,4))
            recon_loss_nll = recon_loss / error_logvar.exp() + error_logvar

            hidden_state_recon_loss = hidden_state_recon_loss + recon_loss
            hidden_state_recon_loss_nll = hidden_state_recon_loss_nll + recon_loss_nll

            hidden_state_logs[f"state_recon_loss/{i}"] = recon_loss.detach()
            hidden_state_logs[f"state_recon_loss_nll/{i}"] = recon_loss_nll.detach()

        kl_loss = latents_kl_loss + samples_kl_loss + hidden_state_kl_loss

        return_dict = {
            "loss": hidden_state_recon_loss_nll + kl_loss * self.config.kl_loss_weight,
            "loss/recon_nll": recon_loss_nll.mean().detach(),
            "loss/recon": hidden_state_recon_loss.mean().detach(),
            "loss/kl": kl_loss.mean().detach(),
            "latents/mean": latents.mean().detach(),
            "latents/std": latents.std().detach(),
        }
        return_dict.update(hidden_state_logs)
        return return_dict

    @torch.no_grad()
    def finish_batch(self) -> dict[str, torch.Tensor]:
        return {}