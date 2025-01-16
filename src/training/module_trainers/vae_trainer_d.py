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

        samples = self.format.raw_to_sample(batch["audio"]).detach().clone()
        if self.trainer.config.enable_channels_last == True:
            samples = samples.to(memory_format=torch.channels_last)

        sample_audio_embeddings = normalize(batch["audio_embeddings"])
        vae_emb = self.vae.get_embeddings(sample_audio_embeddings)
        enc_states, dec_states = self.vae(samples, vae_emb)

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
        latents: torch.Tensor = enc_states[-1][1].float()
        latents_var = latents.var(dim=(2,3,4)).clip(min=0.1)
        latents_mean = latents.mean(dim=(2,3,4))
        latents_kl_loss = (latents_mean.square() + latents_var - 1 - latents_var.log()).mean(dim=1)

        # state kl loss
        output_states = [state[1] for state in enc_states + dec_states]
        state_kl_loss = torch.zeros_like(latents_kl_loss)
        for state in output_states:
            state_var = state.var(dim=(1,2,3,4)).clip(min=0.1)
            state_mean = state.mean(dim=(1,2,3,4))
            state_kl_loss = state_kl_loss + (state_mean.square() + state_var - 1 - state_var.log())
        
        # total kl loss
        kl_loss = latents_kl_loss + state_kl_loss

        logs = {}
        recon_loss = torch.zeros_like(state_kl_loss)
        recon_loss_nll = torch.zeros_like(recon_loss)
        for i, (enc_state, dec_state) in enumerate(zip(enc_states, reversed(dec_states))):

            enc_x_in, enc_x_out = enc_state
            dec_x_in, dec_x_out, error_logvar = dec_state

            recon_loss1 = torch.nn.functional.mse_loss(enc_x_in, dec_x_out,
                                        reduction="none").mean(dim=(1,2,3,4))
            recon_loss2 = torch.nn.functional.mse_loss(enc_x_out, dec_x_in,
                                        reduction="none").mean(dim=(1,2,3,4))
            
            state_recon_loss = recon_loss1 + recon_loss2
            state_recon_loss_nll = state_recon_loss / error_logvar.exp() + error_logvar

            recon_loss = recon_loss + state_recon_loss
            recon_loss_nll = recon_loss_nll + state_recon_loss_nll

            logs[f"state_loss/{i}"] = state_recon_loss
            logs[f"state_loss_nll/{i}"] = state_recon_loss_nll

        logs.update({
            "loss": kl_loss * self.config.kl_loss_weight + recon_loss_nll ,
            "loss/recon_nll": recon_loss_nll.mean().detach(),
            "loss/recon": state_recon_loss.mean().detach(),
            "loss/kl": kl_loss.mean().detach(),
            "latents/mean": latents.mean().detach(),
            "latents/std": latents.std().detach(),
        })
        return logs

    @torch.no_grad()
    def finish_batch(self) -> dict[str, torch.Tensor]:
        return {}