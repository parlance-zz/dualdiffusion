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
from modules.daes.dae_edm2_a1 import DualDiffusionDAE_EDM2_A1
from modules.mp_tools import resample_3d
from training.trainer import DualDiffusionTrainer
from .module_trainer import ModuleTrainerConfig, ModuleTrainer
from utils.dual_diffusion_utils import dict_str, tensor_4d_to_5d


@dataclass
class DAETrainer_Config(ModuleTrainerConfig):
    kl_loss_weight: float = 0.1
    kl_warmup_steps: int  = 1000
    add_latents_noise: float = 0
    
class DAETrainer(ModuleTrainer):
    
    @torch.no_grad()
    def __init__(self, config: DAETrainer_Config, trainer: DualDiffusionTrainer) -> None:

        self.config = config
        self.trainer = trainer
        self.logger = trainer.logger
        self.dae: DualDiffusionDAE_EDM2_A1 = trainer.module
        self.format: DualDiffusionFormat = trainer.pipeline.format.to(trainer.accelerator.device)

        if trainer.config.enable_model_compilation:
            self.format.compile(**trainer.config.compile_params)

        self.logger.info("Training DAE model:")
        self.logger.info(f"DAE Training params: {dict_str(config.__dict__)}")

    @torch.no_grad()
    def init_batch(self, validation: bool = False) -> None:
        pass

    def train_batch(self, batch: dict) -> dict[str, torch.Tensor]:

        samples = self.format.raw_to_sample(batch["audio"]).detach().clone()
        if self.trainer.config.enable_channels_last == True:
            samples = samples.to(memory_format=torch.channels_last)

        latents, hidden_states, output_samples = self.dae(
            samples, add_latents_noise=self.config.add_latents_noise)

        # output state kl loss
        latents: torch.Tensor = tensor_4d_to_5d(latents, self.dae.config.latent_channels)

        for state in hidden_states + [latents]:
            weight = 1 if state is latents else 1 / len(hidden_states)

            state_var = state.var(dim=1).clip(min=0.1)
            state_mean = state.mean(dim=1)
            kl_loss = (state_mean.square() + state_var - 1 - state_var.log()).mean(dim=(1,2,3)) * (weight / 2)

            state_var = latents.var(dim=(2,3,4)).clip(min=0.1)
            state_mean = latents.mean(dim=(2,3,4))
            kl_loss = kl_loss + (state_mean.square() + state_var - 1 - state_var.log()).mean(dim=1) * (weight / 2)

        # input/output sample kl loss
        relative_var = (output_samples.var(dim=(1,2,3)) / samples.var(dim=(1,2,3))).clip(min=0.1, max=10)
        relative_mean = samples.mean(dim=(1,2,3)) - output_samples.mean(dim=(1,2,3))
        kl_loss = kl_loss + (relative_mean.square() + relative_var - 1 - relative_var.log())

        # spectral energy kl loss
        downsampled = latents
        images = []
        while downsampled.shape[-1] % 2 == 0 and downsampled.shape[-2] % 2 == 0:
            images.append(downsampled)
            downsampled = resample_3d(downsampled, "down")

        energy_logs = {}    
        octave_energies = []
        total_octave_energy = torch.zeros_like(kl_loss)
    
        for i in range(len(images) - 1):
            octave_energy = (images[i] - resample_3d(images[i+1], "up")).square().mean(dim=(1,2,3,4))
            energy_logs[f"octave_energies/{i}"] = octave_energy.mean().detach()
            octave_energies.append(octave_energy)
            total_octave_energy = total_octave_energy + octave_energy

        avg_octave_energy = total_octave_energy / (len(images) - 1)
        for energy in octave_energies:
            relative_var = (energy / avg_octave_energy).clip(min=0.1, max=10)
            kl_loss = kl_loss + (relative_var - 1 - relative_var.log()) / len(octave_energies)

        # recon loss
        #recon_loss = torch.nn.functional.mse_loss(output_samples, samples, reduction="none").mean(dim=(1,2,3))
        recon_loss = (samples - output_samples).abs().mean(dim=(1,2,3))
        recon_loss_logvar = self.dae.get_recon_loss_logvar()
        recon_loss_nll = recon_loss / recon_loss_logvar.exp() + recon_loss_logvar

        kl_loss_weight = self.config.kl_loss_weight
        if self.trainer.global_step < self.config.kl_warmup_steps:
            kl_loss_weight *= self.trainer.global_step / self.config.kl_warmup_steps

        return {
            "loss": kl_loss * kl_loss_weight + recon_loss_nll,
            "loss/recon_nll": recon_loss_nll.mean().detach(),
            "loss/recon": recon_loss.mean().detach(),
            "loss/kl": kl_loss.mean().detach(),
            "latents/mean": latents.mean().detach(),
            "latents/std": latents.std().detach(),
            **energy_logs
        }

    @torch.no_grad()
    def finish_batch(self) -> dict[str, torch.Tensor]:
        return {}