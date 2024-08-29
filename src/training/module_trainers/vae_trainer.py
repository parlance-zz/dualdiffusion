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
from typing import Optional

import torch

from training.trainer import DualDiffusionTrainer
from .module_trainer import ModuleTrainerConfig, ModuleTrainer
from training.loss import DualMultiscaleSpectralLoss2DConfig, DualMultiscaleSpectralLoss2D
from utils.dual_diffusion_utils import dict_str

@dataclass
class VAETrainerConfig(ModuleTrainerConfig):

    block_overlap: int = 8
    block_widths: tuple = (8, 16, 32, 64)
    channel_kl_loss_weight: float = 0.1
    imag_loss_weight: float = 0.1
    point_loss_weight: float = 0
    recon_loss_weight: float = 0.1

class VAETrainer(ModuleTrainer):
    
    @torch.no_grad()
    def __init__(self, config: VAETrainerConfig, trainer: DualDiffusionTrainer) -> None:

        self.config = config
        self.trainer = trainer
        self.logger = trainer.logger
        self.module = trainer.module

        if trainer.config.enable_model_compilation:
            self.module.encode = torch.compile(self.module.encode, **trainer.config.compile_params)
            self.module.decode = torch.compile(self.module.decode, **trainer.config.compile_params)

        trainer.pipeline.format = trainer.pipeline.format.to(self.accelerator.device)
        #if trainer.config.enable_model_compilation: # todo: complex operators are not currently supported in compile
        #    trainer.pipeline.format.raw_to_sample = torch.compile(trainer.pipeline.format.raw_to_sample,
        #                                                        **trainer.config.compile_params)

        self.loss = DualMultiscaleSpectralLoss2D(DualMultiscaleSpectralLoss2DConfig(block_widths=config.block_widths,
                                                                                     block_overlap=config.block_overlap))
        self.target_snr = self.module.get_target_snr()
        self.target_noise_std = (1 / (self.target_snr**2 + 1))**0.5

        self.logger.info("Training VAE model:")
        self.logger.info(f"VAE Training params: {dict_str(config.__dict__)}")
        self.logger.info(f"Dropout: {self.module.config.dropout}")
        self.logger.info(f"Target SNR: {self.target_snr:{8}f}")
    
    @staticmethod
    def get_config_class() -> ModuleTrainerConfig:
        return VAETrainerConfig
    
    @torch.no_grad()
    def init_batch(self, validation: bool = False) -> None:
        pass

    def train_batch(self, batch: dict, grad_accum_steps: int) -> dict[str, torch.Tensor]:

        raw_samples = batch["input"]
        sample_game_ids = batch["game_ids"]
        #sample_author_ids = batch["author_ids"]

        class_labels = self.trainer.pipeline.get_class_labels(sample_game_ids, module="vae")
        vae_class_embeddings = self.module.get_class_embeddings(class_labels)

        samples = self.trainer.pipeline.format.raw_to_sample(raw_samples)
        
        posterior = self.module.encode(samples, vae_class_embeddings, self.trainer.pipeline.format)
        latents = posterior.sample()
        latents_mean = latents.mean()
        latents_std = latents.std()

        measured_sample_std = (latents_std**2 - self.target_noise_std**2).clip(min=0)**0.5
        latents_snr = measured_sample_std / self.target_noise_std
        recon_samples = self.module.decode(latents, vae_class_embeddings, self.trainer.pipeline.format)

        point_similarity_loss = (samples - recon_samples).abs().mean()
        
        recon_loss_logvar = self.module.get_recon_loss_logvar()
        real_loss, imag_loss = self.loss(recon_samples, samples)
        real_nll_loss = (real_loss / recon_loss_logvar.exp() + recon_loss_logvar) * self.config.recon_loss_weight
        imag_nll_loss = (imag_loss / recon_loss_logvar.exp() + recon_loss_logvar) * self.config.recon_loss_weight * self.config.imag_loss_weight

        latents_square_norm = (torch.linalg.vector_norm(latents, dim=(1,2,3), dtype=torch.float32) / latents[0].numel()**0.5).square()
        latents_batch_mean = latents.mean(dim=(1,2,3))
        channel_kl_loss = (latents_batch_mean.square() + latents_square_norm - 1 - latents_square_norm.log()).mean()
        
        loss = real_nll_loss + imag_nll_loss + channel_kl_loss * self.config.channel_kl_loss_weight + point_similarity_loss * self.config.point_loss_weight

        return {
            "loss": loss,
            "channel_kl_loss_weight": self.config.channel_kl_loss_weight,
            "point_loss_weight": self.config.point_loss_weight,
            "recon_loss_weight": self.config.recon_loss_weight,
            "imag_loss_weight": self.config.imag_loss_weight,
            "channel_kl_loss": channel_kl_loss,
            "point_similarity_loss": point_similarity_loss,
            "real_loss": real_loss,
            "imag_loss": imag_loss,
            "latents_mean": latents_mean,
            "latents_std": latents_std,
            "latents_snr": latents_snr,
        }

    @torch.no_grad()
    def finish_batch(self) -> dict[str, torch.Tensor]:
        return {}