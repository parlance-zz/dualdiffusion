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
from typing import Union, Any, Optional

import torch

from training.trainer import DualDiffusionTrainer
from training.module_trainers.module_trainer import ModuleTrainerConfig, ModuleTrainer
from training.loss.multiscale_spectral import MSSLoss2D, MSSLoss2DConfig
from training.loss.spectral_regularization import SpecRegLoss, SpecRegLossConfig
from modules.daes.dae_edm2_j3 import DAE_J3
from modules.formats.ms_mdct_dual import MS_MDCT_DualFormat
from modules.mp_tools import normalize
from utils.dual_diffusion_utils import dict_str


def random_stereo_augmentation(x: torch.Tensor) -> torch.Tensor:
    
    output = x.clone()
    flip_mask = (torch.rand(x.shape[0]) > 0.5).to(x.device)
    output[flip_mask] = output[flip_mask].flip(dims=(1,))
    
    return output

@dataclass
class DAETrainer_J1_Config(ModuleTrainerConfig):

    latents_kl_loss_weight: float = 6e-2
    hidden_kl_loss_weight: float = 4e-3
    kl_warmup_steps: int = 1000

    point_loss_weight: float = 1
    point_loss_warmup_steps: int = 100

    mss_loss_weight: float = 1
    mss_loss_2d_config: Optional[dict[str, Any]] = None

    spec_reg_loss_weight: float = 0
    spec_reg_loss_config: Optional[dict[str, Any]] = None

class DAETrainer_J1(ModuleTrainer):
    
    @torch.no_grad()
    def __init__(self, config: DAETrainer_J1_Config, trainer: DualDiffusionTrainer) -> None:
        
        self.config = config
        self.trainer = trainer
        self.logger = trainer.logger

        self.dae: DAE_J3 = trainer.get_train_module("dae")
        self.format: MS_MDCT_DualFormat = trainer.pipeline.format.to(self.trainer.accelerator.device)

        if trainer.config.enable_model_compilation == True:
            self.dae.compile(**trainer.config.compile_params)
            self.format.compile(**trainer.config.compile_params)

        if config.mss_loss_weight > 0:
            config.mss_loss_2d_config = config.mss_loss_2d_config or {}
            self.mss_loss = MSSLoss2D(MSSLoss2DConfig(**config.mss_loss_2d_config), device=trainer.accelerator.device)

            if trainer.config.enable_model_compilation == True:
                self.mss_loss.compile(**trainer.config.compile_params)

        if config.spec_reg_loss_weight > 0:
            config.spec_reg_loss_config = config.spec_reg_loss_config or {}
            self.spec_reg_loss = SpecRegLoss(SpecRegLossConfig(**config.spec_reg_loss_config), 
                latents_shape=trainer.latent_shape, mel_spec_shape=trainer.sample_shape, device=trainer.accelerator.device)

            if trainer.config.enable_model_compilation == True:
                self.spec_reg_loss.compile(**trainer.config.compile_params)

        self.logger.info("Training DAE model:")
        self.logger.info(f"Latents KL loss weight: {self.config.latents_kl_loss_weight}")
        self.logger.info(f"Hidden KL loss weight: {self.config.hidden_kl_loss_weight}")
        self.logger.info(f"KL warmup steps: {self.config.kl_warmup_steps}")
        self.logger.info(f"Point loss weight: {self.config.point_loss_weight}")
        self.logger.info(f"Point loss warmup steps: {self.config.point_loss_warmup_steps}")

        self.logger.info(f"MSS loss weight: {self.config.mss_loss_weight}")
        if self.config.mss_loss_weight > 0:
            self.logger.info("MSS_Loss_2D config:")
            self.logger.info(dict_str(self.mss_loss.config.__dict__))

        self.logger.info(f"Spec reg loss weight: {self.config.spec_reg_loss_weight}")
        if self.config.spec_reg_loss_weight > 0:
            self.logger.info("SpecRegLoss config:")
            self.logger.info(dict_str(self.spec_reg_loss.config.__dict__))
    
    def train_batch(self, batch: dict) -> dict[str, Union[torch.Tensor, float]]:

        if "audio_embeddings" in batch:
            audio_embeddings = normalize(batch["audio_embeddings"]).detach()
            dae_embeddings = self.dae.get_embeddings(audio_embeddings)
        else:
            audio_embeddings = None
            dae_embeddings = None
        
        raw_audio = random_stereo_augmentation(batch["audio"])
        mel_spec = self.format.raw_to_mel_spec(raw_audio).detach()
        latents, reconstructed, latents_kld, hidden_kld = self.dae(mel_spec, dae_embeddings)

        point_loss_weight = self.config.point_loss_weight
        if self.config.point_loss_warmup_steps > 0:
            if self.trainer.global_step < self.config.point_loss_warmup_steps:
                point_loss_weight *= 1 - self.trainer.global_step / self.config.point_loss_warmup_steps
            else:
                point_loss_weight = 0
        point_loss = torch.nn.functional.l1_loss(reconstructed, mel_spec, reduction="none").mean(dim=(1,2,3))
        point_loss_mse = torch.nn.functional.mse_loss(reconstructed, mel_spec, reduction="none").mean(dim=(1,2,3)).detach()

        if point_loss_weight > 0:
            recon_loss =  point_loss * point_loss_weight
        else:
            recon_loss = torch.zeros(mel_spec.shape[0], device=self.trainer.accelerator.device)
        
        if self.config.mss_loss_weight > 0:
            mss_loss = self.mss_loss.mss_loss(reconstructed, mel_spec)
            recon_loss = recon_loss + mss_loss * self.config.mss_loss_weight
        else:
            mss_loss = None

        recon_loss_logvar = self.dae.get_recon_loss_logvar()
        recon_loss_nll = recon_loss / recon_loss_logvar.exp() + recon_loss_logvar

        latents_kl_loss_weight = self.config.latents_kl_loss_weight
        hidden_kl_loss_weight = self.config.hidden_kl_loss_weight
        if self.trainer.global_step < self.config.kl_warmup_steps:
            warmup_scale = self.trainer.global_step / self.config.kl_warmup_steps
            latents_kl_loss_weight *= warmup_scale
            hidden_kl_loss_weight *= warmup_scale

        total_loss = recon_loss_nll + latents_kld * latents_kl_loss_weight + hidden_kld * hidden_kl_loss_weight

        logs = {}

        if self.config.spec_reg_loss_weight > 0:
            spec_reg_loss = self.spec_reg_loss.spec_reg_loss(latents, mel_spec)
            logs["loss/spec_reg"] = spec_reg_loss
            logs["loss_weight/spec_reg"] = self.config.spec_reg_loss_weight
            total_loss = total_loss + spec_reg_loss * self.config.spec_reg_loss_weight

        logs.update({
            "loss": total_loss,
            "loss/recon": recon_loss,
            "loss/point": point_loss,
            "loss/point_mse": point_loss_mse,
            "loss/kl_latents": latents_kld,
            "loss/kl_hidden": hidden_kld,
            "loss_weight/kl_latents": latents_kl_loss_weight,
            "loss_weight/kl_hidden": hidden_kl_loss_weight,
            "loss_weight/point": point_loss_weight,
            "loss_weight/mss": self.config.mss_loss_weight,
            "io_stats/mel_spec_std": mel_spec.std(dim=(1,2,3)),
            "io_stats/mel_spec_mean": mel_spec.mean(dim=(1,2,3)),
            "io_stats/recon_mel_std": reconstructed.std(dim=(1,2,3)),
            "io_stats/recon_mel_mean": reconstructed.mean(dim=(1,2,3)),
            "io_stats/latents_std": latents.std(dim=(1,2,3)),
            "io_stats/latents_mean": latents.mean(dim=(1,2,3))
        })
        
        if mss_loss is not None:
            logs["loss/mss"] = mss_loss

        for name, block in self.dae.encoder.enc.items():
            logs[f"res_t/enc_{name}"] = block.res_balance.sigmoid().detach()

        for name, block in self.dae.dec.items():
            logs[f"res_t/dec_{name}"] = block.res_balance.sigmoid().detach()

        return logs