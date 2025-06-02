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
from training.loss.wavelet import WaveletLoss, WaveletLoss_Config
from modules.daes.dae_edm2_g1 import DAE_G1
from modules.formats.ms_mdct_dual import MS_MDCT_DualFormat
from modules.mp_tools import normalize
from utils.dual_diffusion_utils import dict_str


@dataclass
class DAETrainer_G1_Config(ModuleTrainerConfig):

    add_latents_noise: float = 0

    kl_loss_weight: float = 2e-2
    kl_warmup_steps: int  = 2000

    point_loss_weight: float = 10
    point_loss_warmup_steps: int = 2000

    mss_loss_2d_config: Optional[dict[str, Any]] = None

class DAETrainer_G1(ModuleTrainer):
    
    @torch.no_grad()
    def __init__(self, config: DAETrainer_G1_Config, trainer: DualDiffusionTrainer) -> None:
        
        self.config = config
        self.trainer = trainer
        self.logger = trainer.logger
        self.dae: DAE_G1 = trainer.get_train_module("dae")

        config.mss_loss_2d_config = config.mss_loss_2d_config or {}
        self.mss_loss = MSSLoss2D(MSSLoss2DConfig(**config.mss_loss_2d_config), device=trainer.accelerator.device)
        #self.wavelet_loss = WaveletLoss(WaveletLoss_Config())
        self.format: MS_MDCT_DualFormat = trainer.pipeline.format.to(self.trainer.accelerator.device)

        if trainer.config.enable_model_compilation == True:
            self.dae.compile(**trainer.config.compile_params)
            self.format.compile(**trainer.config.compile_params)
            self.mss_loss.compile(**trainer.config.compile_params)
            #self.wavelet_loss.compile(**trainer.config.compile_params)

        self.logger.info("Training DAE model:")
        self.logger.info(f"KL loss weight: {self.config.kl_loss_weight} KL warmup steps: {self.config.kl_warmup_steps}")
        self.logger.info(f"Point loss weight: {self.config.point_loss_weight}")
        self.logger.info("MSS_Loss_2D Config:")
        self.logger.info(dict_str(self.mss_loss.config.__dict__))

    def train_batch(self, batch: dict) -> dict[str, Union[torch.Tensor, float]]:

        if "audio_embeddings" in batch:
            audio_embeddings = normalize(batch["audio_embeddings"]).detach()
            dae_embeddings = self.dae.get_embeddings(audio_embeddings)
        else:
            dae_embeddings = None
        
        mel_spec = self.format.raw_to_mel_spec(batch["audio"]).detach()
        latents, reconstructed, pre_norm_latents = self.dae(mel_spec, dae_embeddings, add_latents_noise=self.config.add_latents_noise)

        point_loss = torch.nn.functional.l1_loss(reconstructed, mel_spec, reduction="none").mean(dim=(1,2,3))

        recon_loss = self.mss_loss.mss_loss(reconstructed, mel_spec)
        #recon_loss, level_losses = self.wavelet_loss.wavelet_loss(reconstructed, mel_spec)
        recon_loss_logvar = self.dae.get_recon_loss_logvar()
        recon_loss_nll = recon_loss / recon_loss_logvar.exp() + recon_loss_logvar

        pre_norm_latents_var = pre_norm_latents.var(dim=(1,2,3))
        kl_loss = pre_norm_latents.mean(dim=(1,2,3)).square() + pre_norm_latents_var - 1 - pre_norm_latents_var.log()

        kl_loss_weight = self.config.kl_loss_weight
        if self.trainer.global_step < self.config.kl_warmup_steps:
            kl_loss_weight *= self.trainer.global_step / self.config.kl_warmup_steps

        point_loss_weight = self.config.point_loss_weight
        if self.trainer.global_step < self.config.point_loss_warmup_steps:
            point_loss_weight *= 1 - self.trainer.global_step / self.config.point_loss_warmup_steps
        else:
            point_loss_weight = 0

        logs = {
            "loss": recon_loss_nll + kl_loss * kl_loss_weight + point_loss * point_loss_weight,
            "loss/recon": recon_loss,
            "loss/point": point_loss,
            "loss/kl": kl_loss,
            "loss_weight/kl": kl_loss_weight,
            "loss_weight/point": point_loss_weight,
            "io_stats/mel_spec_std": mel_spec.std(dim=(1,2,3)),
            "io_stats/mel_spec_mean": mel_spec.mean(dim=(1,2,3)),
            "io_stats/recon_mel_std": reconstructed.std(dim=(1,2,3)),
            "io_stats/recon_mel_mean": reconstructed.mean(dim=(1,2,3)),
            "io_stats/latents_std": latents.std(dim=(1,2,3)),
            "io_stats/latents_mean": latents.mean(dim=(1,2,3)),
            "io_stats/latents_pre_norm_std": pre_norm_latents_var.sqrt()
        }

        #for i, level_loss in enumerate(level_losses):
        #    logs[f"loss/w_level{i}"] = level_loss

        return logs