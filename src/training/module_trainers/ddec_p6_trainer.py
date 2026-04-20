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
from typing import Union, Optional, Any

import torch

from training.trainer import DualDiffusionTrainer
from training.module_trainers.module_trainer import ModuleTrainer, ModuleTrainerConfig
from training.module_trainers.unet_trainer_p4 import UNetTrainerConfig, UNetTrainer
from training.loss.lipschitz import lipschitz_loss
from training.loss.mss_2d import MSSLoss2D, MSSLoss2DConfig
from modules.daes.dae_edm2_p6 import DAE
from modules.unets.unet_edm2_p6_ddec import UNet
from modules.unets.unet_edm2_p6 import UNet as UNet_LDM
from modules.formats.ms_mdct_dual_3 import MS_MDCT_DualFormat
from modules.mp_tools import normalize


@torch.no_grad()
def random_stereo_augmentation(x: torch.Tensor) -> torch.Tensor:
    
    output = x.clone()
    flip_mask = (torch.rand(x.shape[0]) > 0.5).to(x.device)
    output[flip_mask] = output[flip_mask].flip(dims=(1,))
    
    return output

@dataclass
class DiffusionDecoder_Trainer_Config(ModuleTrainerConfig):

    ddecp: dict[str, Any]
    ddecm: dict[str, Any]
    unet: dict[str, Any]

    mss_2d: dict[str, Any]
    mse_dae_loss_weight: float = 1e-2
    mse_dae_warmup_steps: int = 200

    lipschitz_loss_weight: Optional[float] = None
    lipschitz_loss_eps: float = 1e-2
    add_latents_noise: Optional[float] = 0.08
    add_mel_spec_noise: float = 0.08

    unet_loss_weight: float     = 0.2
    unet_loss_warmup_steps: int = 300

    random_stereo_augmentation: bool = True
    random_phase_augmentation: bool  = True

class DiffusionDecoder_Trainer(ModuleTrainer):
    
    @torch.no_grad()
    def __init__(self, config: DiffusionDecoder_Trainer_Config, trainer: DualDiffusionTrainer) -> None:

        self.config = config
        self.trainer = trainer
        self.logger = trainer.logger

        self.logger.info(f"Training modules: {trainer.config.train_modules}")
        
        self.ddecp: UNet = trainer.get_train_module("ddecp")
        self.ddecm: UNet = trainer.get_train_module("ddecm")
        self.dae: DAE = trainer.get_train_module("dae")
        self.unet: UNet_LDM = trainer.get_train_module("unet")

        self.train_dae = self.dae is not None
        self.train_unet = self.unet is not None
        self.train_ddecm = self.ddecm is not None
        self.train_ddecp = self.ddecp is not None

        if self.train_dae == False:
            if self.train_ddecm == True or self.train_ddecp == True:
                self.dae = trainer.pipeline.dae.to(device=trainer.accelerator.device, dtype=torch.bfloat16).requires_grad_(False)
                assert self.dae.config.last_global_step > 0
        else:
            self.mss_2d = MSSLoss2D(MSSLoss2DConfig(), device=trainer.accelerator.device)

        self.format: MS_MDCT_DualFormat = trainer.pipeline.format.to(self.trainer.accelerator.device)

        if trainer.config.enable_model_compilation:
            self.format.compile(**trainer.config.compile_params)

            if self.dae is not None:
                self.dae.compile(**trainer.config.compile_params)
            if self.ddecp is not None:
                self.ddecp.compile(**trainer.config.compile_params)
            if self.ddecm is not None:
                self.ddecm.compile(**trainer.config.compile_params)
            if self.unet is not None:
                self.unet.compile(**trainer.config.compile_params)

            global lipschitz_loss
            lipschitz_loss = torch.compile(lipschitz_loss, **trainer.config.compile_params)

        if self.train_dae == True or self.train_ddecm == True:
            self.logger.info(f"Add latents noise: {self.config.add_latents_noise}")
            self.logger.info(f"Lipschitz loss weight: {self.config.lipschitz_loss_weight}")
    
        if self.config.random_stereo_augmentation == True:
            self.logger.info("Using random stereo augmentation")
        else: self.logger.info("Random stereo augmentation is disabled")

        if self.train_ddecp == True:
            self.logger.info(f"DDEC-P trainer (add mel spec noise: {self.config.add_mel_spec_noise}):")
            self.ddecp_trainer = UNetTrainer(UNetTrainerConfig(**config.ddecp), trainer, self.ddecp, "ddecmp")

            if self.config.random_phase_augmentation == True:
                self.logger.info("Using random phase augmentation")
            else: self.logger.info("Random phase augmentation is disabled")

        if self.train_ddecm == True:
            self.logger.info(f"DDEC-M trainer:")
            self.ddecm_trainer = UNetTrainer(UNetTrainerConfig(**config.ddecm), trainer, self.ddecm, "ddecm")
        if self.train_unet == True:
            self.logger.info(f"UNet-LDM trainer (loss weight: {self.config.unet_loss_weight}) (warmup steps:{self.config.unet_loss_warmup_steps}):")
            self.unet_trainer = UNetTrainer(UNetTrainerConfig(**config.unet), trainer, self.unet, "unet")

    @torch.no_grad()
    def init_batch(self, validation: bool = False) -> Optional[dict[str, Union[torch.Tensor, float]]]:
        
        if self.train_ddecp == True:
            self.ddecp_trainer.init_batch(validation)
        if self.train_ddecm == True:
            self.ddecm_trainer.init_batch(validation)
        if self.train_unet == True:
            self.unet_trainer.init_batch(validation)

        return None
    
    def train_batch(self, batch: dict) -> Optional[dict[str, Union[torch.Tensor, float]]]:

        logs = {"loss": torch.zeros(self.trainer.config.device_batch_size, device=self.trainer.accelerator.device)}

        # prepare model inputs
        if "audio_embeddings" in batch:
            audio_embeddings = normalize(batch["audio_embeddings"]).detach()
        else:
            audio_embeddings = None

        if self.config.random_stereo_augmentation == True:
            raw_samples = random_stereo_augmentation(batch["audio"])
        else:
            raw_samples = batch["audio"]

        mdct_phase = self.format.raw_to_mdct_phase(raw_samples, random_phase_augmentation=self.config.random_phase_augmentation)
        input_mel_spec = self.format.raw_to_mel_spec(raw_samples)
        dae_input = input_mel_spec

        logs.update({
            "io_stats/input_mel_spec_mean": input_mel_spec.mean(dim=(1,2,3)),
            "io_stats/input_mel_spec_var": input_mel_spec.var(dim=(1,2,3)),
            "io_stats/mdct_phase_var": mdct_phase.var(dim=(1,2,3)),
        })

        if self.train_dae == True:
            
            latents, ddec_cond = self.trainer.get_ddp_module(self.dae)(
                dae_input, audio_embeddings, latents_sigma=self.config.add_latents_noise)
            
            self.dae.latents_stats_tracker(latents)

        elif self.train_ddecm == True or self.train_ddecp == True:
            latents, ddec_cond = self.dae(dae_input, audio_embeddings, latents_sigma=self.config.add_latents_noise)
            latents = latents.detach(); ddec_cond = ddec_cond.detach()
            #latents = ddec_cond = None
        else:
            latents = ddec_cond = None
        
        if latents is not None:
            latents: torch.Tensor = latents.float()
            ddec_cond: torch.Tensor = ddec_cond.float()
            
            logs.update({
                "io_stats/latents_var": latents.var(dim=(1,2,3)).detach(),
                "io_stats/latents_mean": latents.mean(dim=(1,2,3)).detach(),
                "io_stats/latents_per_ch_mean": self.dae.latents_stats_tracker.mean.pow(2).mean().pow(0.5),
                "io_stats/latents_per_ch_var": self.dae.latents_stats_tracker.var.mean(),
                "io_stats/latents_sigma": self.config.add_latents_noise if self.config.add_latents_noise is not None else 0,
                "io_stats/ddec_cond_var": ddec_cond.var(dim=(1,2,3)),
                "io_stats/ddec_cond_mean": ddec_cond.mean(dim=(1,2,3))
            })

            if self.dae.config.latent_channels <= 32:
                for i in range(self.dae.config.latent_channels):
                    logs[f"ch_stats/mean_{i}"] = self.dae.latents_stats_tracker.mean[i].detach()
                    logs[f"ch_stats/var_{i}"]  = self.dae.latents_stats_tracker.var[i].detach()

        if self.train_ddecp == True:
            #ddecp_x_ref = input_mel_spec + torch.randn_like(input_mel_spec) * self.config.add_mel_spec_noise
            ddecp_x_ref = ddec_cond
            logs.update(self.ddecp_trainer.train_batch(mdct_phase, audio_embeddings, ddecp_x_ref))
            logs["loss"] = logs["loss"] + logs["loss/ddecmp"]

        if self.train_ddecm == True:
            logs.update(self.ddecm_trainer.train_batch(input_mel_spec, audio_embeddings, ddec_cond))
            logs["loss"] = logs["loss"] + logs["loss/ddecm"]
        #"""
        #elif self.train_dae == True:
        if self.train_dae == True:
            
            recon_loss_logvar: torch.nn.Parameter = getattr(self.dae, "recon_loss_logvar", None)
            logs["loss/mel_spec_mse"] = torch.nn.functional.mse_loss(ddec_cond, input_mel_spec, reduction="none").mean(dim=(1,2,3))
            logs["loss/mel_spec_mss2d"] = self.mss_2d.mss_loss(ddec_cond, input_mel_spec)

            mse_dae_loss_weight = max(10 - 10 * self.trainer.global_step / self.config.mse_dae_warmup_steps, self.config.mse_dae_loss_weight)
            recon_loss = logs["loss/mel_spec_mss2d"] + logs["loss/mel_spec_mse"] * mse_dae_loss_weight

            if recon_loss_logvar is not None:
                recon_loss = recon_loss / recon_loss_logvar.exp() + recon_loss_logvar
                logs["loss/mel_spec_recon_loss_nll"] = recon_loss

            logs["loss"] = logs["loss"] + recon_loss
            logs["loss_weight/mel_spec_mse"] = mse_dae_loss_weight
        #"""
        
        if self.train_unet == True:
            
            unet_loss_weight = self.config.unet_loss_weight
            if self.trainer.global_step < self.config.unet_loss_warmup_steps:
                unet_loss_weight *= self.trainer.global_step / self.config.unet_loss_warmup_steps

            #if self.config.unet_loss_weight <= 0:
            #    latents = latents.detach()
            #    unet_loss_weight = 1

            logs.update(self.unet_trainer.train_batch(latents, audio_embeddings))
            logs["loss"] = logs["loss"] + logs["loss/unet"] * unet_loss_weight
            logs["loss_weight/unet"] = unet_loss_weight

        if self.train_dae == True and self.config.lipschitz_loss_weight is not None:
            _lipschitz_loss = lipschitz_loss(input_mel_spec, latents, eps=self.config.lipschitz_loss_eps)
            
            logs["loss"] = logs["loss"] + _lipschitz_loss * self.config.lipschitz_loss_weight
            logs["loss/lipschitz"] = _lipschitz_loss.detach()
            logs["loss_weight/lipschitz"] = self.config.lipschitz_loss_weight

        dynamic_range_ddecm = input_mel_spec.amax(dim=(1,2,3)) - input_mel_spec.amin(dim=(1,2,3))
        logs["io_stats_ddecm/dynamic_range"] = dynamic_range_ddecm
        dynamic_range_ddecp = mdct_phase.amax(dim=(1,2,3)) - mdct_phase.amin(dim=(1,2,3))
        logs["io_stats_ddecp/dynamic_range"] = dynamic_range_ddecp

        if self.trainer.config.enable_debug_mode == True:
            print("input_mel_spec.shape:", input_mel_spec.shape)
            print("mdct_phase.shape:", mdct_phase.shape)

            if latents is not None:
                print("ddec_cond.shape:", ddec_cond.shape)
                print("latents.shape:", latents.shape)

        return logs
      
    @torch.no_grad()
    def finish_batch(self) -> Optional[dict[str, Union[torch.Tensor, float]]]:

        logs = {}
        if self.train_ddecp == True:
            logs.update(self.ddecp_trainer.finish_batch())
        if self.train_ddecm == True:
            logs.update(self.ddecm_trainer.finish_batch())
        if self.train_unet == True:
            logs.update(self.unet_trainer.finish_batch())

        return logs