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
from modules.daes.dae_edm2_p5 import DAE
from modules.unets.unet_edm2_p5_ddec import UNet
from modules.unets.unet_edm2_p5 import UNet as UNet_LDM
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

    ddecm: dict[str, Any]
    ddecp: dict[str, Any]
    unet: dict[str, Any]

    kl_loss_weight: float = 1e-2
    kl_warmup_steps: int  = 150

    decoder_loss_multiplier: float = 3
    add_latents_noise: float = 0.08

    random_stereo_augmentation: bool = True
    random_phase_augmentation: bool  = True

    crop_edges: int = 4 # used to avoid artifacts due to mdct lapped blocks at beginning and end of sample

class DiffusionDecoder_Trainer(ModuleTrainer):
    
    @torch.no_grad()
    def __init__(self, config: DiffusionDecoder_Trainer_Config, trainer: DualDiffusionTrainer) -> None:

        self.config = config
        self.trainer = trainer
        self.logger = trainer.logger

        self.unet: UNet_LDM = trainer.get_train_module("unet")
        self.ddecp: UNet = trainer.get_train_module("ddecp")
        self.ddecm: UNet = trainer.get_train_module("ddecm")
        self.dae: DAE = trainer.get_train_module("dae")

        assert self.ddecp is not None and self.ddecm is not None
        
        if self.dae is None:
            assert self.dae.config.last_global_step > 0 and self.unet is None
            self.dae = trainer.pipeline.dae.to(device=trainer.accelerator.device, dtype=torch.bfloat16).requires_grad_(False)
            self.train_dae = False
        else:
            assert self.unet is not None
            self.train_dae = True
        
        self.format: MS_MDCT_DualFormat = trainer.pipeline.format.to(self.trainer.accelerator.device)

        if trainer.config.enable_model_compilation:
            self.ddecp.compile(**trainer.config.compile_params)
            self.ddecm.compile(**trainer.config.compile_params)
            self.dae.compile(**trainer.config.compile_params)
            self.format.compile(**trainer.config.compile_params)

            if self.unet is not None:
                self.unet.compile(**trainer.config.compile_params)

        self.logger.info(f"Training modules: {trainer.config.train_modules}")
        if self.train_dae == True:
            self.logger.info(f"KL loss weight: {self.config.kl_loss_weight} KL warmup steps: {self.config.kl_warmup_steps}")
            self.logger.info(f"Decoder loss multiplier: {config.decoder_loss_multiplier}")
            self.logger.info(f"Add latents noise: {config.add_latents_noise}")
        self.logger.info(f"Crop edges: {self.config.crop_edges}")

        if self.config.random_stereo_augmentation == True:
            self.logger.info("Using random stereo augmentation")
        else: self.logger.info("Random stereo augmentation is disabled")

        self.logger.info("DDEC-P trainer:")
        self.ddecp_trainer = UNetTrainer(UNetTrainerConfig(**config.ddecp), trainer, self.ddecp, "ddecp")
        self.logger.info("DDEC-M trainer:")
        self.ddecm_trainer = UNetTrainer(UNetTrainerConfig(**config.ddecm), trainer, self.ddecm, "ddecm")
        self.logger.info("UNet trainer:")
        self.unet_trainer = UNetTrainer(UNetTrainerConfig(**config.unet), trainer, self.unet, "unet")

    @torch.no_grad()
    def init_batch(self, validation: bool = False) -> Optional[dict[str, Union[torch.Tensor, float]]]:
        
        self.ddecp_trainer.init_batch(validation)
        self.ddecm_trainer.init_batch(validation)
        self.unet_trainer.init_batch(validation)

        return None
    
    def train_batch(self, batch: dict) -> Optional[dict[str, Union[torch.Tensor, float]]]:

        # prepare model inputs
        if "audio_embeddings" in batch:
            audio_embeddings = normalize(batch["audio_embeddings"]).detach()
        else:
            audio_embeddings = None

        if self.config.random_stereo_augmentation == True:
            raw_samples = random_stereo_augmentation(batch["audio"])
        else:
            raw_samples = batch["audio"]

        mdct_phase, mdct_psd = self.format.raw_to_mdct_phase_psd(raw_samples, random_phase_augmentation=self.config.random_phase_augmentation)
        mdct_phase = mdct_phase[..., self.config.crop_edges:-self.config.crop_edges]
        mdct_psd = mdct_psd[..., self.config.crop_edges:-self.config.crop_edges]

        input_mel_spec = self.format.raw_to_mel_spec(raw_samples)
        input_mel_spec = input_mel_spec[..., self.config.crop_edges:-self.config.crop_edges]

        if self.train_dae == True:
            latents, ddec_cond = self.trainer.get_ddp_module(self.dae)(input_mel_spec, audio_embeddings, self.config.add_latents_noise)
        else:
            latents, ddec_cond = self.dae(input_mel_spec, audio_embeddings, self.config.add_latents_noise)
            ddec_cond = ddec_cond.detach()
            
        latents: torch.Tensor = latents.float()

        latents_var = latents.pow(2).mean(dim=(0,2,3)) + 1e-20
        var_kl = latents_var - 1 - latents_var.log()
        kl_loss = var_kl.mean() + latents.mean(dim=(0,2,3)).square().mean()
        kl_loss = kl_loss.expand(latents.shape[0]) # needed for per-sample logging

        kl_loss_weight = self.config.kl_loss_weight
        if self.trainer.global_step < self.config.kl_warmup_steps:
            kl_loss_weight *= self.trainer.global_step / self.config.kl_warmup_steps
        
        logs = {
            "loss": kl_loss * kl_loss_weight if self.train_dae == True else torch.zeros_like(kl_loss),
            "io_stats/ddec_cond_var": ddec_cond.var(dim=(1,2,3)),
            "io_stats/ddec_cond_mean": ddec_cond.mean(dim=(1,2,3)),
            "io_stats/latents_var": latents.var(dim=(1,2,3)).detach(),
            "io_stats/latents_mean": latents.mean(dim=(1,2,3)).detach(),

            "io_stats_ddecp/mdct_phase_var": mdct_phase.var(dim=(1,2,3)),
            "io_stats_ddecm/mdct_psd_var": mdct_psd.var(dim=(1,2,3)),
            "io_stats_ddecm/mdct_psd_mean": mdct_psd.mean(dim=(1,2,3)),

            "loss/kl_latents": kl_loss.detach(),
            "loss_weight/kl_latents": kl_loss_weight,
        }

        noise = torch.randn_like(mdct_psd)
        perturb_noise = torch.randn_like(mdct_psd)

        logs.update(self.ddecp_trainer.train_batch(mdct_phase, audio_embeddings, ddec_cond, noise=noise, perturb_noise=perturb_noise))
        logs.update(self.ddecm_trainer.train_batch(mdct_psd, audio_embeddings, ddec_cond, noise=noise, perturb_noise=perturb_noise))
        logs["loss"] = logs["loss"] + (logs["loss/ddecp"] + logs["loss/ddecm"]) * (self.config.decoder_loss_multiplier / 2)

        if self.train_dae == True:
            logs.update(self.unet_trainer.train_batch(latents, audio_embeddings))
            logs["loss"] = logs["loss"] + logs["loss/unet"]

        dynamic_range_ddecm = mdct_psd.amax(dim=(1,2,3)) - mdct_psd.amin(dim=(1,2,3))
        logs["io_stats_ddecm/dynamic_range"] = dynamic_range_ddecm
        dynamic_range_ddecp = mdct_phase.amax(dim=(1,2,3)) - mdct_phase.amin(dim=(1,2,3))
        logs["io_stats_ddecp/dynamic_range"] = dynamic_range_ddecp

        if self.trainer.config.enable_debug_mode == True:
            print("mdct_phase.shape:", mdct_phase.shape)
            print("mdct_psd.shape:", mdct_psd.shape)
            print("ddec_cond.shape:", ddec_cond.shape)
            print("latents.shape:", latents.shape)
            print("input_mel_spec.shape", input_mel_spec.shape)

        return logs
      
    @torch.no_grad()
    def finish_batch(self) -> Optional[dict[str, Union[torch.Tensor, float]]]:

        logs = {}
        logs.update(self.ddecp_trainer.finish_batch())
        logs.update(self.ddecm_trainer.finish_batch())

        return logs