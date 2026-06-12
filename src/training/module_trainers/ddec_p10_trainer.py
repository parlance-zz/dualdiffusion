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
from training.module_trainers.unet_trainer_p5 import UNetTrainerConfig, UNetTrainer
from training.loss.mss_2d import MSSLoss2D, MSSLoss2DConfig
from training.loss.sigreg import sigreg_strong_loss
from training.loss.mss_1d import MSSLoss1D, MSSLoss1DConfig
from modules.daes.dae_edm2_q7 import DAE
from modules.unets.unet_edm2_q7_ddec import UNet
from modules.unets.unet_edm2_p6 import UNet as UNet_LDM
from modules.formats.ms_mdct_dual_5 import MS_MDCT_DualFormat
from modules.mp_tools import normalize
from utils.dual_diffusion_utils import dict_str


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
    sigreg: dict[str, Any]
    mss_1d: dict[str, Any]

    mss_2d: dict[str, Any]
    mss_2d_leak_pow: float = 4
    mss_2d_leak_steps: int = 200
    mss_2d_loss_weight: float = 0

    latents_sigreg_loss_weight: float = 0
    sigreg_loss_warmup_steps: int = 0
    
    mss_loss_weight: float = 0
    cepstrum_loss_weight: float = 0

    add_x_ref_noise: float = 0
    add_latents_noise: Optional[float] = None
    unet_loss_weight: float     = 0.03
    unet_loss_warmup_steps: int = 2000

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
                #self.dae = trainer.pipeline.dae.to(device=trainer.accelerator.device, dtype=torch.bfloat16).requires_grad_(False)
                #assert self.dae.config.last_global_step > 0
                pass
        else:
            #assert self.train_ddecp == False
            #self.ddecp = trainer.pipeline.ddecp.to(device=trainer.accelerator.device, dtype=torch.bfloat16).requires_grad_(False)
            #assert self.ddecp.config.last_global_step > 0

            #self.train_ddecp = True
            #if config.mss_2d_loss_weight > 0:
            #    self.mss_2d = MSSLoss2D(MSSLoss2DConfig(**config.mss_2d), device=trainer.accelerator.device)
            #else:
            #    self.mss_2d = None
            pass

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

        if self.train_dae == True:
            self.logger.info(f"Add latents noise: {self.config.add_latents_noise}")
            self.logger.info(f"SIGReg loss weight: {self.config.latents_sigreg_loss_weight} (warmup steps: {self.config.sigreg_loss_warmup_steps})")
            self.logger.info(f"SIGReg config: {dict_str(self.config.sigreg)}")

        if self.config.random_stereo_augmentation == True:
            self.logger.info("Using random stereo augmentation")
        else: self.logger.info("Random stereo augmentation is disabled")

        if self.train_ddecp == True:
            self.logger.info(f"Add x_ref noise: {self.config.add_x_ref_noise}")
            self.logger.info(f"MSS-1D loss weight: {self.config.mss_loss_weight} (cepstrum loss weight: {self.config.cepstrum_loss_weight})")

            if self.config.mss_loss_weight > 0 or self.config.cepstrum_loss_weight > 0:
                self.mss_1d = MSSLoss1D(MSSLoss1DConfig(**config.mss_1d), device=trainer.accelerator.device)
                self.logger.info(f"MSS-1D config: {dict_str(self.mss_1d.config.__dict__)}")
            else:
                self.mss_1d = None

            self.logger.info(f"DDEC-P trainer:")
            self.ddecp_trainer = UNetTrainer(UNetTrainerConfig(**config.ddecp), trainer, self.ddecp, "ddecp", mss_1d=self.mss_1d)

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

        mdct_phase_psd = self.format.raw_to_mdct_phase_psd(raw_samples, random_phase_augmentation=self.config.random_phase_augmentation)#, level=-1)
        #mdct_phase_psd = self.format.flatten_mdct_phase_psd(mdct_phase_psd)
        ms_psds = self.format.raw_to_ms_psd(raw_samples, level=-1)

        logs.update({
            "io_stats/mdct_phase_psd_msq": mdct_phase_psd.pow(2).mean(dim=(1,2,3)),
            "io_stats/mdct_phase_psd_mean": mdct_phase_psd.mean(dim=(1,2,3)),
        })
        for i, psd in enumerate(ms_psds):
            logs[f"io_stats/ms_psd_level_{i}_msq"] = psd.pow(2).mean(dim=(1,2,3))
            logs[f"io_stats/ms_psd_level_{i}_mean"] = psd.mean(dim=(1,2,3))

        if self.train_dae == True:
            
            latents, ddec_cond = self.trainer.get_ddp_module(self.dae)(
                ms_psds, audio_embeddings, latents_sigma=self.config.add_latents_noise)
            
            self.dae.latents_stats_tracker(latents)

        #elif self.train_ddecm == True or self.train_ddecp == True:
        #    latents, ddec_cond = self.dae(ms_psds, audio_embeddings, latents_sigma=self.config.add_latents_noise)
        #    latents = latents.detach(); ddec_cond: list[torch.Tensor] = [x.detach() for x in ddec_cond]
        else:
            latents = ddec_cond = None
        
        if latents is not None:
            latents: torch.Tensor = latents.float()
            ddec_cond: list[torch.Tensor] = [x.float() for x in ddec_cond]
            
            logs.update({
                "io_stats/latents_var": latents.var(dim=(1,2,3)).detach(),
                "io_stats/latents_mean": latents.mean(dim=(1,2,3)).detach(),
                "io_stats/latents_per_ch_mean": self.dae.latents_stats_tracker.mean.pow(2).mean().pow(0.5),
                "io_stats/latents_per_ch_var": self.dae.latents_stats_tracker.var.mean(),
                "io_stats/latents_sigma": self.config.add_latents_noise if self.config.add_latents_noise is not None else 0,
            })

            for i, cond in enumerate(ddec_cond):
                logs[f"io_stats/ddec_cond_level_{i}_msq"] = cond.pow(2).mean(dim=(1,2,3)).detach()
                logs[f"io_stats/ddec_cond_level_{i}_mean"] = cond.mean(dim=(1,2,3)).detach()
                
            if self.dae.config.latent_channels <= 8:
                for i in range(self.dae.config.latent_channels):
                    logs[f"ch_stats/mean_{i}"] = self.dae.latents_stats_tracker.mean[i].detach()
                    logs[f"ch_stats/var_{i}"]  = self.dae.latents_stats_tracker.var[i].detach()

        if ddec_cond is not None:
            ddec_x_ref: list[torch.Tensor] = self.format.ms_psd_to_psd_linear(ddec_cond)
        else:
            ddec_x_ref: list[torch.Tensor] = [x + torch.randn_like(x) * self.config.add_x_ref_noise for x in self.format.ms_psd_to_psd_linear(ms_psds)]

        if self.train_ddecp == True:
            logs.update(self.ddecp_trainer.train_batch(mdct_phase_psd, audio_embeddings, ref_samples=ddec_x_ref))
            logs["loss"] = logs["loss"] + logs["loss/ddecp"]

            logs["loss_weight/mss1d"] = self.config.mss_loss_weight
            logs["loss_weight/mss1d_cepstrum"] = self.config.cepstrum_loss_weight
            if self.config.mss_loss_weight > 0 or self.config.cepstrum_loss_weight > 0:
                logs["loss"] = logs["loss"] + logs["loss/mss1d"] * self.config.mss_loss_weight
                logs["loss"] = logs["loss"] + logs["loss/mss1d_cepstrum"] * self.config.cepstrum_loss_weight

        if self.train_ddecm == True:
            raise NotImplementedError()
            #logs.update(self.ddecm_trainer.train_batch(input_mel_spec, audio_embeddings, ddec_x_ref))
            #logs["loss"] = logs["loss"] + logs["loss/ddecm"]
        
        if self.train_dae == True:
            
            latents_sigreg_loss_weight = self.config.latents_sigreg_loss_weight
            if self.trainer.global_step < self.config.sigreg_loss_warmup_steps:
                latents_sigreg_loss_weight *= (self.trainer.global_step + 1) / self.config.sigreg_loss_warmup_steps
            logs["loss_weight/sigreg_latents"] = latents_sigreg_loss_weight

            if latents_sigreg_loss_weight > 0:
                latents_sigreg_loss = sigreg_strong_loss(latents, **self.config.sigreg)
                if latents_sigreg_loss_weight <= 0:
                    latents_sigreg_loss = latents_sigreg_loss.detach()
                logs["loss/latents_sigreg"] = latents_sigreg_loss.detach()
                logs["loss"] = logs["loss"] + latents_sigreg_loss * latents_sigreg_loss_weight

        if self.train_unet == True:
            
            unet_loss_weight = self.config.unet_loss_weight
            if self.trainer.global_step < self.config.unet_loss_warmup_steps:
                unet_loss_weight *= self.trainer.global_step / self.config.unet_loss_warmup_steps

            logs.update(self.unet_trainer.train_batch(latents, audio_embeddings))
            logs["loss"] = logs["loss"] + logs["loss/unet"] * unet_loss_weight
            logs["loss_weight/unet"] = unet_loss_weight

        if self.trainer.config.enable_debug_mode == True:
            print("mdct_phase_psd.shape:", mdct_phase_psd.shape)

            if latents is not None:
                print("latents.shape:", latents.shape)
                for i, cond in enumerate(ddec_cond):
                    print(f"ddec_cond_{i}.shape:", cond.shape)
                for i, x in enumerate(ddec_x_ref):
                    print(f"ddec_x_ref_{i}.shape:", x.shape)

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