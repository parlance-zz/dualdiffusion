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
from training.module_trainers.unet_trainer_p5 import UNetTrainerConfig as UNetTrainerConfig_LDM, UNetTrainer as UNetTrainer_LDM
from training.loss.sigreg import sigreg_strong_loss
from training.loss.mss_1d import MSSLoss1D, MSSLoss1DConfig
from training.loss.mss_2d import MSSLoss2D, MSSLoss2DConfig
from modules.daes.dae_edm2_q4 import DAE
from modules.unets.unet_edm2_q4_ddec import UNet
from modules.unets.unet_edm2_p6 import UNet as UNet_LDM
from modules.formats.ms_mdct_dual_10 import MS_MDCT_DualFormat
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
    mel_density_loss_weight_pow: float = 1

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
            if self.trainer.pipeline.dae.config.last_global_step > 0:
                self.dae = trainer.pipeline.dae.to(device=trainer.accelerator.device, dtype=torch.bfloat16).requires_grad_(False)
            else:
                self.dae = None
        else:
            if self.train_ddecp == False:
                self.ddecp = trainer.pipeline.ddecp.to(device=trainer.accelerator.device, dtype=torch.bfloat16).requires_grad_(False)
                assert self.ddecp.config.last_global_step > 0
                self.train_ddecp = True
            #self.mss_2d = MSSLoss2D(MSSLoss2DConfig(), device=trainer.accelerator.device)

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
            self.ddecp_trainer = UNetTrainer(UNetTrainerConfig(**config.ddecp), trainer, self.ddecp, "ddecp")

            if self.config.random_phase_augmentation == True:
                self.logger.info("Using random phase augmentation")
            else: self.logger.info("Random phase augmentation is disabled")

        if self.train_ddecm == True:
            self.logger.info(f"DDEC-M trainer:")
            self.ddecm_trainer = UNetTrainer(UNetTrainerConfig(**config.ddecm), trainer, self.ddecm, "ddecm")
        if self.train_unet == True:
            self.logger.info(f"UNet-LDM trainer (loss weight: {self.config.unet_loss_weight}) (warmup steps:{self.config.unet_loss_warmup_steps}):")
            self.unet_trainer = UNetTrainer_LDM(UNetTrainerConfig_LDM(**config.unet), trainer, self.unet, "unet")

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

        mdct_phase_psd = self.format.raw_to_mdct_phase_psd(raw_samples, random_phase_augmentation=self.config.random_phase_augmentation)
        #mdct_phase_psd = self.format.flatten_mdct_phase_psd(mdct_phase_psd)
        ms_psd = self.format.raw_to_ms_psd(raw_samples)

        logs.update({
            "io_stats/mdct_phase_psd_msq": mdct_phase_psd.pow(2).mean(dim=(1,2,3)),
            "io_stats/mdct_phase_psd_mean": mdct_phase_psd.mean(dim=(1,2,3)),
            "io_stats/ms_psd_msq": ms_psd.pow(2).mean(dim=(1,2,3)),
            "io_stats/ms_psd_mean": ms_psd.mean(dim=(1,2,3))
        })

        if self.train_dae == True:
            latents, ddec_cond = self.trainer.get_ddp_module(self.dae)(
                ms_psd, audio_embeddings, latents_sigma=self.config.add_latents_noise)
            
            self.dae.latents_stats_tracker(latents)

        elif self.dae is not None:
            latents, ddec_cond = self.dae(ms_psd, audio_embeddings, latents_sigma=self.config.add_latents_noise)
            latents = latents.detach(); ddec_cond: torch.Tensor = ddec_cond.detach()
        else:
            latents = ddec_cond = None
        
        if latents is not None:
            latents: torch.Tensor = latents.float()
            ddec_cond: torch.Tensor = ddec_cond.float()
            
            logs.update({
                "io_stats/latents_msq": latents.pow(2).mean(dim=(1,2,3)).detach(),
                "io_stats/latents_mean": latents.mean(dim=(1,2,3)).detach(),
                "io_stats/latents_per_ch_mean": self.dae.latents_stats_tracker.mean.abs().mean(),
                "io_stats/latents_per_ch_msq": self.dae.latents_stats_tracker.msq.mean(),
                "io_stats/latents_sigma": self.config.add_latents_noise if self.config.add_latents_noise is not None else 0,
            })

            #for i, cond in enumerate(ddec_cond):
            #    logs[f"io_stats/ddec_cond_level_{i}_msq"] = cond.pow(2).mean(dim=(1,2,3)).detach()
            #    logs[f"io_stats/ddec_cond_level_{i}_mean"] = cond.mean(dim=(1,2,3)).detach()
                
            if self.dae.config.latent_channels <= 8:
                for i in range(self.dae.config.latent_channels):
                    logs[f"ch_stats/mean_{i}"] = self.dae.latents_stats_tracker.mean[i].detach()
                    logs[f"ch_stats/msq_{i}"]  = self.dae.latents_stats_tracker.msq[i].detach()

        if ddec_cond is not None:
            ddec_x_ref: torch.Tensor = ddec_cond + torch.randn_like(ddec_cond) * self.config.add_x_ref_noise
        else:
            ddec_x_ref: torch.Tensor = ms_psd.detach() + torch.randn_like(ms_psd) * self.config.add_x_ref_noise
            logs["io_stats/add_x_ref_noise"] = self.config.add_x_ref_noise

        if self.train_ddecp == True:
            loss_weight = self.format.mdct_mel_density.pow(self.config.mel_density_loss_weight_pow)
            loss_weight /= loss_weight.mean()
            logs.update(self.ddecp_trainer.train_batch(
                mdct_phase_psd, audio_embeddings, ref_samples=ddec_x_ref, loss_weight=loss_weight))
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

            """
            leak = 1 - min(self.trainer.global_step / 200, 1)
            if leak <= 0: leak = None

            logs["loss/mss_2d"] = self.mss_2d.mss_loss(ddec_cond, ms_psd, 4, leak_max=leak)
            logs["loss/mss_2d_nll"] = logs["loss/mss_2d"] / self.dae.get_recon_loss_logvar().exp() + self.dae.get_recon_loss_logvar()
            logs["loss"] = logs["loss"] + logs["loss/dae_qh_nll"]

            logs["loss/dae_mse"] = torch.nn.functional.mse_loss(ddec_cond, ms_psd, reduction="none").mean(dim=(1,2,3)).detach()
            """
            
            """
            logs["loss/ms_psd_mse"] = torch.zeros_like(logs["loss"])
            for i, (cond, ms_psd) in enumerate(zip(ddec_cond, ms_psds)):
                logs[f"loss/ms_psd_mse_{i}"] = torch.nn.functional.mse_loss(cond, ms_psd, reduction="none").mean(dim=(1,2,3))
                logs["loss/ms_psd_mse"] = logs["loss/ms_psd_mse"] + logs[f"loss/ms_psd_mse_{i}"] / len(ms_psds)

            ms_psd_mse_loss_weight = 0
            logs["loss_weight/ms_psd_mse"] = ms_psd_mse_loss_weight
            logs["loss"] = logs["loss"] + logs["loss/ms_psd_mse"] * ms_psd_mse_loss_weight
            """
            """
            for i in range(self.dae.num_psd_levels - 1):
                logs[f"io_stats_dae/in_balance_{i+1}"] = self.dae.in_balance[i].sigmoid().detach()

            for i in range(self.dae.num_psd_levels):
                logs[f"io_stats_dae/out_gain_{i}"] = self.dae.out_gain[i].detach()
            """

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