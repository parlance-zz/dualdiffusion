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
#from training.loss.sigreg import sigreg_strong_loss
from training.loss.mss_2d import MSSLoss2D, MSSLoss2DConfig
from modules.daes.dae_edm2_q4 import DAE
from modules.unets.unet_edm2_q4_ddec import UNet
from modules.unets.unet_edm2_p6 import UNet as UNet_LDM
from modules.formats.ms_mdct_dual_10 import MS_MDCT_DualFormat
from modules.formats.frequency_scale import get_mel_density
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
    #sigreg: dict[str, Any]
    mss_2d: dict[str, Any]

    #latents_sigreg_loss_weight: float = 0
    #sigreg_loss_warmup_steps: int = 0
    
    #mss_2d_leak_pow: float = 4
    #mss_2d_leak_steps: int = 200
    #dae_mae_loss_weight: float = 0.5
    #dae_mse_loss_weight: float = 8
    #mel_density_loss_weight_pow_dae: float = 0

    #add_ddecm_x_ref_noise: float = 0
    add_ddecp_x_ref_noise: float = 0
    
    add_latents_noise: Optional[float] = 0.08
    unet_loss_start_weight: float = 0
    unet_loss_start_steps: int    = 0
    unet_loss_weight: float     = 0.075
    unet_loss_warmup_steps: int = 1500

    random_stereo_augmentation: bool = True
    random_phase_augmentation: bool  = True
    mel_density_loss_weight_pow_ddecp: float = 1
    mel_density_loss_weight_pow_ddecm: float = 0.5

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

        """
        if self.train_dae == False:
            if self.train_ddecm == True and self.trainer.pipeline.dae.config.last_global_step > 0:
                self.dae = trainer.pipeline.dae.to(device=trainer.accelerator.device, dtype=torch.bfloat16).requires_grad_(False)
            #else:
            #    self.dae = None
        #else:
        #    if self.train_ddecp == False:
        #        self.ddecp = trainer.pipeline.ddecp.to(device=trainer.accelerator.device, dtype=torch.bfloat16).requires_grad_(False)
        #        assert self.ddecp.config.last_global_step > 0
        #        self.train_ddecp = True
        #"""

        if self.train_dae == False and self.train_ddecp == True:
            assert self.trainer.pipeline.dae.config.last_global_step > 0
            self.dae = trainer.pipeline.dae.to(device=trainer.accelerator.device, dtype=torch.bfloat16).requires_grad_(False)
        
        #if self.train_dae == True or self.train_ddecm == True:
        #    assert self.train_dae == True and self.train_ddecm == True

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
            #self.logger.info(f"SIGReg loss weight: {self.config.latents_sigreg_loss_weight} (warmup steps: {self.config.sigreg_loss_warmup_steps})")
            #self.logger.info(f"SIGReg config: {dict_str(self.config.sigreg)}")

            """
            if self.train_ddecp == False:
                self.mss_2d = MSSLoss2D(MSSLoss2DConfig(**self.config.mss_2d), device=trainer.accelerator.device)
                self.logger.info(f"MSS-2D config: {dict_str(self.mss_2d.config.__dict__)}")
                #self.logger.info(f"MSS-2D leak pow: {self.config.mss_2d_leak_pow} (leak steps: {self.config.mss_2d_leak_steps})")
            else:
                self.mss_2d = None
            """
            self.mss_2d = MSSLoss2D(MSSLoss2DConfig(**config.mss_2d), device=trainer.accelerator.device)
            self.logger.info(f"MSS-2D config: {dict_str(self.mss_2d.config.__dict__)}")
            #hz = torch.linspace(0, 1, self.format.config.ms_psd_num_filters, device=self.trainer.accelerator.device) * self.format.config.sample_rate/2
            #loss_weight = get_mel_density(hz).pow(self.config.mel_density_loss_weight_pow_dae)
            #self.dae_loss_weight = (loss_weight / loss_weight.mean()).view(1, 1,-1, 1)

        if self.train_ddecp == True:
            self.logger.info(f"Add DDEC-P x_ref noise: {self.config.add_ddecp_x_ref_noise}")
            self.logger.info(f"DDEC-P mel-density loss weight pow: {self.config.mel_density_loss_weight_pow_ddecp}")
            self.logger.info(f"DDEC-P trainer:")
            self.ddecp_trainer = UNetTrainer(UNetTrainerConfig(**config.ddecp), trainer, self.ddecp, "ddecp")

            hz = torch.linspace(0, 1, self.format.config.num_frequencies, device=self.trainer.accelerator.device) * self.format.config.sample_rate/2
            loss_weight = get_mel_density(hz).pow(self.config.mel_density_loss_weight_pow_ddecp)
            self.ddecp_loss_weight = (loss_weight / loss_weight.mean()).view(1, 1,-1, 1)
            
            if self.config.random_phase_augmentation == True:
                self.logger.info("Using random phase augmentation")
            else: self.logger.info("Random phase augmentation is disabled")

        if self.config.random_stereo_augmentation == True:
            self.logger.info("Using random stereo augmentation")
        else: self.logger.info("Random stereo augmentation is disabled")

        if self.train_ddecm == True:
            #self.logger.info(f"Add DDEC-M x_ref noise: {self.config.add_ddecm_x_ref_noise}")
            self.logger.info(f"DDEC-M mel-density loss weight pow: {self.config.mel_density_loss_weight_pow_ddecm}")
            self.logger.info(f"DDEC-M trainer:")
            self.ddecm_trainer = UNetTrainer(UNetTrainerConfig(**config.ddecm), trainer, self.ddecm, "ddecm")

            hz = torch.linspace(0, 1, self.format.config.ms_psd_num_filters, device=self.trainer.accelerator.device) * self.format.config.sample_rate/2
            loss_weight = get_mel_density(hz).pow(self.config.mel_density_loss_weight_pow_ddecm)
            self.ddecm_loss_weight = (loss_weight / loss_weight.mean()).view(1, 1,-1, 1)
        
        if self.train_unet == True:
            self.logger.info(f"UNet-LDM trainer (start loss weight: {self.config.unet_loss_start_weight}) (warmup steps:{self.config.unet_loss_start_steps})"
                             f" (loss weight: {self.config.unet_loss_weight}) (warmup steps: {self.config.unet_loss_warmup_steps}):")
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
        ms_psd = self.format.raw_to_ms_psd(raw_samples).detach()
        ms_psd_scaled = self.format.scale_ms_psd(ms_psd).detach()

        logs.update({
            "io_stats/mdct_phase_psd_msq": mdct_phase_psd.pow(2).mean(dim=(1,2,3)),
            "io_stats/mdct_phase_psd_mean": mdct_phase_psd.mean(dim=(1,2,3)),
            "io_stats/ms_psd_msq": ms_psd.pow(2).mean(dim=(1,2,3)),
            "io_stats/ms_psd_mean": ms_psd.mean(dim=(1,2,3)),
            "io_stats/ms_psd_scaled_msq": ms_psd_scaled.pow(2).mean(dim=(1,2,3)),
            "io_stats/ms_psd_scaled_mean": ms_psd_scaled.mean(dim=(1,2,3))
        })

        if self.train_dae == True:
            latents, ddec_cond = self.trainer.get_ddp_module(self.dae)(
                ms_psd_scaled, audio_embeddings, latents_sigma=self.config.add_latents_noise)
            
            self.dae.latents_stats_tracker(latents)

        elif self.dae is not None:
            with torch.no_grad():
                latents, ddec_cond = self.dae(ms_psd_scaled, audio_embeddings, latents_sigma=self.config.add_latents_noise)
        else:
            latents = ddec_cond = None
        
        if latents is not None:
            latents: torch.Tensor = latents.float()
            ddec_cond: torch.Tensor = ddec_cond.float()
            
            logs.update({
                "io_stats_dae/latents_msq": latents.pow(2).mean(dim=(1,2,3)).detach(),
                "io_stats_dae/latents_mean": latents.mean(dim=(1,2,3)).detach(),
                "io_stats_dae/latents_per_ch_mean": self.dae.latents_stats_tracker.mean.abs().mean(),
                "io_stats_dae/latents_per_ch_msq": self.dae.latents_stats_tracker.msq.mean(),
                "io_stats_dae/latents_sigma": self.config.add_latents_noise if self.config.add_latents_noise is not None else 0,
                "io_stats_dae/ddec_cond_msq": ddec_cond.pow(2).mean(dim=(1,2,3)).detach(),
                "io_stats_dae/ddec_cond_mean": ddec_cond.mean(dim=(1,2,3)).detach()
            })
            
            if self.dae.config.latent_channels <= 8:
                for i in range(self.dae.config.latent_channels):
                    logs[f"ch_stats/mean_{i}"] = self.dae.latents_stats_tracker.mean[i].detach()
                    logs[f"ch_stats/msq_{i}"]  = self.dae.latents_stats_tracker.msq[i].detach()

            if self.config.add_latents_noise:
                logs["io_stats_dae/latents_bpd"] = (1 + self.dae.latents_stats_tracker.msq**2 / self.config.add_latents_noise**2).log2().mean() / 2
                logs["io_stats_dae/latents_ms_psd_bpp"] = logs["io_stats_dae/latents_bpd"] / self.dae.downsample_ratio**2 * self.dae.config.latent_channels

        if self.train_ddecp == True:

            #ddecp_x_ref = self.format.unscale_ms_psd(ms_psd_scaled + torch.randn_like(ms_psd_scaled) * self.config.add_ddecp_x_ref_noise).detach()
            #logs["io_stats_ddecp/add_ddecp_x_ref_noise"] = self.config.add_ddecp_x_ref_noise
            ddecp_x_ref = self.format.unscale_ms_psd(ddec_cond).detach()

            logs.update(self.ddecp_trainer.train_batch(
                mdct_phase_psd, audio_embeddings, ref_samples=ddecp_x_ref, loss_weight=self.ddecp_loss_weight))
            logs["loss"] = logs["loss"] + logs["loss/ddecp"]

            """
            logs["loss_weight/mss1d"] = self.config.mss_loss_weight
            logs["loss_weight/mss1d_cepstrum"] = self.config.cepstrum_loss_weight
            if self.config.mss_loss_weight > 0 or self.config.cepstrum_loss_weight > 0:
                logs["loss"] = logs["loss"] + logs["loss/mss1d"] * self.config.mss_loss_weight
                logs["loss"] = logs["loss"] + logs["loss/mss1d_cepstrum"] * self.config.cepstrum_loss_weight
            """
        else:
            ddecp_x_ref = None

        if self.train_ddecm == True:

            logs.update(self.ddecm_trainer.train_batch(
                ms_psd_scaled, audio_embeddings, ref_samples=ddec_cond, loss_weight=self.ddecm_loss_weight))
            
            logs["loss"] = logs["loss"] + logs["loss/ddecm"]
        
        if self.train_dae == True:
            
            pass
            """
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

            """
            if self.config.mss_2d_leak_steps > 0:
                leak_max = 1 - min(self.trainer.global_step / self.config.mss_2d_leak_steps, 1)
                if leak_max <= 0: leak_max = None
            else:
                leak_max = None
            logs["io_stats_dae/mss_2d_leak_max"] = leak_max if leak_max is not None else 0
            logs["loss/mss_2d"] = self.mss_2d.mss_loss(ddec_cond, ms_psd_scaled, self.config.mss_2d_leak_pow, leak_max=leak_max)

            logs["loss/dae_mae"] = (torch.nn.functional.l1_loss(ddec_cond, ms_psd_scaled, reduction="none") * self.dae_loss_weight).mean(dim=(1,2,3)).detach()
            logs["loss/dae_mse"] = (torch.nn.functional.mse_loss(ddec_cond, ms_psd_scaled, reduction="none") * self.dae_loss_weight).mean(dim=(1,2,3))
            
            dae_recon_loss = logs["loss/dae_mse"] * self.config.dae_mse_loss_weight + logs["loss/mss_2d"]
            logs["loss/dae_recon_nll"] = dae_recon_loss / self.dae.get_recon_loss_logvar().exp() + self.dae.get_recon_loss_logvar()
            logs["loss"] = logs["loss"] + logs["loss/dae_recon_nll"]
            """

            """
            if self.config.mss_2d_leak_steps > 0:
                leak_max = 1 - min(self.trainer.global_step / self.config.mss_2d_leak_steps, 1)
                if leak_max <= 0: leak_max = None
            else:
                leak_max = None
            logs["io_stats_dae/mss_2d_leak_max"] = leak_max if leak_max is not None else 0
            """

            logs["loss/mss_2d"] = self.mss_2d.mss_loss(ddec_cond, ms_psd_scaled)#, leak_pow=self.config.mss_2d_leak_pow, leak_max=leak_max)

            logs["loss/dae_mae"] = torch.nn.functional.l1_loss( ddec_cond, ms_psd_scaled, reduction="none").mean(dim=(1,2,3)).detach()
            logs["loss/dae_mse"] = torch.nn.functional.mse_loss(ddec_cond, ms_psd_scaled, reduction="none").mean(dim=(1,2,3)).detach()
            
            dae_recon_loss = logs["loss/mss_2d"]
            logs["loss/dae_recon_nll"] = dae_recon_loss / self.dae.get_recon_loss_logvar().exp() + self.dae.get_recon_loss_logvar()
            logs["loss"] = logs["loss"] + logs["loss/dae_recon_nll"]

        if self.train_unet == True:
            
            if self.trainer.global_step < self.config.unet_loss_start_steps:
                unet_loss_weight = self.config.unet_loss_start_weight
            else:
                t = min((self.trainer.global_step - self.config.unet_loss_start_steps) / self.config.unet_loss_warmup_steps, 1)
                unet_loss_weight = self.config.unet_loss_start_weight * (1 - t) + self.config.unet_loss_weight * t

            logs.update(self.unet_trainer.train_batch(latents, audio_embeddings))
            logs["loss"] = logs["loss"] + logs["loss/unet"] * unet_loss_weight
            logs["loss_weight/unet"] = unet_loss_weight

        if self.trainer.config.enable_debug_mode == True:
            print("mdct_phase_psd.shape:", mdct_phase_psd.shape)
            print("ms_psd.shape:", ms_psd.shape)
            print("ms_psd_scaled.shape:", ms_psd_scaled.shape)

            if latents is not None:
                print("latents.shape:", latents.shape)
                print(f"ddec_cond.shape:", ddec_cond.shape)
            
            if ddecp_x_ref is not None:
                print(f"ddecp_x_ref.shape:", ddecp_x_ref.shape)

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