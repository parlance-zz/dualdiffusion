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
from training.module_trainers.unet_trainer_p6 import UNetTrainerConfig, UNetTrainer
from training.module_trainers.unet_trainer_p6 import UNetTrainerConfig as UNetTrainerConfig_LDM, UNetTrainer as UNetTrainer_LDM
from training.loss.sigreg import sigreg_strong_loss
from training.loss.mss_2d import MSSLoss2D, MSSLoss2DConfig
from training.loss.mss_1d import MSSLoss1D, MSSLoss1DConfig
from modules.daes.dae_edm2_q43 import DAE
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
    unet: dict[str, Any]
    sigreg: dict[str, Any]
    mss_1d: dict[str, Any]
    mss_2d: dict[str, Any]

    latents_sigreg_loss_weight: float = 1e-3
    sigreg_loss_warmup_steps: int = 300
    
    use_mss_1d_loss: bool = True
    mss_1d_loss_weight: float          = 1
    mss_1d_cepstrum_loss_weight: float = 1

    use_mss_2d_loss: bool = False
    mss_2d_leak_pow: float = 2
    mss_2d_leak_steps: int = 350
    
    unet_loss_start_weight: float = 0
    unet_loss_start_steps: int    = 0
    unet_loss_weight: float     = 0.15
    unet_loss_warmup_steps: int = 350

    random_stereo_augmentation: bool = True
    random_phase_augmentation: bool  = True
    mel_density_loss_weight_pow_ddecp: float = 1
    add_ddecp_x_ref_noise: float = 0.02

class DiffusionDecoder_Trainer(ModuleTrainer):
    
    @torch.no_grad()
    def __init__(self, config: DiffusionDecoder_Trainer_Config, trainer: DualDiffusionTrainer) -> None:

        self.config = config
        self.trainer = trainer
        self.logger = trainer.logger

        self.logger.info(f"Training modules: {trainer.config.train_modules}")
        
        self.dae: DAE = trainer.get_train_module("dae")
        self.ddecp: UNet = trainer.get_train_module("ddecp")
        self.unet: UNet_LDM = trainer.get_train_module("unet")

        self.train_dae = self.dae is not None
        self.train_ddecp = self.ddecp is not None
        self.train_unet = self.unet is not None

        if self.train_ddecp == True:
            assert self.train_dae == self.train_unet == False                            
            self.dae  = trainer.pipeline.dae.to( device=trainer.accelerator.device, dtype=torch.bfloat16).requires_grad_(False)
            self.unet = trainer.pipeline.unet.to(device=trainer.accelerator.device, dtype=torch.bfloat16).requires_grad_(False).train()
            assert self.dae.config.last_global_step > 0 and self.unet.config.last_global_step > 0

        if self.train_dae == True or self.train_unet == True:
            assert self.train_dae == True and self.train_unet == True

        self.format: MS_MDCT_DualFormat = trainer.pipeline.format.to(self.trainer.accelerator.device)

        if trainer.config.enable_model_compilation:
            self.format.compile(**trainer.config.compile_params)

            if self.dae is not None:
                self.dae.compile(**trainer.config.compile_params)
            if self.ddecp is not None:
                self.ddecp.compile(**trainer.config.compile_params)
            if self.unet is not None:
                self.unet.compile(**trainer.config.compile_params)

        if self.train_dae == True:
            self.logger.info(f"SIGReg loss weight: {self.config.latents_sigreg_loss_weight} (warmup steps: {self.config.sigreg_loss_warmup_steps})")
            self.logger.info(f"SIGReg config: {dict_str(self.config.sigreg)}")

            if config.use_mss_2d_loss == True:
                self.mss_2d = MSSLoss2D(MSSLoss2DConfig(**config.mss_2d), device=trainer.accelerator.device)
                self.logger.info(f"MSS-2D config: {dict_str(self.mss_2d.config.__dict__)}")

        if self.train_ddecp == True:
            self.logger.info(f"DDEC-P mel-density loss weight pow: {self.config.mel_density_loss_weight_pow_ddecp}")
            self.logger.info(f"DDEC-P add x_ref noise: {self.config.add_ddecp_x_ref_noise}")
            self.logger.info(f"DDEC-P trainer:")
            self.ddecp_trainer = UNetTrainer(UNetTrainerConfig(**config.ddecp), trainer, self.ddecp, "ddecp")

            hz = torch.linspace(0, 1, self.format.config.num_frequencies, device=self.trainer.accelerator.device) * self.format.config.sample_rate/2
            loss_weight = get_mel_density(hz).pow(self.config.mel_density_loss_weight_pow_ddecp)
            self.ddecp_loss_weight = (loss_weight / loss_weight.mean()).view(1, 1,-1, 1)
            
            if self.config.random_phase_augmentation == True:
                self.logger.info("Using random phase augmentation")
            else: self.logger.info("Random phase augmentation is disabled")

            if config.use_mss_1d_loss == True:
                self.logger.info(f"MSS-1D loss weight: {self.config.mss_1d_loss_weight} (cepstrum loss weight: {self.config.mss_1d_cepstrum_loss_weight})")
                self.mss_1d = MSSLoss1D(MSSLoss1DConfig(**self.config.mss_1d), device=trainer.accelerator.device)
                self.logger.info(f"MSS-1D config: {dict_str(self.mss_1d.config.__dict__)}")

        if self.config.random_stereo_augmentation == True:
            self.logger.info("Using random stereo augmentation")
        else: self.logger.info("Random stereo augmentation is disabled")
        
        if self.train_unet == True or self.unet is not None:
            self.logger.info(f"UNet-LDM trainer (start loss weight: {self.config.unet_loss_start_weight}) (warmup steps:{self.config.unet_loss_start_steps})"
                             f" (loss weight: {self.config.unet_loss_weight}) (warmup steps: {self.config.unet_loss_warmup_steps}):")
            self.unet_trainer = UNetTrainer_LDM(UNetTrainerConfig_LDM(**config.unet, skip_bucket_loss_logging=True), trainer, self.unet, "unet")
        else:
            self.unet_trainer = None

    @torch.no_grad()
    def init_batch(self, validation: bool = False) -> Optional[dict[str, Union[torch.Tensor, float]]]:
        
        if self.train_ddecp == True:
            self.ddecp_trainer.init_batch(validation)
        if self.train_unet == True or self.unet is not None:
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
            
            latents, ddec_cond, unet_logs, ext_logs = self.trainer.get_ddp_module(self.dae)(
                ms_psd_scaled, audio_embeddings, unet_trainer=self.unet_trainer)

            logs.update(unet_logs)
            self.unet_trainer.unet_loss_buckets.log_buckets(ext_logs["bucket_log_loss"], ext_logs["batch_sigma"])

        elif self.dae is not None:

            with torch.no_grad():
                latents, ddec_cond, unet_logs, ext_logs = self.trainer.get_ddp_module(self.dae)(
                    ms_psd_scaled, audio_embeddings, unet_trainer=self.unet_trainer)

                logs.update(unet_logs)
                self.unet_trainer.unet_loss_buckets.log_buckets(ext_logs["bucket_log_loss"], ext_logs["batch_sigma"])
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
                "io_stats_dae/ddec_cond_msq": ddec_cond.pow(2).mean(dim=(1,2,3)).detach(),
                "io_stats_dae/ddec_cond_mean": ddec_cond.mean(dim=(1,2,3)).detach()
            })

        if self.train_dae == True:
            
            if self.config.use_mss_2d_loss == True:
                if self.config.mss_2d_leak_steps > 0:
                    leak_max = 1 - min(self.trainer.global_step / self.config.mss_2d_leak_steps, 1)
                    if leak_max <= 0: leak_max = None
                else:
                    leak_max = None
                
                logs["io_stats_dae/mss_2d_leak_max"] = leak_max if leak_max is not None else 0
                
                logs["loss/mss_2d"] = self.mss_2d.mss_loss(ddec_cond, ms_psd_scaled, leak_pow=self.config.mss_2d_leak_pow, leak_max=leak_max)
                dae_recon_loss = logs["loss/mss_2d"]
                logs["loss/dae_recon_nll"] = dae_recon_loss / self.dae.get_recon_loss_logvar().exp() + self.dae.get_recon_loss_logvar()
                logs["loss"] = logs["loss"] + logs["loss/dae_recon_nll"]
            
            logs["loss/dae_mse"] = torch.nn.functional.mse_loss(ddec_cond, ms_psd_scaled, reduction="none").mean(dim=(1,2,3)).detach()
            
            if self.trainer.global_step < self.config.unet_loss_start_steps:
                unet_loss_weight = self.config.unet_loss_start_weight
            else:
                t = min((self.trainer.global_step - self.config.unet_loss_start_steps) / (self.config.unet_loss_warmup_steps + 1), 1)
                unet_loss_weight = self.config.unet_loss_start_weight * (1 - t) + self.config.unet_loss_weight * t
            logs["loss"] = logs["loss"] + logs["loss/unet"] * unet_loss_weight
            logs["loss_weight/unet"] = unet_loss_weight

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

            self.dae.latents_stats_tracker(latents)

        if self.train_ddecp == True:

            def ddecp_loss_fn(denoised: torch.Tensor, samples: torch.Tensor) -> torch.Tensor:
                
                denoised_phase, denoised_psd = denoised.chunk(2, dim=1)
                samples_phase, samples_psd = samples.chunk(2, dim=1)
                denoised_raw1 = self.format.mdct_phase_psd_to_raw(torch.cat((samples_phase, denoised_psd), dim=1))
                denoised_raw2 = self.format.mdct_phase_psd_to_raw(torch.cat((denoised_phase, samples_psd), dim=1))
                denoised_raw3 = self.format.mdct_phase_psd_to_raw(denoised)
                denoised_raw = torch.cat((denoised_raw1, denoised_raw2, denoised_raw3), dim=0)
                
                input_raw = self.format.mdct_phase_psd_to_raw(samples).detach().repeat(3, 1, 1)
                
                mss_logs = self.mss_1d.mss_loss(denoised_raw, input_raw)

                mss_1d_loss1, mss_1d_loss2, mss_1d_loss3 = mss_logs["loss/mss_1d"].chunk(3, dim=0)
                mss_1d_cepstrum_loss1, mss_1d_cepstrum_loss2, mss_1d_cepstrum_loss3 = mss_logs["loss/mss_1d_cepstrum"].chunk(3, dim=0)
                loss = (mss_1d_loss1 + mss_1d_loss2 + mss_1d_loss3) / 2 + (mss_1d_cepstrum_loss1 + mss_1d_cepstrum_loss2 + mss_1d_cepstrum_loss3) / 2

                return loss
            
            if ddec_cond is not None:
                ddecp_x_ref = self.format.unscale_ms_psd(ddec_cond)
            else:
                ddecp_x_ref = self.format.unscale_ms_psd(ms_psd_scaled)
            ddecp_x_ref = (ddecp_x_ref + torch.randn_like(ddecp_x_ref) * self.config.add_ddecp_x_ref_noise).detach()
            logs["io_stats_ddecp/ddecp_x_ref_noise"] = self.config.add_ddecp_x_ref_noise
            
            ddecp_logs, ext_logs = self.ddecp_trainer.train_batch(
                mdct_phase_psd, audio_embeddings, ref_samples=ddecp_x_ref, loss_weight=self.ddecp_loss_weight)
            
            logs.update(ddecp_logs)
            logs["loss"] = logs["loss"] + logs["loss/ddecp"]

            if self.config.use_mss_1d_loss == True:
                logs["loss/mss_1d"] = ddecp_loss_fn(ext_logs["denoised"], mdct_phase_psd)
                logs["loss"] = logs["loss"] + logs["loss/mss_1d"] * self.config.mss_1d_loss_weight

        if self.trainer.config.enable_debug_mode == True:
            print("mdct_phase_psd.shape:", mdct_phase_psd.shape)
            print("ms_psd.shape:", ms_psd.shape)
            print("ms_psd_scaled.shape:", ms_psd_scaled.shape)

            if latents is not None:
                print("latents.shape:", latents.shape)
                print(f"ddec_cond.shape:", ddec_cond.shape)

        return logs
      
    @torch.no_grad()
    def finish_batch(self) -> Optional[dict[str, Union[torch.Tensor, float]]]:

        logs = {}
        if self.train_ddecp == True:
            logs.update(self.ddecp_trainer.finish_batch())
        if self.train_unet == True or self.unet is not None:
            logs.update(self.unet_trainer.finish_batch())

        return logs