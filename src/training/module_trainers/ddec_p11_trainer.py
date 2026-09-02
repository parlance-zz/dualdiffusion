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
from training.loss.sigreg import sigreg_strong_loss
from training.loss.mss_2d import MSSLoss2D, MSSLoss2DConfig
from training.loss.mss_1d import MSSLoss1D, MSSLoss1DConfig
from modules.daes.dae_edm2_q4112 import DAE
from modules.unets.unet_edm2_q4112_ddec import UNet
from modules.formats.ms_mdct_dual_9 import MS_MDCT_DualFormat
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

    latents_sigreg_loss_weight: float = 0
    sigreg_loss_warmup_steps: int = 350

    use_mss_1d_loss: bool = False
    mss_1d_loss_weight: float          = 0.5
    mss_1d_cepstrum_loss_weight: float = 0.5

    use_mss_2d_loss: bool = True
    mss_2d_leak_pow: float = 1
    mss_2d_leak_steps: int = 500
    
    unet_loss_start_weight: float = 0
    unet_loss_start_steps: int    = 0
    unet_loss_weight: float     = 0.075
    unet_loss_warmup_steps: int = 1500

    random_stereo_augmentation: bool = False
    random_phase_augmentation: bool  = False
    mel_density_loss_weight_pow_ddecp: float = 0
    add_ddecp_x_ref_noise: float = 0

class DiffusionDecoder_Trainer(ModuleTrainer):
    
    @torch.no_grad()
    def __init__(self, config: DiffusionDecoder_Trainer_Config, trainer: DualDiffusionTrainer) -> None:

        self.config = config
        self.trainer = trainer
        self.logger = trainer.logger

        self.logger.info(f"Training modules: {trainer.config.train_modules}")
        
        self.dae: DAE = trainer.get_train_module("dae")
        self.ddecp: UNet = trainer.get_train_module("ddecp")

        self.train_dae = self.dae is not None
        self.train_ddecp = self.ddecp is not None

        if self.train_ddecp == True:
            assert self.train_dae == False
            #self.dae  = trainer.pipeline.dae.to( device=trainer.accelerator.device, dtype=torch.bfloat16).requires_grad_(False)
            #assert self.dae.config.last_global_step > 0 and self.unet.config.last_global_step > 0

        if self.train_dae == True:

            self.ddecp = trainer.pipeline.ddecp.to(device=trainer.accelerator.device, dtype=torch.bfloat16).requires_grad_(True).train()
            assert self.ddecp.config.last_global_step > 0
            self.train_ddecp = True

        self.format: MS_MDCT_DualFormat = trainer.pipeline.format.to(self.trainer.accelerator.device)

        if trainer.config.enable_model_compilation:
            self.format.compile(**trainer.config.compile_params)

            if self.dae is not None:
                self.dae.compile(**trainer.config.compile_params)
            if self.ddecp is not None:
                self.ddecp.compile(**trainer.config.compile_params)

        if self.train_dae == True:
            self.logger.info(f"SIGReg loss weight: {self.config.latents_sigreg_loss_weight} (warmup steps: {self.config.sigreg_loss_warmup_steps})")
            self.logger.info(f"SIGReg config: {dict_str(self.config.sigreg)}")

            if config.use_mss_2d_loss == True:
                self.mss_2d = MSSLoss2D(MSSLoss2DConfig(**config.mss_2d), device=trainer.accelerator.device)
                self.logger.info(f"MSS-2D config: {dict_str(self.mss_2d.config.__dict__)}")

            if self.dae.config.unet is not None:
                self.logger.info(f"DAE UNet-LDM trainer (start loss weight: {self.config.unet_loss_start_weight}) (delayed start steps:{self.config.unet_loss_start_steps})"
                                f" (loss weight: {self.config.unet_loss_weight}) (warmup steps: {self.config.unet_loss_warmup_steps}):")
                self.unet_trainer = UNetTrainer(UNetTrainerConfig(**config.unet), trainer, self.dae.unet, "unet")
            else:
                self.unet_trainer = None

        if self.train_ddecp == True:
            self.logger.info(f"DDEC-P mel-density loss weight pow: {self.config.mel_density_loss_weight_pow_ddecp}")
            self.logger.info(f"DDEC-P add x_ref noise: {self.config.add_ddecp_x_ref_noise}")
            self.logger.info(f"DDEC-P trainer:")
            self.ddecp_trainer = UNetTrainer(UNetTrainerConfig(**config.ddecp), trainer, self.ddecp, "ddecp")

            
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

    @torch.no_grad()
    def init_batch(self, validation: bool = False) -> Optional[dict[str, Union[torch.Tensor, float]]]:
        
        if self.train_ddecp == True:
            self.ddecp_trainer.init_batch(validation)
        if self.train_dae == True and self.unet_trainer is not None:
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

        mdct_phase_psd = self.format.raw_to_mdct_phase_psd(raw_samples,
            random_phase_augmentation=self.config.random_phase_augmentation, level=-1)
        mdct_phase_psd_flattened = self.format.flatten_mdct_phase_psd(mdct_phase_psd)
        ms_psd = self.format.raw_to_ms_psd(raw_samples, level=-1)

        for i, (_mdct_phase_psd, _ms_psd) in enumerate(zip(mdct_phase_psd, ms_psd)):
            logs[f"io_stats/mdct_phase_psd_{i}_msq"] = _mdct_phase_psd.pow(2).mean(dim=(1,2,3))
            logs[f"io_stats/mdct_phase_psd_{i}_mean"] = _mdct_phase_psd.mean(dim=(1,2,3))
            logs[f"io_stats/ms_psd_{i}_msq"] = _ms_psd.pow(2).mean(dim=(1,2,3))
            logs[f"io_stats/ms_psd_{i}_mean"] = _ms_psd.mean(dim=(1,2,3))

        if self.train_dae == True:
            
            dae_unet_batch_sigma = self.unet_trainer.get_batch_sigma() if self.unet_trainer is not None else None
            latents, ddec_cond, dae_recon_logvar, dae_unet_batch_loss, bucket_log_loss = self.trainer.get_ddp_module(self.dae)(ms_psd, audio_embeddings, batch_sigma=dae_unet_batch_sigma)

            if self.unet_trainer is not None:
                if self.unet_trainer.config.num_loss_buckets > 0:
                    self.unet_trainer.unet_loss_buckets.log_buckets(bucket_log_loss, dae_unet_batch_sigma)

                logs["loss/dae_unet"] = dae_unet_batch_loss.detach()
                logs["io_stats_dae/batch_sigma"] = dae_unet_batch_sigma

        elif self.dae is not None:

            with torch.no_grad():
                dae_unet_batch_sigma = self.unet_trainer.get_batch_sigma() if self.unet_trainer is not None else None
                latents, ddec_cond, dae_recon_logvar, dae_unet_batch_loss, bucket_log_loss = self.dae(ms_psd, audio_embeddings, batch_sigma=dae_unet_batch_sigma)

            if self.unet_trainer is not None:
                if self.unet_trainer.config.num_loss_buckets > 0:
                    self.unet_trainer.unet_loss_buckets.log_buckets(bucket_log_loss, dae_unet_batch_sigma)

                logs["loss/dae_unet"] = dae_unet_batch_loss.detach()
                logs["io_stats_dae/batch_sigma"] = dae_unet_batch_sigma
        else:
            latents = ddec_cond = None
        
        if latents is not None:
            latents: torch.Tensor = latents.float()
            ddec_cond: list[torch.Tensor] = [x.float() for x in ddec_cond]
            
            logs.update({
                "io_stats_dae/latents_msq": latents.pow(2).mean(dim=(1,2,3)).detach(),
                "io_stats_dae/latents_mean": latents.mean(dim=(1,2,3)).detach(),
                "io_stats_dae/latents_per_ch_mean": self.dae.latents_stats_tracker.mean.abs().mean(),
                "io_stats_dae/latents_per_ch_msq": self.dae.latents_stats_tracker.msq.mean(),
            })

            for i, _ddec_cond in enumerate(ddec_cond):
                logs[f"io_stats_dae/ddec_cond_{i}_msq"]  = _ddec_cond.pow(2).mean(dim=(1,2,3)).detach()
                logs[f"io_stats_dae/ddec_cond_{i}_mean"] = _ddec_cond.mean(dim=(1,2,3)).detach()

        if self.train_dae == True:
            
            if self.config.mss_2d_leak_steps > 0:
                leak_max = 1 - min(self.trainer.global_step / self.config.mss_2d_leak_steps, 1)
                if leak_max <= 0: leak_max = None
            else:
                leak_max = None
            logs["io_stats_dae/mss_2d_leak_max"] = leak_max if leak_max is not None else 0

            for i, (_ddec_cond, _ms_psd) in enumerate(zip(ddec_cond, ms_psd)):
                logs[f"loss/mss_2d_{i}"] = self.mss_2d.mss_loss(_ddec_cond, _ms_psd, leak_pow=self.config.mss_2d_leak_pow, leak_max=leak_max)
                logs[f"loss/dae_mse_{i}"] = torch.nn.functional.mse_loss(_ddec_cond, _ms_psd, reduction="none").mean(dim=(1,2,3)).detach()

                if i == 0:
                    logs["loss/mss_2d"]  = logs[f"loss/mss_2d_{i}"] / len(ddec_cond)
                    logs["loss/dae_mse"] = logs[f"loss/dae_mse_{i}"] / len(ddec_cond)
                else:
                    logs["loss/mss_2d"]  = logs["loss/mss_2d"] + logs[f"loss/mss_2d_{i}"] / len(ddec_cond)
                    logs["loss/dae_mse"] = logs["loss/dae_mse"] + logs[f"loss/dae_mse_{i}"] / len(ddec_cond)
            
            dae_recon_loss = logs["loss/mss_2d"]
            logs["loss/dae_recon_nll"] = dae_recon_loss / self.dae.get_recon_loss_logvar().exp() + self.dae.get_recon_loss_logvar()
            logs["loss"] = logs["loss"] + logs["loss/dae_recon_nll"]

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

            ddecp_x_ref = ms_psd
            #ddecp_x_ref = ddec_cond

            logs["io_stats_ddecp/add_x_ref_noise"] = self.config.add_ddecp_x_ref_noise
            if self.config.add_ddecp_x_ref_noise > 0:
                ddecp_x_ref = [x + torch.randn_like(x) * self.config.add_ddecp_x_ref_noise for x in ddecp_x_ref]
            mdct_phase_flattened, _ = mdct_phase_psd_flattened.chunk(2, dim=1)

            loss_fn = lambda x, y: self.format.get_mdct_phase_psd_loss(x, y, mel_density_pow=self.config.mel_density_loss_weight_pow_ddecp)
            ddecp_logs, ext_logs = self.ddecp_trainer.train_batch(
                mdct_phase_flattened, audio_embeddings, ref_samples=ddecp_x_ref, loss_fn=loss_fn)
            
            logs.update(ddecp_logs)
            logs["loss"] = logs["loss"] + logs["loss/ddecp"]

            """
            denoised = self.format.unflatten_mdct_phase_psd(denoised)

            denoised_raw: list[torch.Tensor] = []; input_raw: list[torch.Tensor] = []
            for i, (_denoised, _input) in enumerate(zip(denoised, mdct_phase_psd)):
                denoised_raw.append(self.format.mdct_phase_psd_to_raw(_denoised, level=i))
                input_raw.append(self.format.mdct_phase_psd_to_raw(_input, level=i))

            crop_length = min(x.shape[-1] for x in denoised_raw)
            denoised_raw = torch.cat([x[..., :crop_length] for x in denoised_raw], dim=0)
            input_raw = torch.cat([x[..., :crop_length] for x in input_raw], dim=0)

            #for i, (_denoised_raw, _input_raw) in enumerate(zip(denoised_raw, input_raw)):
            #    logs[f"loss/mss_1d_{i}"], logs[f"loss/mss_1d_cepstrum_{i}"] = self.mss_1d.mss_loss(_denoised_raw, _input_raw)

            logs.update(self.mss_1d.mss_loss(denoised_raw, input_raw.detach()))
            logs["loss"] = logs["loss"] + logs["loss/mss_1d"].mean() + logs["loss/mss_1d_cepstrum"].mean()
            """
        else:
            ddecp_x_ref = None
        
        if self.trainer.config.enable_debug_mode == True:
            print("mdct_phase_psd_flattened.shape:", mdct_phase_psd_flattened.shape)

            for i, (_mdct_phase_psd, _ms_psd) in enumerate(zip(mdct_phase_psd, ms_psd)):
                print(f"mdct_phase_psd[{i}].shape:", _mdct_phase_psd.shape)
                print(f"ms_psd[{i}].shape:", _ms_psd.shape)

            if latents is not None:
                print("latents.shape:", latents.shape)
                for i, _ddec_cond in enumerate(ddec_cond):
                    print(f"ddec_cond[{i}].shape:", _ddec_cond.shape)

        return logs
      
    @torch.no_grad()
    def finish_batch(self) -> Optional[dict[str, Union[torch.Tensor, float]]]:

        logs = {}
        if self.train_ddecp == True:
            logs.update(self.ddecp_trainer.finish_batch())
        if self.train_dae == True and self.unet_trainer is not None:
            logs.update(self.unet_trainer.finish_batch())

        return logs