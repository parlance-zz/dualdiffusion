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
from training.loss.mss_1d import MSSLoss1D, MSSLoss1DConfig
from modules.phaseret.phaseret_q432 import PhaseRet
from modules.formats.ms_mdct_dual_10 import MS_MDCT_DualFormat
from utils.dual_diffusion_utils import dict_str


@torch.no_grad()
def random_stereo_augmentation(x: torch.Tensor) -> torch.Tensor:
    
    output = x.clone()
    flip_mask = (torch.rand(x.shape[0]) > 0.5).to(x.device)
    output[flip_mask] = output[flip_mask].flip(dims=(1,))
    
    return output

@dataclass
class PhaseRet_Trainer_Config(ModuleTrainerConfig):

    mss_1d: dict[str, Any]
    
    mss_1d_loss_weight: float          = 1
    mss_1d_cepstrum_loss_weight: float = 1

    random_stereo_augmentation: bool = False
    random_phase_augmentation: bool  = False

    add_ms_psd_noise: float = 0

class PhaseRet_Trainer(ModuleTrainer):
    
    @torch.no_grad()
    def __init__(self, config: PhaseRet_Trainer_Config, trainer: DualDiffusionTrainer) -> None:

        self.config = config
        self.trainer = trainer
        self.logger = trainer.logger

        self.logger.info(f"Training modules: {trainer.config.train_modules}")
        
        self.phaseret: PhaseRet = trainer.get_train_module("phaseret")
        self.format: MS_MDCT_DualFormat = trainer.pipeline.format.to(self.trainer.accelerator.device)

        if trainer.config.enable_model_compilation:
            self.format.compile(**trainer.config.compile_params)

            if self.phaseret is not None:
                self.phaseret.compile(**trainer.config.compile_params)        
        
        if self.config.random_phase_augmentation == True:
            self.logger.info("Using random phase augmentation")
        else: self.logger.info("Random phase augmentation is disabled")

        if self.config.random_stereo_augmentation == True:
            self.logger.info("Using random stereo augmentation")
        else: self.logger.info("Random stereo augmentation is disabled")

        self.logger.info(f"MS-PSD add noise: {self.config.add_ms_psd_noise}")
        self.logger.info(f"MSS-1D loss weight: {self.config.mss_1d_loss_weight} (cepstrum loss weight: {self.config.mss_1d_cepstrum_loss_weight})")
        self.mss_1d = MSSLoss1D(MSSLoss1DConfig(**self.config.mss_1d), device=trainer.accelerator.device)
        self.logger.info(f"MSS-1D config: {dict_str(self.mss_1d.config.__dict__)}")
    
    def phaseret_loss_fn(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        
        output_phase, output_psd = output.chunk(2, dim=1)
        target_phase, target_psd = target.chunk(2, dim=1)

        output_raw1 = self.format.mdct_phase_psd_to_raw(torch.cat((target_phase, output_psd), dim=1))
        output_raw2 = self.format.mdct_phase_psd_to_raw(torch.cat((output_phase, target_psd), dim=1))
        output_raw3 = self.format.mdct_phase_psd_to_raw(output)

        output_raw = torch.cat((output_raw1, output_raw2, output_raw3), dim=0)
        input_raw = self.format.mdct_phase_psd_to_raw(target).repeat(3, 1, 1).detach()
        
        return self.mss_1d.mss_loss(output_raw, input_raw)

    def train_batch(self, batch: dict) -> Optional[dict[str, Union[torch.Tensor, float]]]:

        logs = {"loss": torch.zeros(self.trainer.config.device_batch_size * 3, device=self.trainer.accelerator.device)}

        if self.config.random_stereo_augmentation == True:
            raw_samples = random_stereo_augmentation(batch["audio"])
        else:
            raw_samples = batch["audio"]

        with torch.no_grad():
            mdct_phase_psd = self.format.raw_to_mdct_phase_psd(raw_samples, random_phase_augmentation=self.config.random_phase_augmentation)
            ms_psd = self.format.raw_to_ms_psd(raw_samples)
            ms_psd_scaled = self.format.scale_ms_psd(ms_psd)

        logs.update({
            "io_stats/mdct_phase_psd_msq": mdct_phase_psd.pow(2).mean(dim=(1,2,3)),
            "io_stats/mdct_phase_psd_mean": mdct_phase_psd.mean(dim=(1,2,3)),
            "io_stats/ms_psd_msq": ms_psd.pow(2).mean(dim=(1,2,3)),
            "io_stats/ms_psd_mean": ms_psd.mean(dim=(1,2,3)),
            "io_stats/ms_psd_scaled_msq": ms_psd_scaled.pow(2).mean(dim=(1,2,3)),
            "io_stats/ms_psd_scaled_mean": ms_psd_scaled.mean(dim=(1,2,3))
        })

        x_ref = self.format.unscale_ms_psd(ms_psd_scaled)
        x_ref = (x_ref + torch.randn_like(x_ref) * self.config.add_ms_psd_noise).detach()
        logs["io_stats/add_ms_psd_noise"] = self.config.add_ms_psd_noise
        
        output_mdct_phase_psd, recon_loss_logvar = self.trainer.get_ddp_module(self.phaseret)(x_ref)
        
        mss_logs = self.phaseret_loss_fn(output_mdct_phase_psd, mdct_phase_psd)
        logs.update(mss_logs)

        recon_loss = logs["loss/mss_1d"] * self.config.mss_1d_loss_weight + logs["loss/mss_1d_cepstrum"] * self.config.mss_1d_cepstrum_loss_weight
        recon_loss_nll = recon_loss / recon_loss_logvar.exp() + recon_loss_logvar
        
        logs["loss"] = logs["loss"] + recon_loss_nll

        if self.trainer.config.enable_debug_mode == True:
            print("mdct_phase_psd.shape:", mdct_phase_psd.shape)
            print("ms_psd.shape:", ms_psd.shape)
            print("ms_psd_scaled.shape:", ms_psd_scaled.shape)

        return logs
