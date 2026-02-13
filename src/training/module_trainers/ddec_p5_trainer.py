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
from training.sigma_sampler import SigmaSamplerConfig, SigmaSampler
from modules.daes.dae_edm2_p5 import DAE
from modules.unets.unet_edm2_p5_ddec import UNet
from modules.formats.ms_mdct_dual_3 import MS_MDCT_DualFormat
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

    ddec: dict[str, Any]

    kl_loss_weight: float = 0.1
    kl_warmup_steps: int  = 300

    random_stereo_augmentation: bool = True
    random_phase_augmentation: bool  = True

    crop_edges: int = 4 # used to avoid artifacts due to mdct lapped blocks at beginning and end of sample

class DiffusionDecoder_Trainer(ModuleTrainer):
    
    @torch.no_grad()
    def __init__(self, config: DiffusionDecoder_Trainer_Config, trainer: DualDiffusionTrainer) -> None:

        self.config = config
        self.trainer = trainer
        self.logger = trainer.logger

        self.ddec: UNet = trainer.get_train_module("ddec")
        self.dae: DAE = trainer.get_train_module("dae")
        
        if self.dae is None:
            self.train_dae = False
            assert self.ddec is not None
        else:
            self.train_dae = True
            assert self.ddec is None

            self.ddec = trainer.pipeline.ddec.to(device=trainer.accelerator.device, dtype=torch.bfloat16).requires_grad_(False)
            assert self.ddec.config.last_global_step > 0
        
        self.format: MS_MDCT_DualFormat = trainer.pipeline.format.to(self.trainer.accelerator.device)

        if trainer.config.enable_model_compilation:
            self.ddec.compile(**trainer.config.compile_params)
            self.format.compile(**trainer.config.compile_params)

            if self.dae is not None:
                self.dae.compile(**trainer.config.compile_params)

        self.logger.info(f"Training modules: {trainer.config.train_modules}")
        if self.train_dae == True:
            self.logger.info(f"KL loss weight: {self.config.kl_loss_weight} KL warmup steps: {self.config.kl_warmup_steps}")
        self.logger.info(f"Crop edges: {self.config.crop_edges}")

        if self.config.random_stereo_augmentation == True:
            self.logger.info("Using random stereo augmentation")
        else: self.logger.info("Random stereo augmentation is disabled")

        self.logger.info("DDEC trainer:")
        self.ddec_trainer = UNetTrainer(UNetTrainerConfig(**config.ddec), trainer, self.ddec, "ddec")

        if self.train_dae == True:
            latents_sigma_sampler_config = SigmaSamplerConfig(
                sigma_max=100,
                sigma_min=0.01,
                sigma_data=1,
                distribution="linear",
                dist_scale=-1,
                dist_offset=0,
                use_stratified_sigma_sampling=True
            )
            self.latents_sigma_sampler = SigmaSampler(latents_sigma_sampler_config)
            if self.train_dae == True:
                self.logger.info("Latents SigmaSamplerConfig:")
                self.logger.info(dict_str(latents_sigma_sampler_config.__dict__))
        else:
            self.latents_sigma_sampler = None

    @torch.no_grad()
    def init_batch(self, validation: bool = False) -> Optional[dict[str, Union[torch.Tensor, float]]]:
        
        self.ddec_trainer.init_batch(validation)

        if self.train_dae == True:
            self.global_sigma = self.latents_sigma_sampler.sample(self.trainer.total_batch_size, device=self.trainer.accelerator.device)
            self.global_sigma = self.trainer.accelerator.gather(self.global_sigma.unsqueeze(0))[0]
        else:
            self.global_sigma = None

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

        logs = {"loss": torch.zeros(raw_samples.shape[0], device=self.trainer.accelerator.device)}

        input_mdct_phase, _ = self.format.raw_to_mdct_phase_psd(raw_samples, random_phase_augmentation=self.config.random_phase_augmentation)
        input_mdct_phase = input_mdct_phase[..., self.config.crop_edges:-self.config.crop_edges]
        input_mel_spec = self.format.raw_to_mel_spec(raw_samples)
        input_mel_spec = input_mel_spec[..., self.config.crop_edges:-self.config.crop_edges]

        logs.update({
            "io_stats/mdct_phase_var": input_mdct_phase.var(dim=(1,2,3)),
            "io_stats/mdct_phase_mean": input_mdct_phase.mean(dim=(1,2,3)),
            "io_stats/mel_spec_var": input_mel_spec.var(dim=(1,2,3)),
            "io_stats/mel_spec_mean": input_mel_spec.mean(dim=(1,2,3)),
        })

        if self.train_dae == True:
            # get the noise level for this sub-batch from the pre-calculated whole-batch sigma (required for stratified sampling)
            device_bsz = self.trainer.config.device_batch_size
            local_sigma = self.global_sigma[self.trainer.accelerator.process_index::self.trainer.accelerator.num_processes]
            latents_batch_sigma = local_sigma[self.trainer.accum_step * device_bsz:(self.trainer.accum_step+1) * device_bsz]
            
            latents, ddec_cond, pre_norm_latents = self.trainer.get_ddp_module(self.dae)(
                input_mel_spec, audio_embeddings, latents_sigma=latents_batch_sigma)
            
            latents: torch.Tensor = latents.float()
            pre_norm_latents: torch.Tensor = pre_norm_latents.float()

            pre_norm_latents_var = pre_norm_latents.pow(2).mean() + 1e-20
            var_kl = pre_norm_latents_var - 1 - pre_norm_latents_var.log()
            kl_loss = var_kl.mean() + 0.5 * pre_norm_latents.mean().square().mean()
            kl_loss = kl_loss.expand(latents.shape[0]) # needed for per-sample logging

            kl_loss_weight = self.config.kl_loss_weight
            if self.trainer.global_step < self.config.kl_warmup_steps:
                kl_loss_weight *= self.trainer.global_step / self.config.kl_warmup_steps
            
            logs["loss"] = logs["loss"] + kl_loss * kl_loss_weight

            logs.update({
                "io_stats/prenorm_latents_var": pre_norm_latents.var(dim=(1,2,3)).detach(),
                "io_stats/latents_var": latents.var(dim=(1,2,3)).detach(),
                "io_stats/latents_mean": latents.mean(dim=(1,2,3)).detach(),
                "io_stats/latents_sigma": latents_batch_sigma.detach(),
                "loss/kl_latents": kl_loss.detach(),
                "loss_weight/kl_latents": kl_loss_weight
            })

        else:
            latents = pre_norm_latents = latents_batch_sigma = kl_loss = kl_loss_weight = None
            ddec_cond = input_mel_spec.detach()
        
        logs.update({
            "io_stats/ddec_cond_var": ddec_cond.var(dim=(1,2,3)),
            "io_stats/ddec_cond_mean": ddec_cond.mean(dim=(1,2,3)),
        })

        logs.update(self.ddec_trainer.train_batch(input_mdct_phase, audio_embeddings, ddec_cond))
        logs["loss"] = logs["loss"] + logs["loss/ddec"]

        if self.trainer.config.enable_debug_mode == True:
            print("input_mdct_phase.shape:", input_mdct_phase.shape)
            print("input_mel_spec.shape:", input_mel_spec.shape)
            print("ddec_cond.shape:", ddec_cond.shape)

            if self.train_dae == True:
                print("latents.shape:", latents.shape)
                print("pre_norm_latents.shape:", pre_norm_latents.shape)
                print("latents_batch_sigma.shape:", latents_batch_sigma.shape)

        return logs
    
    @torch.no_grad()
    def finish_batch(self) -> Optional[dict[str, Union[torch.Tensor, float]]]:

        logs = {}
        logs.update(self.ddec_trainer.finish_batch())

        return logs