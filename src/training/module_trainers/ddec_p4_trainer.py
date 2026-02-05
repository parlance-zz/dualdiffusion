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
from modules.daes.dae_edm2_p4 import DAE
from modules.unets.unet_edm2_p4_ddec import UNet
from modules.formats.ms_mdct_dual_2 import MS_MDCT_DualFormat
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

    ddecm: dict[str, Any]
    ddecp: dict[str, Any]

    kl_loss_weight: float = 1
    kl_warmup_steps: int  = 2000

    phase_invariance_loss_weight: float = 0
    phase_invariance_warmup_steps: int = 0

    random_stereo_augmentation: bool = True
    random_phase_augmentation: bool  = True

    crop_edges: int = 4 # used to avoid artifacts due to mdct lapped blocks at beginning and end of sample

class DiffusionDecoder_Trainer(ModuleTrainer):
    
    @torch.no_grad()
    def __init__(self, config: DiffusionDecoder_Trainer_Config, trainer: DualDiffusionTrainer) -> None:

        self.config = config
        self.trainer = trainer
        self.logger = trainer.logger

        self.ddecp: UNet = trainer.get_train_module("ddecp")
        self.ddecm: UNet = trainer.get_train_module("ddecm")
        self.dae: DAE = trainer.get_train_module("dae")

        assert self.ddecp is not None
        assert self.ddecm is not None
        assert self.dae is not None
        
        self.format: MS_MDCT_DualFormat = trainer.pipeline.format.to(self.trainer.accelerator.device)

        if trainer.config.enable_model_compilation:
            self.ddecp.compile(**trainer.config.compile_params)
            self.ddecm.compile(**trainer.config.compile_params)
            self.dae.compile(**trainer.config.compile_params)
            self.format.compile(**trainer.config.compile_params)

        self.logger.info(f"Training modules: {trainer.config.train_modules}")
        self.logger.info(f"KL loss weight: {self.config.kl_loss_weight} KL warmup steps: {self.config.kl_warmup_steps}")
        self.logger.info(f"Latents phase-invariance loss weight: {self.config.phase_invariance_loss_weight} Warmup steps: {self.config.phase_invariance_warmup_steps}")
        self.logger.info(f"Crop edges: {self.config.crop_edges}")

        if self.config.random_stereo_augmentation == True:
            self.logger.info("Using random stereo augmentation")
        else: self.logger.info("Random stereo augmentation is disabled")

        self.logger.info("DDEC-P trainer:")
        self.ddecp_trainer = UNetTrainer(UNetTrainerConfig(**config.ddecp), trainer, self.ddecp, "ddecp")
        self.logger.info("DDEC-M trainer:")
        self.ddecm_trainer = UNetTrainer(UNetTrainerConfig(**config.ddecm), trainer, self.ddecm, "ddecm")

        sigma_sampler_config = SigmaSamplerConfig(
            sigma_max=100,
            sigma_min=0.004,
            sigma_data=1,
            distribution="linear",
            dist_scale=-1,
            dist_offset=0,
            use_stratified_sigma_sampling=True
        )
        self.sigma_sampler = SigmaSampler(sigma_sampler_config)
        self.logger.info("Latents noise SigmaSampler config:")
        self.logger.info(dict_str(sigma_sampler_config.__dict__))

        #self.trainer.optimizer.optimizer.zero_momentum()

    """
    def shift_equivariance_loss(self, mdct_phase: torch.Tensor, mdct_psd: torch.Tensor,
            dae_embeddings: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:

        if self.config.phase_invariance_loss_bsz == 0: return None

        latents = latents[:self.config.phase_invariance_loss_bsz]
        mdct_phase = mdct_phase[:self.config.phase_invariance_loss_bsz]
        mdct_psd = mdct_psd[:self.config.phase_invariance_loss_bsz]
        dae_embeddings = dae_embeddings[:self.config.phase_invariance_loss_bsz] if dae_embeddings is not None else None

        crop_left = np.random.randint(1, self.config.crop_edges * 2)
        crop_right = self.config.crop_edges * 2 - crop_left
        
        mdct_phase = mdct_phase[..., crop_left:-crop_right]
        mdct_psd = mdct_psd[..., crop_left:-crop_right]

        dae_input = torch.cat((mdct_phase, mdct_psd), dim=1).detach()
        with torch.autocast(device_type="cuda", dtype=self.trainer.mixed_precision_dtype, enabled=self.trainer.mixed_precision_enabled):
            latents2 = self.dae.encode(dae_input, dae_embeddings)

        latents_up: torch.Tensor = torch.repeat_interleave(latents, self.dae.downsample_ratio, dim=-1)
        latents_up_cropped = latents_up[..., crop_left:-crop_right]
        latents_down: torch.Tensor = torch.nn.functional.avg_pool2d(latents_up_cropped, kernel_size=(1,self.dae.downsample_ratio))

        return (latents_down - latents2.float())[..., 2:-2].pow(2).mean().expand(latents.shape[0])
    """

    @torch.no_grad()
    def init_batch(self, validation: bool = False) -> Optional[dict[str, Union[torch.Tensor, float]]]:
        
        self.ddecp_trainer.init_batch(validation)
        self.ddecm_trainer.init_batch(validation)

        self.global_sigma = self.sigma_sampler.sample(self.trainer.total_batch_size, device=self.trainer.accelerator.device)
        self.global_sigma = self.trainer.accelerator.gather(self.global_sigma.unsqueeze(0))[0]

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

        dae_input = torch.cat((mdct_phase, input_mel_spec), dim=1).detach()

        # get the noise level for this sub-batch from the pre-calculated whole-batch sigma (required for stratified sampling)
        device_bsz = self.trainer.config.device_batch_size
        local_sigma = self.global_sigma[self.trainer.accelerator.process_index::self.trainer.accelerator.num_processes]
        batch_sigma = local_sigma[self.trainer.accum_step * device_bsz:(self.trainer.accum_step+1) * device_bsz]
        latents_sigma = batch_sigma

        if self.config.phase_invariance_loss_weight > 0:
            mdct_phase2, _ = self.format.raw_to_mdct_phase_psd(raw_samples, random_phase_augmentation=True)
            mdct_phase2 = mdct_phase2[..., self.config.crop_edges:-self.config.crop_edges]
            dae_input2 = torch.cat((mdct_phase2, input_mel_spec), dim=1).detach()
        else:
            dae_input2 = None

        latents, ddec_cond, pre_norm_latents, phase_invariance_loss = self.trainer.get_ddp_module(self.dae)(
            dae_input, audio_embeddings, latents_sigma=latents_sigma, samples2=dae_input2)
        latents: torch.Tensor = latents.float()
        pre_norm_latents: torch.Tensor = pre_norm_latents.float()

        pre_norm_latents_var = pre_norm_latents.pow(2).mean() + 1e-20
        var_kl = pre_norm_latents_var - 1 - pre_norm_latents_var.log()
        kl_loss = var_kl.mean() + 0.5 * pre_norm_latents.mean().square().mean()
        kl_loss = kl_loss.expand(latents.shape[0]) # needed for per-sample logging

        phase_invariance_loss_weight = self.config.phase_invariance_loss_weight
        if self.trainer.global_step < self.config.phase_invariance_warmup_steps:
            warmup_factor = self.trainer.global_step / self.config.phase_invariance_warmup_steps
            phase_invariance_loss_weight *= warmup_factor

        kl_loss_weight = self.config.kl_loss_weight
        if self.trainer.global_step < self.config.kl_warmup_steps:
            kl_loss_weight *= self.trainer.global_step / self.config.kl_warmup_steps
        
        logs = {
            "loss": kl_loss * kl_loss_weight,
            "io_stats/ddec_cond_var": ddec_cond.var(dim=(1,2,3)),
            "io_stats/ddec_cond_mean": ddec_cond.mean(dim=(1,2,3)),
            "io_stats/prenorm_latents_var": pre_norm_latents.var(dim=(1,2,3)).detach(),
            "io_stats/latents_var": latents.var(dim=(1,2,3)).detach(),
            "io_stats/latents_mean": latents.mean(dim=(1,2,3)).detach(),
            "io_stats/latents_sigma": latents_sigma.detach(),

            "io_stats_ddecp/mdct_phase_var": mdct_phase.var(dim=(1,2,3)),
            "io_stats_ddecm/mdct_psd_var": mdct_psd.var(dim=(1,2,3)),
            "io_stats_ddecm/mdct_psd_mean": mdct_psd.mean(dim=(1,2,3)),

            "loss/kl_latents": kl_loss.detach(),
            "loss_weight/kl_latents": kl_loss_weight,
            "loss_weight/phase_invariance": phase_invariance_loss_weight,
        }

        if self.config.phase_invariance_loss_weight > 0:
            logs["loss"] = logs["loss"] + phase_invariance_loss * phase_invariance_loss_weight
            logs["loss/phase_invariance"] = phase_invariance_loss.detach()

        noise = torch.randn_like(mdct_psd)
        perturb_noise = torch.randn_like(mdct_psd)

        logs.update(self.ddecp_trainer.train_batch(mdct_phase, audio_embeddings, ddec_cond, noise=noise, perturb_noise=perturb_noise))
        logs.update(self.ddecm_trainer.train_batch(mdct_psd, audio_embeddings, ddec_cond, noise=noise, perturb_noise=perturb_noise))
        logs["loss"] = logs["loss"] + logs["loss/ddecp"] + logs["loss/ddecm"]

        dynamic_range_ddecm = mdct_psd.amax(dim=(1,2,3)) - mdct_psd.amin(dim=(1,2,3))
        logs["io_stats_ddecm/dynamic_range"] = dynamic_range_ddecm
        dynamic_range_ddecp = mdct_phase.amax(dim=(1,2,3)) - mdct_phase.amin(dim=(1,2,3))
        logs["io_stats_ddecp/dynamic_range"] = dynamic_range_ddecp

        if self.trainer.config.enable_debug_mode == True:
            print("mdct_phase.shape:", mdct_phase.shape)
            print("ddec_cond.shape:", ddec_cond.shape)
            print("latents.shape:", latents.shape)

        return logs
      
    @torch.no_grad()
    def finish_batch(self) -> Optional[dict[str, Union[torch.Tensor, float]]]:

        logs = {}
        logs.update(self.ddecp_trainer.finish_batch())
        logs.update(self.ddecm_trainer.finish_batch())

        return logs