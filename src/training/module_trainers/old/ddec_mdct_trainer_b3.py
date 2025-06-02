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
from typing import Literal, Optional

import torch
import numpy as np

from training.sigma_sampler import SigmaSamplerConfig, SigmaSampler
from training.trainer import DualDiffusionTrainer
from training.module_trainers.module_trainer import ModuleTrainerConfig, ModuleTrainer
from modules.daes.dae_edm2_h1 import DAE_H1
from modules.unets.unet_edm2_ddec_mdct_b3 import DDec_MDCT_UNet_B3
from modules.formats.ms_mdct_dual import MS_MDCT_DualFormat


from modules.mp_tools import normalize
from utils.dual_diffusion_utils import dict_str


@dataclass
class DiffusionDecoder_MDCT_Trainer_B3_Config(ModuleTrainerConfig):

    sigma_distribution: Literal["ln_normal", "ln_sech", "ln_sech^2",
                                "ln_linear", "ln_pdf"] = "ln_pdf"
    sigma_override_max: Optional[float] = 16
    sigma_override_min: Optional[float] = 4e-5
    sigma_dist_scale: float = 3.0
    sigma_dist_offset: float = 0
    use_stratified_sigma_sampling: bool = True
    sigma_pdf_resolution: Optional[int] = 127
    sigma_pdf_warmup_steps: Optional[int] = 5000

    validation_sigma_distribution: Literal["ln_normal", "ln_sech", "ln_sech^2",
                                            "ln_linear", "ln_pdf"] = "ln_sech"
    validation_sigma_override_max: Optional[float] = None
    validation_sigma_override_min: Optional[float] = None
    validation_sigma_dist_scale: float = 1.0
    validation_sigma_dist_offset: float = 0

    num_loss_buckets: int = 12
    loss_buckets_sigma_min: float = 0.0005
    loss_buckets_sigma_max: float = 100

    latents_perturbation: float = 0
    conditioning_dropout: float = 0.1

    kl_loss_weight: float = 2e-3
    kl_warmup_steps: int  = 5000

class DiffusionDecoder_MDCT_Trainer_B3(ModuleTrainer):
    
    @torch.no_grad()
    def __init__(self, config: DiffusionDecoder_MDCT_Trainer_B3_Config, trainer: DualDiffusionTrainer) -> None:

        self.config = config
        self.trainer = trainer
        self.logger = trainer.logger
        self.ddec: DDec_MDCT_UNet_B3 = trainer.modules[trainer.config.train_modules.index("ddec")]
        self.dae: DAE_H1 = trainer.modules[trainer.config.train_modules.index("dae")]

        self.is_validation_batch = False
        self.device_generator = None
        self.cpu_generator = None

        self.format: MS_MDCT_DualFormat = trainer.pipeline.format.to(self.trainer.accelerator.device)
        self.loss_weight = self.format.mdct_mel_density / torch.linalg.vector_norm(self.format.mdct_mel_density)

        if trainer.config.enable_model_compilation:
            self.ddec.compile(**trainer.config.compile_params)
            self.dae.compile(**trainer.config.compile_params)

        if self.config.num_loss_buckets > 0:
            self.logger.info(f"Using {self.config.num_loss_buckets} loss buckets")
            self.unet_loss_buckets = torch.zeros(
                self.config.num_loss_buckets, device="cpu", dtype=torch.float32)
            self.unet_loss_bucket_counts = torch.zeros(
                self.config.num_loss_buckets, device="cpu", dtype=torch.float32)
        else:
            self.logger.info("UNet loss buckets are disabled")

        if self.config.latents_perturbation > 0:
            self.logger.info(f"Using latents perturbation: {self.config.latents_perturbation}")
        else: self.logger.info("Latents perturbation is disabled")

        self.logger.info(f"Conditioning dropout: {self.config.conditioning_dropout}")

        # sigma schedule / distribution for train batches
        sigma_sampler_config = SigmaSamplerConfig(
            sigma_max=self.config.sigma_override_max or self.ddec.config.sigma_max,
            sigma_min=self.config.sigma_override_min or self.ddec.config.sigma_min,
            sigma_data=self.ddec.config.sigma_data,
            distribution=self.config.sigma_distribution,
            dist_scale=self.config.sigma_dist_scale,
            dist_offset=self.config.sigma_dist_offset,
            use_stratified_sigma_sampling=self.config.use_stratified_sigma_sampling,
            sigma_pdf_resolution=self.config.sigma_pdf_resolution,
        )
        self.sigma_sampler = SigmaSampler(sigma_sampler_config)
        self.logger.info("SigmaSampler config:")
        self.logger.info(dict_str(sigma_sampler_config.__dict__))

        # separate noise schedule / sigma distribution for validation batches
        validation_sigma_sampler_config = SigmaSamplerConfig(
            sigma_max=self.config.validation_sigma_override_max or self.ddec.config.sigma_max,
            sigma_min=self.config.validation_sigma_override_min or self.ddec.config.sigma_min,
            sigma_data=self.ddec.config.sigma_data,
            distribution=self.config.validation_sigma_distribution,
            dist_scale=self.config.validation_sigma_dist_scale,
            dist_offset=self.config.validation_sigma_dist_offset,
            use_static_sigma_sampling=True,
            sigma_pdf_resolution=self.config.sigma_pdf_resolution,
        )
        self.validation_sigma_sampler = SigmaSampler(validation_sigma_sampler_config)
        self.logger.info("Validation SigmaSampler config:")
        self.logger.info(dict_str(validation_sigma_sampler_config.__dict__))

        # pre-calculate the per-sigma loss bucket names
        if self.config.num_loss_buckets > 0:
            bucket_sigma = torch.linspace(np.log(self.config.loss_buckets_sigma_min),
                np.log(self.config.loss_buckets_sigma_max), self.config.num_loss_buckets + 1).exp()
            bucket_sigma[0] = 0; bucket_sigma[-1] = float("inf")
            self.bucket_names = [f"loss_σ_buckets/{bucket_sigma[i]:.4f} - {bucket_sigma[i+1]:.4f}"
                                 for i in range(self.config.num_loss_buckets)]

        self.logger.info("Training DAE/DDEC modules:")
        self.logger.info(f"KL loss weight: {self.config.kl_loss_weight} KL warmup steps: {self.config.kl_warmup_steps}")

    @torch.no_grad()
    def init_batch(self, validation: bool = False) -> None:
        
        if validation == True: # use the same random values for every validation batch
            self.is_validation_batch = True
            total_batch_size = self.trainer.validation_total_batch_size
            sigma_sampler = self.validation_sigma_sampler

            self.device_generator = torch.Generator(device=self.trainer.accelerator.device).manual_seed(0)
            self.cpu_generator = torch.Generator(device="cpu").manual_seed(0)
        else:
            self.is_validation_batch = False
            total_batch_size = self.trainer.total_batch_size
            sigma_sampler = self.sigma_sampler

            self.device_generator = None
            self.cpu_generator = None

        # reset sigma-bucketed loss for new batch
        if self.config.num_loss_buckets > 0:
            self.unet_loss_buckets.zero_()
            self.unet_loss_bucket_counts.zero_()

        # if using dynamic sigma sampling with ln_pdf, update the pdf using the learned per-sigma error estimate
        if self.config.sigma_distribution == "ln_pdf" and validation == False:
            ln_sigma = torch.linspace(self.sigma_sampler.config.ln_sigma_min,
                                      self.sigma_sampler.config.ln_sigma_max,
                                      self.config.sigma_pdf_resolution,
                                      device=self.trainer.accelerator.device)
            ln_sigma_error = self.ddec.logvar_linear(
                self.ddec.logvar_fourier(ln_sigma/4)).float().flatten().detach()
            pdf_warmup_factor = min(1, self.trainer.global_step / (self.config.sigma_pdf_warmup_steps or 1))
            sigma_distribution_pdf = (-pdf_warmup_factor * self.config.sigma_dist_scale * ln_sigma_error).exp()
            sigma_distribution_pdf = (sigma_distribution_pdf - 0.8).clip(min=0.2)
            self.sigma_sampler.update_pdf(sigma_distribution_pdf)
        
        # sample whole-batch sigma and sync across all ranks / processes
        self.global_sigma = sigma_sampler.sample(total_batch_size, device=self.trainer.accelerator.device)
        self.global_sigma = self.trainer.accelerator.gather(self.global_sigma.unsqueeze(0))[0]

    def train_batch(self, batch: dict) -> dict[str, torch.Tensor]:

        if self.is_validation_batch == False:
            device_batch_size = self.trainer.config.device_batch_size
        else:
            device_batch_size = self.trainer.config.validation_device_batch_size

        if "audio_embeddings" in batch:
            sample_audio_embeddings = normalize(batch["audio_embeddings"])

            conditioning_mask = (torch.rand(device_batch_size, generator=self.device_generator,
                device=self.trainer.accelerator.device) > self.config.conditioning_dropout).float()
            unet_class_embeddings = self.ddec.get_embeddings(sample_audio_embeddings, conditioning_mask)

            dae_class_embeddings = self.dae.get_embeddings(sample_audio_embeddings.to(dtype=self.dae.dtype))
        else:
            unet_class_embeddings = None
            dae_class_embeddings = None

        # prepare model inputs
        raw_samples: torch.Tensor = batch["audio"].clone().requires_grad_(False).detach()
        mdct_samples: torch.Tensor = self.format.raw_to_mdct(raw_samples, random_phase_augmentation=True).requires_grad_(False).detach()

        latents, ref_samples, pre_norm_latents = self.dae(mdct_samples,
            dae_class_embeddings, add_latents_noise=self.config.latents_perturbation)

        # get the noise level for this sub-batch from the pre-calculated whole-batch sigma (required for stratified sampling)
        local_sigma = self.global_sigma[self.trainer.accelerator.local_process_index::self.trainer.accelerator.num_processes]
        batch_sigma = local_sigma[self.trainer.accum_step * device_batch_size:(self.trainer.accum_step+1) * device_batch_size]

        noise = torch.randn(mdct_samples.shape, device=mdct_samples.device, generator=self.device_generator)
        noise = (noise * batch_sigma.view(-1, 1, 1, 1)).detach()

        denoised: torch.Tensor = self.ddec(mdct_samples.detach() + noise.detach(), batch_sigma, self.format, unet_class_embeddings, ref_samples)
        batch_loss_weight = (batch_sigma ** 2 + self.ddec.config.sigma_data ** 2) / (batch_sigma * self.ddec.config.sigma_data) ** 2
        batch_weighted_loss = torch.nn.functional.mse_loss(denoised, mdct_samples, reduction="none") * batch_loss_weight.view(-1, 1, 1, 1)
        
        if self.is_validation_batch == True:
            batch_loss = batch_weighted_loss
        else: # for train batches this takes the loss as the gaussian NLL with sigma-specific learned variance
            error_logvar = self.ddec.get_sigma_loss_logvar(sigma=batch_sigma).view(-1, 1, 1, 1)
            batch_loss = batch_weighted_loss / error_logvar.exp() + error_logvar
            batch_loss = (batch_loss * self.loss_weight).mean(dim=(1,2,3))

        # log loss bucketed by noise level range
        if self.config.num_loss_buckets > 0:
            batch_weighted_loss = batch_weighted_loss.mean(dim=(1,2,3))
            global_weighted_loss = self.trainer.accelerator.gather(batch_weighted_loss.detach()).cpu()
            global_sigma_quantiles = (batch_sigma.detach().log().cpu() - np.log(self.config.loss_buckets_sigma_min)) / (
                np.log(self.config.loss_buckets_sigma_max) - np.log(self.config.loss_buckets_sigma_min))
            target_buckets = (global_sigma_quantiles * self.unet_loss_buckets.shape[0]).long().clip(min=0, max=self.unet_loss_buckets.shape[0] - 1)
            self.unet_loss_buckets.index_add_(0, target_buckets, global_weighted_loss)
            self.unet_loss_bucket_counts.index_add_(0, target_buckets, torch.ones_like(global_weighted_loss))

        pre_norm_latents_var: torch.Tensor = pre_norm_latents.var(dim=(1,2,3))
        kl_loss = pre_norm_latents.mean(dim=(1,2,3)).square() + pre_norm_latents_var - 1 - pre_norm_latents_var.log()

        kl_loss_weight = self.config.kl_loss_weight
        if self.trainer.global_step < self.config.kl_warmup_steps:
            kl_loss_weight *= self.trainer.global_step / self.config.kl_warmup_steps

        return {"loss": kl_loss * kl_loss_weight + batch_loss,
                "loss/kl": kl_loss,
                "loss_weight/kl": kl_loss_weight,
                "io_stats/mdct_std": mdct_samples.std(dim=(1,2,3)),
                "io_stats/mdct_mean": mdct_samples.mean(dim=(1,2,3)),
                "io_stats/x_ref_std": ref_samples.std(dim=(1,2,3)),
                "io_stats/x_ref_mean": ref_samples.mean(dim=(1,2,3)),
                "io_stats/denoised_std": denoised.std(dim=(1,2,3)),
                "io_stats/denoised_mean": denoised.mean(dim=(1,2,3)),
                "io_stats/latents_std": latents.std(dim=(1,2,3)),
                "io_stats/latents_mean": latents.mean(dim=(1,2,3)),
                "io_stats/latents_pre_norm_std": pre_norm_latents_var.sqrt()
            }

    @torch.no_grad()
    def finish_batch(self) -> dict[str, torch.Tensor]:
        logs = {}

        # added sigma-bucketed loss to logs
        if self.config.num_loss_buckets > 0:
            for i in range(self.config.num_loss_buckets):
                if self.unet_loss_bucket_counts[i].item() > 0:
                    logs[self.bucket_names[i]] = (self.unet_loss_buckets[i] / self.unet_loss_bucket_counts[i]).item()

        return logs