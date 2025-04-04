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

from utils import config

from dataclasses import dataclass
from typing import Literal, Optional
from traceback import format_exception
import os

import torch
import numpy as np

from training.sigma_sampler import SigmaSamplerConfig, SigmaSampler
from training.trainer import DualDiffusionTrainer
from .module_trainer import ModuleTrainerConfig, ModuleTrainer
from modules.unets.unet import DualDiffusionUNet
from modules.daes.dae import DualDiffusionDAE
from modules.mp_tools import mp_sum
from utils.dual_diffusion_utils import dict_str, normalize, save_img


@dataclass
class UNetTrainerConfig(ModuleTrainerConfig):

    sigma_distribution: Literal["ln_normal", "ln_sech", "ln_sech^2",
                                "ln_linear", "ln_pdf"] = "ln_sech"
    sigma_override_max: Optional[float] = None
    sigma_override_min: Optional[float] = None
    sigma_dist_scale: float = 1.0
    sigma_dist_offset: float = 0.3
    use_stratified_sigma_sampling: bool = True
    sigma_pdf_resolution: Optional[int] = 127
    sigma_pdf_warmup_steps: Optional[int] = 2000

    validation_sigma_distribution: Literal["ln_normal", "ln_sech", "ln_sech^2",
                                            "ln_linear", "ln_pdf"] = "ln_sech"
    validation_sigma_override_max: Optional[float] = None
    validation_sigma_override_min: Optional[float] = None
    validation_sigma_dist_scale: float = 1.0
    validation_sigma_dist_offset: float = 0.3

    num_loss_buckets: int = 12
    loss_buckets_sigma_min: float = 0.01
    loss_buckets_sigma_max: float = 200

    invert_stereo_augmentation: bool = False
    input_perturbation: float = 0.
    noise_sample_bias: float = 0.
    conditioning_perturbation: float = 0.
    conditioning_dropout: float = 0.1
    continuous_conditioning_dropout: bool = False

    inpainting_probability: float = 0
    inpainting_extend_probability: float = 0.2
    inpainting_prepend_probability: float = 0.1
    inpainting_outpaint_min_width: int = 172
    inpainting_outpaint_max_width: int = 516
    inpainting_min_width: int = 8
    inpainting_max_width: int = 516
    inpainting_random_probability: float = 0.2

class UNetTrainer(ModuleTrainer):
    
    @torch.no_grad()
    def __init__(self, config: UNetTrainerConfig, trainer: DualDiffusionTrainer) -> None:

        self.config = config
        self.trainer = trainer
        self.logger = trainer.logger
        self.module: DualDiffusionUNet = trainer.module

        self.is_validation_batch = False
        self.device_generator = None
        self.cpu_generator = None

        if trainer.config.enable_model_compilation:
            self.module.compile(**trainer.config.compile_params)

        self.saved_debug_latents = False

        self.logger.info(f"Training diffusion model with pre-encoded latents and embeddings")
    
        if self.config.num_loss_buckets > 0:
            self.logger.info(f"Using {self.config.num_loss_buckets} loss buckets")
            self.unet_loss_buckets = torch.zeros(
                self.config.num_loss_buckets, device="cpu", dtype=torch.float32)
            self.unet_loss_bucket_counts = torch.zeros(
                self.config.num_loss_buckets, device="cpu", dtype=torch.float32)
        else:
            self.logger.info("UNet loss buckets are disabled")

        # log unet trainer specific config / settings
        if self.config.invert_stereo_augmentation == True:
            self.logger.info(f"Invert-stereo Augmentation: {self.config.input_perturbation}")
        if self.config.input_perturbation > 0:
            self.logger.info(f"Using input perturbation: {self.config.input_perturbation}")
        else: self.logger.info("Input perturbation is disabled")
        if self.config.conditioning_perturbation > 0:
            self.config.conditioning_perturbation = min(self.config.conditioning_perturbation, 1)
            self.logger.info(f"Using conditioning perturbation: {self.config.conditioning_perturbation}")
        else: self.logger.info("Conditioning perturbation is disabled")
        self.logger.info(f"Dropout: {self.module.config.dropout} Conditioning dropout: {self.config.conditioning_dropout}")
        if self.config.continuous_conditioning_dropout == True: self.logger.info("Continuous conditioning dropout is enabled")
        if self.config.inpainting_probability > 0:
            self.logger.info(f"Using inpainting (probability: {self.config.inpainting_probability})")
            self.logger.info(f"  extend probability: {self.config.inpainting_extend_probability}")
            self.logger.info(f"  prepend probability: {self.config.inpainting_prepend_probability}")
            self.logger.info(f"  outpaint min width: {self.config.inpainting_outpaint_min_width}")
            self.logger.info(f"  outpaint max width: {self.config.inpainting_outpaint_max_width}")
            self.logger.info(f"  inpainting min width: {self.config.inpainting_min_width}")
            self.logger.info(f"  inpainting max width: {self.config.inpainting_max_width}")
            self.logger.info(f"  random probability: {self.config.inpainting_random_probability}")
        else:
            self.logger.info("Inpainting training is disabled")

        self.logger.info(f"Using sample biased noise: {self.config.noise_sample_bias > 0}")
        if self.config.noise_sample_bias > 0:
            self.logger.info(f"  noise sample bias: {self.config.noise_sample_bias}")

        # sigma schedule / distribution for train batches
        sigma_sampler_config = SigmaSamplerConfig(
            sigma_max=self.config.sigma_override_max or self.module.config.sigma_max,
            sigma_min=self.config.sigma_override_min or self.module.config.sigma_min,
            sigma_data=self.module.config.sigma_data,
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
            sigma_max=self.config.validation_sigma_override_max or self.module.config.sigma_max,
            sigma_min=self.config.validation_sigma_override_min or self.module.config.sigma_min,
            sigma_data=self.module.config.sigma_data,
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

    @torch.no_grad()
    def get_inpainting_ref_samples(self, samples: torch.Tensor) -> None:
        mask = torch.ones_like(samples[:, 0:1])
        
        for i in range(samples.shape[0]):
            if torch.rand(1, generator=self.cpu_generator).item() < self.config.inpainting_probability:
                # extension / out-painting
                if torch.rand(1, generator=self.cpu_generator).item() < self.config.inpainting_extend_probability:
                    mask_start = samples.shape[-1] - torch.randint(self.config.inpainting_outpaint_min_width,
                                                                   self.config.inpainting_outpaint_max_width + 1,
                                                                   (1,), generator=self.cpu_generator).item()
                    mask_end = samples.shape[-1]
                elif torch.rand(1, generator=self.cpu_generator).item() < self.config.inpainting_prepend_probability:
                    # prepend / out-painting
                    mask_end = torch.randint(self.config.inpainting_outpaint_min_width,
                                             self.config.inpainting_outpaint_max_width + 1,
                                             (1,), generator=self.cpu_generator).item()
                    mask_start = 0
                    
                else: # normal inpainting
                    mask_width = torch.randint(self.config.inpainting_min_width,
                                               self.config.inpainting_max_width + 1,
                                               (1,), generator=self.cpu_generator).item()
                    mask_start = torch.randint(0, samples.shape[-1] - mask_width + 1, (1,), generator=self.cpu_generator).item()
                    mask_end = mask_start + mask_width

                mask[i, :] = 0
                mask[i, :, :, mask_start:mask_end] = 1

        # dropout/inpaint random pixels with configured probability
        if self.config.inpainting_random_probability > 0:
            mask *= torch.rand(mask.shape, generator=self.cpu_generator).to(device=mask.device) > self.config.inpainting_random_probability

        return torch.cat((samples * (1 - mask), mask), dim=1).detach()

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
            ln_sigma_error = self.module.logvar_linear(
                self.module.logvar_fourier(ln_sigma/4)).float().flatten().detach()
            pdf_warmup_factor = min(1, self.trainer.global_step / (self.config.sigma_pdf_warmup_steps or 1))
            sigma_distribution_pdf = (-pdf_warmup_factor * self.config.sigma_dist_scale * ln_sigma_error).exp()
            sigma_distribution_pdf = (sigma_distribution_pdf - 0.8).clip(min=0.2)
            self.sigma_sampler.update_pdf(sigma_distribution_pdf)
        
        # sample whole-batch sigma and sync across all ranks / processes
        self.global_sigma = sigma_sampler.sample(total_batch_size, device=self.trainer.accelerator.device)
        self.global_sigma = self.trainer.accelerator.gather(self.global_sigma.unsqueeze(0))[0]

    def train_batch(self, batch: dict) -> dict[str, torch.Tensor]:

        samples: torch.Tensor = batch["latents"].float().clone()

        # in our first train batch save the input latents to debug image files
        if self.trainer.accelerator.is_main_process == True and self.saved_debug_latents == False:
            try:
                latents_debug_img_path = None
                if config.DEBUG_PATH is not None:
                    dae: DualDiffusionDAE = self.trainer.pipeline.dae
                    for i in range(samples.shape[0]):
                        latents_debug_img_path = os.path.join(config.DEBUG_PATH, "unet_trainer", f"latents_{i}.png")
                        save_img(dae.latents_to_img(samples[i:i+1]), latents_debug_img_path)

            except Exception as e:
                self.logger.error("".join(format_exception(type(e), e, e.__traceback__)))
                self.logger.error(f"Error saving debug lantents to '{latents_debug_img_path}': {e}")

            self.saved_debug_latents = True

        audio_embeddings = normalize(batch["audio_embeddings"]).float().clone()

        if self.is_validation_batch == False:
            device_batch_size = self.trainer.config.device_batch_size
        else:
            device_batch_size = self.trainer.config.validation_device_batch_size
        
        # with continuous conditioning dropout enabled the conditioning embedding is interpolated smoothly to the unconditional embedding
        if self.config.continuous_conditioning_dropout == True and self.is_validation_batch == False:
            conditioning_mask = (torch.rand(device_batch_size,
                generator=self.device_generator, device=self.trainer.accelerator.device) > (self.config.conditioning_dropout * 2)).float()
            conditioning_mask = 1 - ((1 - conditioning_mask) * torch.rand(
                conditioning_mask.shape, generator=self.device_generator, device=self.trainer.accelerator.device))
        else: # normal conditioning dropout
            conditioning_mask = (torch.rand(device_batch_size,
                generator=self.device_generator, device=self.trainer.accelerator.device) > self.config.conditioning_dropout).float()
            
        unet_embeddings = self.module.get_embeddings(audio_embeddings, conditioning_mask)

        if self.config.conditioning_perturbation > 0 and self.is_validation_batch == False: # adds noise to the conditioning embedding while preserving variance
            conditioning_perturbation = torch.randn(unet_embeddings.shape, device=unet_embeddings.device, generator=self.device_generator)
            unet_embeddings = mp_sum(unet_embeddings, conditioning_perturbation, self.config.conditioning_perturbation)

        if self.is_validation_batch == False:
            assert samples.shape == self.trainer.latent_shape, f"Expected shape {self.trainer.latent_shape}, got {samples.shape}"
        else:
            assert samples.shape == self.trainer.validation_latent_shape, f"Expected shape {self.trainer.validation_latent_shape}, got {samples.shape}"
        
        # add extra noise to the sample while preserving variance if input_perturbation is enabled
        if self.config.input_perturbation > 0 and self.is_validation_batch == False:
            input_perturbation = torch.randn(samples.shape, device=samples.device, generator=self.device_generator)
            samples = mp_sum(samples, input_perturbation, self.config.input_perturbation)

        # get the noise level for this sub-batch from the pre-calculated whole-batch sigma (required for stratified sampling)
        local_sigma = self.global_sigma[self.trainer.accelerator.local_process_index::self.trainer.accelerator.num_processes]
        batch_sigma = local_sigma[self.trainer.accum_step * device_batch_size:(self.trainer.accum_step+1) * device_batch_size]

        # prepare model inputs
        noise = torch.randn(samples.shape, device=samples.device, generator=self.device_generator)
        samples = (samples * self.module.config.sigma_data).detach()
        ref_samples = self.get_inpainting_ref_samples(samples) if self.config.inpainting_probability > 0 else None
        if self.is_validation_batch == False and self.config.noise_sample_bias > 0: # this has an effect similar to immiscible diffusion / rectified flow
            noise = (mp_sum(noise, samples, t=self.config.noise_sample_bias) * batch_sigma.view(-1, 1, 1, 1)).detach()
        else:
            noise = (noise * batch_sigma.view(-1, 1, 1, 1)).detach()

        # convert model inputs to channels_last memory format for performance, if enabled
        if self.trainer.config.enable_channels_last == True:
            samples = samples.to(memory_format=torch.channels_last)
            noise = noise.to(memory_format=torch.channels_last)
            ref_samples = ref_samples.to(memory_format=torch.channels_last) if ref_samples is not None else None

        denoised = self.module(samples + noise, batch_sigma, self.trainer.pipeline.format, unet_embeddings, ref_samples)
        batch_loss_weight = (batch_sigma ** 2 + self.module.config.sigma_data ** 2) / (batch_sigma * self.module.config.sigma_data) ** 2
        batch_weighted_loss = torch.nn.functional.mse_loss(denoised, samples, reduction="none").mean(dim=(1,2,3)) * batch_loss_weight

        # normalize loss wrt inpainting mask for comparability/consistency in validation loss
        if self.config.inpainting_probability > 0 and self.is_validation_batch == True:
            batch_weighted_loss = batch_weighted_loss / ref_samples[:, -1:].mean(dim=(1,2,3))

        if self.is_validation_batch == True:
            batch_loss = batch_weighted_loss
        else: # for train batches this takes the loss as the gaussian NLL with sigma-specific learned variance
            error_logvar = self.module.get_sigma_loss_logvar(sigma=batch_sigma)
            batch_loss = batch_weighted_loss / error_logvar.exp() + error_logvar
        
        # log loss bucketed by noise level range
        if self.config.num_loss_buckets > 0:
            global_weighted_loss = self.trainer.accelerator.gather(batch_weighted_loss.detach()).cpu()
            global_sigma_quantiles = (batch_sigma.detach().log().cpu() - np.log(self.config.loss_buckets_sigma_min)) / (
                np.log(self.config.loss_buckets_sigma_max) - np.log(self.config.loss_buckets_sigma_min))
            target_buckets = (global_sigma_quantiles * self.unet_loss_buckets.shape[0]).long().clip(min=0, max=self.unet_loss_buckets.shape[0] - 1)
            self.unet_loss_buckets.index_add_(0, target_buckets, global_weighted_loss)
            self.unet_loss_bucket_counts.index_add_(0, target_buckets, torch.ones_like(global_weighted_loss))

        return {
            "loss": batch_loss,
            "latents/mean": samples.mean(),
            "latents/std": samples.std()
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