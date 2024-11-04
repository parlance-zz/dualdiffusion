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

from training.sigma_sampler import SigmaSamplerConfig, SigmaSampler
from training.trainer import DualDiffusionTrainer
from .module_trainer import ModuleTrainerConfig, ModuleTrainer
from modules.unets.unet import DualDiffusionUNet
from utils.dual_diffusion_utils import dict_str

@dataclass
class UNetTrainerConfig(ModuleTrainerConfig):

    sigma_distribution: Literal["ln_normal", "ln_sech", "ln_sech^2",
                                "ln_linear", "ln_pdf"] = "ln_sech"
    sigma_dist_scale: float = 1.0
    sigma_dist_offset: float = 0.1
    use_stratified_sigma_sampling: bool = True
    sigma_pdf_resolution: Optional[int] = 127

    validation_sigma_distribution: Literal["ln_normal", "ln_sech", "ln_sech^2",
                                            "ln_linear", "ln_pdf"] = "ln_sech"
    validation_sigma_dist_scale: float = 1.0
    validation_sigma_dist_offset: float = 0.3

    num_loss_buckets: int = 10
    input_perturbation: float = 0.
    conditioning_perturbation: float = 0.
    conditioning_dropout: float = 0.1
    continuous_conditioning_dropout: bool = False

    inpainting_probability: float = 0.7
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

        #if config.inpainting_probability > 0 and self.module.config.inpainting == False:
        #    self.logger.error(f"UNet model does not support inpainting, aborting training..."); exit(1)

        if trainer.config.enable_model_compilation:
            self.module.compile(**trainer.config.compile_params)

        if not trainer.config.dataloader.use_pre_encoded_latents:
            
            trainer.pipeline.format = trainer.pipeline.format.to(self.accelerator.device)
            if trainer.config.enable_model_compilation:
                trainer.pipeline.format.compile(**trainer.config.compile_params)

            if hasattr(trainer.pipeline, "vae"):
                if trainer.pipeline.vae.config.last_global_step == 0:
                    self.logger.error("VAE model has not been trained, aborting training..."); exit(1)
                    
                trainer.pipeline.vae = trainer.pipeline.vae.to(trainer.accelerator.device).requires_grad_(False).eval()
                if trainer.mixed_precision_enabled == True:
                    trainer.pipeline.vae = trainer.pipeline.vae.to(dtype=trainer.mixed_precision_dtype)
                if trainer.config.enable_model_compilation:
                    trainer.pipeline.vae.compile(**trainer.config.compile_params)
                
                self.logger.info(f"Training diffusion model with VAE")
            else:
                self.logger.info(f"Training diffusion model without VAE")
        else:
            self.logger.info(f"Training diffusion model with pre-encoded latents")
        
        if self.config.num_loss_buckets > 0:
            self.logger.info(f"Using {self.config.num_loss_buckets} loss buckets")
            self.unet_loss_buckets = torch.zeros(
                self.config.num_loss_buckets, device="cpu", dtype=torch.float32)
            self.unet_loss_bucket_counts = torch.zeros(
                self.config.num_loss_buckets, device="cpu", dtype=torch.float32)
        else:
            self.logger.info("UNet loss buckets are disabled")

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

        sigma_sampler_config = SigmaSamplerConfig(
            sigma_max=self.module.config.sigma_max,
            sigma_min=self.module.config.sigma_min,
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

        validation_sigma_sampler_config = SigmaSamplerConfig(
            sigma_max=self.module.config.sigma_max,
            sigma_min=self.module.config.sigma_min,
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

        if self.config.num_loss_buckets > 0:
            bucket_ln_sigma = (1 / torch.linspace(torch.pi/2, 0, self.config.num_loss_buckets+1).tan()).log()
            bucket_ln_sigma[0] = float("-inf"); bucket_ln_sigma[-1] = float("inf")
            self.bucket_names = [f"unet_buckets_loss/b{i} s:{bucket_ln_sigma[i]:.3f} ~ {bucket_ln_sigma[i+1]:.3f}"
                                 for i in range(self.config.num_loss_buckets)]

    @staticmethod
    def get_config_class() -> ModuleTrainerConfig:
        return UNetTrainerConfig
    
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
        
        if validation == True:
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

        if self.config.num_loss_buckets > 0:
            self.unet_loss_buckets.zero_()
            self.unet_loss_bucket_counts.zero_()

        if self.config.sigma_distribution == "ln_pdf" and validation == False:
            sigma_sample_temperature = 1 / self.config.sigma_dist_scale
            ln_sigma = torch.linspace(self.sigma_sampler.config.ln_sigma_min,
                                      self.sigma_sampler.config.ln_sigma_max,
                                      self.config.sigma_pdf_resolution,
                                      device=self.trainer.accelerator.device)
            ln_sigma_error = self.module.logvar_linear(
                self.module.logvar_fourier(ln_sigma/4)).float().flatten().detach()
            sigma_distribution_pdf = (-sigma_sample_temperature * ln_sigma_error).exp()
            self.sigma_sampler.update_pdf(sigma_distribution_pdf)
        
        self.global_sigma = sigma_sampler.sample(total_batch_size, device=self.trainer.accelerator.device)
        self.global_sigma = self.trainer.accelerator.gather(self.global_sigma.unsqueeze(0))[0] # sync sigma across all ranks / processes

    def train_batch(self, batch: dict, accum_step: int) -> dict[str, torch.Tensor]:

        raw_samples = batch["input"]
        sample_game_ids = batch["game_ids"]
        sample_t_ranges = batch["t_ranges"] if self.trainer.dataset.config.t_scale is not None else None
        #sample_author_ids = batch["author_ids"]

        class_labels = self.trainer.pipeline.get_class_labels(sample_game_ids, module_name="unet")
        if self.config.continuous_conditioning_dropout == True and self.is_validation_batch == False:
            conditioning_mask = (torch.rand(self.trainer.config.device_batch_size,
                generator=self.device_generator, device=self.trainer.accelerator.device) > (self.config.conditioning_dropout * 2)).float()
            conditioning_mask = 1 - ((1 - conditioning_mask) * torch.rand(
                conditioning_mask.shape, generator=self.device_generator, device=self.trainer.accelerator.device))
        else:
            conditioning_mask = (torch.rand(self.trainer.config.device_batch_size,
                generator=self.device_generator, device=self.trainer.accelerator.device) > self.config.conditioning_dropout).float()

        unet_class_embeddings = self.module.get_class_embeddings(class_labels, conditioning_mask)
        if self.config.conditioning_perturbation > 0 and self.is_validation_batch == False:
            conditioning_perturbation = torch.randn(unet_class_embeddings.shape, device=unet_class_embeddings.device, generator=self.device_generator)
            unet_class_embeddings = unet_class_embeddings + conditioning_perturbation * self.config.conditioning_perturbation
            
        if self.trainer.config.dataloader.use_pre_encoded_latents:
            samples = raw_samples.float()
            assert samples.shape == self.trainer.latent_shape, f"Expected shape {self.trainer.latent_shape}, got {samples.shape}"
        else:
            samples = self.trainer.pipeline.format.raw_to_sample(raw_samples)
            vae_class_embeddings = self.trainer.pipeline.vae.get_class_embeddings(class_labels)
            samples = self.trainer.pipeline.vae.encode(samples.to(self.trainer.pipeline.vae.dtype),
                                                       vae_class_embeddings, self.trainer.pipeline.format).mode().float()
            assert samples.shape == self.trainer.sample_shape, f"Expected shape {self.trainer.sample_shape}, got {samples.shape}"
        
        if self.config.input_perturbation > 0 and self.is_validation_batch == False:
            input_perturbation = torch.randn(samples.shape, device=samples.device, generator=self.device_generator)
            samples += input_perturbation * self.config.input_perturbation

        local_sigma = self.global_sigma[self.trainer.accelerator.local_process_index::self.trainer.accelerator.num_processes]
        batch_sigma = local_sigma[accum_step * self.trainer.config.device_batch_size:(accum_step+1) * self.trainer.config.device_batch_size]

        noise = torch.randn(samples.shape, device=samples.device, generator=self.device_generator) * batch_sigma.view(-1, 1, 1, 1)
        samples = (samples * self.module.config.sigma_data).detach()
        ref_samples = self.get_inpainting_ref_samples(samples) if self.config.inpainting_probability > 0 or self.module.config.inpainting == True else None

        denoised = self.module(samples + noise, batch_sigma, self.trainer.pipeline.format, unet_class_embeddings, sample_t_ranges, ref_samples)
        batch_loss_weight = (batch_sigma ** 2 + self.module.config.sigma_data ** 2) / (batch_sigma * self.module.config.sigma_data) ** 2
        batch_weighted_loss = torch.nn.functional.mse_loss(denoised, samples, reduction="none").mean(dim=(1,2,3)) * batch_loss_weight

        # normalize loss wrt inpainting mask for comparability/consistency in validation loss
        if self.config.inpainting_probability > 0 and self.is_validation_batch == True:
            batch_weighted_loss = batch_weighted_loss / ref_samples[:, -1:].mean(dim=(1,2,3))

        if self.is_validation_batch == True:
            batch_loss = batch_weighted_loss
        else:
            error_logvar = self.module.get_sigma_loss_logvar(sigma=batch_sigma, class_embeddings=unet_class_embeddings)
            batch_loss = batch_weighted_loss / error_logvar.exp() + error_logvar
        
        if self.config.num_loss_buckets > 0:
            global_weighted_loss = self.trainer.accelerator.gather(batch_weighted_loss.detach()).cpu()
            global_sigma_quantiles = self.trainer.accelerator.gather(self.module.config.sigma_data / batch_sigma.detach()).cpu().arctan() / (torch.pi/2)

            target_buckets = (global_sigma_quantiles * self.unet_loss_buckets.shape[0]).long().clip(min=0, max=self.unet_loss_buckets.shape[0] - 1)
            self.unet_loss_buckets.index_add_(0, target_buckets, global_weighted_loss)
            self.unet_loss_bucket_counts.index_add_(0, target_buckets, torch.ones_like(global_weighted_loss))

        return {"loss": batch_loss}

    @torch.no_grad()
    def finish_batch(self) -> dict[str, torch.Tensor]:
        logs = {}

        if self.config.num_loss_buckets > 0:
            for i in range(self.config.num_loss_buckets):
                if self.unet_loss_bucket_counts[i].item() > 0:
                    logs[self.bucket_names[i]] = (self.unet_loss_buckets[i] / self.unet_loss_bucket_counts[i]).item()

        return logs