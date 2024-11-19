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
import os

import torch

from training.sigma_sampler import SigmaSamplerConfig, SigmaSampler
from training.trainer import DualDiffusionTrainer
from .module_trainer import ModuleTrainerConfig, ModuleTrainer
from modules.unets.unet import DualDiffusionUNet
from modules.mp_tools import mp_sum
from utils.dual_diffusion_utils import dict_str, normalize, load_safetensors

@dataclass
class UNetTrainerConfig(ModuleTrainerConfig):

    sigma_distribution: Literal["ln_normal", "ln_sech", "ln_sech^2",
                                "ln_linear", "ln_pdf"] = "ln_sech"
    sigma_override_max: Optional[float] = None
    sigma_override_min: Optional[float] = None
    sigma_dist_scale: float = 1.0
    sigma_dist_offset: float = 0.1
    use_stratified_sigma_sampling: bool = True
    sigma_pdf_resolution: Optional[int] = 127

    validation_sigma_distribution: Literal["ln_normal", "ln_sech", "ln_sech^2",
                                            "ln_linear", "ln_pdf"] = "ln_sech"
    validation_sigma_override_max: Optional[float] = None
    validation_sigma_override_min: Optional[float] = None
    validation_sigma_dist_scale: float = 1.0
    validation_sigma_dist_offset: float = 0.3

    num_loss_buckets: int = 10
    input_perturbation: float = 0.
    noise_sample_bias: float = 0.
    conditioning_perturbation: float = 0.
    conditioning_dropout: float = 0.1
    continuous_conditioning_dropout: bool = False
    text_embedding_weight: float = 0.

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

            if trainer.config.dataloader.use_pre_encoded_audio_embeddings == True:
                self.logger.info(f"  Using pre-encoded audio embeddings")
            if trainer.config.dataloader.use_pre_encoded_text_embeddings == True:
                self.logger.info(f"  Using pre-encoded text embeddings")
                self.logger.info(f"  Text embedding weight: {self.config.text_embedding_weight}")
            
            if (trainer.config.dataloader.use_pre_encoded_audio_embeddings == True
                or trainer.config.dataloader.use_pre_encoded_text_embeddings == True):
                self.unconditional_audio_embedding = None
                self.unconditional_text_embedding = None

                dataset_embeddings_path = os.path.join(self.trainer.dataset.config.data_dir, "dataset_infos", "dataset_embeddings.safetensors")
                if not os.path.isfile(dataset_embeddings_path):
                    self.logger.error(f"  Dataset embeddings not found at {dataset_embeddings_path}, aborting training..."); exit(1)
                dataset_embeddings = load_safetensors(dataset_embeddings_path)
                if trainer.config.dataloader.use_pre_encoded_audio_embeddings == True:
                    self.unconditional_audio_embedding = normalize(dataset_embeddings["_unconditional_audio"][:].to(self.trainer.accelerator.device).unsqueeze(0)).float()
                if trainer.config.dataloader.use_pre_encoded_text_embeddings == True:
                    self.unconditional_text_embedding = normalize(dataset_embeddings["_unconditional_text"][:].to(self.trainer.accelerator.device).unsqueeze(0)).float()

                self.logger.info(f"  Loaded dataset embeddings successfully from {dataset_embeddings_path}")
        
        if self.config.num_loss_buckets > 0:
            self.logger.info(f"Using {self.config.num_loss_buckets} loss buckets")
            self.unet_loss_buckets = torch.zeros(
                self.config.num_loss_buckets, device="cpu", dtype=torch.float32)
            self.unet_loss_bucket_counts = torch.zeros(
                self.config.num_loss_buckets, device="cpu", dtype=torch.float32)
        else:
            self.logger.info("UNet loss buckets are disabled")

        # log unet trainer specific config / settings
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
            bucket_sigma = 2 / torch.linspace(torch.pi/2, 0, self.config.num_loss_buckets+1).tan()
            bucket_sigma[0] = 0; bucket_sigma[-1] = float("inf")
            self.bucket_names = [f"loss_σ_buckets/{bucket_sigma[i]:.2f} - {bucket_sigma[i+1]:.2f}"
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
            sigma_distribution_pdf = (-self.config.sigma_dist_scale * ln_sigma_error).exp()
            self.sigma_sampler.update_pdf(sigma_distribution_pdf)
        
        # sample whole-batch sigma and sync across all ranks / processes
        self.global_sigma = sigma_sampler.sample(total_batch_size, device=self.trainer.accelerator.device)
        self.global_sigma = self.trainer.accelerator.gather(self.global_sigma.unsqueeze(0))[0]

    def train_batch(self, batch: dict, accum_step: int) -> dict[str, torch.Tensor]:

        raw_samples = batch["input"]
        sample_game_ids = batch["game_ids"]
        sample_t_ranges = batch["t_ranges"] if self.trainer.dataset.config.t_scale is not None else None
        sample_audio_embeddings = normalize(batch["audio_embeddings"]) if self.trainer.dataset.config.use_pre_encoded_audio_embeddings else None
        sample_text_embeddings = normalize(batch["text_embeddings"]) if self.trainer.dataset.config.use_pre_encoded_text_embeddings else None

        # calculate sample embeddings if using pre-encoded audio/text embeddings
        sample_embeddings = None
        if sample_audio_embeddings is not None:
            if sample_text_embeddings is not None and self.config.text_embedding_weight > 0:
                sample_embeddings = normalize(mp_sum(sample_audio_embeddings, sample_text_embeddings, self.config.text_embedding_weight)).float()
            else:
                sample_embeddings = sample_audio_embeddings.float()
        else:
            if sample_text_embeddings is not None:
                sample_embeddings = sample_text_embeddings.float()

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

        if sample_embeddings is None:
            class_labels = self.trainer.pipeline.get_class_labels(sample_game_ids, module_name="unet")
            unet_class_embeddings = self.module.get_class_embeddings(class_labels, conditioning_mask)
        else:
            unet_class_embeddings = mp_sum(self.unconditional_audio_embedding, sample_embeddings,
                t=conditioning_mask.unsqueeze(1).to(self.trainer.accelerator.device, self.module.dtype))

        if self.config.conditioning_perturbation > 0 and self.is_validation_batch == False: # adds noise to the conditioning embedding while preserving variance
            conditioning_perturbation = torch.randn(unet_class_embeddings.shape, device=unet_class_embeddings.device, generator=self.device_generator)
            unet_class_embeddings = mp_sum(unet_class_embeddings, conditioning_perturbation, self.config.conditioning_perturbation)
        
        # pre-encoding latents is strongly recommended for performance / training efficiency
        if self.trainer.config.dataloader.use_pre_encoded_latents:
            samples = raw_samples.float()
        else: # otherwise convert audio to spectrogram/format and encode latents with VAE
            samples = self.trainer.pipeline.format.raw_to_sample(raw_samples)
            vae_class_embeddings = self.trainer.pipeline.vae.get_class_embeddings(class_labels)
            samples = self.trainer.pipeline.vae.encode(samples.to(self.trainer.pipeline.vae.dtype),
                                                       vae_class_embeddings, self.trainer.pipeline.format).mode().float()

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
        batch_sigma = local_sigma[accum_step * device_batch_size:(accum_step+1) * device_batch_size]

        # prepare model inputs
        noise = torch.randn(samples.shape, device=samples.device, generator=self.device_generator)
        samples = (samples * self.module.config.sigma_data).detach()
        ref_samples = self.get_inpainting_ref_samples(samples) if self.config.inpainting_probability > 0 or self.module.config.inpainting == True else None
        if self.is_validation_batch == False and self.config.noise_sample_bias > 0: # this has an effect similar to immiscible diffusion / rectified flow
            noise = (mp_sum(noise, samples, t=self.config.noise_sample_bias) * batch_sigma.view(-1, 1, 1, 1)).detach()
        else:
            noise = (noise * batch_sigma.view(-1, 1, 1, 1)).detach()

        # convert model inputs to channels_last memory format for performance, if enabled
        if self.trainer.config.enable_channels_last == True:
            samples = samples.to(memory_format=torch.channels_last)
            noise = noise.to(memory_format=torch.channels_last)
            ref_samples = ref_samples.to(memory_format=torch.channels_last) if ref_samples is not None else None

        denoised = self.module(samples + noise, batch_sigma, self.trainer.pipeline.format, unet_class_embeddings, sample_t_ranges, ref_samples)
        batch_loss_weight = (batch_sigma ** 2 + self.module.config.sigma_data ** 2) / (batch_sigma * self.module.config.sigma_data) ** 2
        batch_weighted_loss = torch.nn.functional.mse_loss(denoised, samples, reduction="none").mean(dim=(1,2,3)) * batch_loss_weight

        # normalize loss wrt inpainting mask for comparability/consistency in validation loss
        if self.config.inpainting_probability > 0 and self.is_validation_batch == True:
            batch_weighted_loss = batch_weighted_loss / ref_samples[:, -1:].mean(dim=(1,2,3))

        if self.is_validation_batch == True:
            batch_loss = batch_weighted_loss
        else: # for train batches this takes the loss as the gaussian NLL with sigma-specific learned variance
            error_logvar = self.module.get_sigma_loss_logvar(sigma=batch_sigma, class_embeddings=unet_class_embeddings)
            batch_loss = batch_weighted_loss / error_logvar.exp() + error_logvar
        
        # log loss bucketed by noise level range
        if self.config.num_loss_buckets > 0:
            global_weighted_loss = self.trainer.accelerator.gather(batch_weighted_loss.detach()).cpu()
            global_sigma_quantiles = 1 - self.trainer.accelerator.gather(self.module.config.sigma_data / batch_sigma.detach() * 2).cpu().arctan() / (torch.pi/2)

            target_buckets = (global_sigma_quantiles * self.unet_loss_buckets.shape[0]).long().clip(min=0, max=self.unet_loss_buckets.shape[0] - 1)
            self.unet_loss_buckets.index_add_(0, target_buckets, global_weighted_loss)
            self.unet_loss_bucket_counts.index_add_(0, target_buckets, torch.ones_like(global_weighted_loss))

        return {"loss": batch_loss}

    @torch.no_grad()
    def finish_batch(self) -> dict[str, torch.Tensor]:
        logs = {}

        # added sigma-bucketed loss to logs
        if self.config.num_loss_buckets > 0:
            for i in range(self.config.num_loss_buckets):
                if self.unet_loss_bucket_counts[i].item() > 0:
                    logs[self.bucket_names[i]] = (self.unet_loss_buckets[i] / self.unet_loss_bucket_counts[i]).item()

        return logs