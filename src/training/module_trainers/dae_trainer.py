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

from training.sigma_sampler import SigmaSamplerConfig, SigmaSampler
from training.trainer import DualDiffusionTrainer
from .module_trainer import ModuleTrainerConfig, ModuleTrainer
from modules.daes.dae import DiffusionAutoencoder_EDM2_D1
from modules.formats.spectrogram import SpectrogramFormat
from modules.mp_tools import normalize
from utils.dual_diffusion_utils import dict_str, save_img, tensor_5d_to_4d


@dataclass
class DAE_TrainerConfig(ModuleTrainerConfig):

    sigma_distribution: Literal["ln_normal", "ln_sech", "ln_sech^2",
                                "ln_linear", "ln_pdf"] = "ln_sech"
    sigma_override_max: Optional[float] = None
    sigma_override_min: Optional[float] = None
    sigma_dist_scale: float = 1.0
    sigma_dist_offset: float = 0
    use_stratified_sigma_sampling: bool = True
    sigma_pdf_resolution: Optional[int] = 128
    sigma_pdf_warmup_steps: Optional[int] = 30000

    validation_sigma_distribution: Literal["ln_normal", "ln_sech", "ln_sech^2",
                                            "ln_linear", "ln_pdf"] = "ln_sech"
    validation_sigma_override_max: Optional[float] = None
    validation_sigma_override_min: Optional[float] = None
    validation_sigma_dist_scale: float = 1.0
    validation_sigma_dist_offset: float = 0

    num_loss_buckets: int = 10
    conditioning_dropout: float = 0.1
    latents_perturbation: float = 0.004
    kl_loss_weight: float = 0.25
    kl_loss_weight_warmup_steps: Optional[int] = 1000

class DAE_Trainer(ModuleTrainer):
    
    @torch.no_grad()
    def __init__(self, config: DAE_TrainerConfig, trainer: DualDiffusionTrainer) -> None:

        self.config = config
        self.trainer = trainer
        self.logger = trainer.logger
        self.module: DiffusionAutoencoder_EDM2_D1 = trainer.module

        self.is_validation_batch = False
        self.device_generator = None
        self.cpu_generator = None

        self.format: SpectrogramFormat = trainer.pipeline.format.to(self.trainer.accelerator.device)

        if trainer.config.enable_model_compilation:
            self.module.compile(**trainer.config.compile_params)
            self.format.compile(**trainer.config.compile_params)
        
        if self.config.num_loss_buckets > 0:
            self.logger.info(f"Using {self.config.num_loss_buckets} loss buckets")
            self.unet_loss_buckets = torch.zeros(
                self.config.num_loss_buckets, device="cpu", dtype=torch.float32)
            self.unet_loss_bucket_counts = torch.zeros(
                self.config.num_loss_buckets, device="cpu", dtype=torch.float32)
        else:
            self.logger.info("UNet loss buckets are disabled")

        if self.config.kl_loss_weight_warmup_steps is not None:
            self.logger.info(f"Using KL loss weight: {self.config.kl_loss_weight} "
                             f"Warmup steps: {self.config.kl_loss_weight_warmup_steps}")
        else:
            self.logger.info(f"Using KL loss weight: {self.config.kl_loss_weight}")

        if self.config.latents_perturbation > 0:
            self.logger.info(f"Using latents perturbation: {self.config.latents_perturbation}")
        else:
            self.logger.info("Latents perturbation disabled")

        self.logger.info(f"Dropout: {self.module.config.dropout} Conditioning dropout: {self.config.conditioning_dropout}")

        # sigma schedule / distribution for train batches
        sigma_sampler_config = SigmaSamplerConfig(
            sigma_max=self.config.sigma_override_max or self.module.config.ddec_sigma_max,
            sigma_min=self.config.sigma_override_min or self.module.config.ddec_sigma_min,
            sigma_data=self.module.config.ddec_sigma_data,
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
            sigma_max=self.config.validation_sigma_override_max or self.module.config.ddec_sigma_max,
            sigma_min=self.config.validation_sigma_override_min or self.module.config.ddec_sigma_min,
            sigma_data=self.module.config.ddec_sigma_data,
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

        self.saved_debug_latents = 0

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
            ln_sigma_error = self.module.ddec_logvar_linear(
                self.module.ddec_logvar_fourier(ln_sigma/4)).float().flatten().detach()
            pdf_warmup_factor = min(1, self.trainer.global_step / (self.config.sigma_pdf_warmup_steps or 1))
            sigma_distribution_pdf = (-pdf_warmup_factor * self.config.sigma_dist_scale * ln_sigma_error).exp()
            self.sigma_sampler.update_pdf(sigma_distribution_pdf)
        
        # sample whole-batch sigma and sync across all ranks / processes
        self.global_sigma = sigma_sampler.sample(total_batch_size, device=self.trainer.accelerator.device)
        self.global_sigma = self.trainer.accelerator.gather(self.global_sigma.unsqueeze(0))[0]

    def train_batch(self, batch: dict) -> dict[str, torch.Tensor]:

        samples = self.format.raw_to_sample(batch["audio"]).detach().clone()
        sample_audio_embeddings = normalize(batch["audio_embeddings"])
        vae_embeddings = self.module.get_embeddings(sample_audio_embeddings)
        
        if self.is_validation_batch == False:
            device_batch_size = self.trainer.config.device_batch_size
            #assert latents.shape == self.trainer.latent_shape, f"Expected shape {self.trainer.latent_shape}, got {latents.shape}"
            #assert samples.shape == self.trainer.sample_shape, f"Expected shape {self.trainer.sample_shape}, got {samples.shape}"
        else:
            device_batch_size = self.trainer.config.validation_device_batch_size
            #assert latents.shape == self.trainer.validation_latent_shape, f"Expected shape {self.trainer.validation_latent_shape}, got {latents.shape}"
            #assert samples.shape == self.trainer.validation_sample_shape, f"Expected shape {self.trainer.validation_sample_shape}, got {samples.shape}"

        conditioning_mask = (torch.rand(device_batch_size,
            generator=self.device_generator, device=self.trainer.accelerator.device) > self.config.conditioning_dropout).float()
        ddec_embeddings = self.module.get_ddec_embeddings(sample_audio_embeddings, conditioning_mask)

        # get the noise level for this sub-batch from the pre-calculated whole-batch sigma (required for stratified sampling)
        local_sigma = self.global_sigma[self.trainer.accelerator.local_process_index::self.trainer.accelerator.num_processes]
        batch_sigma = local_sigma[self.trainer.accum_step * device_batch_size:(self.trainer.accum_step+1) * device_batch_size]

        # prepare model inputs
        noise = torch.randn(samples.shape, device=samples.device, generator=self.device_generator)
        noise = (noise * batch_sigma.view(-1, 1, 1, 1)).detach()

        enc_states, dec_states, denoised = self.module(samples, vae_embeddings,
            self.config.latents_perturbation, samples + noise, batch_sigma, self.format, ddec_embeddings)
        
        batch_loss_weight = (batch_sigma ** 2 + self.module.config.ddec_sigma_data ** 2) / (batch_sigma * self.module.config.ddec_sigma_data) ** 2
        batch_weighted_loss = torch.nn.functional.mse_loss(denoised, samples, reduction="none").mean(dim=(1,2,3)) * batch_loss_weight

        if self.is_validation_batch == True:
            batch_loss = batch_weighted_loss
        else: # for train batches this takes the loss as the gaussian NLL with sigma-specific learned variance
            error_logvar = self.module.get_sigma_loss_logvar(sigma=batch_sigma)
            batch_loss = batch_weighted_loss / error_logvar.exp() + error_logvar
        
        # log loss bucketed by noise level range
        if self.config.num_loss_buckets > 0:
            global_weighted_loss = self.trainer.accelerator.gather(batch_weighted_loss.detach()).cpu()
            global_sigma_quantiles = 1 - self.trainer.accelerator.gather(self.module.config.ddec_sigma_data / batch_sigma.detach() * 2).cpu().arctan() / (torch.pi/2)

            target_buckets = (global_sigma_quantiles * self.unet_loss_buckets.shape[0]).long().clip(min=0, max=self.unet_loss_buckets.shape[0] - 1)
            self.unet_loss_buckets.index_add_(0, target_buckets, global_weighted_loss)
            self.unet_loss_bucket_counts.index_add_(0, target_buckets, torch.ones_like(global_weighted_loss))

        # in our first train batch save the input latents to debug image files
        latents: torch.Tensor = enc_states[-1][1]
        if self.trainer.accelerator.is_main_process == True and self.saved_debug_latents < 10:
            try:
                latents_debug_img_path = None
                if config.DEBUG_PATH is not None:
                    for i in range(samples.shape[0]):
                        latents_debug_img_path = os.path.join(config.DEBUG_PATH, "dae_trainer", f"latents_{self.saved_debug_latents+i}.png")
                        save_img(self.module.latents_to_img(tensor_5d_to_4d(latents[i:i+1])), latents_debug_img_path)
                        sample_c_debug_img_path = os.path.join(config.DEBUG_PATH, "dae_trainer", f"sample_c_{self.saved_debug_latents+i}.png")
                        save_img(self.module.latents_to_img(tensor_5d_to_4d(dec_states[-1][1][i:i+1])), sample_c_debug_img_path)
                        sample_debug_img_path = os.path.join(config.DEBUG_PATH, "dae_trainer", f"sample_{self.saved_debug_latents+i}.png")
                        save_img(self.module.latents_to_img(samples[i:i+1]), sample_debug_img_path)

            except Exception as e:
                self.logger.error("".join(format_exception(type(e), e, e.__traceback__)))
                self.logger.error(f"Error saving debug lantents to '{latents_debug_img_path}': {e}")

            self.saved_debug_latents += latents.shape[0]

        # hidden state encoder/decoder kl loss
        output_states = [state[1] for state in enc_states + dec_states[:-1]]
        kl_loss = torch.zeros(samples.shape[0], device=self.trainer.accelerator.device)
        for state in output_states:

            loss_weight = 1 if state is latents else 1/len(output_states)

            state_var = state.var(dim=1).clip(min=0.1)
            state_mean = state.mean(dim=1)
            kl_loss = kl_loss + (state_mean.square() + state_var - 1 - state_var.log()).mean(dim=(1,2,3)) * (loss_weight / 2)

            state_var = state.var(dim=(2,3,4)).clip(min=0.1)
            state_mean = state.mean(dim=(2,3,4))
            kl_loss = kl_loss + (state_mean.square() + state_var - 1 - state_var.log()).mean(dim=1) * (loss_weight / 2)

        kl_loss_weight = self.config.kl_loss_weight
        if self.config.kl_loss_weight_warmup_steps is not None:
            kl_loss_weight *= min(self.trainer.global_step / self.config.kl_loss_weight_warmup_steps, 1)

        return {
            "loss": kl_loss * kl_loss_weight + batch_loss,
            "loss/ddec": batch_loss.mean().detach(),
            "loss/kl": kl_loss.mean().detach(),
            "latents/mean": latents.mean().detach(),
            "latents/std": latents.std().detach(),
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