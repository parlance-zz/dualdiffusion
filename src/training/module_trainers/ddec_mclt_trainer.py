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
from .module_trainer import ModuleTrainerConfig, ModuleTrainer
from modules.unets.unet_edm2_ddec_mclt import DDec_MCLT_UNet
from modules.formats.spectrogram import SpectrogramFormat
from modules.formats.mclt import DualMCLTFormatConfig, DualMCLTFormat
from modules.mp_tools import normalize, resample_1d, resample_2d
from utils.dual_diffusion_utils import dict_str


def ref_dropout(x_ref: torch.Tensor, threshold: float, generator: torch.Generator) -> torch.Tensor:

    b, c, h, w = x_ref.shape
    rand_values = torch.rand(b, device=x_ref.device, generator=generator)
    mask = (rand_values > threshold).float().view(b, 1, 1, 1)
    avg_tensor = torch.zeros_like(x_ref)#resample_1d(resample_1d(x_ref, "down"), "up")
    
    return torch.lerp(avg_tensor, x_ref, mask)

@dataclass
class DiffusionDecoder_MCLT_TrainerConfig(ModuleTrainerConfig):

    sigma_distribution: Literal["ln_normal", "ln_sech", "ln_sech^2",
                                "ln_linear", "ln_pdf"] = "ln_sech"
    sigma_override_max: Optional[float] = None
    sigma_override_min: Optional[float] = None
    sigma_dist_scale: float = 1.0
    sigma_dist_offset: float = 0
    use_stratified_sigma_sampling: bool = True
    sigma_pdf_resolution: Optional[int] = 127
    sigma_pdf_warmup_steps: Optional[int] = 30000

    validation_sigma_distribution: Literal["ln_normal", "ln_sech", "ln_sech^2",
                                            "ln_linear", "ln_pdf"] = "ln_sech"
    validation_sigma_override_max: Optional[float] = None
    validation_sigma_override_min: Optional[float] = None
    validation_sigma_dist_scale: float = 1.0
    validation_sigma_dist_offset: float = 0

    num_loss_buckets: int = 15
    loss_buckets_sigma_min: float = 0.002
    loss_buckets_sigma_max: float = 150

    latents_perturbation: float = 0.03
    conditioning_dropout: float = 0.1

class DiffusionDecoder_MCLT_Trainer(ModuleTrainer):
    
    @torch.no_grad()
    def __init__(self, config: DiffusionDecoder_MCLT_TrainerConfig, trainer: DualDiffusionTrainer) -> None:

        self.config = config
        self.trainer = trainer
        self.logger = trainer.logger
        self.module: DDec_MCLT_UNet = trainer.module

        self.is_validation_batch = False
        self.device_generator = None
        self.cpu_generator = None

        self.format: SpectrogramFormat = trainer.pipeline.format.to(self.trainer.accelerator.device)
        self.mclt: DualMCLTFormat = DualMCLTFormat(DualMCLTFormatConfig()).to(self.trainer.accelerator.device)
        #self.vae: DualDiffusionVAE = trainer.pipeline.vae.to(
        #    device=self.trainer.accelerator.device, dtype=trainer.mixed_precision_dtype).requires_grad_(False).eval()

        #if trainer.config.enable_channels_last == True:
        #    self.vae = self.vae.to(memory_format=torch.channels_last)
        if trainer.config.enable_model_compilation:
            self.module.compile(**trainer.config.compile_params)
            self.format.compile(**trainer.config.compile_params)
            self.mclt.compile(**trainer.config.compile_params)

        #if self.vae.config.last_global_step == 0:
        #    self.logger.error("VAE model has not been trained, aborting training..."); exit(1) 

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

        self.logger.info(f"Dropout: {self.module.config.dropout} Conditioning dropout: {self.config.conditioning_dropout}")

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

        if self.is_validation_batch == False:
            device_batch_size = self.trainer.config.device_batch_size
            #assert latents.shape == self.trainer.latent_shape, f"Expected shape {self.trainer.latent_shape}, got {latents.shape}"
            #assert samples.shape == self.trainer.sample_shape, f"Expected shape {self.trainer.sample_shape}, got {samples.shape}"
        else:
            device_batch_size = self.trainer.config.validation_device_batch_size
            #assert latents.shape == self.trainer.validation_latent_shape, f"Expected shape {self.trainer.validation_latent_shape}, got {latents.shape}"
            #assert samples.shape == self.trainer.validation_sample_shape, f"Expected shape {self.trainer.validation_sample_shape}, got {samples.shape}"

        if "audio_embeddings" in batch:
            sample_audio_embeddings = normalize(batch["audio_embeddings"])
            conditioning_mask = (torch.rand(device_batch_size, generator=self.device_generator,
                device=self.trainer.accelerator.device) > self.config.conditioning_dropout).float()
            unet_class_embeddings = self.module.get_embeddings(sample_audio_embeddings, conditioning_mask)
        else:
            unet_class_embeddings = None
        
        #vae_emb = self.vae.get_embeddings(sample_audio_embeddings.to(dtype=self.vae.dtype))
        #enc_states, dec_states, sigma = self.vae(samples.to(dtype=self.vae.dtype),
        #                            vae_emb, add_latents_noise=self.config.latents_perturbation)

        #mclt_samples = self.mclt.raw_to_sample(batch["audio"]).clone().detach()
        mclt_samples = self.mclt.raw_to_sample(batch["audio"], random_phase_augmentation=True).clone().detach()
        ref_samples = self.format.convert_to_abs_exp1(self.format.raw_to_sample(batch["audio"])).clone().detach()
        #ref_samples = ref_dropout(ref_samples, self.config.conditioning_dropout, generator=self.device_generator)

        # get the noise level for this sub-batch from the pre-calculated whole-batch sigma (required for stratified sampling)
        local_sigma = self.global_sigma[self.trainer.accelerator.local_process_index::self.trainer.accelerator.num_processes]
        batch_sigma = local_sigma[self.trainer.accum_step * device_batch_size:(self.trainer.accum_step+1) * device_batch_size]

        # prepare model inputs
        noise = torch.randn(mclt_samples.shape, device=mclt_samples.device, generator=self.device_generator)
        mclt_samples = mclt_samples / self.module.mel_density
        noise = (noise * batch_sigma.view(-1, 1, 1, 1)).detach()

        denoised: torch.Tensor = self.module(mclt_samples + noise, batch_sigma, self.format, unet_class_embeddings, ref_samples)
        #denoised = denoised * self.module.mel_density
        #mclt_samples = mclt_samples * self.module.mel_density

        #mclt_psd = (mclt_samples * self.module.mel_density).std(dim=2, keepdim=True).clip(min=0.01)
        #mclt_psd = torch.min(mclt_psd[..., 1:], mclt_psd[..., :-1])
        #mclt_psd = torch.cat((mclt_psd[..., 0:1], mclt_psd), dim=-1)

        #loss_weight = 1 / mclt_psd
        #loss_weight = 1 / (mclt_samples * self.module.mel_density).std(dim=2, keepdim=True).clip(min=0.001)**0.25
        
        #loss_weight = (2 / (mclt_psd ** 2 + mclt_psd.square().mean(dim=2, keepdim=True)) ** 0.5).detach()
        #denoised = denoised * loss_weight
        #mclt_samples = mclt_samples * loss_weight
        batch_loss_weight = (batch_sigma ** 2 + self.module.config.sigma_data ** 2) / (batch_sigma * self.module.config.sigma_data) ** 2
        batch_weighted_loss = torch.nn.functional.mse_loss(denoised, mclt_samples, reduction="none").mean(dim=(1,2,3)) * batch_loss_weight
        
        #_batch_sigma = batch_sigma.view(-1, 1, 1, 1)
        #_batch_sigma_data = mclt_samples.std(dim=(1,3), keepdim=True).clip(min=0.01, max=7)
        #_batch_sigma_data = self.module.config.sigma_data
        #batch_loss_weight = (_batch_sigma ** 2 + _batch_sigma_data ** 2) / (_batch_sigma * _batch_sigma_data) ** 2
        #batch_weighted_loss = mse_loss * batch_loss_weight

        #_batch_sigma_data = mclt_samples.std(dim=2, keepdim=True).clip(min=0.0146, max=6.2)
        #batch_loss_weight = (_batch_sigma ** 2 + _batch_sigma_data ** 2) / (_batch_sigma * _batch_sigma_data) ** 2
        #batch_weighted_loss = (batch_weighted_loss + (mse_loss * batch_loss_weight).mean(dim=(1,2,3))) * 0.5

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

        return {"loss": batch_loss,
                "std/input_samples": mclt_samples.std(),
                "std/ref_samples": ref_samples.std(),
                "std/output_samples": denoised.std()}

    @torch.no_grad()
    def finish_batch(self) -> dict[str, torch.Tensor]:
        logs = {}

        # added sigma-bucketed loss to logs
        if self.config.num_loss_buckets > 0:
            for i in range(self.config.num_loss_buckets):
                if self.unet_loss_bucket_counts[i].item() > 0:
                    logs[self.bucket_names[i]] = (self.unet_loss_buckets[i] / self.unet_loss_bucket_counts[i]).item()

        return logs