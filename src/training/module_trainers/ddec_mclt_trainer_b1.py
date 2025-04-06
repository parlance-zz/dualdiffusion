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
from modules.unets.unet_edm2_ddec_mclt_b1 import DDec_MCLT_UNet_B1
from modules.formats.spectrogram import SpectrogramFormat
from modules.formats.mclt import DualMCLTFormatConfig, DualMCLTFormat
from modules.daes.dae_edm2_d3 import DAE_D3
from modules.mp_tools import normalize
#from modules.formats.frequency_scale import _hz_to_mel, _mel_to_hz
from utils.dual_diffusion_utils import dict_str


@dataclass
class DiffusionDecoder_MCLT_Trainer_B1_Config(ModuleTrainerConfig):

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

    num_loss_buckets: int = 12
    loss_buckets_sigma_min: float = 0.0005
    loss_buckets_sigma_max: float = 100

    latents_perturbation: float = 0.01
    conditioning_dropout: float = 0.1
    noise_level_bias: bool = False
    expected_sample_std: float = 1

class DiffusionDecoder_MCLT_Trainer_B1(ModuleTrainer):
    
    @torch.no_grad()
    def __init__(self, config: DiffusionDecoder_MCLT_Trainer_B1_Config, trainer: DualDiffusionTrainer) -> None:

        self.config = config
        self.trainer = trainer
        self.logger = trainer.logger
        self.module: DDec_MCLT_UNet_B1 = trainer.modules[0]
        assert len(trainer.modules) == 1

        self.is_validation_batch = False
        self.device_generator = None
        self.cpu_generator = None

        self.format: SpectrogramFormat = trainer.pipeline.format.to(self.trainer.accelerator.device)
        self.mclt: DualMCLTFormat = DualMCLTFormat(DualMCLTFormatConfig()).to(self.trainer.accelerator.device)
        self.dae: DAE_D3 = trainer.pipeline.dae.to(
            device=self.trainer.accelerator.device, dtype=trainer.mixed_precision_dtype).requires_grad_(False).eval()

        if trainer.config.enable_channels_last == True:
            self.dae = self.dae.to(memory_format=torch.channels_last_3d)

        if trainer.config.enable_model_compilation:
            self.module.compile(**trainer.config.compile_params)
            self.format.compile(**trainer.config.compile_params)
            self.mclt.compile(**trainer.config.compile_params)
            self.dae.compile(**trainer.config.compile_params)

        if self.dae.config.last_global_step == 0:
            self.logger.error("DAE model has not been trained, aborting training..."); exit(1)

        if self.config.noise_level_bias == True:
            self.logger.info(f"Noise level bias enabled - expected sample std: {self.config.expected_sample_std}")

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
        else:
            device_batch_size = self.trainer.config.validation_device_batch_size

        if "audio_embeddings" in batch:
            sample_audio_embeddings = normalize(batch["audio_embeddings"])

            conditioning_mask = (torch.rand(device_batch_size, generator=self.device_generator,
                device=self.trainer.accelerator.device) > self.config.conditioning_dropout).float()
            unet_class_embeddings = self.module.get_embeddings(sample_audio_embeddings, conditioning_mask)

            dae_class_embeddings = self.dae.get_embeddings(sample_audio_embeddings.to(dtype=self.dae.dtype))
        else:
            unet_class_embeddings = None
            dae_class_embeddings = None
        
        raw_samples: torch.Tensor = batch["audio"].clone()

        mel_spec = self.format.raw_to_sample(raw_samples)
        _, dae_mel_spec, _ = self.dae(mel_spec.to(dtype=self.dae.dtype),
            dae_class_embeddings, add_latents_noise=self.config.latents_perturbation)

        mclt_samples = self.mclt.raw_to_sample(raw_samples, random_phase_augmentation=True).detach()
        ref_samples = self.format.convert_to_unscaled_psd(dae_mel_spec.float()).detach()

        # get the noise level for this sub-batch from the pre-calculated whole-batch sigma (required for stratified sampling)
        local_sigma = self.global_sigma[self.trainer.accelerator.local_process_index::self.trainer.accelerator.num_processes]
        batch_sigma = local_sigma[self.trainer.accum_step * device_batch_size:(self.trainer.accum_step+1) * device_batch_size]

        # prepare model inputs
        noise = torch.randn(mclt_samples.shape, device=mclt_samples.device, generator=self.device_generator)
        mclt_samples = mclt_samples / self.module.mel_density.squeeze(0)
        #quantiles = self.sigma_sampler._sample_uniform_stratified(device_batch_size)
        #quantiles = _mel_to_hz(_hz_to_mel(self.format.config.sample_rate / 2) * quantiles)
        #freq_indices = (quantiles / quantiles.amax() * (mclt_samples.shape[2] - 1e-4)).to(dtype=torch.int32)
        #freq_indices = (quantiles * (mclt_samples.shape[2] - 1e-4)).to(dtype=torch.int32)
        #freq_indices = (quantiles * (mclt_samples.shape[3] - 1e-4)).to(dtype=torch.int32)
        #freq_indices = (quantiles * (mclt_samples.shape[3] + mclt_samples.shape[2] - 1e-4)).to(dtype=torch.int32)
        
        #freq_stds = mclt_samples.std(dim=(1,3))
        #freq_stds = mclt_samples.std(dim=(1,2))
        #freq_stds = torch.cat((mclt_samples.std(dim=(1,3)), mclt_samples.std(dim=(1,2))), dim=1)
        #b = torch.arange(0, mclt_samples.shape[0], device=mclt_samples.device)
        #batch_sigma = (freq_stds[b, freq_indices]).clip(min=self.sigma_sampler.config.sigma_min,
        #                                                max=self.sigma_sampler.config.sigma_max)
        
        #if self.config.noise_level_bias == True: # bias the noise level a bit by the std of each input sample
        #    batch_sigma = batch_sigma * mclt_samples.std(dim=(1,2,3)) / self.config.expected_sample_std
        noise = (noise * batch_sigma.view(-1, 1, 1, 1)).detach()

        denoised: torch.Tensor = self.module(mclt_samples + noise, batch_sigma, self.format, unet_class_embeddings, ref_samples)
        batch_loss_weight = (batch_sigma ** 2 + self.module.config.sigma_data ** 2) / (batch_sigma * self.module.config.sigma_data) ** 2
        batch_weighted_loss = torch.nn.functional.mse_loss(denoised, mclt_samples, reduction="none").mean(dim=(1,2,3)) * batch_loss_weight
        
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
                "std/input_samples": mclt_samples.std(dim=(1,2,3)),
                "std/ref_samples": ref_samples.square().mean(dim=(1,2,3)).sqrt(),
                "std/output_samples": denoised.std(dim=(1,2,3))}

    @torch.no_grad()
    def finish_batch(self) -> dict[str, torch.Tensor]:
        logs = {}

        # added sigma-bucketed loss to logs
        if self.config.num_loss_buckets > 0:
            for i in range(self.config.num_loss_buckets):
                if self.unet_loss_bucket_counts[i].item() > 0:
                    logs[self.bucket_names[i]] = (self.unet_loss_buckets[i] / self.unet_loss_bucket_counts[i]).item()

        return logs