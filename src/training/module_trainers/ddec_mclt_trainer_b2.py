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
from modules.daes.dae_edm2_g1 import DAE_G1
from modules.unets.unet_edm2_ddec_mclt_b2 import DDec_MCLT_UNet_B2
from modules.formats.spectrogram import SpectrogramFormat
from modules.formats.mclt import DualMCLTFormatConfig, DualMCLTFormat

from modules.mp_tools import normalize
from utils.dual_diffusion_utils import dict_str


@dataclass
class MSSLoss2DConfig:

    block_widths: tuple[int] = (8, 16, 32, 64)
    block_overlap: int = 8
    loss_scale: float = 3e-7

class MSSLoss2D:

    @torch.no_grad()
    def __init__(self, config: MSSLoss2DConfig, device: torch.device) -> None:

        self.config = config
        self.device = device

        self.steps = []
        self.windows = []
        self.loss_weights = []
        
        for block_width in self.config.block_widths:

            self.steps.append(max(block_width // self.config.block_overlap, 1))
            window = self.get_flat_top_window_2d(block_width)
            window /= window.square().mean().sqrt()
            self.windows.append(window.to(device=device).requires_grad_(False).detach())

            blockfreq_y = torch.fft.fftfreq(block_width, 1/block_width, device=device)
            blockfreq_x = torch.arange(block_width//2 + 1, device=device)
            wavelength = 1 / ((blockfreq_y.square().view(-1, 1) + blockfreq_x.square().view(1, -1)).sqrt() + 1)
            loss_weight = (1 / wavelength * wavelength.amin()) * block_width**2
            self.loss_weights.append(loss_weight.requires_grad_(False).detach())

    @torch.no_grad()
    def _flat_top_window(self, x: torch.Tensor) -> torch.Tensor:
        return (0.21557895 - 0.41663158 * torch.cos(x) + 0.277263158 * torch.cos(2*x)
                - 0.083578947 * torch.cos(3*x) + 0.006947368 * torch.cos(4*x))

    @torch.no_grad()
    def get_flat_top_window_2d(self, block_width: int) -> torch.Tensor:
        wx = (torch.arange(block_width) + 0.5) / block_width * 2 * torch.pi
        return self._flat_top_window(wx.view(1, 1,-1, 1)) * self._flat_top_window(wx.view(1, 1, 1,-1))
    
    def stft2d(self, x: torch.Tensor, block_width: int,
               step: int, window: torch.Tensor) -> torch.Tensor:
        
        padding = block_width // 2
        x = torch.nn.functional.pad(x, (padding, padding, padding, padding), mode="reflect")
        x = x.unfold(2, block_width, step).unfold(3, block_width, step)

        x = torch.fft.rfft2(x * window, norm="ortho")

        #if x.shape[1] == 2: # mid-side t-form
            #x = torch.stack((x[:, 0] + x[:, 1], x[:, 0] - x[:, 1]), dim=1)
            #x = torch.cat((x, (x[:, 0:1] + x[:, 1:2])*0.5**0.5, (x[:, 0:1] - x[:, 1:2])*0.5**0.5), dim=1)

        return x
    
    def mss_loss(self, sample: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        loss = torch.zeros(target.shape[0], device=self.device)

        for i, block_width in enumerate(self.config.block_widths):
            
            step = self.steps[i]
            window = self.windows[i]
            loss_weight = self.loss_weights[i]

            with torch.no_grad():
                target_fft = self.stft2d(target, block_width, step, window)
                target_fft_abs = target_fft.abs().requires_grad_(False).detach()

            sample_fft = self.stft2d(sample, block_width, step, window)
            sample_fft_abs = sample_fft.abs()
            
            l1_loss = torch.nn.functional.l1_loss(sample_fft_abs.float(), target_fft_abs.float(), reduction="none")
            loss = loss + (l1_loss * loss_weight).mean(dim=(1,2,3,4,5))
   
        return loss * self.config.loss_scale

    def compile(self, **kwargs) -> None:
        self.mss_loss = torch.compile(self.mss_loss, **kwargs)

@dataclass
class DiffusionDecoder_MCLT_Trainer_B2_Config(ModuleTrainerConfig):

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

    latents_perturbation: float = 0.1
    conditioning_dropout: float = 0.1

    kl_loss_weight: float = 2e-3
    kl_warmup_steps: int  = 5000
    use_random_crop: bool = True

class DiffusionDecoder_MCLT_Trainer_B2(ModuleTrainer):
    
    @torch.no_grad()
    def __init__(self, config: DiffusionDecoder_MCLT_Trainer_B2_Config, trainer: DualDiffusionTrainer) -> None:

        self.config = config
        self.trainer = trainer
        self.logger = trainer.logger
        self.ddec: DDec_MCLT_UNet_B2 = trainer.modules[trainer.config.train_modules.index("ddec")]
        self.dae: DAE_G1 = trainer.modules[trainer.config.train_modules.index("dae")]

        self.is_validation_batch = False
        self.device_generator = None
        self.cpu_generator = None

        self.format: SpectrogramFormat = trainer.pipeline.format.to(self.trainer.accelerator.device)
        self.mclt: DualMCLTFormat = DualMCLTFormat(DualMCLTFormatConfig()).to(self.trainer.accelerator.device)
        #self.mss_loss = MSSLoss2D(MSSLoss2DConfig(), device=trainer.accelerator.device)

        if trainer.config.enable_model_compilation:
            self.ddec.compile(**trainer.config.compile_params)
            self.dae.compile(**trainer.config.compile_params)
            self.format.compile(**trainer.config.compile_params)
            self.mclt.compile(**trainer.config.compile_params)
            #self.mss_loss.compile(**trainer.config.compile_params)

            self.random_crop = torch.compile(self.random_crop, **trainer.config.compile_params)

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
        self.logger.info(f"Random sub-pixel latent crop: {self.config.use_random_crop}")

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

    def random_crop(self, tensors: list[torch.Tensor], psd_freqs_per_freq: int) -> list[torch.Tensor]:
        b, c, h, w = tensors[1].shape
        new_h, new_w = h - 8, w - 8
        
        # generate random offsets for each sample in the batch
        tops = torch.randint(0, h - new_h, (b,), device=tensors[1].device)
        lefts = torch.randint(0, w - new_w, (b,), device=tensors[1].device)
        
        # create a meshgrid of batch indices
        batch_indices = torch.arange(b, device=tensors[1].device).view(b, 1, 1, 1).expand(-1, c, new_h, new_w)
        channel_indices = torch.arange(c, device=tensors[1].device).view(1, c, 1, 1).expand(b, -1, new_h, new_w)

        # create a meshgrid of height and width indices based on offsets
        height_indices = torch.arange(new_h, device=tensors[1].device).view(1, 1, new_h, 1) + tops.view(b, 1, 1, 1)
        width_indices = torch.arange(new_w, device=tensors[1].device).view(1, 1, 1, new_w) + lefts.view(b, 1, 1, 1)
        
        # expand height and width indices to match batch dimension
        height_indices = height_indices.expand(b, c, -1, new_w)
        width_indices = width_indices.expand(b, c, new_h, -1)
        
        # return the cropped tensors using the indices
        cropped_tensors = []
        for i in range (1, len(tensors)):
            cropped_tensor = tensors[i][batch_indices, channel_indices, height_indices, width_indices]
            cropped_tensors.append(cropped_tensor)

        # create a meshgrid of batch indices
        new_h *= psd_freqs_per_freq
        tops *= psd_freqs_per_freq

        batch_indices = torch.arange(b, device=tensors[0].device).view(b, 1, 1, 1).expand(-1, c, new_h, new_w)
        channel_indices = torch.arange(c, device=tensors[0].device).view(1, c, 1, 1).expand(b, -1, new_h, new_w)

        height_indices = torch.arange(new_h, device=tensors[0].device).view(1, 1, new_h, 1) + tops.view(b, 1, 1, 1)
        width_indices = torch.arange(new_w, device=tensors[0].device).view(1, 1, 1, new_w) + lefts.view(b, 1, 1, 1)
        height_indices = height_indices.expand(b, c, -1, new_w)
        width_indices = width_indices.expand(b, c, new_h, -1)

        return [tensors[0][batch_indices, channel_indices, height_indices, width_indices]] + cropped_tensors

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
        
        raw_samples: torch.Tensor = batch["audio"].clone()

        mel_spec = self.format.raw_to_sample(raw_samples).clone().detach()
        latents, recon_mel_spec, pre_norm_latents = self.dae(mel_spec,
            dae_class_embeddings, add_latents_noise=self.config.latents_perturbation)

        # prepare model inputs
        mclt_samples: torch.Tensor = self.mclt.raw_to_sample(raw_samples, random_phase_augmentation=True).detach()
        mclt_samples = mclt_samples / self.ddec.mel_density.squeeze(0)

        # get the noise level for this sub-batch from the pre-calculated whole-batch sigma (required for stratified sampling)
        local_sigma = self.global_sigma[self.trainer.accelerator.local_process_index::self.trainer.accelerator.num_processes]
        batch_sigma = local_sigma[self.trainer.accum_step * device_batch_size:(self.trainer.accum_step+1) * device_batch_size]

        noise = torch.randn(mclt_samples.shape, device=mclt_samples.device, generator=self.device_generator)

        ref_samples = self.format.convert_to_unscaled_psd(recon_mel_spec.float())
        noise = (noise * batch_sigma.view(-1, 1, 1, 1)).detach()

        if self.config.use_random_crop == True:
            ref_samples, mclt_samples, noise = self.random_crop(
                [ref_samples, mclt_samples, noise], self.ddec.psd_freqs_per_freq)

        denoised: torch.Tensor = self.ddec(mclt_samples.detach() + noise.detach(), batch_sigma, self.format, unet_class_embeddings, ref_samples)
        batch_loss_weight = (batch_sigma ** 2 + self.ddec.config.sigma_data ** 2) / (batch_sigma * self.ddec.config.sigma_data) ** 2
        batch_weighted_loss = torch.nn.functional.mse_loss(denoised, mclt_samples, reduction="none").mean(dim=(1,2,3)) * batch_loss_weight

        #mss_weighted_loss = self.mss_loss.mss_loss(denoised, mclt_samples) * batch_loss_weight
        
        if self.is_validation_batch == True:
            batch_loss = batch_weighted_loss
        else: # for train batches this takes the loss as the gaussian NLL with sigma-specific learned variance
            error_logvar = self.ddec.get_sigma_loss_logvar(sigma=batch_sigma)
            batch_loss = batch_weighted_loss / error_logvar.exp() + error_logvar
            #batch_loss = batch_loss * self.config.mse_loss_weight + mss_weighted_loss * self.config.mss_loss_weight

        # log loss bucketed by noise level range
        if self.config.num_loss_buckets > 0:
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
                "io_stats/mel_spec_std": mel_spec.std(dim=(1,2,3)),
                "io_stats/mel_spec_mean": mel_spec.mean(dim=(1,2,3)),
                "io_stats/mclt_spec_std": mclt_samples.std(dim=(1,2,3)),
                "io_stats/mclt_spec_mean": mclt_samples.mean(dim=(1,2,3)),
                "io_stats/recon_mel_spec_std": recon_mel_spec.std(dim=(1,2,3)),
                "io_stats/recon_mel_spec_mean": recon_mel_spec.mean(dim=(1,2,3)),
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