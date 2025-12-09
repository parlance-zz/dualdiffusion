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
from typing import Literal, Optional, Union

import torch
import numpy as np

from training.sigma_sampler import SigmaSamplerConfig, SigmaSampler
from training.trainer import DualDiffusionTrainer
from training.module_trainers.module_trainer import ModuleTrainerConfig, ModuleTrainer
from modules.unets.unet import DualDiffusionUNet
from utils.dual_diffusion_utils import dict_str


@dataclass
class UNetTrainerConfig(ModuleTrainerConfig):

    sigma_distribution: Literal["ln_normal", "ln_sech", "ln_sech^2", "ln_linear", "ln_pdf"] = "ln_sech"
    sigma_override_max: Optional[float] = None
    sigma_override_min: Optional[float] = None
    sigma_dist_scale: float = 1.
    sigma_dist_offset: float = 0
    use_stratified_sigma_sampling: bool = True
    sigma_pdf_resolution: Optional[int] = 127
    sigma_pdf_sanitization: bool = True
    sigma_pdf_warmup_steps: int = 1000
    sigma_pdf_offset: float = -0.8
    sigma_pdf_min: float = 0.2

    num_loss_buckets: int = 12
    loss_buckets_sigma_max: float = 200
    loss_buckets_sigma_min: float = 0.005
    
    input_perturbation: float   = 0.1
    conditioning_dropout: float = 0.1

class UNetTrainer(ModuleTrainer):
    
    @torch.no_grad()
    def __init__(self, config: UNetTrainerConfig, trainer: DualDiffusionTrainer, unet: DualDiffusionUNet, flavor: str) -> None:

        self.config = config
        self.trainer = trainer
        self.logger = trainer.logger
        self.unet = unet
        self.flavor = flavor

        if trainer.config.enable_model_compilation == True:
            self.unet.compile(**trainer.config.compile_params)
    
        if self.config.num_loss_buckets > 0: # buckets for sigma-range-specific loss tracking
            self.logger.info(f"Using {self.config.num_loss_buckets} loss buckets")
            self.unet_loss_buckets = torch.zeros(
                self.config.num_loss_buckets, device="cpu", dtype=torch.float32)
            self.unet_loss_bucket_counts = torch.zeros(
                self.config.num_loss_buckets, device="cpu", dtype=torch.float32)
            
            bucket_sigma = torch.linspace(np.log(self.config.loss_buckets_sigma_min),
                np.log(self.config.loss_buckets_sigma_max), self.config.num_loss_buckets + 1).exp()
            bucket_sigma[0] = 0; bucket_sigma[-1] = float("inf")

            self.bucket_names = [f"{flavor}_loss_σ_buckets/{bucket_sigma[i]:.4f} - {bucket_sigma[i+1]:.4f}"
                                 for i in range(self.config.num_loss_buckets)]
        else:
            self.logger.info("UNet loss buckets are disabled")

        # log unet trainer specific config / settings
        if self.config.input_perturbation > 0:
            self.logger.info(f"Using input perturbation: {self.config.input_perturbation}")
        else:
            self.logger.info("Input perturbation is disabled")
        
        self.logger.info(f"Conditioning dropout: {self.config.conditioning_dropout}")

        # sigma schedule / distribution for train batches
        sigma_sampler_config = SigmaSamplerConfig(
            sigma_max=self.config.sigma_override_max or self.unet.config.sigma_max,
            sigma_min=self.config.sigma_override_min or self.unet.config.sigma_min,
            sigma_data=self.unet.config.sigma_data,
            distribution=self.config.sigma_distribution,
            dist_scale=self.config.sigma_dist_scale,
            dist_offset=self.config.sigma_dist_offset,
            use_stratified_sigma_sampling=self.config.use_stratified_sigma_sampling,
            sigma_pdf_resolution=self.config.sigma_pdf_resolution,
            sigma_pdf_sanitization=self.config.sigma_pdf_sanitization,
            sigma_pdf_warmup_steps=self.config.sigma_pdf_warmup_steps,
            sigma_pdf_offset=self.config.sigma_pdf_offset,
            sigma_pdf_min=self.config.sigma_pdf_min
        )
        self.sigma_sampler = SigmaSampler(sigma_sampler_config)
        self.logger.info("SigmaSampler config:")
        self.logger.info(dict_str(sigma_sampler_config.__dict__))
            
    @torch.no_grad()
    def init_batch(self, validation: bool = False) -> Optional[dict[str, Union[torch.Tensor, float]]]:
        
        assert validation == False

        total_batch_size = self.trainer.total_batch_size
        sigma_sampler = self.sigma_sampler

        # reset sigma-bucketed loss for new batch
        if self.config.num_loss_buckets > 0:
            self.unet_loss_buckets.zero_()
            self.unet_loss_bucket_counts.zero_()

        # if using dynamic sigma sampling with ln_pdf, update the pdf using the learned per-sigma error estimate
        if self.config.sigma_distribution == "ln_pdf":
            self.sigma_sampler.update_pdf_from_logvar(self.unet, self.trainer.global_step)
        
        # sample whole-batch sigma and sync across all ranks / processes
        self.global_sigma = sigma_sampler.sample(total_batch_size, device=self.trainer.accelerator.device)
        self.global_sigma = self.trainer.accelerator.gather(self.global_sigma.unsqueeze(0))[0]

        return None

    def train_batch(self, samples: torch.Tensor, embeddings: Optional[Union[torch.Tensor, list[torch.Tensor]]] = None,
            ref_samples: Optional[torch.Tensor] = None, loss_weight: Optional[torch.Tensor] = None, noise: Optional[torch.Tensor] = None) -> dict[str, Union[torch.Tensor, float]]:

        device_bsz = self.trainer.config.device_batch_size

        # normal conditioning dropout
        conditioning_mask = (torch.rand(device_bsz, device=self.trainer.accelerator.device) > self.config.conditioning_dropout).requires_grad_(False).detach()
        unet_embeddings = self.unet.get_embeddings(embeddings, conditioning_mask)
        
        # get the noise level for this sub-batch from the pre-calculated whole-batch sigma (required for stratified sampling)
        local_sigma = self.global_sigma[self.trainer.accelerator.local_process_index::self.trainer.accelerator.num_processes]
        batch_sigma = local_sigma[self.trainer.accum_step * device_bsz:(self.trainer.accum_step+1) * device_bsz]

        # prepare model inputs
        samples = samples.detach()
        if noise is None:
            noise = torch.randn(samples.shape, device=samples.device)
        noise = (noise * batch_sigma.view(-1, 1, 1, 1)).detach()

        if self.config.input_perturbation > 0:
            input_perturbation = torch.randn(samples.shape, device=samples.device)
            perturbed_input = samples + noise + input_perturbation * batch_sigma.view(-1, 1, 1, 1) * self.config.input_perturbation
        else:
            perturbed_input = None

        denoised: torch.Tensor = self.unet(samples + noise, batch_sigma, None, unet_embeddings, ref_samples, perturbed_input)
        
        sigma_data = self.sigma_sampler.config.sigma_data
        batch_loss_weight = (batch_sigma ** 2 + sigma_data ** 2) / (batch_sigma * sigma_data) ** 2
        batch_weighted_loss = torch.nn.functional.mse_loss(denoised, samples, reduction="none")
        if loss_weight is not None: # use custom loss weight if provided
            batch_weighted_loss = batch_weighted_loss * loss_weight
        batch_weighted_loss = batch_weighted_loss.mean(dim=(1,2,3)) * batch_loss_weight

        error_logvar = self.unet.get_sigma_loss_logvar(sigma=batch_sigma)
        batch_loss = batch_weighted_loss / error_logvar.exp() + error_logvar
        
        if self.config.num_loss_buckets > 0: # log loss bucketed by noise level range

            global_weighted_loss = self.trainer.accelerator.gather(batch_weighted_loss.detach()).cpu()
            global_sigma_quantiles = (batch_sigma.detach().log().cpu() - np.log(self.config.loss_buckets_sigma_min)) / (
                np.log(self.config.loss_buckets_sigma_max) - np.log(self.config.loss_buckets_sigma_min))
            
            target_buckets = (global_sigma_quantiles * self.unet_loss_buckets.shape[0]).long().clip(min=0, max=self.unet_loss_buckets.shape[0] - 1)
            self.unet_loss_buckets.index_add_(0, target_buckets, global_weighted_loss)
            self.unet_loss_bucket_counts.index_add_(0, target_buckets, torch.ones_like(global_weighted_loss))

        return {
            f"loss/{self.flavor}": batch_loss,
            f"io_stats_{self.flavor}/denoised_var": denoised.var(dim=(1,2,3)),
            f"io_stats_{self.flavor}/denoised_mean": denoised.mean(dim=(1,2,3))
        }
    
    @torch.no_grad()
    def finish_batch(self) -> Optional[dict[str, Union[torch.Tensor, float]]]:
        logs = {}

        # added sigma-bucketed loss to logs
        if self.config.num_loss_buckets > 0:
            for i in range(self.config.num_loss_buckets):
                if self.unet_loss_bucket_counts[i].item() > 0:
                    logs[self.bucket_names[i]] = (self.unet_loss_buckets[i] / self.unet_loss_bucket_counts[i]).item()

        return logs