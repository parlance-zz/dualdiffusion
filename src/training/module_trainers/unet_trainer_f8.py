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
from training.loss.mss_1d import MSSLoss1D
from modules.unets.unet_edm2_q7_ddec_f8 import UNet
from modules.formats.ms_mdct_dual_8 import MS_MDCT_DualFormat
from utils.dual_diffusion_utils import dict_str


@dataclass
class UNetTrainerConfig(ModuleTrainerConfig):

    sigma_distribution: Literal["ln_normal", "ln_sech", "ln_sech^2", "ln_linear", "ln_pdf"] = "ln_sech"
    sigma_override_max: Optional[float] = None
    sigma_override_min: Optional[float] = None
    sigma_dist_scale: float = 1.
    sigma_dist_offset: float = 0
    use_stratified_sigma_sampling: bool = True
    use_stratified_sigma_shuffling: bool = False
    sigma_pdf_resolution: Optional[int] = 127
    sigma_pdf_sanitization: bool = True
    sigma_pdf_warmup_steps: int = 1000
    sigma_pdf_offset: float = -0.8
    sigma_pdf_min: float = 0.2

    # for loss logging within bucketed sigma ranges
    num_loss_buckets: int = 12
    loss_buckets_sigma_max: float = 200
    loss_buckets_sigma_min: float = 0.005
    linear_buckets: bool = False
    
    input_perturbation: float   = 0.1 # from https://arxiv.org/pdf/2301.11706
    conditioning_dropout: float = 0.1

    disable_loss_weight: bool = False

class UNetLossBuckets(torch.nn.Module):

    def __init__(self, num_buckets: int, sigma_min: float, sigma_max: float, trainer: DualDiffusionTrainer, log_prefix: str = "ddec", linear_buckets: bool = False) -> None:
        super().__init__()

        self.num_buckets = num_buckets
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.log_prefix = log_prefix
        self.trainer = trainer

        self.loss_buckets: torch.Tensor; self.loss_bucket_counts: torch.Tensor
        self.register_buffer("loss_buckets", torch.zeros(num_buckets, dtype=torch.float32))
        self.register_buffer("loss_bucket_counts", torch.zeros(num_buckets, dtype=torch.float32))

        if linear_buckets == False:
            bucket_sigma = torch.linspace(np.log(self.sigma_min), np.log(self.sigma_max), self.num_buckets + 1).exp()
        else:
            bucket_sigma = torch.linspace(self.sigma_min, self.sigma_max, self.num_buckets + 1)
        bucket_sigma[0] = 0; bucket_sigma[-1] = float("inf")

        self.bucket_names = [f"{log_prefix}_loss_σ_buckets/{bucket_sigma[i]:.4f} - {bucket_sigma[i+1]:.4f}" for i in range(num_buckets)]

    def log_buckets(self, loss: torch.Tensor, sigma: torch.Tensor) -> None:
        
        global_loss = self.trainer.accelerator.gather(loss.detach()).cpu()
        sigma = self.trainer.accelerator.gather(sigma.detach()).cpu()
        sigma_quantiles = (sigma.detach().log().cpu() - np.log(self.sigma_min)) / (np.log(self.sigma_max) - np.log(self.sigma_min))
        
        target_buckets = (sigma_quantiles * self.loss_buckets.shape[0]).long().clip(min=0, max=self.loss_buckets.shape[0] - 1)
        self.loss_buckets.index_add_(0, target_buckets, global_loss)
        self.loss_bucket_counts.index_add_(0, target_buckets, torch.ones_like(global_loss))

    def clear(self) -> None:
        self.loss_buckets.zero_()
        self.loss_bucket_counts.zero_()

    def get_logs(self) -> dict[str, float]:
        logs = {}

        for i in range(self.num_buckets):
            if self.loss_bucket_counts[i].item() > 0:
                logs[self.bucket_names[i]] = (self.loss_buckets[i] / self.loss_bucket_counts[i]).item()

        return logs

class UNetTrainer(ModuleTrainer):
    
    @torch.no_grad()
    def __init__(self, config: UNetTrainerConfig, trainer: DualDiffusionTrainer, unet: UNet, flavor: str, mss_1d: Optional[MSSLoss1D] = None) -> None:

        self.config = config
        self.trainer = trainer
        self.logger = trainer.logger
        self.unet = unet
        self.flavor = flavor
        self.mss_1d = mss_1d
        self.format: MS_MDCT_DualFormat = trainer.pipeline.format.to(trainer.accelerator.device)

        if trainer.config.enable_model_compilation == True:
            self.unet.compile(**trainer.config.compile_params)

        if config.disable_loss_weight == True:
            self.logger.info("Loss weighting is disabled")

        if self.config.num_loss_buckets > 0: # buckets for sigma-range-specific loss tracking
            if config.linear_buckets == True:
                self.logger.info("Using linear loss buckets")
            self.logger.info(f"Using {self.config.num_loss_buckets} loss buckets")
            self.unet_loss_buckets: list[UNetLossBuckets] = []
            
            for i in range(self.format.config.num_mdcts):

                buckets = UNetLossBuckets(
                    num_buckets=self.config.num_loss_buckets,
                    sigma_min=self.config.loss_buckets_sigma_min,
                    sigma_max=self.config.loss_buckets_sigma_max,
                    trainer=trainer,
                    log_prefix=f"{flavor}_{i}",
                    linear_buckets=config.linear_buckets
                )
                self.unet_loss_buckets.append(buckets)

            if mss_1d is not None:
                self.mss1d_loss_buckets = UNetLossBuckets(
                    num_buckets=self.config.num_loss_buckets,
                    sigma_min=self.config.loss_buckets_sigma_min,
                    sigma_max=self.config.loss_buckets_sigma_max,
                    trainer=trainer,
                    log_prefix=f"{flavor}_mss1d",
                    linear_buckets=config.linear_buckets
                )
                self.mss1d_ceptrum_loss_buckets = UNetLossBuckets(
                    num_buckets=self.config.num_loss_buckets,
                    sigma_min=self.config.loss_buckets_sigma_min,
                    sigma_max=self.config.loss_buckets_sigma_max,
                    trainer=trainer,
                    log_prefix=f"{flavor}_mss1d_cepstrum",
                    linear_buckets=config.linear_buckets
                )
                
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
            use_stratified_sigma_shuffling=self.config.use_stratified_sigma_shuffling,
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
            for bucket in self.unet_loss_buckets:
                bucket.clear()

            if self.mss_1d is not None:
                self.mss1d_loss_buckets.clear()
                self.mss1d_ceptrum_loss_buckets.clear()

        # if using dynamic sigma sampling with ln_pdf, update the pdf using the learned per-sigma error estimate
        if self.config.sigma_distribution == "ln_pdf":
            self.sigma_sampler.update_pdf_from_logvar(self.unet, self.trainer.global_step)
        
        # sample whole-batch sigma and sync across all ranks / processes
        self.global_sigma = sigma_sampler.sample(total_batch_size, device=self.trainer.accelerator.device)
        self.global_sigma = self.trainer.accelerator.gather(self.global_sigma.unsqueeze(0))[0]

        return None

    def train_batch(self, samples: torch.Tensor, embeddings: Optional[Union[torch.Tensor, list[torch.Tensor]]] = None,
            ref_samples: Optional[torch.Tensor] = None) -> dict[str, Union[torch.Tensor, float]]:

        device_bsz = self.trainer.config.device_batch_size

        # normal conditioning dropout
        conditioning_mask = (torch.rand(device_bsz, device=self.trainer.accelerator.device) > self.config.conditioning_dropout).requires_grad_(False).detach()
        
        # get the noise level for this sub-batch from the pre-calculated whole-batch sigma (required for stratified sampling)
        local_sigma = self.global_sigma[self.trainer.accelerator.process_index::self.trainer.accelerator.num_processes]
        batch_sigma = local_sigma[self.trainer.accum_step * device_bsz:(self.trainer.accum_step+1) * device_bsz]

        # prepare model inputs
        noise = torch.randn(samples.shape, device=samples.device)
        noise = (noise * batch_sigma.view(-1, 1, 1, 1)).detach()

        if self.config.input_perturbation > 0:
            input_perturbation = torch.randn(samples.shape, device=samples.device)
            perturbed_input = samples + noise + input_perturbation * batch_sigma.view(-1, 1, 1, 1) * self.config.input_perturbation
        else:
            perturbed_input = None

        try: unet_module = self.trainer.get_ddp_module(self.unet)
        except: unet_module = self.unet

        denoised, error_logvar = unet_module(samples + noise, batch_sigma, self.format, embeddings,
            x_ref=ref_samples, perturbed_input=perturbed_input, conditioning_mask=conditioning_mask)
        
        samples = self.format.unflatten_mdct_phase_psd(samples)
        denoised = self.format.unflatten_mdct_phase_psd(denoised)
        mel_density = self.format.get_mdct_mel_density(level=-1)
        error_logvar: torch.Tensor = error_logvar[:, :, 0, 0]
        
        assert len(samples) == len(denoised) == len(mel_density)
        assert error_logvar.ndim == 2 and error_logvar.shape[0] == samples[0].shape[0] and error_logvar.shape[1] == len(samples)

        sigma_data = self.sigma_sampler.config.sigma_data

        if self.config.disable_loss_weight == True:
            batch_loss_weight = 2 / batch_sigma**2
        else:
            batch_loss_weight = (batch_sigma ** 2 + sigma_data ** 2) / (batch_sigma * sigma_data) ** 2
        
        batch_weighted_loss = []
        for i, (x, y, loss_weight) in enumerate(zip(denoised, samples, mel_density)):
            loss_weight = loss_weight / loss_weight.mean()
            loss = (torch.nn.functional.mse_loss(x, y.detach(), reduction="none") * loss_weight).mean(dim=(1,2,3)) * batch_loss_weight
            batch_weighted_loss.append(loss)
        batch_weighted_loss = torch.stack(batch_weighted_loss, dim=1)

        if self.mss_1d is not None:
            
            mss_loss_logs = None

            for i, (_sample, _denoised) in enumerate(zip(samples, denoised)):

                denoised_raw = self.trainer.module_trainer.format.mdct_phase_psd_to_raw(_denoised, level=i)
                target_raw = self.trainer.module_trainer.format.mdct_phase_psd_to_raw(_sample, level=i).detach()
                
                _mss_loss_logs = self.mss_1d.mss_loss(denoised_raw, target_raw)
                if mss_loss_logs is None:
                    mss_loss_logs = _mss_loss_logs
                else:
                    for k in mss_loss_logs.keys():
                        mss_loss_logs[k] = mss_loss_logs[k] + _mss_loss_logs[k]

            #mss_loss_logs["loss/mss1d"] = mss_loss_logs["loss/mss1d"] / batch_sigma.clip(min=0.2)
            #mss_loss_logs["loss/mss1d_cepstrum"] = mss_loss_logs["loss/mss1d_cepstrum"] / batch_sigma.clip(min=0.2)

            #t = 1 / (sigma_data ** 2 + batch_sigma ** 2).pow(0.5)
            #mss_loss_logs = self.mss_1d.mss_loss(denoised_raw, sample_raw, t=t)

            #for k, v in mss_loss_logs.items():
            #    if k.startswith("loss/"):
            #        mss_loss_logs[k] = v * batch_loss_weight

        if self.config.disable_loss_weight == True:
            error_logvar = self.unet.get_sigma_loss_logvar(torch.ones_like(batch_sigma))
            batch_loss = batch_weighted_loss / error_logvar.exp() + error_logvar
            #bucket_log_loss = batch_weighted_loss
            bucket_log_loss = (0.5 * batch_weighted_loss * batch_sigma**2) * (batch_sigma ** 2 + sigma_data ** 2) / (batch_sigma * sigma_data) ** 2
        else:
            batch_loss = batch_weighted_loss / error_logvar.exp() + error_logvar
            bucket_log_loss = batch_weighted_loss
        
        if self.config.num_loss_buckets > 0:
            for i in range(bucket_log_loss.shape[1]):
                self.unet_loss_buckets[i].log_buckets(bucket_log_loss[:, i], batch_sigma)

            if self.mss_1d is not None:
                self.mss1d_loss_buckets.log_buckets(mss_loss_logs["loss/mss1d"], batch_sigma)
                self.mss1d_ceptrum_loss_buckets.log_buckets(mss_loss_logs["loss/mss1d_cepstrum"], batch_sigma)

        #if self.mss_1d is not None:
        #    for k, v in mss_loss_logs.items():
        #        if k.startswith("loss/"):
        #            batch_loss = batch_loss + v / error_logvar.exp() + error_logvar

        logs = {
            f"loss/{self.flavor}": batch_loss.mean(dim=1),
            #f"io_stats_{self.flavor}/denoised_var": denoised.var(dim=(1,2,3)),
            #f"io_stats_{self.flavor}/denoised_mean": denoised.mean(dim=(1,2,3))
        }

        for i in range(batch_loss.shape[1]):
            logs[f"loss/{self.flavor}_{i}"] = batch_loss[:, i]
            
        if self.mss_1d is not None:
            logs.update(mss_loss_logs)

        return logs
    
    @torch.no_grad()
    def finish_batch(self) -> Optional[dict[str, Union[torch.Tensor, float]]]:

        if self.config.num_loss_buckets > 0:
            logs = {}
            for bucket in self.unet_loss_buckets:
                logs.update(bucket.get_logs())

            if self.mss_1d is not None:
                logs.update(self.mss1d_loss_buckets.get_logs())
                logs.update(self.mss1d_ceptrum_loss_buckets.get_logs())

            return logs
        
        return None