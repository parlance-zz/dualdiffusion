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
from typing import Optional, Union

import torch
import numpy as np

from sampling.schedule import SamplingSchedule
from training.trainer import DualDiffusionTrainer
from training.module_trainers.module_trainer import ModuleTrainerConfig, ModuleTrainer
from training.loss.mss_1d import MSSLoss1D
from modules.unets.unet_edm2_q6_ddec import UNet
from modules.formats.ms_mdct_dual_7 import MS_MDCT_DualFormat


@dataclass
class UNetTrainerConfig(ModuleTrainerConfig):

    sigma_schedule: str = "cos"
    sigma_schedule_rho: float = 1.
    sigma_max: float = 100
    sigma_min: float = 0.01
    sigma_data: float = 1.
    num_sampling_steps: int = 7

    use_gradient_checkpointing: bool = True

    # for loss logging within bucketed sigma ranges
    num_loss_buckets: int = 12
    loss_buckets_sigma_max: float = 100
    loss_buckets_sigma_min: float = 0.01
    linear_buckets: bool = False
    
    conditioning_dropout: float = 0.1

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

        assert mss_1d is not None

        self.config = config
        self.trainer = trainer
        self.logger = trainer.logger
        self.unet = unet
        self.flavor = flavor
        self.mss_1d = mss_1d
        self.format: MS_MDCT_DualFormat = trainer.pipeline.format.to(trainer.accelerator.device)

        if trainer.config.enable_model_compilation == True:
            self.unet.compile(**trainer.config.compile_params)

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
                
        else:
            self.logger.info("UNet loss buckets are disabled")

        # log unet trainer specific config / settings
        self.logger.info(f"Use gradient checkpointing: {self.config.use_gradient_checkpointing}")
        self.logger.info(f"Conditioning dropout: {self.config.conditioning_dropout}")

        # sigma schedule / distribution for train batches
        self.logger.info(f"Using sigma schedule: {self.config.sigma_schedule} (rho={self.config.sigma_schedule_rho})")
        self.logger.info(f"Steps: {self.config.num_sampling_steps}  Sigma max: {self.config.sigma_max}  Sigma min: {self.config.sigma_min}")
            
    @torch.no_grad()
    def init_batch(self, validation: bool = False) -> Optional[dict[str, Union[torch.Tensor, float]]]:
        
        assert validation == False

        # reset sigma-bucketed loss for new batch
        if self.config.num_loss_buckets > 0:
            for bucket in self.unet_loss_buckets:
                bucket.clear()

        return None

    def train_batch(self, target_samples: torch.Tensor, embeddings: Optional[Union[torch.Tensor, list[torch.Tensor]]] = None,
                                    ref_samples: Optional[torch.Tensor] = None) -> dict[str, Union[torch.Tensor, float]]:

        device_bsz = self.trainer.config.device_batch_size
        logs = {f"loss/{self.flavor}": torch.zeros(self.trainer.config.device_batch_size, device=self.trainer.accelerator.device)}

        # normal conditioning dropout
        conditioning_mask = (torch.rand(device_bsz, device=self.trainer.accelerator.device) > self.config.conditioning_dropout).requires_grad_(False).detach()
        
        # get the noise level for this sub-batch from the pre-calculated whole-batch sigma (required for stratified sampling)
        sigma_schedule = SamplingSchedule.get_schedule(
            name=self.config.sigma_schedule,
            steps=self.config.num_sampling_steps,
            t_start=1, device=target_samples.device,
            sigma_max=self.config.sigma_max,
            sigma_min=self.config.sigma_min,
            rho=self.config.sigma_schedule_rho,
        ).clone().detach()
        
        # prepare model inputs
        target_samples_unflattened = self.format.unflatten_mdct_phase_psd(target_samples)
        loss_weights = [md / md.mean() for md in self.format.get_mdct_mel_density(level=-1)]

        noise = torch.randn(target_samples.shape, device=target_samples.device)
        noise = (noise * sigma_schedule[0].view(-1, 1, 1, 1)).detach()
        samples = target_samples + noise

        unet_module = self.trainer.get_ddp_module(self.unet)

        def unet_forward(_samples: torch.Tensor, _sigma: torch.Tensor, _embeddings: torch.Tensor, _ref_samples, _conditioning_mask):
            return unet_module(_samples, _sigma, self.format, _embeddings,
                x_ref=_ref_samples, perturbed_input=None, conditioning_mask=_conditioning_mask)

        for i in range(self.config.num_sampling_steps):
            sigma = sigma_schedule[i].expand(samples.shape[0])

            if self.config.use_gradient_checkpointing == True:
                denoised, error_logvar = torch.utils.checkpoint.checkpoint(
                    unet_forward, samples, sigma, embeddings, ref_samples, conditioning_mask, use_reentrant=False)
            else:
                denoised, error_logvar = unet_forward(samples, sigma, embeddings, ref_samples, conditioning_mask)

            logs[f"io_stats/{self.flavor}_denoised_std_{i}"] = denoised.std()

            batch_loss_weight = (sigma ** 2 + self.config.sigma_data ** 2) / (sigma * self.config.sigma_data) ** 2
            batch_weighted_loss = []
            for x, y, loss_weight in zip(self.format.unflatten_mdct_phase_psd(denoised), target_samples_unflattened, loss_weights):
                loss = (torch.nn.functional.mse_loss(x, y.detach(), reduction="none") * loss_weight).mean(dim=(1,2,3)) * batch_loss_weight
                batch_weighted_loss.append(loss)
            
            batch_weighted_loss = torch.stack(batch_weighted_loss, dim=1)
            error_logvar: torch.Tensor = error_logvar[:, :, 0, 0]
            batch_loss = batch_weighted_loss / error_logvar.exp() + error_logvar
            logs[f"loss/{self.flavor}"] = logs[f"loss/{self.flavor}"] + batch_loss.mean(dim=1) / self.config.num_sampling_steps

            bucket_log_loss = batch_weighted_loss
            if self.config.num_loss_buckets > 0:
                for b in range(bucket_log_loss.shape[1]):
                    self.unet_loss_buckets[b].log_buckets(bucket_log_loss[:, b], sigma)

            if i < self.config.num_sampling_steps - 1:
                sigma_next = sigma_schedule[i+1].expand(samples.shape[0])
            else:
                sigma_next = torch.zeros_like(sigma)
            
            #old_sigma_next = sigma_next
            #effective_ip = sigma.log().tanh() / 2 + 0.5
            #sigma_next = sigma_next * (1 - effective_ip)

            samples = torch.lerp(denoised, samples.clone(), (sigma_next / sigma).view(-1, 1, 1, 1))

            #if i < self.config.num_sampling_steps - 1:
            #    p = (old_sigma_next**2 - sigma_next**2).clip(min=0)**0.5
            #    samples = samples + torch.randn_like(samples) * p.view(-1, 1, 1, 1)
    
        mss_loss_logs = None

        for i, (_target, _denoised) in enumerate(zip(target_samples_unflattened, self.format.unflatten_mdct_phase_psd(samples))):
            
            """
            target_phase, target_psd = torch.chunk(_target, 2, dim=1)
            denoised_phase, denoised_psd = torch.chunk(_denoised, 2, dim=1)

            denoised1 = torch.cat((denoised_phase, target_psd), dim=1)
            denoised2 = torch.cat((target_phase, denoised_psd), dim=1)

            denoised1_raw = self.trainer.module_trainer.format.mdct_phase_psd_to_raw(denoised1, level=i)
            denoised2_raw = self.trainer.module_trainer.format.mdct_phase_psd_to_raw(denoised2, level=i)
            target_raw = self.trainer.module_trainer.format.mdct_phase_psd_to_raw(_target, level=i).detach()
            
            _mss_loss_logs = self.mss_1d.mss_loss(denoised1_raw, denoised2_raw, target_raw)
            """
            
            target_raw = self.format.mdct_phase_psd_to_raw(_target, level=i).detach()
            denoised_raw = self.format.mdct_phase_psd_to_raw(_denoised, level=i)
            _mss_loss_logs = self.mss_1d.mss_loss(denoised_raw, target_raw)

            if mss_loss_logs is None:
                mss_loss_logs = _mss_loss_logs
            else:
                for k in mss_loss_logs.keys():
                    mss_loss_logs[k] = mss_loss_logs[k] + _mss_loss_logs[k]

        logs.update(mss_loss_logs)
        logs.update({
            f"io_stats_{self.flavor}/denoised_var": samples.var(dim=(1,2,3)),
            f"io_stats_{self.flavor}/denoised_mean": samples.mean(dim=(1,2,3))
        })
        
        return logs
    
    @torch.no_grad()
    def finish_batch(self) -> Optional[dict[str, Union[torch.Tensor, float]]]:

        if self.config.num_loss_buckets > 0:
            logs = {}
            for bucket in self.unet_loss_buckets:
                logs.update(bucket.get_logs())

            return logs
        
        return None