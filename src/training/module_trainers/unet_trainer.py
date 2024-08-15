from dataclasses import dataclass
from typing import Literal, Optional

import torch

from training.sigma_sampler import SigmaSamplerConfig, SigmaSampler
from training.trainer import DualDiffusionTrainer
from .module_trainer import ModuleTrainerConfig, ModuleTrainer
from utils.dual_diffusion_utils import normalize, dict_str

@dataclass
class UNetTrainerConfig(ModuleTrainerConfig):

    sigma_distribution: Literal["ln_normal", "ln_sech", "ln_sech^2",
                                "ln_linear", "ln_pdf"] = "ln_sech"
    sigma_dist_scale: float = 1.0
    sigma_dist_offset: float = 0.1
    use_stratified_sigma_sampling: bool = True
    sigma_pdf_resolution: Optional[int] = 127

    num_loss_buckets: int = 10
    input_perturbation: Optional[float] = None

class UNetTrainer(ModuleTrainer):
    
    @torch.no_grad()
    def __init__(self, config: UNetTrainerConfig, trainer: DualDiffusionTrainer) -> None:

        self.config = config
        self.trainer = trainer
        self.logger = self.trainer.logger
        self.module = self.trainer.module

        self.logger.info("Training UNet model:")

        if self.config.num_loss_buckets > 0:
            self.logger.info(f"Using {self.config.num_loss_buckets} loss buckets")
            self.unet_loss_buckets = torch.zeros(
                self.config.num_loss_buckets, device="cpu", dtype=torch.float32)
            self.unet_loss_bucket_counts = torch.zeros(
                self.config.num_loss_buckets, device="cpu", dtype=torch.float32)
        else:
            self.logger.info("UNet loss buckets are disabled")

        if self.trainer.vae.config.last_global_step == 0 and self.trainer.config.dataloader.use_pre_encoded_latents == False:
            self.logger.error("VAE model has not been trained, aborting training..."); exit(1)

        if self.config.input_perturbation > 0:
            self.logger.info(f"Using input perturbation of {self.config.input_perturbation}")
        else: self.logger.info("Input perturbation is disabled")

        self.logger.info(f"Dropout: {self.module.config.dropout} Conditioning dropout: {self.module.config.label_dropout}")

        sigma_sampler_config = SigmaSamplerConfig(
            sigma_max=self.module.config.sigma_max,
            sigma_min=self.module.config.sigma_min,
            sigma_data=self.module.config.sigma_data,
            distribution=self.config.sigma_distribution,
            sigma_dist_scale=self.config.sigma_dist_scale,
            sigma_dist_offset=self.config.sigma_dist_offset,
            use_stratified_sigma_sampling=self.config.use_stratified_sigma_sampling,
            sigma_pdf_resolution=self.config.sigma_pdf_resolution,
        )
        self.sigma_sampler = SigmaSampler(sigma_sampler_config)
        self.logger("Sigma sampler config:")
        self.logger(dict_str(sigma_sampler_config))

        if self.config.num_loss_buckets > 0:
            bucket_ln_sigma = (1 / torch.linspace(torch.pi/2, 0, self.config.num_loss_buckets+1).tan()).log()
            self.bucket_names = [f"unet_loss_buckets/b{i} s:{bucket_ln_sigma[i]:.3f} ~ {bucket_ln_sigma[i+1]:.3f}"
                                 for i in range(self.config.num_loss_buckets)]

    @staticmethod
    def get_config_class() -> ModuleTrainerConfig:
        return UNetTrainerConfig
    
    @torch.no_grad()
    def init_batch(self) -> None:
        
        if self.config.num_loss_buckets > 0:
            self.unet_loss_buckets.zero_()
            self.unet_loss_bucket_counts.zero_()

        if self.config.sigma_distribution == "ln_pdf":
            sigma_sample_temperature = 1 / self.config.sigma_dist_scale
            ln_sigma = torch.linspace(self.sigma_sampler.config.ln_sigma_min,
                                      self.sigma_sampler.config.ln_sigma_max,
                                      self.config.sigma_pdf_resolution,
                                      device=self.trainer.accelerator.device)
            ln_sigma_error = self.module.logvar_linear(
                self.module.logvar_fourier(ln_sigma/4)).float().flatten().detach()
            sigma_distribution_pdf = (-sigma_sample_temperature * ln_sigma_error).exp()
            self.sigma_sampler.update_pdf(sigma_distribution_pdf)
        
        self.global_sigma = self.sigma_sampler.sample(self.trainer.total_batch_size,
                                                      device=self.trainer.accelerator.device)
        self.global_sigma = self.trainer.accelerator.gather(self.global_sigma.unsqueeze(0))[0] # sync sigma across all ranks / processes

    def train_batch(self, batch: dict, grad_accum_steps: int) -> dict[str, torch.Tensor]:

        raw_samples = batch["input"]
        sample_game_ids = batch["game_ids"]
        sample_t_ranges = batch["t_ranges"] if self.trainer.dataset.config.t_scale is not None else None
        #sample_author_ids = batch["author_ids"]

        class_labels = self.trainer.pipeline.get_class_labels(sample_game_ids)
        unet_class_embeddings = self.module.get_class_embeddings(class_labels)

        if self.trainer.config.dataloader.use_pre_encoded_latents:
            samples = normalize(raw_samples).float()
            assert samples.shape == self.trainer.latent_shape
        else:
            samples = self.trainer.pipeline.format.raw_to_sample(raw_samples)
            vae_class_embeddings = self.trainer.vae.get_class_embeddings(class_labels)
            samples = self.trainer.vae.encode(samples.to(self.trainer.vae.dtype),
                                              vae_class_embeddings,
                                              self.trainer.pipeline.format).mode().detach()
            samples = normalize(samples).float()

        process_sigma = self.global_sigma[self.trainer.accelerator.local_process_index::self.trainer.accelerator.num_processes]
        batch_sigma = process_sigma[grad_accum_steps * self.trainer.config.train_batch_size:(grad_accum_steps+1) * self.trainer.config.train_batch_size]

        noise = torch.randn_like(samples) * batch_sigma.view(-1, 1, 1, 1)
        samples = samples * self.module.config.sigma_data

        denoised, error_logvar = self.module(samples + noise,
                                             batch_sigma,
                                             unet_class_embeddings,
                                             sample_t_ranges,
                                             self.trainer.pipeline.format,
                                             return_logvar=True)
        
        batch_loss_weight = (batch_sigma ** 2 + self.module.config.sigma_data ** 2) / (batch_sigma * self.module.config.sigma_data) ** 2
        batch_weighted_loss = torch.nn.functional.mse_loss(denoised, samples, reduction="none").mean(dim=(1,2,3)) * batch_loss_weight
        batch_loss = batch_weighted_loss / error_logvar.exp() + error_logvar
        
        if self.config.num_loss_buckets > 0:
            global_weighted_loss = self.trainer.accelerator.gather(batch_weighted_loss.detach().cpu())
            global_sigma_quantiles = self.trainer.accelerator.gather(self.module.config.sigma_data / batch_sigma.detach().cpu()).arctan() / (torch.pi/2)

            target_buckets = (global_sigma_quantiles * self.unet_loss_buckets.shape[0]).long().clip(min=0, max=self.unet_loss_buckets.shape[0] - 1)
            self.unet_loss_buckets.index_add_(0, target_buckets, global_weighted_loss)
            self.unet_loss_bucket_counts.index_add_(0, target_buckets, torch.ones_like(global_weighted_loss))

        return {"loss": batch_loss}

    @torch.no_grad()
    def finish_batch(self) -> None:    
        logs = {}

        if self.config.num_loss_buckets > 0:
            for i in range(self.config.num_loss_buckets):
                if self.unet_loss_bucket_counts[i].item() > 0:
                    logs[self.bucket_names[i]] = (self.unet_loss_buckets[i] / self.unet_loss_bucket_counts[i]).item()

        return logs