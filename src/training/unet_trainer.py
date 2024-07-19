from dataclasses import dataclass
from typing import Literal, Optional

import torch

from .sigma_sampler import SigmaSamplerConfig, SigmaSampler
from .trainer import ModuleTrainerConfig, ModuleTrainer, DualDiffusionTrainer
from utils.dual_diffusion_utils import dict_str

@dataclass
class UNetTrainerConfig(ModuleTrainerConfig):

    sigma_distribution: Literal["ln_normal", "ln_sech", "ln_sech^2", "ln_linear", "ln_pdf"] = "ln_sech"
    sigma_dist_scale: float = 1.0
    sigma_dist_offset: float = 0.1
    use_stratified_sigma_sampling: bool = True

    num_loss_buckets: int = 10
    input_perturbation: Optional[float] = None

class UNetTrainer(ModuleTrainer):
    
    def __init__(self, config: UNetTrainerConfig, trainer: DualDiffusionTrainer) -> None:

        self.config = config
        self.trainer = trainer
        self.logger = self.trainer.logger
        self.module = self.trainer.module

        self.logger.info("Training UNet model:")

        if self.config.num_loss_buckets > 0:
            self.logger.info(f"Using {self.config.num_loss_buckets} loss buckets")
            self.unet_loss_buckets = torch.zeros(self.config.num_loss_buckets, device="cpu", dtype=torch.float32)
            self.unet_loss_bucket_counts = torch.zeros(self.config.num_loss_buckets, device="cpu", dtype=torch.float32)
        else:
            self.logger.info("UNet loss buckets are disabled")

        if self.trainer.vae.config["last_global_step"] == 0 and self.trainer.config.dataloader_config.use_pre_encoded_latents == False:
            self.logger.error("VAE model has not been trained, aborting training..."); exit(1)

        if self.config.input_perturbation > 0:
            self.logger.info(f"Using input perturbation of {self.config.input_perturbation}")
        else: self.logger.info("Input perturbation is disabled")

        self.logger.info(f"Dropout: {self.module.dropout} Conditioning dropout: {self.module.label_dropout}")

        sigma_ln_std = torch.tensor(sigma_ln_std, device=self.trainer.accelerator.device, dtype=torch.float32)
        sigma_ln_mean = torch.tensor(sigma_ln_mean, device=self.trainer.accelerator.device, dtype=torch.float32)
        module_log_channels = [
            "sigma_ln_std",
            "sigma_ln_mean",
        ]

        sigma_sampler_config = SigmaSamplerConfig(
            sigma_max=self.module.config["sigma_max"],
            sigma_min=self.module.config["sigma_min"],
            sigma_data=self.module.config["sigma_data"],
            distribution=self.config.sigma_distribution,
            sigma_dist_scale=self.config.sigma_dist_scale,
            sigma_dist_offset=self.config.sigma_dist_offset,
            use_stratified_sigma_sampling=self.config.use_stratified_sigma_sampling,
        )
        self.sigma_sampler = SigmaSampler(sigma_sampler_config)
        self.logger("Sigma sampler config:")
        self.logger(dict_str(sigma_sampler_config))

    @staticmethod
    def get_config_class() -> ModuleTrainerConfig:
        return UNetTrainerConfig
    
    def get_log_channels(self):
        pass
    
    def init_batch(self) -> None:

        self.batch_sigma = self.sigma_sampler.sample(self.trainer.total_batch_size, device=self.trainer.accelerator.device)
        self.batch_sigma = self.trainer.accelerator.gather(self.batch_sigma.unsqueeze(0))[0] # sync sigma across all ranks / processes
        
        if self.config.num_loss_buckets > 0:
            self.unet_loss_buckets.zero_()
            self.unet_loss_bucket_counts.zero_()

        #sigma_sample_temperature = min(global_step / sigma_temperature_ref_steps, 1) * sigma_sample_max_temperature
        #ln_sigma = torch.linspace(np.log(sigma_min), np.log(sigma_max), sigma_sample_resolution, device=accelerator.device)
        #ln_sigma_error = module.logvar_linear(module.logvar_fourier(ln_sigma/4)).float().flatten().detach()
        #sigma_distribution_pdf = (-sigma_sample_temperature * ln_sigma_error).exp() + torch.linspace(sigma_sample_pdf_skew**0.5, 0, sigma_sample_resolution, device=accelerator.device).square()
        #sigma_sampler.update_pdf(sigma_distribution_pdf)
                    
    def train_batch(self, batch) -> dict:

        raw_samples = batch["input"]
        sample_game_ids = batch["game_ids"]
        sample_t_ranges = batch["t_ranges"] if self.dataset.config.t_scale is not None else None
        raw_sample_paths = batch["sample_paths"]
        #sample_author_ids = batch["author_ids"]

        class_labels = pipeline.get_class_labels(sample_game_ids)
        unet_class_embeddings = module.get_class_embeddings(class_labels)

        if args.train_data_format == ".safetensors":
            #samples = normalize(raw_samples + torch.randn_like(raw_samples) * np.exp(0.5 * vae.get_noise_logvar())).float()
            samples = normalize(raw_samples).float()
            assert samples.shape == latent_shape
        else:
            samples = pipeline.format.raw_to_sample(raw_samples)
            if vae is not None:
                vae_class_embeddings = vae.get_class_embeddings(class_labels)
                samples = vae.encode(samples.to(torch.bfloat16), vae_class_embeddings, pipeline.format).mode().detach()
                samples = normalize(samples).float()

        if use_stratified_sigma_sampling:
            process_batch_quantiles = global_quantiles[accelerator.local_process_index::accelerator.num_processes]
            quantiles = process_batch_quantiles[grad_accum_steps * args.train_batch_size:(grad_accum_steps+1) * args.train_batch_size]
        else:
            quantiles = None

        #if use_stratified_sigma_sampling:
        #    batch_normal = sigma_ln_mean + (sigma_ln_std * (2 ** 0.5)) * (quantiles * 2 - 1).erfinv().clip(min=-5, max=5)
        #else:
        #    batch_normal = torch.randn(samples.shape[0], device=accelerator.device) * sigma_ln_std + sigma_ln_mean
        #sigma = batch_normal.exp().clip(min=module.sigma_min, max=module.sigma_max)
        sigma = sigma_sampler.sample(samples.shape[0], quantiles=quantiles).to(accelerator.device)
        noise = torch.randn_like(samples) * sigma.view(-1, 1, 1, 1)
        samples = samples * module.sigma_data

        denoised, error_logvar = module(samples + noise,
                                        sigma,
                                        unet_class_embeddings,
                                        sample_t_ranges,
                                        pipeline.format,
                                        return_logvar=True)
        
        mse_loss = torch.nn.functional.mse_loss(denoised, samples, reduction="none")
        loss_weight = (sigma ** 2 + module.sigma_data ** 2) / (sigma * module.sigma_data) ** 2
        loss = (loss_weight.view(-1, 1, 1, 1) / error_logvar.exp() * mse_loss + error_logvar).mean()
        #loss = (loss_weight.view(-1, 1, 1, 1) * mse_loss).mean()
        
        if args.num_unet_loss_buckets > 0:
            batch_loss = mse_loss.mean(dim=(1,2,3)) * loss_weight

            global_step_quantiles = (accelerator.gather(sigma.detach()).cpu().log() - np.log(module.sigma_min)) / (np.log(module.sigma_max) - np.log(module.sigma_min))
            global_step_batch_loss = accelerator.gather(batch_loss.detach()).cpu()
            target_buckets = (global_step_quantiles * unet_loss_buckets.shape[0]).long().clip(min=0, max=unet_loss_buckets.shape[0]-1)
            unet_loss_buckets.index_add_(0, target_buckets, global_step_batch_loss)
            unet_loss_bucket_counts.index_add_(0, target_buckets, torch.ones_like(global_step_batch_loss))

    def finish_batch(self) -> None:

        #if args.module == "unet" and args.num_unet_loss_buckets > 0:
        #    for i in range(unet_loss_buckets.shape[0]):
        #        if unet_loss_bucket_counts[i].item() > 0:
        #            bucket_ln_sigma_start = np.log(module.sigma_min) + i * (np.log(module.sigma_max) - np.log(module.sigma_min)) / unet_loss_buckets.shape[0]
        #            bucket_ln_sigma_end = np.log(module.sigma_min) + (i+1) * (np.log(module.sigma_max) - np.log(module.sigma_min)) / unet_loss_buckets.shape[0]
        #            logs[f"unet_loss_buckets/b{i} s:{bucket_ln_sigma_start:.3f} ~ {bucket_ln_sigma_end:.3f}"] = (unet_loss_buckets[i] / unet_loss_bucket_counts[i]).item()
        pass