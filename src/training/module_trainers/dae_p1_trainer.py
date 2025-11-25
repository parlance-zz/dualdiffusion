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
from typing import Union, Optional, Literal

import torch
import numpy as np

from training.trainer import DualDiffusionTrainer
from training.module_trainers.module_trainer import ModuleTrainer, ModuleTrainerConfig
from modules.daes.dae_edm2_p1 import DAE
from modules.formats.ms_mdct_dual_2 import MS_MDCT_DualFormat
from modules.mp_tools import normalize

def vicreg_regularization(z: torch.Tensor, 
                          lambda_: float = 25.0, 
                          mu: float = 25.0, 
                          nu: float = 1.0) -> torch.Tensor:
    """
    VICReg regularization for preventing latent collapse in autoencoders.
    
    Args:
        z: Latent tensor of shape (b, c, h, w)
        lambda_: Weight for variance loss
        mu: Weight for invariance loss (not used here, set to 0)
        nu: Weight for covariance loss
    
    Returns:
        Regularization loss
    """
    b, c, h, w = z.shape
    z_flat = z.permute(0, 2, 3, 1).reshape(-1, c)  # (b*h*w, c)
    
    # Standardize
    z_norm = (z_flat - z_flat.mean(dim=0)) / (z_flat.std(dim=0) + 1e-6)
    
    # Variance loss: encourage non-zero variance across batch
    #var_loss = torch.mean(torch.relu(1.0 - z_norm.std(dim=0)))
    
    # Covariance loss: decorrelate dimensions
    cov_z = (z_norm.T @ z_norm) / (b * h * w - 1)
    cov_z_off_diag = cov_z - torch.diag(torch.diag(cov_z))
    cov_loss = torch.mean(cov_z_off_diag ** 2)
    
    # Total regularization
    #reg_loss = lambda_ * var_loss + nu * cov_loss
    reg_loss = nu * cov_loss
    
    return reg_loss

def _is_prime(n: int) -> bool:
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

@dataclass
class MSSLoss2DConfig:

    block_low:  int = 9
    block_high: int = 254

    block_sampling_replace: bool = True
    block_sampling_scale: Literal["linear", "ln_linear"] = "ln_linear"

    num_iterations: int = 100
    midside_probability: float = 0.5
    psd_eps: float = 1e-4
    loss_scale: float = 3

class MSSLoss2D:

    @torch.no_grad()
    def __init__(self, config: MSSLoss2DConfig, device: torch.device) -> None:

        self.config = config
        self.device = device

        primes = [i for i in range(self.config.block_low, self.config.block_high+1) if _is_prime(i)]

        n = 25000

        if self.config.block_sampling_scale == "ln_linear":
            targets = np.exp(np.linspace(np.log(self.config.block_low), np.log(self.config.block_high), n))
        elif self.config.block_sampling_scale == "linear":
            targets = np.linspace(self.config.block_low, self.config.block_high, n)
        else:
            raise ValueError(f"Invalid block_sampling_scale: {self.config.block_sampling_scale}")

        spaced_primes = []
        for t in targets:
            closest = min(primes, key=lambda p: abs(p - t))
            spaced_primes.append(closest)

        block_sizes = []
        block_weights = []

        for b in sorted(set(spaced_primes)):
            count = spaced_primes.count(b)

            block_sizes.append(b)
            block_weights.append(float(count))

        self.block_sizes = np.array(block_sizes)
        self.block_weights = np.array(block_weights)
        self.block_weights /= self.block_weights.sum()

        for i in range(len(self.block_sizes)):
            print(f"Block size: {self.block_sizes[i]:3d} Weight: {(self.block_weights[i]*100):.3f}%")
        print(f"total unique block sizes: {len(block_sizes)}\n")

        self.loss_scale = config.loss_scale / self.config.num_iterations

    @torch.no_grad()
    def _flat_top_window(self, x: torch.Tensor) -> torch.Tensor:
        return (0.21557895 - 0.41663158 * torch.cos(x) + 0.277263158 * torch.cos(2*x)
                - 0.083578947 * torch.cos(3*x) + 0.006947368 * torch.cos(4*x))

    @torch.no_grad()
    def get_flat_top_window_2d(self, block_width: int, block_height: int) -> torch.Tensor:

        if block_height <= 3: hx = torch.ones(block_height, device=self.device)
        else: hx = self._flat_top_window((torch.arange(block_height, device=self.device) + 0.5) / block_height * 2 * torch.pi)

        if block_width <= 3: wx = torch.ones(block_width, device=self.device)
        else: wx = self._flat_top_window((torch.arange(block_width,  device=self.device) + 0.5) / block_width  * 2 * torch.pi)

        window = hx.view(1, 1,-1, 1) * wx.view(1, 1, 1,-1)
        window /= window.square().mean().sqrt()
        return window.detach().requires_grad_(False)
    
    def stft2d(self, x: torch.Tensor, block_width: int, block_height: int, order: tuple[int],
               step_w: int, step_h: int, window: torch.Tensor, offset_h: int, offset_w: int, midside: bool) -> torch.Tensor:
        
        padding_h = block_height
        padding_w = block_width
        x = torch.nn.functional.pad(x, (padding_w, padding_w, padding_h, padding_h), mode="reflect")
        x = x[:, :, offset_h:, offset_w:]
        x = x.unfold(2, block_height, step_h).unfold(3, block_width, step_w)

        x = torch.fft.rfft2(x * window, norm="ortho", dim=order)
        if midside == True:
            x = torch.stack((x[:, 0] + x[:, 1], x[:, 0] - x[:, 1]), dim=1) / 2**0.5

        return x
    
    def mss_loss(self, sample: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        loss = torch.zeros(target.shape[0], device=self.device)

        block_widths  = np.random.choice(self.block_sizes, size=self.config.num_iterations, replace=self.config.block_sampling_replace, p=self.block_weights)
        block_heights = np.random.choice(self.block_sizes, size=self.config.num_iterations, replace=self.config.block_sampling_replace, p=self.block_weights)

        for i in range(self.config.num_iterations):

            block_width = int(block_widths[i])
            block_height = int(block_heights[i])

            step_w = block_width
            step_h = block_height
            window = self.get_flat_top_window_2d(block_width, block_height)

            offset_h = int(np.random.randint(0, step_h))
            offset_w = int(np.random.randint(0, step_w))
            
            order = (-1, -2) if np.random.randint(0, 2) == 0 else (-2, -1)
            midside = np.random.rand() < self.config.midside_probability
            #r_dims = (0, 2, 3) if midside == True else (0, 1, 2, 3)
            r_dims = (0, 3) if midside == True else (0, 1, 3)

            with torch.no_grad():
                target_fft = self.stft2d(target, block_width, block_height, order, step_w, step_h, window, offset_h, offset_w, midside)
                target_fft_abs = target_fft.abs().requires_grad_(False).detach()
                loss_weight = target_fft_abs.pow(2).mean(dim=r_dims, keepdim=True).clip(min=self.config.psd_eps).pow(0.5).requires_grad_(False).detach()

            sample_fft = self.stft2d(sample, block_width, block_height, order, step_w, step_h, window, offset_h, offset_w, midside)
            sample_fft_abs = sample_fft.abs()
            
            mse_loss = torch.nn.functional.mse_loss(sample_fft_abs.float(), target_fft_abs.float(), reduction="none")
            loss = loss + (mse_loss / loss_weight).mean(dim=(1,2,3,4,5)) #** 2

        return loss * self.loss_scale

    def compile(self, **kwargs) -> None:
        self.mss_loss = torch.compile(self.mss_loss, **kwargs)

def get_cos_angle(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    dot = torch.einsum("bchw,bchw->bhw", x, y)
    return dot / x.shape[1]

@torch.no_grad()
def random_stereo_augmentation(x: torch.Tensor) -> torch.Tensor:
    
    output = x.clone()
    flip_mask = (torch.rand(x.shape[0]) > 0.5).to(x.device)
    output[flip_mask] = output[flip_mask].flip(dims=(1,))
    
    return output

@dataclass
class DAE_Trainer_Config(ModuleTrainerConfig):

    kl_loss_weight: float = 1-2
    kl_mean_weight: float = 1
    kl_warmup_steps: int  = 20000

    phase_invariance_loss_weight: float = 1
    phase_invariance_loss_bsz: int = -1
    latents_dispersion_loss_weight: float = 0
    latents_dispersion_loss_bsz: int = -1
    latents_dispersion_num_iterations: int = 1
    latents_regularization_warmup_steps: int = 20000

    point_loss_weight: float = 2
    point_loss_warmup_steps: int = 100

    random_stereo_augmentation: bool = True

    crop_edges: int = 4 # used to avoid artifacts due to mdct lapped blocks at beginning and end of sample

class DAE_Trainer(ModuleTrainer):
    
    @torch.no_grad()
    def __init__(self, config: DAE_Trainer_Config, trainer: DualDiffusionTrainer) -> None:

        self.config = config
        self.trainer = trainer
        self.logger = trainer.logger

        self.dae: DAE = trainer.get_train_module("dae")                                                                        

        if self.config.phase_invariance_loss_weight > 0:
            assert self.config.phase_invariance_loss_bsz != 0, "phase_invariance_loss_weight > 0 but phase_invariance_loss_bsz is 0"
        if self.config.phase_invariance_loss_bsz == -1:
            self.config.phase_invariance_loss_bsz = self.trainer.config.device_batch_size
        
        if self.config.latents_dispersion_loss_weight > 0:
            assert self.config.latents_dispersion_loss_bsz != 0, "latents_dispersion_loss_weight > 0 but latents_dispersion_loss_bsz is 0"
        if self.config.latents_dispersion_loss_bsz == -1:
            self.config.latents_dispersion_loss_bsz = self.trainer.config.device_batch_size
        else:
            assert self.config.latents_dispersion_loss_bsz <= self.trainer.config.device_batch_size, "latents_dispersion_loss_bsz cannot be larger than device_batch_size"
        
        self.format: MS_MDCT_DualFormat = trainer.pipeline.format.to(self.trainer.accelerator.device)

        if trainer.config.enable_model_compilation:
            self.dae.compile(**trainer.config.compile_params)
            self.format.compile(**trainer.config.compile_params)

        self.logger.info(f"Training modules: {trainer.config.train_modules}")
        self.logger.info(f"KL loss weight: {self.config.kl_loss_weight} KL warmup steps: {self.config.kl_warmup_steps}")
        self.logger.info(f"Latents phase-invariance loss weight: {self.config.phase_invariance_loss_weight} Batch size: {self.config.phase_invariance_loss_bsz}")
        self.logger.info(f"Latents dispersion loss weight: {self.config.latents_dispersion_loss_weight} Batch size: {self.config.latents_dispersion_loss_bsz}")
        self.logger.info(f"Latents dispersion loss num iterations: {self.config.latents_dispersion_num_iterations}")
        self.logger.info(f"Latents regularization loss warmup steps: {self.config.latents_regularization_warmup_steps}")
        self.logger.info(f"Point loss weight: {self.config.point_loss_weight} Point loss warmup steps: {self.config.point_loss_warmup_steps}")
        self.logger.info(f"Crop edges: {self.config.crop_edges}")

        if self.config.random_stereo_augmentation == True:
            self.logger.info("Using random stereo augmentation")
        else: self.logger.info("Random stereo augmentation is disabled")

        self.mss_loss = MSSLoss2D(MSSLoss2DConfig(), device=trainer.accelerator.device)

    def train_batch(self, batch: dict) -> Optional[dict[str, Union[torch.Tensor, float]]]:

        # prepare model inputs
        if "audio_embeddings" in batch:
            audio_embeddings = normalize(batch["audio_embeddings"]).detach()
            dae_embeddings = self.dae.get_embeddings(audio_embeddings)
        else:
            audio_embeddings = dae_embeddings = None

        if self.config.random_stereo_augmentation == True:
            raw_samples = random_stereo_augmentation(batch["audio"])
        else:
            raw_samples = batch["audio"]

        ms_samples: torch.Tensor = self.format.raw_to_mel_spec(raw_samples)
        ms_samples = ms_samples[:, :, :, self.config.crop_edges:-self.config.crop_edges].detach()
        
        latents, recon_ms_samples, pre_norm_latents = self.dae(ms_samples, dae_embeddings)
        latents: torch.Tensor = latents.float()
        pre_norm_latents: torch.Tensor = pre_norm_latents.float()

        mss_loss = self.mss_loss.mss_loss(recon_ms_samples, ms_samples)
        recon_loss = mss_loss

        point_loss_weight = self.config.point_loss_weight
        if self.trainer.global_step < self.config.point_loss_warmup_steps:
            point_loss_weight = point_loss_weight * (1 - self.trainer.global_step / self.config.point_loss_warmup_steps)
        else:
            point_loss_weight = 0

        point_loss = torch.nn.functional.l1_loss(recon_ms_samples, ms_samples, reduction="none").mean(dim=(1,2,3))
        if point_loss_weight > 0:
            recon_loss = recon_loss + point_loss * point_loss_weight

        recon_loss_logvar = self.dae.get_recon_loss_logvar()
        recon_loss_nll = recon_loss / recon_loss_logvar.exp() + recon_loss_logvar

        if self.config.phase_invariance_loss_bsz > 0:

            mdct_samples_dae2: torch.Tensor = self.format.raw_to_mdct(raw_samples[:self.config.phase_invariance_loss_bsz], random_phase_augmentation=True)
            mdct_samples_dae2 = mdct_samples_dae2[:, :, :, self.config.crop_edges:-self.config.crop_edges].detach()
            dae_embeddings2 = dae_embeddings[:self.config.phase_invariance_loss_bsz] if dae_embeddings is not None else None
            with torch.autocast(device_type="cuda", dtype=self.trainer.mixed_precision_dtype, enabled=self.trainer.mixed_precision_enabled):
                latents2 = self.dae.encode(mdct_samples_dae2, dae_embeddings2)

            cos_angle = get_cos_angle(latents[:self.config.phase_invariance_loss_bsz], latents2.float())
            phase_invariance_loss = (1 - cos_angle).mean() / 2

            phase_invariance_loss = phase_invariance_loss.expand(latents.shape[0]) # needed for per-sample logging
        else:
            phase_invariance_loss = None

        if self.config.latents_dispersion_loss_bsz > 0:
            #dispersion_loss = vicreg_regularization(latents)

            #"""
            dispersion_loss = torch.zeros(1, device=latents.device)
            total_dispersion_iterations = 0

            for i in range(self.config.latents_dispersion_loss_bsz - 1):
                repulse_latents = latents.roll(shifts=i+1, dims=0)

                for j in range(self.config.latents_dispersion_num_iterations):

                    repulse_latents = repulse_latents.roll(shifts=np.random.randint(1, repulse_latents.shape[3]), dims=3)
                    if repulse_latents.shape[2] > 1:
                        repulse_latents = repulse_latents.roll(shifts=np.random.randint(1, repulse_latents.shape[2]), dims=2)

                    cos_angle = get_cos_angle(latents, repulse_latents)
                    dispersion_loss = dispersion_loss + (cos_angle**2).mean()

                total_dispersion_iterations += self.config.latents_dispersion_num_iterations

            if total_dispersion_iterations > 0:
                dispersion_loss = dispersion_loss / total_dispersion_iterations
            dispersion_loss = dispersion_loss.expand(latents.shape[0]) # needed for per-sample logging
            #"""
        else:
            dispersion_loss = None

        pre_norm_latents_var = pre_norm_latents.pow(2).mean(dim=(0,2,3)) + 1e-20
        var_kl = pre_norm_latents_var - 1 - pre_norm_latents_var.log()
        kl_loss = var_kl.mean() + pre_norm_latents.mean(dim=(0,2,3)).square().mean() * self.config.kl_mean_weight

        per_row_pre_norm_latents_var = pre_norm_latents.pow(2).mean(dim=(0,2,3))
        per_row_pre_norm_latents_mean = pre_norm_latents.mean(dim=(0,2,3)).abs().mean()

        #pre_norm_latents_var = pre_norm_latents.pow(2).mean(dim=1) + 1e-20
        #var_kl = pre_norm_latents_var - 1 - pre_norm_latents_var.log()
        #kl_loss = var_kl.mean() + pre_norm_latents.mean().square() * self.config.kl_mean_weight
        kl_loss = kl_loss.expand(latents.shape[0]) # needed for per-sample logging

        phase_invariance_loss_weight = self.config.phase_invariance_loss_weight
        dispersion_loss_weight = self.config.latents_dispersion_loss_weight
        if self.trainer.global_step < self.config.latents_regularization_warmup_steps:
            warmup_factor = self.trainer.global_step / self.config.latents_regularization_warmup_steps
            phase_invariance_loss_weight *= warmup_factor
            dispersion_loss_weight *= warmup_factor

        kl_loss_weight = self.config.kl_loss_weight
        if self.trainer.global_step < self.config.kl_warmup_steps:
            kl_loss_weight *= self.trainer.global_step / self.config.kl_warmup_steps

        logs = {
            "loss": recon_loss_nll + kl_loss * kl_loss_weight,
            "io_stats/recon_ms_samples_std": recon_ms_samples.std(dim=(1,2,3)),
            "io_stats/recon_ms_samples_mean": recon_ms_samples.mean(dim=(1,2,3)),
            "io_stats/ms_samples_std": ms_samples.std(dim=(1,2,3)),
            "io_stats/ms_samples_mean": ms_samples.mean(dim=(1,2,3)),
            "io_stats/latents_std": latents.std(dim=1).detach(),
            "io_stats/latents_mean": latents.mean(dim=1).detach(),
            "io_stats/latents_pre_norm_std": pre_norm_latents_var.sqrt(),
            "io_stats/per_row_latents_mean": per_row_pre_norm_latents_mean,
            "io_stats/per_row_latents_pre_norm_std": per_row_pre_norm_latents_var.sqrt(),
            "loss/kl_latents": kl_loss.detach(),
            "loss/recon": recon_loss.detach(),
            "loss/point": point_loss.detach(),
            "loss/mss": mss_loss.detach(),
            "loss_weight/kl_latents": kl_loss_weight,
            "loss_weight/point": point_loss_weight,
            "loss_weight/phase_invariance": phase_invariance_loss_weight,
            "loss_weight/dispersion": dispersion_loss_weight
        }

        if self.config.phase_invariance_loss_weight > 0:
            logs["loss"] = logs["loss"] + phase_invariance_loss * phase_invariance_loss_weight
            logs["loss/phase_invariance"] = phase_invariance_loss.detach()

        if self.config.latents_dispersion_loss_weight > 0:
            logs["loss"] = logs["loss"] + dispersion_loss * dispersion_loss_weight
        if dispersion_loss is not None:
            logs["loss/latents_dispersion"] = dispersion_loss.detach()

        if self.trainer.config.enable_debug_mode == True:
            print("ms_samples.shape:", ms_samples.shape)
            print("recon_ms_samples.shape:", recon_ms_samples.shape)
            print("latents.shape:", latents.shape)

        return logs
    

"""
def _is_prime(n: int) -> bool:
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

if __name__ == "__main__":
        
        import numpy as np

        block_low = 5
        block_high = 251
        primes = [i for i in range(block_low, block_high+1) if _is_prime(i)]

        n = 25000

        targets = np.exp(np.linspace(np.log(self.config.block_low), np.log(self.config.block_high), n))
        #targets = np.linspace(self.config.block_low, self.config.block_high, n)

        spaced_primes = []
        for t in targets:
            closest = min(primes, key=lambda p: abs(p - t))
            spaced_primes.append(closest)

        block_sizes = []
        block_weights = []

        for b in sorted(set(spaced_primes)):
            count = spaced_primes.count(b)
            print(f"{b:3d}: {count}")

            block_sizes.append(b)
            block_weights.append(float(count))
        print(f"total unique block sizes: {len(block_sizes)}\n")

        self.block_sizes = np.array(block_sizes)
        self.block_weights = np.array(block_weights)
        self.block_weights /= self.block_weights.sum()

"""