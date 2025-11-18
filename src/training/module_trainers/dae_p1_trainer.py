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
from typing import Union, Optional

import torch
import numpy as np

from training.trainer import DualDiffusionTrainer
from training.module_trainers.module_trainer import ModuleTrainer, ModuleTrainerConfig
from modules.daes.dae_edm2_p1 import DAE
from modules.formats.mdct import MDCT_Format
from modules.mp_tools import normalize


@dataclass
class MSSLoss2DConfig:

    #block_sizes: tuple[tuple[int, int]] = (
    #    (19, 5), (29, 7), (43, 11), (53, 13), (67, 17), (79, 19), (89, 23), (113, 29), (127, 31)
    #)

    block_widths: tuple[int] = (11, 13, 17, 19, 23, 29, 31)
    block_steps: tuple[int] =  ( 2,  3,  5,  7, 11, 13, 17)
    #block_steps: tuple[int] = ( 1,  2,  3,  5,  7, 11, 13)
    
    #block_widths: tuple[int] = (5, 7, 11, 19, 37, 71)
    #block_steps:  tuple[int] = (2, 3,  5,  7, 17, 31)
    loss_scale: float = 1/42

class MSSLoss2D:

    @torch.no_grad()
    def __init__(self, config: MSSLoss2DConfig, device: torch.device) -> None:

        self.config = config
        self.device = device

        self.steps = []
        self.windows = []
        
        for block_width, block_step in zip(config.block_widths, config.block_steps):

            self.steps.append(block_step)
            window = self.get_flat_top_window_2d(block_width)
            window /= window.square().mean().sqrt()
            self.windows.append(window.to(device=device).requires_grad_(False).detach())

    @torch.no_grad()
    def _flat_top_window(self, x: torch.Tensor) -> torch.Tensor:
        return (0.21557895 - 0.41663158 * torch.cos(x) + 0.277263158 * torch.cos(2*x)
                - 0.083578947 * torch.cos(3*x) + 0.006947368 * torch.cos(4*x))

    @torch.no_grad()
    def get_flat_top_window_2d(self, block_width: int) -> torch.Tensor:
        wx = (torch.arange(block_width) + 0.5) / block_width * 2 * torch.pi
        return self._flat_top_window(wx.view(1, 1,-1, 1)) * self._flat_top_window(wx.view(1, 1, 1,-1))
    
    def stft2d(self, x: torch.Tensor, block_width: int,
               step: int, window: torch.Tensor, offset_h: int, offset_w: int, midside: bool = False) -> torch.Tensor:
        
        padding = block_width // 2
        x = torch.nn.functional.pad(x, (padding+1+step, padding, padding+1+step, padding), mode="reflect")
        x = x[:, :, offset_h:, offset_w:]
        x = x.unfold(2, block_width, step).unfold(3, block_width, step)

        x = torch.fft.rfft2(x * window, norm="ortho")
        if midside == True:
            x = torch.stack((x[:, 0] + x[:, 1], x[:, 0] - x[:, 1]), dim=1) / 2

        return x
    
    def mss_loss(self, sample: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        sample = sample.permute(0, 2, 1, 3)
        target = target.permute(0, 2, 1, 3)
        
        loss = torch.zeros(target.shape[0], device=self.device)

        for i, block_width in enumerate(self.config.block_widths):
            
            step = self.steps[i]
            window = self.windows[i]

            offset_h = np.random.randint(0, step)
            offset_w = np.random.randint(0, step)

            midside = np.random.randint(0, 2) == 0
            
            with torch.no_grad():
                target_fft = self.stft2d(target, block_width, step, window, offset_h, offset_w, midside)
                target_fft_abs = target_fft.abs().requires_grad_(False).detach()
                loss_weight = block_width / target_fft_abs.square().mean(dim=(0,1,2,3), keepdim=True)
                #print(block_width, loss_weight.amin().item())
                loss_weight = (loss_weight.clip(min=1e-4).sqrt()).requires_grad_(False).detach()
                #loss_weight = (block_width / target_fft_abs.mean(dim=(0,1,2,3), keepdim=True).clip(min=1e-2)).requires_grad_(False).detach()

            sample_fft = self.stft2d(sample, block_width, step, window, offset_h, offset_w, midside)
            sample_fft_abs = sample_fft.abs()
            
            mse_loss = torch.nn.functional.mse_loss(sample_fft_abs, target_fft_abs, reduction="none")
            loss = loss + (mse_loss * loss_weight).mean(dim=(1,2,3,4,5))

        return loss * self.config.loss_scale

    def compile(self, **kwargs) -> None:
        self.mss_loss = torch.compile(self.mss_loss, **kwargs)

def get_cos_angle(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    dot = torch.einsum('bchw,bchw->bhw', x, y)
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
        
        self.format: MDCT_Format = trainer.pipeline.format.to(self.trainer.accelerator.device)

        if trainer.config.enable_model_compilation:
            self.dae.compile(**trainer.config.compile_params)
            self.format.compile(**trainer.config.compile_params)

        self.logger.info(f"Training modules: {trainer.config.train_modules}")
        self.logger.info(f"KL loss weight: {self.config.kl_loss_weight} KL warmup steps: {self.config.kl_warmup_steps}")
        self.logger.info(f"Latents phase-invariance loss weight: {self.config.phase_invariance_loss_weight} Batch size: {self.config.phase_invariance_loss_bsz}")
        self.logger.info(f"Latents dispersion loss weight: {self.config.latents_dispersion_loss_weight} Batch size: {self.config.latents_dispersion_loss_bsz}")
        self.logger.info(f"Latents dispersion loss num iterations: {self.config.latents_dispersion_num_iterations}")
        self.logger.info(f"Latents regularization loss warmup steps: {self.config.latents_regularization_warmup_steps}")
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

        mdct_samples: torch.Tensor = self.format.raw_to_mdct(raw_samples, random_phase_augmentation=True)
        mdct_samples = mdct_samples[:, :, :, self.config.crop_edges:-self.config.crop_edges].detach()
        
        latents, recon_mdct, pre_norm_latents = self.dae(mdct_samples, dae_embeddings)
        latents: torch.Tensor = latents.float()
        pre_norm_latents: torch.Tensor = pre_norm_latents.float()

        mdct_samples_raw = self.format.mdct_to_raw(mdct_samples)
        recon_mdct_raw = self.format.mdct_to_raw(recon_mdct)
        mdct_win_len = self.format.config.mdct_window_len
        rnd_offset = np.random.randint(0, mdct_win_len)
        mdct_samples_raw = mdct_samples_raw[..., rnd_offset:]
        recon_mdct_raw = recon_mdct_raw[..., rnd_offset:]
        mdct_samples = self.format.raw_to_mdct(mdct_samples_raw)[..., 1:-1]
        recon_mdct = self.format.raw_to_mdct(recon_mdct_raw)[..., 1:-1]

        recon_loss = self.mss_loss.mss_loss(recon_mdct, mdct_samples)
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
        else:
            dispersion_loss = None

        pre_norm_latents_var = pre_norm_latents.pow(2).mean(dim=(0,2,3)) + 1e-20
        var_kl = pre_norm_latents_var - 1 - pre_norm_latents_var.log()
        kl_loss = var_kl.mean() + pre_norm_latents.mean(dim=(0,2,3)).square().mean() * self.config.kl_mean_weight

        per_row_pre_norm_latents_var = pre_norm_latents.pow(2).mean(dim=(0,2,3))
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
            "io_stats/recon_mdct_std": recon_mdct.std(dim=(1,2,3)),
            "io_stats/recon_mdct_mean": recon_mdct.mean(dim=(1,2,3)),
            "io_stats/mdct_std": mdct_samples.std(dim=(1,2,3)),
            "io_stats/mdct_mean": mdct_samples.mean(dim=(1,2,3)),
            "io_stats/latents_std": latents.std(dim=1).detach(),
            "io_stats/latents_mean": latents.mean(dim=1).detach(),
            "io_stats/latents_pre_norm_std": pre_norm_latents_var.sqrt(),
            "io_stats/per_row_latents_pre_norm_std": per_row_pre_norm_latents_var.sqrt(),
            "loss/kl_latents": kl_loss.detach(),
            "loss/recon": recon_loss.detach(),
            "loss_weight/kl_latents": kl_loss_weight,
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
            print("mdct_samples.shape:", mdct_samples.shape)
            print("recon_mdct.shape:", recon_mdct.shape)
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