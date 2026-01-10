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
from modules.daes.dae_edm2_q1 import DAE
from modules.formats.ms_mdct_dual_2 import MS_MDCT_DualFormat
from modules.mp_tools import normalize


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

@torch.no_grad()
def random_stereo_augmentation(x: torch.Tensor) -> torch.Tensor:
    
    output = x.clone()
    flip_mask = (torch.rand(x.shape[0]) > 0.5).to(x.device)
    output[flip_mask] = output[flip_mask].flip(dims=(1,))
    
    return output

@dataclass
class DAE_Trainer_Config(ModuleTrainerConfig):

    kl_loss_weight: float = 1e-2
    kl_warmup_steps: int  = 2000

    point_loss_weight: float = 2
    point_loss_warmup_steps: int = 100

    shift_equivariance_loss_weight: float = 0
    shift_equivariance_warmup_steps: int = 2000

    input_perturbation: float = 0
    
    crop_edges: int = 4 # used to avoid artifacts due to mdct lapped blocks at beginning and end of sample
    random_stereo_augmentation: bool = True

class DAE_Trainer(ModuleTrainer):
    
    @torch.no_grad()
    def __init__(self, config: DAE_Trainer_Config, trainer: DualDiffusionTrainer) -> None:

        self.config = config
        self.trainer = trainer
        self.logger = trainer.logger

        self.dae: DAE = trainer.get_train_module("dae")                                                                        
        self.format: MS_MDCT_DualFormat = trainer.pipeline.format.to(self.trainer.accelerator.device)

        if trainer.config.enable_model_compilation:
            self.dae.compile(**trainer.config.compile_params)
            self.format.compile(**trainer.config.compile_params)

        self.logger.info(f"Training modules: {trainer.config.train_modules}")
        self.logger.info(f"KL loss weight: {self.config.kl_loss_weight} KL warmup steps: {self.config.kl_warmup_steps}")
        self.logger.info(f"Latents shift-equivariance loss weight: {self.config.shift_equivariance_loss_weight} Warmup steps: {self.config.shift_equivariance_warmup_steps}")
        self.logger.info(f"Input perturbation: {self.config.input_perturbation}")
        self.logger.info(f"Point loss weight: {self.config.point_loss_weight} Point loss warmup steps: {self.config.point_loss_warmup_steps}")
        self.logger.info(f"Crop edges: {self.config.crop_edges}")

        if self.config.random_stereo_augmentation == True:
            self.logger.info("Using random stereo augmentation")
        else: self.logger.info("Random stereo augmentation is disabled")

        self.mss_loss = MSSLoss2D(MSSLoss2DConfig(), device=trainer.accelerator.device)

    def shift_equivariance_loss(self, mel_spec: torch.Tensor, dae_embeddings: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:

        crop_left = np.random.randint(1, self.config.crop_edges * 2)
        crop_right = self.config.crop_edges * 2 - crop_left
        mel_spec = mel_spec[..., crop_left:-crop_right]

        with torch.autocast(device_type="cuda", dtype=self.trainer.mixed_precision_dtype, enabled=self.trainer.mixed_precision_enabled):
            latents2 = self.dae.encode(mel_spec, dae_embeddings)

        latents_up: torch.Tensor = torch.repeat_interleave(latents, self.dae.downsample_ratio, dim=-1)
        latents_up_cropped = latents_up[..., crop_left:-crop_right]
        latents_down: torch.Tensor = torch.nn.functional.avg_pool2d(latents_up_cropped, kernel_size=(1,self.dae.downsample_ratio))

        return (latents_down - latents2.float())[..., 2:-2].pow(2).mean().expand(latents.shape[0])
    
    def train_batch(self, batch: dict) -> Optional[dict[str, Union[torch.Tensor, float]]]:

        # prepare model inputs
        if "audio_embeddings" in batch:
            audio_embeddings = normalize(batch["audio_embeddings"]).detach()
            dae_embeddings = self.dae.get_embeddings(audio_embeddings)
        else:
            audio_embeddings = dae_embeddings = None

        with torch.no_grad():
            if self.config.random_stereo_augmentation == True:
                raw_samples = random_stereo_augmentation(batch["audio"])
            else:
                raw_samples = batch["audio"]

            input_mel_spec: torch.Tensor = self.format.raw_to_mel_spec(raw_samples)
            input_mel_spec = input_mel_spec[:, :, :, self.config.crop_edges:-self.config.crop_edges]
            target_mel_spec = input_mel_spec.clone()

            if self.config.input_perturbation > 0:
                input_mel_spec += torch.randn_like(input_mel_spec) * self.config.input_perturbation
                
        latents, recon_mel_spec, pre_norm_latents = self.dae(input_mel_spec, dae_embeddings)
        latents: torch.Tensor = latents.float()
        pre_norm_latents: torch.Tensor = pre_norm_latents.float()

        # reconstruction losses
        mss_loss = self.mss_loss.mss_loss(recon_mel_spec, target_mel_spec)
        recon_loss = mss_loss

        point_loss_weight = self.config.point_loss_weight
        if self.trainer.global_step < self.config.point_loss_warmup_steps:
            point_loss_weight = point_loss_weight * (1 - self.trainer.global_step / self.config.point_loss_warmup_steps)
        else:
            point_loss_weight = 0

        point_loss = torch.nn.functional.l1_loss(recon_mel_spec, target_mel_spec, reduction="none").mean(dim=(1,2,3))
        if point_loss_weight > 0:
            recon_loss = recon_loss + point_loss * point_loss_weight

        recon_loss_logvar = self.dae.get_recon_loss_logvar()
        recon_loss_nll = recon_loss / recon_loss_logvar.exp() + recon_loss_logvar

        # shift equivariance loss for latents
        shift_equivariance_loss_weight = self.config.shift_equivariance_loss_weight
        if self.trainer.global_step < self.config.shift_equivariance_warmup_steps:
            warmup_factor = self.trainer.global_step / self.config.shift_equivariance_warmup_steps
            shift_equivariance_loss_weight *= warmup_factor

        if shift_equivariance_loss_weight > 0:
            shift_equivariance_loss = self.shift_equivariance_loss(input_mel_spec, dae_embeddings, latents)
        else: shift_equivariance_loss = None

        # KL loss for latents
        kl_loss_weight = self.config.kl_loss_weight
        if self.trainer.global_step < self.config.kl_warmup_steps:
            kl_loss_weight *= self.trainer.global_step / self.config.kl_warmup_steps

        pre_norm_latents_var = pre_norm_latents.pow(2).mean() + 1e-20
        var_kl = pre_norm_latents_var - 1 - pre_norm_latents_var.log()
        kl_loss = var_kl.mean() + pre_norm_latents.mean().square()
        kl_loss = kl_loss.expand(latents.shape[0]) # needed for per-sample logging

        logs = {
            "loss": recon_loss_nll + kl_loss * kl_loss_weight,
            "loss/kl_latents": kl_loss,
            "loss/recon": recon_loss,
            "loss/point": point_loss,
            "loss/mss": mss_loss,
            "io_stats/recon_mel_spec_var": recon_mel_spec.var(dim=(1,2,3)),
            "io_stats/recon_mel_spec_mean": recon_mel_spec.mean(dim=(1,2,3)),
            "io_stats/mel_spec_var": target_mel_spec.var(dim=(1,2,3)),
            "io_stats/mel_spec_mean": target_mel_spec.mean(dim=(1,2,3)),
            "io_stats/latents_var": latents.var(dim=(1,2,3)),
            "io_stats/latents_mean": latents.mean(dim=(1,2,3)),
            "io_stats/latents_pre_norm_var": pre_norm_latents_var,
            "loss_weight/kl_latents": kl_loss_weight,
            "loss_weight/point": point_loss_weight,
            "loss_weight/shift_equivariance": shift_equivariance_loss_weight,
        }

        if self.config.input_perturbation > 0:
            logs["io_stats/input_perturbation"] = self.config.input_perturbation

        if shift_equivariance_loss_weight > 0:
            logs["loss"] = logs["loss"] + shift_equivariance_loss * shift_equivariance_loss_weight
            logs["loss/shift_equivariance"] = shift_equivariance_loss


        if self.trainer.config.enable_debug_mode == True:
            print("mel_spec.shape:", input_mel_spec.shape)
            print("recon_mel_spec.shape:", recon_mel_spec.shape)
            print("latents.shape:", latents.shape)

        return logs
