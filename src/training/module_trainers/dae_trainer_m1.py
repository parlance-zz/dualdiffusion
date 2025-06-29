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
from typing import Union, Any, Optional

import torch

from training.trainer import DualDiffusionTrainer
from training.module_trainers.module_trainer import ModuleTrainerConfig, ModuleTrainer
from training.loss.spectral_regularization import SpecRegLoss, SpecRegLossConfig
from training.loss.wavelet import WaveletLoss, WaveletLoss_Config
from modules.daes.dae_edm2_m1 import DAE_M1
from modules.formats.ms_mdct_dual import MS_MDCT_DualFormat
from modules.mp_tools import normalize
from utils.dual_diffusion_utils import dict_str


@dataclass
class MSSLoss2DConfig:

    block_widths: tuple[int] = (11, 13, 17, 19, 23, 29, 31)
    block_steps:  tuple[int] = ( 1,  2,  3,  5,  7, 11, 13)
    #block_steps: tuple[int] = ( 2,  3,  5,  7, 11, 13, 17)

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
               step: int, window: torch.Tensor) -> torch.Tensor:
        
        padding = block_width // 2
        x = torch.nn.functional.pad(x, (padding, padding, padding, padding), mode="reflect")
        x = x.unfold(2, block_width, step).unfold(3, block_width, step)

        x = torch.fft.fft2(x * window, norm="ortho")
        #x = torch.fft.fftn(x * window, dim=(-3, -2, -1), norm="ortho")

        if x.shape[1] == 2:
            #x = torch.stack((x[:, 0] + x[:, 1], x[:, 0] - x[:, 1]), dim=1)
            #x = torch.cat((x, x[:, 0:1] + x[:, 1:2], x[:, 0:1] - x[:, 1:2]), dim=1)
            x = torch.cat((x, (x[:, 0:1] + x[:, 1:2]) * 0.5**0.5), dim=1)

        return x
    
    def mss_loss(self, sample: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        loss = torch.zeros(target.shape[0], device=self.device)
        
        sample = torch.complex(*sample.chunk(2, dim=1))
        with torch.no_grad():
            target = torch.complex(*target.chunk(2, dim=1))

        for i, block_width in enumerate(self.config.block_widths):
            
            step = self.steps[i]
            window = self.windows[i]

            with torch.no_grad():
                target_fft = self.stft2d(target, block_width, step, window)
                target_fft_abs = target_fft.abs().requires_grad_(False).detach()
                loss_weight = (block_width / target_fft_abs.square().mean(dim=(0,1,2,3), keepdim=True).clip(min=1e-4).sqrt()).requires_grad_(False).detach()
                #loss_weight = (block_width / target_fft_abs.square().mean(dim=(0,2,3), keepdim=True).clip(min=1e-4).sqrt()).requires_grad_(False).detach()

            sample_fft = self.stft2d(sample, block_width, step, window)
            sample_fft_abs = sample_fft.abs()
            
            mse_loss = torch.nn.functional.mse_loss(sample_fft_abs.float(), target_fft_abs.float(), reduction="none")
            loss = loss + (mse_loss * loss_weight).mean(dim=(1,2,3,4,5))
   
        return loss

    def compile(self, **kwargs) -> None:
        self.mss_loss = torch.compile(self.mss_loss, **kwargs)

@dataclass
class MSSLoss1DConfig:

    block_widths: tuple[int] = (31, 53, 83, 137, 223, 359, 577, 937, 1511, 2447, 3967, 6397)
    block_steps:  tuple[int] = ( 7, 11, 17,  29,  47,  79, 127, 211,  337,  547,  887, 1433)

class MSSLoss1D:

    @torch.no_grad()
    def __init__(self, config: MSSLoss1DConfig, device: torch.device) -> None:

        self.config = config
        self.device = device

        self.windows = []
        
        for block_width in config.block_widths:

            window = self.get_flat_top_window(block_width)

            window /= window.square().mean().sqrt()
            self.windows.append(window.to(device=device).requires_grad_(False).detach())

    @torch.no_grad()
    def _flat_top_window(self, x: torch.Tensor) -> torch.Tensor:
        return (0.21557895 - 0.41663158 * torch.cos(x) + 0.277263158 * torch.cos(2*x)
                - 0.083578947 * torch.cos(3*x) + 0.006947368 * torch.cos(4*x))

    @torch.no_grad()
    def get_flat_top_window(self, block_width: int) -> torch.Tensor:
        wx = (torch.arange(block_width) + 0.5) / block_width * 2 * torch.pi
        return self._flat_top_window(wx)
    
    def stft1d(self, x: torch.Tensor, block_width: int,
               step: int, window: torch.Tensor) -> torch.Tensor:
        
        x = torch.fft.rfft2(x.unfold(2, block_width, step) * window, norm="ortho")
        #if x.shape[1] == 2:
            #x = torch.stack((x[:, 0] + x[:, 1], x[:, 0] - x[:, 1]), dim=1)
            #x = torch.cat((x, (x[:, 0:1] + x[:, 1:2])*0.5**0.5, (x[:, 0:1] - x[:, 1:2])*0.5**0.5), dim=1)

        return x
    
    def mss_loss(self, sample: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        padding = self.config.block_widths[-1] // 2
        sample = torch.nn.functional.pad(sample, (padding, padding), mode="reflect")
        with torch.no_grad():
            target = torch.nn.functional.pad(target, (padding, padding), mode="reflect")

        loss = torch.zeros(target.shape[0], device=self.device)

        for i, block_width in enumerate(self.config.block_widths):
            
            step = self.config.block_steps[i]
            window = self.windows[i]

            with torch.no_grad():
                target_fft = self.stft1d(target, block_width, step, window)
                target_fft_abs = target_fft.abs().requires_grad_(False).detach()
                #print(target_fft_abs.square().mean(dim=(0,1,2)).amin())
                loss_weight = (block_width / target_fft_abs.square().mean(dim=(0,1,2), keepdim=True).clip(min=1e-5).sqrt()).requires_grad_(False).detach()

            sample_fft = self.stft1d(sample, block_width, step, window)
            sample_fft_abs = sample_fft.abs()
            
            mse_loss = torch.nn.functional.mse_loss(sample_fft_abs.float(), target_fft_abs.float(), reduction="none")
            loss = loss + (mse_loss * loss_weight).mean(dim=(1,2,3))
   
        return loss

    def compile(self, **kwargs) -> None:
        self.mss_loss = torch.compile(self.mss_loss, **kwargs)

def random_stereo_augmentation(x: torch.Tensor) -> torch.Tensor:
    
    output = x.clone()
    flip_mask = (torch.rand(x.shape[0]) > 0.5).to(x.device)
    output[flip_mask] = output[flip_mask].flip(dims=(1,))
    
    return output

def freeze_module(optimizer: torch.optim.Optimizer, module_to_freeze: torch.nn.Module) -> None:
    
    params_to_freeze: set[torch.Tensor] = set(module_to_freeze.parameters())

    for param in params_to_freeze:

        param.requires_grad = False

        if param.grad is not None:
            param.grad = None

        if param in optimizer.state:
            del optimizer.state[param]

@dataclass
class DAETrainer_J1_Config(ModuleTrainerConfig):

    latents_kl_loss_weight: float = 3e-2
    kl_warmup_steps: int = 250

    add_latents_noise: float = 0
    latents_noise_warmup_steps: int = 500

    point_loss_weight: float = 0
    point_loss_warmup_steps: int = 0

    mss_loss_weight: float = 1

    spec_reg_loss_weight: float = 0
    spec_reg_loss_config: Optional[dict[str, Any]] = None

    wavelet_loss_weight: float = 0
    wavelet_loss_config: Optional[dict[str, Any]] = None

class DAETrainer_J1(ModuleTrainer):
    
    @torch.no_grad()
    def __init__(self, config: DAETrainer_J1_Config, trainer: DualDiffusionTrainer) -> None:
        
        self.config = config
        self.trainer = trainer
        self.logger = trainer.logger

        self.dae: DAE_M1 = trainer.get_train_module("dae")
        self.format: MS_MDCT_DualFormat = trainer.pipeline.format.to(self.trainer.accelerator.device)
        self.mss = MSSLoss2D(MSSLoss2DConfig(), device=trainer.accelerator.device)

        if trainer.config.enable_model_compilation == True:
            self.dae.compile(**trainer.config.compile_params)
            self.format.compile(**trainer.config.compile_params)

        if config.mss_loss_weight > 0:
            if trainer.config.enable_model_compilation == True:
                #self.mss.compile(**trainer.config.compile_params)
                pass

        if config.spec_reg_loss_weight > 0:
            config.spec_reg_loss_config = config.spec_reg_loss_config or {}
            self.spec_reg_loss = SpecRegLoss(SpecRegLossConfig(**config.spec_reg_loss_config), 
                latents_shape=trainer.latent_shape, mel_spec_shape=trainer.sample_shape, device=trainer.accelerator.device)

            if trainer.config.enable_model_compilation == True:
                self.spec_reg_loss.compile(**trainer.config.compile_params)

        if config.wavelet_loss_weight > 0:
            config.wavelet_loss_config = config.wavelet_loss_config or {}
            self.wavelet_loss = WaveletLoss(WaveletLoss_Config(**config.wavelet_loss_config), device=trainer.accelerator.device)

            if trainer.config.enable_model_compilation == True:
                self.wavelet_loss.compile(**trainer.config.compile_params)

        self.logger.info("Training DAE model:")
        self.logger.info(f"Latents KL loss weight: {self.config.latents_kl_loss_weight}")
        self.logger.info(f"KL warmup steps: {self.config.kl_warmup_steps}")
        self.logger.info(f"Point loss weight: {self.config.point_loss_weight}")
        self.logger.info(f"Point loss warmup steps: {self.config.point_loss_warmup_steps}")

        self.logger.info(f"MSS loss weight: {self.config.mss_loss_weight}")
        if self.config.mss_loss_weight > 0:
            self.logger.info("MSS_Loss_1D config:")
            self.logger.info(dict_str(self.mss.config.__dict__))

        self.logger.info(f"Spec reg loss weight: {self.config.spec_reg_loss_weight}")
        if self.config.spec_reg_loss_weight > 0:
            self.logger.info("SpecRegLoss config:")
            self.logger.info(dict_str(self.spec_reg_loss.config.__dict__))

        self.logger.info(f"Wavelet loss weight: {self.config.wavelet_loss_weight}")
        if self.config.wavelet_loss_weight > 0:
            self.logger.info("WaveletLoss config:")
            self.logger.info(dict_str(self.wavelet_loss.config.__dict__))

    def train_batch(self, batch: dict) -> dict[str, Union[torch.Tensor, float]]:

        logs = {}

        if "audio_embeddings" in batch:
            audio_embeddings = normalize(batch["audio_embeddings"]).detach()
            dae_embeddings = self.dae.get_embeddings(audio_embeddings)
        else:
            audio_embeddings = None
            dae_embeddings = None
        
        if self.config.add_latents_noise > 0:
            if self.trainer.global_step < self.config.latents_noise_warmup_steps:
                latents_sigma = self.config.add_latents_noise * (self.trainer.global_step / self.config.latents_noise_warmup_steps)
            else:
                latents_sigma = self.config.add_latents_noise
        else:
            latents_sigma = 0
        latents_sigma = torch.Tensor([latents_sigma]).to(device=self.trainer.accelerator.device)

        raw_audio = random_stereo_augmentation(batch["audio"])
        mdct_samples = self.format.raw_to_mdct(raw_audio, random_phase_augmentation=True).detach()
        latents, reconstructed, mdct_samples, latents_kld = self.dae(
            mdct_samples, dae_embeddings, latents_sigma)
        
        point_loss_weight = self.config.point_loss_weight
        if self.config.point_loss_warmup_steps > 0:
            if self.trainer.global_step < self.config.point_loss_warmup_steps:
                point_loss_weight *= 1 - self.trainer.global_step / self.config.point_loss_warmup_steps
            else:
                point_loss_weight = 0
        point_loss = torch.nn.functional.l1_loss(reconstructed, mdct_samples, reduction="none").mean(dim=(1,2,3))
        point_loss_mse = torch.nn.functional.mse_loss(reconstructed, mdct_samples, reduction="none").mean(dim=(1,2,3)).detach()

        if point_loss_weight > 0:
            recon_loss =  point_loss * point_loss_weight
        else:
            recon_loss = torch.zeros(mdct_samples.shape[0], device=self.trainer.accelerator.device)
        
        if self.config.mss_loss_weight > 0:
            #reconstructed_raw = self.format.mdct_to_raw(reconstructed, preserve_mel_density_scaling=True)
            #with torch.no_grad():
            #    target_raw = self.format.mdct_to_raw(mdct_samples, preserve_mel_density_scaling=True).requires_grad_(False).detach()
            #mss_abs_loss = self.mss.mss_loss(reconstructed_raw, target_raw)
            mss_abs_loss = self.mss.mss_loss(reconstructed, mdct_samples)
            recon_loss = recon_loss + mss_abs_loss * self.config.mss_loss_weight
        else:
            mss_abs_loss = None

        if self.config.wavelet_loss_weight > 0:
            wavelet_loss, wavelet_level_losses = self.wavelet_loss.wavelet_loss(reconstructed, mdct_samples)
            recon_loss = recon_loss + wavelet_loss * self.config.wavelet_loss_weight
        else:
            wavelet_loss = None
        
        recon_loss_logvar = self.dae.get_recon_loss_logvar()
        recon_loss_nll = recon_loss / recon_loss_logvar.exp() + recon_loss_logvar

        latents_kl_loss_weight = self.config.latents_kl_loss_weight
        if self.trainer.global_step < self.config.kl_warmup_steps:
            warmup_scale = self.trainer.global_step / self.config.kl_warmup_steps
            latents_kl_loss_weight *= warmup_scale

        total_loss = recon_loss_nll + latents_kld * latents_kl_loss_weight

        if self.config.spec_reg_loss_weight > 0:
            spec_reg_loss = self.spec_reg_loss.spec_reg_loss(latents, mdct_samples)
            total_loss = total_loss + spec_reg_loss * self.config.spec_reg_loss_weight
            logs["loss/spec_reg"] = spec_reg_loss.detach()

        logs.update({
            "loss": total_loss,
            "loss/recon": recon_loss.detach(),
            "loss/point": point_loss.detach(),
            "loss/point_mse": point_loss_mse.detach(),
            "loss/kl_latents": latents_kld.detach(),
            "loss_weight/kl_latents": latents_kl_loss_weight,
            "loss_weight/point": point_loss_weight,
            "loss_weight/mss": self.config.mss_loss_weight,
            "loss_weight/wavelet": self.config.wavelet_loss_weight,
            "loss_weight/spec_reg": self.config.spec_reg_loss_weight,
            "io_stats/mdct_samples_std": mdct_samples.std(dim=(1,2,3)),
            "io_stats/mdct_samples_mean": mdct_samples.mean(dim=(1,2,3)),
            "io_stats/recon_mel_std": reconstructed.std(dim=(1,2,3)),
            "io_stats/recon_mel_mean": reconstructed.mean(dim=(1,2,3)),
            "io_stats/latents_std": latents.std(dim=(1,2,3)),
            "io_stats/latents_mean": latents.mean(dim=(1,2,3)),
            "io_stats/latents_sigma": latents_sigma
        })
        
        if mss_abs_loss is not None:
            logs["loss/mss_abs"] = mss_abs_loss.detach()

        if wavelet_loss is not None:
            for i, level_loss in enumerate(wavelet_level_losses):
                logs[f"loss/w_level_{i}"] = level_loss

        if self.trainer.config.enable_debug_mode == True:
            print("mdct_samples.shape:", mdct_samples.shape)
            print("reconstructed.shape:", reconstructed.shape)
            print("latents.shape:", latents.shape)

        for name, block in self.dae.dec.items():
            if block.noise_channels is not None:
                logs[f"io_stats/dec_{name}_noise_gain"] = block.noise_channels_gain.detach()

        #for name, block in self.dae.encoder.enc.items():
        #    logs[f"res_t/enc_{name}"] = block.get_res_balance().detach()

        #for name, block in self.dae.dec.items():
        #    logs[f"res_t/dec_{name}"] = block.get_res_balance().detach()

        return logs