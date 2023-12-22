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

import os
from typing import Union
import numpy as np
import torch

from diffusers.schedulers import DPMSolverMultistepScheduler, DDIMScheduler, DPMSolverSDEScheduler
from diffusers.schedulers import EulerAncestralDiscreteScheduler, KDPM2AncestralDiscreteScheduler
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from unet_dual import UNetDualModel
from autoencoder_kl_dual import AutoencoderKLDual, DiagonalGaussianDistribution
from dual_diffusion_utils import compute_snr, mdct, imdct, normalize_lufs, save_raw

class DualMCLTFormat:

    @staticmethod
    def get_sample_crop_width(model_params):
        block_width = model_params["num_chunks"] * 2
        return model_params["sample_raw_length"] + block_width
    
    @staticmethod
    def get_num_channels(model_params):
        #in_channels = model_params["sample_raw_channels"] * model_params["num_chunks"] * 3   #1d
        #out_channels = model_params["sample_raw_channels"] * model_params["num_chunks"] * 4  #1d
        in_channels = model_params["sample_raw_channels"] * 2   #2d
        out_channels = model_params["sample_raw_channels"] * 4  #2d
        return (in_channels, out_channels)

    @staticmethod
    @torch.no_grad()
    def raw_to_sample(raw_samples, model_params):
        
        num_chunks = model_params["num_chunks"]
        block_width = num_chunks * 2
        
        samples = mdct(raw_samples, block_width, window_degree=1)[..., 1:-2, :]
        samples = samples.permute(0, 2, 1)

        samples *= torch.exp(2j * torch.pi * torch.rand(1, device=samples.device))
        
        if model_params.get("add_abs_input", False):
            samples_abs = samples.abs()
            samples_abs_max = samples_abs.amax(dim=(1,2), keepdim=True).clamp(min=1e-8)
            samples_abs /= samples_abs_max
            samples /= samples_abs_max
            samples /= samples.abs().sqrt().clamp(min=1e-8)

            u = model_params["u"]
            samples_abs = (samples_abs * u).log1p() / np.log(u + 1)

        samples = torch.view_as_real(samples).permute(0, 3, 1, 2).contiguous()

        if model_params.get("add_abs_input", False):
            samples = torch.cat([samples_abs.unsqueeze(1), samples], dim=1)
        else:
            samples /= samples.std(dim=(1,2,3), keepdim=True).clamp(min=1e-8)
            
        #samples = samples.view(samples.shape[0], -1, samples.shape[3]) #1d

        return samples

    @staticmethod
    def sample_to_raw(samples, model_params):
        
        #samples = samples.view(samples.shape[0], -1, model_params["num_chunks"], samples.shape[2])  #1d

        """
        samples_abs = samples[:, 0, :, :].permute(0, 2, 1).contiguous().sigmoid()
        samples_phase = samples[:, 2:, :, :].permute(0, 3, 2, 1).contiguous().tanh()
        samples_phase = torch.view_as_complex(samples_phase)
        samples_waveform = samples_abs * samples_phase
        samples_wave = imdct(samples_waveform, window_degree=2).real

        samples_noise_abs = samples[:, 1, :, :].permute(0, 2, 1).contiguous().sigmoid()
        samples_noise_phase = torch.randn_like(samples_noise_abs)
        samples_noise = samples_noise_abs * samples_noise_phase * 0.5
        samples_noise = imdct(samples_noise, window_degree=2).real

        return samples_wave + samples_noise
        """

        samples_abs = samples[:, 0, :, :].permute(0, 2, 1).contiguous().sigmoid()
        samples_noise_abs = samples[:, 1, :, :].permute(0, 2, 1).contiguous().sigmoid()
        samples_phase = samples[:, 2:, :, :].permute(0, 3, 2, 1).contiguous().tanh()
        samples_phase = torch.view_as_complex(samples_phase) * torch.exp(4j * torch.pi * torch.randn_like(samples_noise_abs) * samples_noise_abs)
        samples_waveform = samples_abs * samples_phase
        samples_wave = imdct(samples_waveform, window_degree=1).real

        return samples_wave

    
    @staticmethod
    def get_sample_shape(model_params, bsz=1, length=1):
        _, num_output_channels = DualMCLTFormat.get_num_channels(model_params)

        crop_width = DualMCLTFormat.get_sample_crop_width(model_params)
        num_chunks = model_params["num_chunks"]
        chunk_len = crop_width // num_chunks - 2

        return (bsz, num_output_channels, num_chunks, chunk_len*length,)   #2d
        #return (bsz, num_output_channels * num_chunks, chunk_len*length,) #1d

class DualDiffusionPipeline(DiffusionPipeline):

    @torch.no_grad()
    def __init__(
        self,
        unet: UNetDualModel,
        scheduler: DDIMScheduler,
        vae: AutoencoderKLDual, 
        model_params: dict = None,
    ):
        super().__init__()

        modules = {
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
        }
        self.register_modules(**modules)
        
        if model_params is not None:
            self.config["model_params"] = model_params
        else:
            model_params = self.config["model_params"]
            
        self.tiling_mode = False

        self.format = DualDiffusionPipeline.get_sample_format(model_params)

    @staticmethod
    @torch.no_grad()
    def get_sample_format(model_params):
        sample_format = model_params["sample_format"]

        if sample_format == "mclt":
            return DualMCLTFormat
        else:
            raise ValueError(f"Unknown sample format '{sample_format}'")
        
    @staticmethod
    @torch.no_grad()
    def create_new(model_params, unet_params, vae_params=None):
        
        unet = UNetDualModel(**unet_params)
        
        beta_schedule = model_params["beta_schedule"]
        if beta_schedule == "trained_betas":
            def alpha_bar_fn(t):
                a = 2 #4
                y = -1/(1 + a*t)**4 + 2/(1 + a*t)**2
                y -= -1/(1 + a)**4 + 2/(1 + a)**2
                return y
            
            trained_betas = []
            num_diffusion_timesteps = 1000
            for i in range(num_diffusion_timesteps):
                t1 = i / num_diffusion_timesteps
                t2 = (i + 1) / num_diffusion_timesteps
                trained_betas.append(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1))
        else:
            trained_betas = None

        scheduler = DDIMScheduler(clip_sample_range=20.,
                                  prediction_type=model_params["prediction_type"],
                                  beta_schedule=beta_schedule,
                                  beta_start=model_params["beta_start"],
                                  beta_end=model_params["beta_end"],
                                  trained_betas=trained_betas,
                                  rescale_betas_zero_snr=model_params["rescale_betas_zero_snr"],)
        
        snr = compute_snr(scheduler, scheduler.timesteps)
        debug_path = os.environ.get("DEBUG_PATH", None)
        if debug_path is not None:
            os.makedirs(debug_path, exist_ok=True)
            snr.log().cpu().numpy().tofile(os.path.join(debug_path, "debug_schedule_ln_snr.raw"))
            np.array(trained_betas).astype(np.float32).tofile(os.path.join(debug_path, "debug_schedule_betas.raw"))

        if vae_params is not None:
            vae = AutoencoderKLDual(**vae_params)
        else:
            vae = None

        return DualDiffusionPipeline(unet, scheduler, vae, model_params=model_params)

    @staticmethod
    @torch.no_grad()
    def add_embeddings(freq_samples, freq_embedding_dim, time_embedding_dim, format_hint="normal", pitch_augmentation=1., tempo_augmentation=1.):
        raise NotImplementedError()
    
    @torch.no_grad()
    def upscale(self, raw_sample):
        raise NotImplementedError()
    
    @torch.no_grad()
    def __call__(
        self,
        steps: int = 100,
        scheduler="dpms++",
        seed: Union[int, torch.Generator]=None,
        loops: int = 0,
        batch_size: int = 1,
        length: int = 1,
    ):
        if (steps <= 0) or (steps > 1000):
            raise ValueError(f"Steps must be between 1 and 1000, got {steps}")
        if loops < 0:
            raise ValueError(f"Loops must be greater than or equal to 0, got {loops}")
        if length <= 0:
            raise ValueError(f"Length must be greater than or equal to 1, got {length}")

        self.set_tiling_mode(loops > 0)

        prediction_type = self.scheduler.config["prediction_type"]
        beta_schedule = self.scheduler.config["beta_schedule"]
        if beta_schedule == "trained_betas":
            trained_betas = self.scheduler.config["trained_betas"]
        else:
            trained_betas = None

        scheduler = scheduler.lower().strip()
        if scheduler == "ddim":
            noise_scheduler = self.scheduler
        elif scheduler == "dpms++":
            noise_scheduler = DPMSolverMultistepScheduler(prediction_type=prediction_type,
                                                          solver_order=3,
                                                          beta_schedule=beta_schedule,
                                                          trained_betas=trained_betas)
        elif scheduler == "kdpm2_a":
            noise_scheduler = KDPM2AncestralDiscreteScheduler(prediction_type=prediction_type,
                                                              beta_schedule=beta_schedule,
                                                              trained_betas=trained_betas)
        elif scheduler == "euler_a":
            noise_scheduler = EulerAncestralDiscreteScheduler(prediction_type=prediction_type,
                                                              beta_schedule=beta_schedule,
                                                              trained_betas=trained_betas)
        elif scheduler == "dpms++_sde":
            if self.unet.dtype != torch.float32:
                raise ValueError("dpms++_sde scheduler requires float32 precision")
            
            noise_scheduler = DPMSolverSDEScheduler(prediction_type=prediction_type,
                                                    beta_schedule=beta_schedule,
                                                    trained_betas=trained_betas)
        else:
            raise ValueError(f"Unknown scheduler '{scheduler}'")
        noise_scheduler.set_timesteps(steps)
        timesteps = noise_scheduler.timesteps

        if isinstance(seed, int):
            if seed == 0: seed = np.random.randint(100000,999999)
            generator = torch.Generator(device=self.device).manual_seed(seed)
        elif isinstance(seed, torch.Generator):
            generator = seed

        model_params = self.config["model_params"]
        
        if getattr(self, "vae", None) is None:
            sample_shape = self.format.get_sample_shape(model_params, bsz=batch_size, length=length)
        else:
            raise NotImplementedError()
        print(f"Sample shape: {sample_shape}")

        sample = torch.randn(sample_shape, device=self.device, dtype=self.unet.dtype, generator=generator)
        sample *= noise_scheduler.init_noise_sigma

        for _, t in enumerate(self.progress_bar(timesteps)):
            
            model_input = sample
            model_input = noise_scheduler.scale_model_input(model_input, t)
            model_output = self.unet(model_input, t).sample

            scheduler_args = {
                "model_output": model_output,
                "timestep": t,
                "sample": sample,
            }
            if scheduler != "dpms++_sde":
                scheduler_args["generator"] = generator
            sample = noise_scheduler.step(**scheduler_args)["prev_sample"]
        
        sample = sample.float()

        debug_path = os.environ.get("DEBUG_PATH", None)
        if debug_path is not None:
            print("Sample std: ", sample.std(dim=(1,2,3)).item())
            os.makedirs(debug_path, exist_ok=True)
            sample.cpu().numpy().tofile(os.path.join(debug_path, "debug_sample.raw"))
        
        if getattr(self, "vae", None) is None:
            raw_sample = self.format.sample_to_raw(sample, model_params).real
        else:
            #raw_sample = self.vae.decode(sample / self.vae.config.scaling_factor).sample
            raise NotImplementedError()

        raw_sample *= 0.18215 / raw_sample.std(dim=1, keepdim=True).clip(min=1e-5)
        if loops > 0: raw_sample = raw_sample.repeat(1, loops+1)
        return raw_sample
    
    @torch.no_grad()
    def set_module_tiling(self, module):
        F, _pair = torch.nn.functional, torch.nn.modules.utils._pair

        padding_modeX = "circular"
        padding_modeY = "constant"

        rprt = module._reversed_padding_repeated_twice
        paddingX = (rprt[0], rprt[1], 0, 0)
        paddingY = (0, 0, rprt[2], rprt[3])

        def _conv_forward(self, input, weight, bias):
            padded = F.pad(input, paddingX, mode=padding_modeX)
            padded = F.pad(padded, paddingY, mode=padding_modeY)
            return F.conv2d(
                padded, weight, bias, self.stride, _pair(0), self.dilation, self.groups
            )

        module._conv_forward = _conv_forward.__get__(module)

    @torch.no_grad()
    def remove_module_tiling(self, module):
        try:
            del module._conv_forward
        except AttributeError:
            pass

    @torch.no_grad()
    def set_tiling_mode(self, tiling: bool):

        if self.tiling_mode == tiling:
            return
        
        modules = [self.unet]
        if getattr(self, "vae", None) is not None:
            modules.append(self.vae)
            
        modules = filter(lambda module: isinstance(module, torch.nn.Module), modules)

        for module in modules:
            for submodule in module.modules():
                if isinstance(submodule, torch.nn.Conv2d | torch.nn.ConvTranspose2d):
                    if isinstance(submodule, torch.nn.ConvTranspose2d):
                        raise NotImplementedError(
                            "Assymetric tiling doesn't support this module"
                        )

                    if tiling is False:
                        self.remove_module_tiling(submodule)
                    else:
                        self.set_module_tiling(submodule)

        self.tiling_mode = tiling