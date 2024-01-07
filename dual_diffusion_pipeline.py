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
from autoencoder_kl_dual import AutoencoderKLDual
from dual_diffusion_utils import compute_snr, mdct, imdct, save_raw

class DualMCLTFormat:

    @staticmethod
    def get_sample_crop_width(model_params):
        block_width = model_params["num_chunks"] * 2
        return model_params["sample_raw_length"] + block_width
    
    @staticmethod
    def get_num_channels(model_params):
        if model_params.get("qphase_input", False):
            in_channels = model_params["sample_raw_channels"] * (1 + model_params["qphase_nquants"])
            if model_params.get("qphase_nquants", None) is not None:
                out_channels = in_channels
            else:
                out_channels = model_params["sample_raw_channels"] * 3
        else:
            in_channels = model_params["sample_raw_channels"] * 2
            out_channels = model_params["sample_raw_channels"] * 3

        return (in_channels, out_channels)

    @staticmethod
    @torch.no_grad()
    def raw_to_sample(raw_samples, model_params, return_mdct=False):
        
        block_width = model_params["num_chunks"] * 2
        samples_mdct = mdct(raw_samples, block_width, window_degree=1)[..., 1:-2, :]

        samples_mdct *= torch.exp(2j * torch.pi * torch.rand(1, device=samples_mdct.device))

        samples_mdct_abs = samples_mdct.abs()
        samples_mdct_abs_max = samples_mdct_abs.amax(dim=(1,2), keepdim=True).clamp(min=1e-10)
        samples_mdct /= samples_mdct_abs_max
        samples_mdct_abs /= samples_mdct_abs_max
        samples_abs_underflow = samples_mdct_abs < 1e-3
        samples_mdct[samples_abs_underflow] = 1e-3# * torch.exp(2j*torch.pi * torch.rand_like(samples_mdct_abs[samples_abs_underflow]))
        
        if model_params.get("qphase_input", False):
            samples = samples_mdct.permute(0, 2, 1).contiguous()

            u = model_params.get("u", 8000.)
            samples_abs_ln = ((samples.abs() * u).log1p() / np.log1p(u)).unsqueeze(1)

            qphase_nquants = model_params["qphase_nquants"]
            samples_phase = ((samples.angle()+torch.pi) / (2*torch.pi)*qphase_nquants + 0.5).long() % qphase_nquants
            samples_phase = torch.nn.functional.one_hot(samples_phase, num_classes=qphase_nquants).float().permute(0, 3, 1, 2).contiguous()

            samples = torch.cat((samples_abs_ln, samples_phase), dim=1)
        else:
            samples = torch.view_as_real(samples_mdct).permute(0, 3, 2, 1).contiguous()
            samples /= samples.std(dim=(1,2,3), keepdim=True).clamp(min=1e-10)

        if return_mdct:
            return samples, samples_mdct
        else:
            return samples

    @staticmethod
    def sample_to_raw(samples, model_params, return_mixed=True):
        
        samples_abs = samples[:, 0, :, :].permute(0, 2, 1).contiguous().sigmoid()
        #samples_noise_abs = samples[:, 1, :, :].permute(0, 2, 1).contiguous().sigmoid()
        #sample_noise_phase = torch.exp(2j * torch.pi * torch.randn_like(samples_noise_abs) * samples_noise_abs)

        qphase_nquants = model_params.get("qphase_nquants", None)
        if qphase_nquants is None:
            samples_phase = torch.view_as_complex(samples[:, 1:3, :, :].permute(0, 3, 2, 1).contiguous().tanh())
        else:
            samples_phase = torch.distributions.Categorical(logits=samples[:, 3:, :, :].permute(0, 3, 2, 1).contiguous(), validate_args=False).sample().view(samples_abs.shape)
            samples_phase = samples_phase * (2*torch.pi / qphase_nquants)
            samples_phase = samples_phase.cos() + 1j*samples_phase.sin()

        samples_waveform = samples_abs * samples_phase

        if return_mixed:
            return imdct(samples_waveform, window_degree=1).real
        else:
            return samples_waveform

    @staticmethod
    def get_loss(sample, target, model_params):
        
        sample_abs = sample.abs()
        sample_abs_underflow = sample_abs < 1e-3
        sample[sample_abs_underflow] = 1e-3# * torch.exp(2j*torch.pi * torch.rand_like(sample_abs[sample_abs_underflow]))

        error = (sample / target).log()

        error_real = error.real - error.real.mean(dim=(1,2), keepdim=True)
        error_real = error_real.square().mean()

        error_imag_mask = (target.abs() > 1e-3).float()
        error_imag_numel = error_imag_mask.sum(dim=(1,2), keepdim=True).clip(min=1)
        error_imag = error.imag * error_imag_mask
        error_imag = (error_imag - error_imag.sum(dim=(1,2), keepdim=True)/error_imag_numel + torch.pi) % (2*torch.pi) - torch.pi
        wavelength = torch.linspace(error.shape[-1], 1, error.shape[-1], device=error.device)
        error_imag = ((error_imag / torch.pi).abs() * wavelength.view(1, 1, -1)).clip(min=1).log().square().mean()

        #error_imag = ((error_imag + torch.pi) / (2*torch.pi) * wavelength.view(1, 1, -1).clip(min=-1).log2()).round().mean()

        return error_real, error_imag
    
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