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
from dual_diffusion_utils import compute_snr, mdct, imdct, save_raw, get_mel_density, MSPSD

class DualMSPSDFormat:

    @staticmethod
    def get_sample_crop_width(model_params):
        return model_params["sample_raw_length"]
    
    @staticmethod
    def get_num_channels(model_params):
        mspsd_params = model_params["mspsd_params"]
        num_scales = mspsd_params["high_scale"] - mspsd_params["low_scale"]
        
        in_channels =  model_params["sample_raw_channels"]
        out_channels = model_params["sample_raw_channels"] * num_scales

        return (in_channels, out_channels)

    @staticmethod
    @torch.no_grad()
    def raw_to_sample(raw_samples, model_params, return_dict=False):

        sample_rate = model_params["sample_rate"] 
        mspsd_params = model_params["mspsd_params"]
        mspsd = MSPSD(**mspsd_params, sample_rate=sample_rate)

        psd = mspsd.get_sample_mspsd(raw_samples).requires_grad_(False)
        #cepstrum = mspsd.mspsd_to_cepstrum(psd)
        
        samples = (raw_samples / raw_samples.std(dim=-1, keepdim=True)).unsqueeze(1).requires_grad_(False)

        if return_dict:
            samples_dict = {
                #"samples": cepstrum.requires_grad_(False),
                #"samples_psd": psd.requires_grad_(False),
                #"samples": psd.requires_grad_(False),
                "samples": samples,
                "samples_psd": psd,
            }
            return samples_dict
        else:
            #return cepstrum.requires_grad_(False)
            #return psd.requires_grad_(False)
            return samples

    @staticmethod
    def sample_to_raw(samples, model_params, return_dict=False, num_iterations=400):
        
        sample_rate = model_params["sample_rate"]
        mspsd_params = model_params["mspsd_params"]
        #noise_floor = mspsd_params["noise_floor"]
        mspsd = MSPSD(**mspsd_params, sample_rate=sample_rate)
        #psd = mspsd.cepstrum_to_mspsd(samples)
        #samples = samples.clip(min=np.log(noise_floor), max=0)#.sigmoid().clip(min=noise_floor).log()
        #psd = samples
        psd = mspsd.get_sample_mspsd(samples, separate_scale_channels=True)
        
        if not return_dict:
            return mspsd.get_sample(psd, num_iterations=num_iterations)
        else:
            samples_dict = {
                "samples": samples,
                "samples_psd": psd,
            }
            return samples_dict

    @staticmethod
    def get_loss(sample, target, model_params):
        
        #sample_rate = model_params["sample_rate"]
        #mspsd_params = model_params["mspsd_params"]
        #mspsd = MSPSD(**mspsd_params, sample_rate=sample_rate)
        
        #sample = mspsd.get_mel_weighted_mspsd(sample["samples_psd"])
        #target = mspsd.get_mel_weighted_mspsd(target["samples_psd"])
        #sample = mspsd.get_mel_weighted_mspsd(sample["samples"])
        #target = mspsd.get_mel_weighted_mspsd(target["samples"])
        sample = sample["samples_psd"]
        target = target["samples_psd"]        

        loss_real = (sample - target).abs().mean()

        loss_imag = torch.zeros_like(loss_real)
        return loss_real, loss_imag
    
    @staticmethod
    def get_sample_shape(model_params, bsz=1, length=1):
        _, num_output_channels = DualMSPSDFormat.get_num_channels(model_params)
        crop_width = DualMSPSDFormat.get_sample_crop_width(model_params)

        return (bsz, num_output_channels, crop_width)

class DualMCLTFormat:

    @staticmethod
    def get_sample_crop_width(model_params):
        block_width = model_params["num_chunks"] * 2
        return model_params["sample_raw_length"] + block_width
    
    @staticmethod
    def get_num_channels(model_params):
        in_channels = model_params["sample_raw_channels"] * 2
        out_channels = model_params["sample_raw_channels"] * 3

        return (in_channels, out_channels)

    @staticmethod
    @torch.no_grad()
    def raw_to_sample(raw_samples, model_params, return_dict=False):
        
        noise_floor = model_params["noise_floor"]
        u = model_params["u"]
        block_width = model_params["num_chunks"] * 2

        samples_mdct = mdct(raw_samples, block_width, window_degree=1)[:, :, 1:-2, :]
        samples_mdct = samples_mdct.permute(0, 1, 3, 2)

        samples_mdct *= torch.exp(2j * torch.pi * torch.rand(1, device=samples_mdct.device))
        samples_mdct_abs = samples_mdct.abs()
        samples_mdct_abs = (samples_mdct_abs / samples_mdct_abs.amax(dim=(1,2,3), keepdim=True).clip(min=1e-8)).clip(min=noise_floor)
        
        samples_abs_ln = (samples_mdct_abs * u).log1p() / np.log1p(u) * 2 - 1
        samples_qphase1 = samples_mdct.angle().abs() / torch.pi * 2 - 1
        #samples_qphase2 = (1j*samples_mdct).angle().abs() / torch.pi * 2 - 1
        #samples = torch.cat((samples_abs_ln, samples_qphase1, samples_qphase2), dim=1).requires_grad_(False)
        samples = torch.cat((samples_abs_ln, samples_qphase1), dim=1).requires_grad_(False)

        raw_samples = imdct(samples_mdct.permute(0, 1, 3, 2), window_degree=1).real.requires_grad_(False)

        if return_dict:
            samples_dict = {
                "samples": samples,
                "raw_samples": raw_samples,
                "samples_abs_ln": samples_abs_ln.requires_grad_(False),
                "samples_qphase1": samples_qphase1.requires_grad_(False),
                #"samples_qphase2": samples_qphase2.requires_grad_(False),
            }
            return samples_dict
        else:
            return samples

    @staticmethod
    def sample_to_raw(samples, model_params, return_dict=False):
        
        u = model_params["u"]
        num_channels = model_params["sample_raw_channels"]

        samples = samples.tanh()

        samples_abs_ln, samples_qphase1, samples_qphase2 = samples.chunk(num_channels, dim=1)        
        #phase_norm = (samples_qphase1.abs() + samples_qphase2.abs()).clip(min=1e-10)
        phase_norm = (samples_qphase1.abs() + (samples_qphase2+1)/2).clip(min=1e-10)
        phase = (((samples_qphase1 / phase_norm) + 1) / 2 * torch.pi).cos()
        abs = ((1 + u) ** (samples_abs_ln+1)/2 - 1) / u
        raw_samples = imdct((abs * phase).permute(0, 2, 1), window_degree=1).real

        if not return_dict:
            return raw_samples
        else:
            samples_dict = {
                "samples": samples,
                "raw_samples": raw_samples,
                "samples_abs_ln": samples_abs_ln,
                "samples_qphase1": samples_qphase1,
                #"samples_qphase2": samples_qphase2,
            }
            return samples_dict

    @staticmethod
    def get_loss(sample, target, model_params):
        
        sample_rate = model_params["sample_rate"]
        num_chunks = model_params["num_chunks"]

        samples_abs_ln = sample["samples_abs_ln"]
        samples_qphase1 = sample["samples_qphase1"]
        #samples_qphase2 = sample["samples_qphase2"]

        target_abs_ln = target["samples_abs_ln"]
        target_qphase1 = target["samples_qphase1"]
        #target_qphase2 = target["samples_qphase2"]

        block_hz = torch.arange(1, num_chunks+1, device=target_abs_ln.device) * (sample_rate/2 / num_chunks)
        mel_density = get_mel_density(block_hz).square().view(1, 1,-1, 1).requires_grad_(False)
        mel_density /= mel_density.mean()

        imag_loss_mask = (target_abs_ln + 1) > ((target_abs_ln.amin(dim=1, keepdim=True) + 1) * 1.05)
        real_loss = (samples_abs_ln - target_abs_ln).square() * 2 #4
        #imag_loss = ((samples_qphase1 - target_qphase1).square() + (samples_qphase2 - target_qphase2).square()) * (imag_loss_mask * mel_density)
        imag_loss = (samples_qphase1 - target_qphase1).square() * (imag_loss_mask * mel_density) 

        #real_loss = real_loss.clip(min=1/32).log() - np.log(1/32) # i should try this with a non-log format
        #imag_loss = imag_loss.clip(min=1/32).log() - np.log(1/32)

        return real_loss, imag_loss
    
    @staticmethod
    def get_sample_shape(model_params, bsz=1, length=1):
        _, num_output_channels = DualMCLTFormat.get_num_channels(model_params)

        crop_width = DualMCLTFormat.get_sample_crop_width(model_params)
        num_chunks = model_params["num_chunks"]
        chunk_len = crop_width // num_chunks - 2

        return (bsz, num_output_channels, num_chunks, chunk_len*length,)

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
        elif sample_format == "mspsd":
            return DualMSPSDFormat
        else:
            raise ValueError(f"Unknown sample format '{sample_format}'")
        
    @staticmethod
    @torch.no_grad()
    def create_new(model_params, unet_params, scheduler_params, vae_params=None):
        
        if scheduler_params["beta_schedule"] == "trained_betas":
            raise NotImplementedError()
        scheduler = DDIMScheduler(clip_sample_range=10., **scheduler_params)
        
        snr = compute_snr(scheduler, scheduler.timesteps)
        debug_path = os.environ.get("DEBUG_PATH", None)
        if debug_path is not None:
            save_raw(snr.log(), os.path.join(debug_path, "debug_schedule_ln_snr.raw"))

        unet = UNetDualModel(**unet_params, num_diffusion_timesteps=scheduler.config.num_train_timesteps)

        if vae_params is not None:
            vae = AutoencoderKLDual(**vae_params)
        else:
            vae = None

        return DualDiffusionPipeline(unet, scheduler, vae, model_params=model_params)
    
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
        beta_schedule = "linear"

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
        
        sample_shape = self.format.get_sample_shape(model_params, bsz=batch_size, length=length)
        if getattr(self, "vae", None) is not None:    
            sample_shape = self.vae.get_latent_shape(sample_shape)
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

        debug_path = os.environ.get("DEBUG_PATH", None)
        if debug_path is not None:
            print("Sample std: ", sample.std(dim=(1,2,3)).item())
            print("Sample mean: ", sample.mean(dim=(1,2,3)).item())
            save_raw(sample, os.path.join(debug_path, "debug_sample.raw"))
        
        if getattr(self, "vae", None) is not None:
            sample = sample - sample.mean(dim=(1,2,3), keepdim=True)
            sample = sample / sample.std(dim=(1,2,3), keepdim=True).clip(min=1e-8)
            sample = self.vae.decode(sample).sample
            save_raw(sample, os.path.join(debug_path, "debug_decoded_sample.raw"))

        raw_sample = self.format.sample_to_raw(sample.float(), model_params)
        raw_sample /= raw_sample.std(dim=(1,2), keepdim=True).clip(min=1e-8)
        if loops > 0: raw_sample = raw_sample.repeat(2, loops+1)
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
                if isinstance(submodule, torch.nn.Conv2d | torch.nn.ConvTranspose2d | torch.nn.Conv1d | torch.nn.ConvTranspose1d):

                    if isinstance(submodule, torch.nn.ConvTranspose2d | torch.nn.ConvTranspose1d):
                        continue
                        raise NotImplementedError(
                            "Assymetric tiling doesn't support this module"
                        )

                    if tiling is False:
                        self.remove_module_tiling(submodule)
                    else:
                        self.set_module_tiling(submodule)

        self.tiling_mode = tiling