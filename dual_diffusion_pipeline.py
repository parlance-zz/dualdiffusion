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
        out_channels = model_params["sample_raw_channels"] * 2

        return (in_channels, out_channels)

    @staticmethod
    @torch.no_grad()
    def raw_to_sample(raw_samples, model_params, return_dict=False):
        
        noise_floor = model_params["noise_floor"]
        u = model_params["u"]
        block_width = model_params["num_chunks"] * 2

        samples_mdct = mdct(raw_samples, block_width, window_degree=1)[..., 1:-2, :]
        samples_mdct = samples_mdct.permute(0, 2, 1)

        samples_mdct *= torch.exp(2j * torch.pi * torch.rand(1, device=samples_mdct.device))
        samples_mdct_abs = samples_mdct.abs()
        samples_mdct_abs = (samples_mdct_abs / samples_mdct_abs.amax(dim=(1,2), keepdim=True).clip(min=1e-8)).clip(min=noise_floor)
        
        samples_abs_ln = ((samples_mdct_abs * u).log1p() / np.log1p(u)).unsqueeze(1)
        samples_qphase = (samples_mdct.angle().abs() / torch.pi).unsqueeze(1)
        samples = torch.cat((samples_abs_ln, samples_qphase), dim=1)

        if return_dict:
            samples_dict = {
                "samples": samples,
                "raw_samples": imdct(samples_mdct.permute(0, 2, 1), window_degree=1).real.requires_grad_(False),
                #"samples_abs": samples_mdct_abs.requires_grad_(False),
                #"samples_phase": samples_qphase.squeeze(1).requires_grad_(False),
            }
            return samples_dict
        else:
            return samples

    @staticmethod
    def sample_to_raw(samples, model_params, return_dict=False):
        
        noise_floor = model_params["noise_floor"]

        samples_abs = samples[:, 0, :, :].sigmoid()
        samples_qphase = samples[:, 1, :, :].sigmoid()

        samples_abs = (samples_abs.permute(0, 2, 1) / samples_abs.amax(dim=(1,2), keepdim=True).clip(min=1e-8)).clip(min=noise_floor)
        samples_phase = (samples_qphase.permute(0, 2, 1) * torch.pi).cos()
        raw_samples = imdct(samples_abs * samples_phase, window_degree=1).real

        if not return_dict:
            return raw_samples
        else:
            samples_dict = {
                #"samples_abs": samples_abs.permute(0, 2, 1),
                #"samples_phase": samples_qphase,
                "raw_samples": raw_samples,
            }
            return samples_dict

    @staticmethod
    def get_loss(sample, target, model_params):
        
        loss = torch.zeros(1, device=sample["raw_samples"].device)
        return loss, loss

        noise_floor = model_params["noise_floor"]
        block_width = model_params["num_chunks"] * 2
        sample_rate = model_params["sample_rate"]
        #u = model_params["u"]

        samples_abs = sample["samples_abs"]
        samples_phase = sample["samples_phase"]
        target_abs = target["samples_abs"]
        target_phase = target["samples_phase"]

        block_hz = torch.arange(1, block_width//2+1, device=target_abs.device) * (sample_rate/2 / (block_width//2))
        mel_density = get_mel_density(block_hz).requires_grad_(False).view(1, -1, 1).requires_grad_(False)

        #samples_wave = samples_abs * (samples_phase * torch.pi).cos()
        #target_wave = target_abs * (target_phase * torch.pi).cos()
        #loss = (((samples_wave - target_wave).abs() * (u / 2)).log1p() / np.log1p(u) * mel_density).mean()

        samples_wave = samples_abs * (samples_phase * torch.pi).cos()
        target_wave = target_abs * (target_phase * torch.pi).cos()

        #target_freq_pow = target_abs.square().mean(dim=1, keepdim=True)
        #target_time_pow = target_abs.square().mean(dim=2, keepdim=True)
        #target_pow = ((target_freq_pow * target_time_pow) ** (1/4)).clip(min=noise_floor*10)

        #target_mask = target_abs > noise_floor
        #loss = ((samples_wave - target_wave).abs() / target_pow * mel_density * target_mask).mean() / 16

        target_mask = target_abs > noise_floor
        loss = ((samples_wave - target_wave).abs() / (samples_wave.abs().detach() + target_wave.abs() + noise_floor*10) * mel_density * target_mask).mean() / 8


        #target_abs_ln = target_abs.log()
        #target_abs_ln -= target_abs_ln.amin(dim=(1,2), keepdim=True)
        #target_abs_ln /= -np.log(noise_floor)
        #loss = ((samples_phase - target_phase).abs() * target_abs_ln * mel_density).mean()

        return loss, torch.zeros_like(loss)
    
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
        elif sample_format == "mspsd":
            return DualMSPSDFormat
        else:
            raise ValueError(f"Unknown sample format '{sample_format}'")
        
    @staticmethod
    @torch.no_grad()
    def create_new(model_params, unet_params, vae_params=None):
        
        beta_schedule = model_params["beta_schedule"]
        if beta_schedule == "trained_betas":
            raise NotImplementedError()

        scheduler = DDIMScheduler(clip_sample_range=20.,
                                  prediction_type=model_params["prediction_type"],
                                  beta_schedule=beta_schedule,
                                  beta_start=model_params["beta_start"],
                                  beta_end=model_params["beta_end"],
                                  rescale_betas_zero_snr=model_params["rescale_betas_zero_snr"],)
        
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

            vae_latent_channels = self.vae.config.latent_channels
            vae_downsample_ratio = self.vae.config.downsample_ratio
            vae_num_blocks = len(self.vae.config.block_out_channels)

            if len(sample_shape) == 4:
                sample_shape = (sample_shape[0],
                                vae_latent_channels,
                                sample_shape[2] // vae_downsample_ratio[0] ** (vae_num_blocks-1),
                                sample_shape[3] // vae_downsample_ratio[1] ** (vae_num_blocks-1))
            else:
                if isinstance(vae_downsample_ratio, tuple):
                    vae_downsample_ratio = vae_downsample_ratio[0]
                sample_shape = (sample_shape[0], vae_latent_channels, sample_shape[2] // vae_downsample_ratio ** (vae_num_blocks-1))

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
            sample = self.vae.decode(sample).sample
            save_raw(sample, os.path.join(debug_path, "debug_decoded_sample.raw"))

        raw_sample = self.format.sample_to_raw(sample.float(), model_params)
        raw_sample *= 0.18215 / raw_sample.std(dim=1, keepdim=True).clip(min=1e-8)
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