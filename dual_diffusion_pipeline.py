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
import json

import numpy as np
import torch

from diffusers.schedulers import DPMSolverMultistepScheduler, DDIMScheduler, DPMSolverSDEScheduler
from diffusers.schedulers import EulerAncestralDiscreteScheduler, KDPM2AncestralDiscreteScheduler
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from unet_dual import UNetDualModel
from autoencoder_kl_dual import AutoencoderKLDual
from dual_diffusion_utils import compute_snr, mdct, imdct, save_raw, save_raw_img, dict_str

class DualMCLTFormat:

    @staticmethod
    def get_sample_crop_width(model_params, length=0):
        block_width = model_params["num_chunks"] * 2
        if length <= 0: length = model_params["sample_raw_length"]
        return length // block_width // 64 * 64 * block_width + block_width
    
    @staticmethod
    def get_num_channels(model_params):
        in_channels = model_params["sample_raw_channels"] * 2
        out_channels = model_params["sample_raw_channels"] * 2
        return (in_channels, out_channels)

    @staticmethod
    def multichannel_transform(wave):
        if wave.shape[1] == 1:
            return wave
        elif wave.shape[1] == 2:
            return torch.stack((wave[:, 0] + wave[:, 1], wave[:, 0] - wave[:, 1]), dim=1) / (2 ** 0.5)
        else: # we would need to do a (ortho normalized) dct/idct over the channel dim here
            raise NotImplementedError("Multichannel transform not implemented for > 2 channels")

    """
    @staticmethod
    def cos_angle_to_norm_angle(x):
        return (x / torch.pi * 2 - 1).clip(min=-.99999, max=.99999).erfinv()
    
    @staticmethod
    def norm_angle_to_cos_angle(x):
        return (x.erf() + 1) / 2 * torch.pi
    """

    @staticmethod
    @torch.no_grad()
    def raw_to_sample(raw_samples, model_params, return_dict=False):
        
        noise_floor = model_params["noise_floor"]
        block_width = model_params["num_chunks"] * 2

        samples_mdct = mdct(raw_samples, block_width, window_degree=1)[:, :, 1:-2, :]
        samples_mdct = samples_mdct.permute(0, 1, 3, 2)
        samples_mdct *= torch.exp(2j * torch.pi * torch.rand(1, device=samples_mdct.device))
        samples_mdct = DualMCLTFormat.multichannel_transform(samples_mdct)

        samples_mdct_abs = samples_mdct.abs()
        samples_mdct_abs_amax = samples_mdct_abs.amax(dim=(1,2,3), keepdim=True).clip(min=1e-5)
        samples_mdct_abs = (samples_mdct_abs / samples_mdct_abs_amax).clip(min=noise_floor)
        samples_abs_ln = samples_mdct_abs.log()
        samples_qphase1 = samples_mdct.angle().abs()
        samples = torch.cat((samples_abs_ln, samples_qphase1), dim=1)

        samples_mdct /= samples_mdct_abs_amax
        raw_samples = imdct(DualMCLTFormat.multichannel_transform(samples_mdct).permute(0, 1, 3, 2), window_degree=1).real.requires_grad_(False)

        if return_dict:
            samples_dict = {
                "samples": samples,
                "raw_samples": raw_samples,
            }
            return samples_dict
        else:
            return samples

    @staticmethod
    def sample_to_raw(samples, model_params, return_dict=False, original_samples_dict=None):
        
        samples_abs, samples_phase1 = samples.chunk(2, dim=1)
        samples_abs = samples_abs.exp()
        samples_phase = samples_phase1.cos()
        raw_samples = imdct(DualMCLTFormat.multichannel_transform(samples_abs * samples_phase).permute(0, 1, 3, 2), window_degree=1).real

        if original_samples_dict is not None:
            orig_samples_abs, orig_samples_phase1 = original_samples_dict["samples"].chunk(2, dim=1)
            orig_samples_abs = orig_samples_abs.exp()
            orig_samples_phase = orig_samples_phase1.cos()

            raw_samples_orig_phase = imdct(DualMCLTFormat.multichannel_transform(samples_abs * orig_samples_phase).permute(0, 1, 3, 2), window_degree=1).real
            raw_samples_orig_abs = imdct(DualMCLTFormat.multichannel_transform(orig_samples_abs * samples_phase).permute(0, 1, 3, 2), window_degree=1).real
        else:
            raw_samples_orig_phase = None
            raw_samples_orig_abs = None

        if not return_dict:         
            return raw_samples
        else:
            samples_dict = {
                "samples": samples,
                "raw_samples": raw_samples,
                "raw_samples_orig_phase": raw_samples_orig_phase,
                "raw_samples_orig_abs": raw_samples_orig_abs,
            }
            return samples_dict

    @staticmethod
    def get_sample_shape(model_params, bsz=1, length=0):
        _, num_output_channels = DualMCLTFormat.get_num_channels(model_params)

        crop_width = DualMCLTFormat.get_sample_crop_width(model_params, length=length)
        num_chunks = model_params["num_chunks"]
        chunk_len = crop_width // num_chunks - 2

        return (bsz, num_output_channels, num_chunks, chunk_len,)

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
    
    @staticmethod
    @torch.no_grad()
    def from_pretrained(model_path,
                        torch_dtype=torch.float32,
                        device="cpu",
                        load_latest_checkpoints=False,
                        requires_grad=False):
        
        with open(os.path.join(model_path, "model_index.json"), "r") as f:
            model_index = json.load(f)
        model_params = model_index["model_params"]

        vae_path = os.path.join(model_path, "vae")
        if load_latest_checkpoints:
            vae_checkpoints = [f for f in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, f)) and f.startswith("vae_checkpoint")]
            if len(vae_checkpoints) > 0:
                vae_checkpoints = sorted(vae_checkpoints, key=lambda x: int(x.split("-")[1]))
                vae_path = os.path.join(model_path, vae_checkpoints[-1], "vae")
        vae = AutoencoderKLDual.from_pretrained(vae_path,
                                                torch_dtype=torch_dtype,
                                                device=device).requires_grad_(requires_grad).train(requires_grad)

        unet_path = os.path.join(model_path, "unet")
        if load_latest_checkpoints:
            unet_checkpoints = [f for f in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, f)) and f.startswith("unet_checkpoint")]
            if len(unet_checkpoints) > 0:
                unet_checkpoints = sorted(unet_checkpoints, key=lambda x: int(x.split("-")[1]))
                unet_path = os.path.join(model_path, unet_checkpoints[-1], "unet")
        unet = UNetDualModel.from_pretrained(unet_path,
                                             torch_dtype=torch_dtype,
                                             device=device).requires_grad_(requires_grad).train(requires_grad)

        scheduler_path = os.path.join(model_path, "scheduler")
        scheduler = DDIMScheduler.from_pretrained(scheduler_path, torch_dtype=torch_dtype, device=device)
        
        return DualDiffusionPipeline(unet, scheduler, vae, model_params=model_params)
        
    @torch.no_grad()
    def __call__(
        self,
        steps: int = 100,
        scheduler="dpms++",
        seed: Union[int, torch.Generator]=None,
        loops: int = 0,
        batch_size: int = 1,
        length: int = 0,
        beta_schedule: str = None,
        beta_start: float = None,
        beta_end: float = None,
    ):
        if (steps <= 0) or (steps > 1000):
            raise ValueError(f"Steps must be between 1 and 1000, got {steps}")
        if loops < 0:
            raise ValueError(f"Loops must be greater than or equal to 0, got {loops}")
        if length < 0:
            raise ValueError(f"Length must be greater than or equal to 0, got {length}")

        self.set_tiling_mode(loops > 0)

        scheduler_args = {
            "prediction_type": self.scheduler.config["prediction_type"],
            "beta_schedule": beta_schedule or self.scheduler.config["beta_schedule"],
            "trained_betas": self.scheduler.config["trained_betas"],
            "beta_start": beta_start or self.scheduler.config["beta_start"],
            "beta_end": beta_end or self.scheduler.config["beta_end"],
            "rescale_betas_zero_snr": self.scheduler.config["rescale_betas_zero_snr"],
        }
        scheduler = scheduler.lower().strip()
        if scheduler == "ddim":
            noise_scheduler = DDIMScheduler(**scheduler_args, clip_sample_range=10)
        elif scheduler == "dpms++":
            scheduler_args.pop("rescale_betas_zero_snr")
            noise_scheduler = DPMSolverMultistepScheduler(**scheduler_args,solver_order=3)
        elif scheduler == "kdpm2_a":
            scheduler_args.pop("rescale_betas_zero_snr")
            noise_scheduler = KDPM2AncestralDiscreteScheduler(**scheduler_args)
        elif scheduler == "euler_a":
            noise_scheduler = EulerAncestralDiscreteScheduler(**scheduler_args)
        elif scheduler == "dpms++_sde":
            if self.unet.dtype != torch.float32:
                raise ValueError("dpms++_sde scheduler requires float32 precision")
            scheduler_args.pop("rescale_betas_zero_snr")
            noise_scheduler = DPMSolverSDEScheduler(**scheduler_args)
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

        sample = torch.randn(sample_shape, device=self.device,
                             dtype=self.unet.dtype,
                             generator=generator) * noise_scheduler.init_noise_sigma

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
            save_raw(sample, os.path.join(debug_path, "debug_latents.raw"))
        
        if getattr(self, "vae", None) is not None:
            #sample = sample - sample.mean(dim=(1,2,3), keepdim=True)
            #sample = sample / sample.std(dim=(1,2,3), keepdim=True).clip(min=1e-8)
            sample = sample * model_params["latent_std"] + model_params["latent_mean"]
            save_raw_img(sample, os.path.join(debug_path, "debug_latents.png"))
            sample = self.vae.decode(sample).sample
            save_raw(sample, os.path.join(debug_path, "debug_decoded_sample.raw"))

        raw_sample = self.format.sample_to_raw(sample.float(), model_params)
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