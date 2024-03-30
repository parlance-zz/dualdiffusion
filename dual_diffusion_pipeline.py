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

from diffusers.pipelines.pipeline_utils import DiffusionPipeline

#from unet_dual import UNetDualModel
from unet_edm2 import UNet
from autoencoder_kl_dual import AutoencoderKLDual
from spectrogram import SpectrogramParams, SpectrogramConverter
from dual_diffusion_utils import mdct, imdct, save_raw, save_raw_img, slerp
from loss import DualMultiscaleSpectralLoss, DualMultiscaleSpectralLoss2D

class DualMCLTFormat(torch.nn.Module):

    def __init__(self, model_params):
        super(DualMCLTFormat, self).__init__()

        self.model_params = model_params
        self.loss = DualMultiscaleSpectralLoss(model_params["loss_params"])

    def get_sample_crop_width(self, length=0):
        block_width = self.model_params["num_chunks"] * 2
        if length <= 0: length = self.model_params["sample_raw_length"]
        return length // block_width // 64 * 64 * block_width + block_width
    
    def get_num_channels(self):
        in_channels = self.model_params["sample_raw_channels"] * 2
        out_channels = self.model_params["sample_raw_channels"] * 2
        return (in_channels, out_channels)

    def multichannel_transform(self, wave):
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

    @torch.no_grad()
    def raw_to_sample(self, raw_samples, return_dict=False):
        
        noise_floor = self.model_params["noise_floor"]
        block_width = self.model_params["num_chunks"] * 2

        samples_mdct = mdct(raw_samples, block_width, window_degree=1)[:, :, 1:-2, :]
        samples_mdct = samples_mdct.permute(0, 1, 3, 2)
        samples_mdct *= torch.exp(2j * torch.pi * torch.rand(1, device=samples_mdct.device))
        samples_mdct = self.multichannel_transform(samples_mdct)

        samples_mdct_abs = samples_mdct.abs()
        samples_mdct_abs_amax = samples_mdct_abs.amax(dim=(1,2,3), keepdim=True).clip(min=1e-5)
        samples_mdct_abs = (samples_mdct_abs / samples_mdct_abs_amax).clip(min=noise_floor)
        samples_abs_ln = samples_mdct_abs.log()
        samples_qphase1 = samples_mdct.angle().abs()
        samples = torch.cat((samples_abs_ln, samples_qphase1), dim=1)

        samples_mdct /= samples_mdct_abs_amax
        raw_samples = imdct(self.multichannel_transform(samples_mdct).permute(0, 1, 3, 2), window_degree=1).real.requires_grad_(False)

        if return_dict:
            samples_dict = {
                "samples": samples,
                "raw_samples": raw_samples,
            }
            return samples_dict
        else:
            return samples

    def sample_to_raw(self, samples, return_dict=False, original_samples_dict=None):
        
        samples_abs, samples_phase1 = samples.chunk(2, dim=1)
        samples_abs = samples_abs.exp()
        samples_phase = samples_phase1.cos()
        raw_samples = imdct(self.multichannel_transform(samples_abs * samples_phase).permute(0, 1, 3, 2), window_degree=1).real

        if original_samples_dict is not None:
            orig_samples_abs, orig_samples_phase1 = original_samples_dict["samples"].chunk(2, dim=1)
            orig_samples_abs = orig_samples_abs.exp()
            orig_samples_phase = orig_samples_phase1.cos()

            raw_samples_orig_phase = imdct(self.multichannel_transform(samples_abs * orig_samples_phase).permute(0, 1, 3, 2), window_degree=1).real
            raw_samples_orig_abs = imdct(self.multichannel_transform(orig_samples_abs * samples_phase).permute(0, 1, 3, 2), window_degree=1).real
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

    def get_loss(self, sample, target):
        return self.loss(sample, target, self.model_params)

    def get_sample_shape(self, bsz=1, length=0):
        _, num_output_channels = self.get_num_channels()

        crop_width = self.get_sample_crop_width(length=length)
        num_chunks = self.model_params["num_chunks"]
        chunk_len = crop_width // num_chunks - 2

        return (bsz, num_output_channels, num_chunks, chunk_len,)

class DualSpectrogramFormat(torch.nn.Module):

    def __init__(self, model_params):
        super(DualSpectrogramFormat, self).__init__()

        self.model_params = model_params
        self.spectrogram_params = SpectrogramParams(sample_rate=model_params["sample_rate"],
                                                    stereo=model_params["sample_raw_channels"] == 2,
                                                    **model_params["spectrogram_params"])
        
        self.spectrogram_converter = SpectrogramConverter(self.spectrogram_params)
        self.loss = DualMultiscaleSpectralLoss2D(model_params["loss_params"])

        self.mels_min = DualSpectrogramFormat._hz_to_mel(self.spectrogram_params.min_frequency)
        self.mels_max = DualSpectrogramFormat._hz_to_mel(self.spectrogram_params.max_frequency)

    def get_sample_crop_width(self, length=0):
        if length <= 0: length = self.model_params["sample_raw_length"]
        return self.spectrogram_converter.get_crop_width(length)
    
    def get_num_channels(self):
        in_channels = out_channels = self.model_params["sample_raw_channels"]
        return (in_channels, out_channels)

    @torch.no_grad()
    def raw_to_sample(self, raw_samples, return_dict=False):
        
        noise_floor = self.model_params["noise_floor"]
        samples = self.spectrogram_converter.audio_to_spectrogram(raw_samples)
        samples /= samples.std(dim=(1,2,3), keepdim=True).clip(min=noise_floor)

        if return_dict:
            samples_dict = {
                "samples": samples,
                "raw_samples": raw_samples,
            }
            return samples_dict
        else:
            return samples

    def sample_to_raw(self, samples, return_dict=False, decode=True):
        
        if decode:
            raw_samples = self.spectrogram_converter.spectrogram_to_audio(samples.clip(min=0))
        else:
            raw_samples = None

        if not return_dict:         
            return raw_samples
        else:
            samples_dict = {
                "samples": samples,
                "raw_samples": raw_samples,
            }
            return samples_dict

    def get_loss(self, sample, target):
        return self.loss(sample, target, self.model_params)
    
    def get_sample_shape(self, bsz=1, length=0):
        _, num_output_channels = self.get_num_channels()
        crop_width = self.get_sample_crop_width(length=length)
        audio_shape = torch.Size((bsz, num_output_channels, crop_width))
        
        spectrogram_shape = self.spectrogram_converter.get_spectrogram_shape(audio_shape)
        return tuple(spectrogram_shape)

    @staticmethod    
    def _hz_to_mel(freq, mel_scale="htk"):
        if mel_scale != "htk":
            raise NotImplementedError("Only HTK mel scale is supported")
        return 2595. * np.log10(1. + (freq / 700.))
        
    @staticmethod
    def _mel_to_hz(mels, mel_scale="htk"):
        if mel_scale != "htk":
            raise NotImplementedError("Only HTK mel scale is supported")
        return 700. * (10. ** (mels / 2595.) - 1.)

    #@torch.no_grad()
    def get_positional_embedding(self, x, n_channels, mode="linear"):

        mels = torch.linspace(self.mels_min, self.mels_max, x.shape[2] + 2, device=x.device)[1:-1]
        ln_freqs = DualSpectrogramFormat._mel_to_hz(mels, mel_scale=self.spectrogram_params.mel_scale_type).log2()

        if mode == "linear":
            emb = ln_freqs.view(1, 1,-1, 1).repeat(x.shape[0], 1, 1, x.shape[3])
            return ((emb - emb.mean()) / emb.std())#.requires_grad_(False)
        elif mode == "fourier":
            raise NotImplementedError("Fourier positional embeddings not implemented")
        else:
            raise ValueError(f"Unknown mode '{mode}'")

class DualDiffusionPipeline(DiffusionPipeline):

    @torch.no_grad()
    def __init__(
        self,
        #unet: UNetDualModel,
        unet: UNet,
        vae: AutoencoderKLDual, 
        model_params: dict = None,
    ):
        super().__init__()

        modules = {
            "unet": unet,
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
            return DualMCLTFormat(model_params)
        elif sample_format == "spectrogram":
            return DualSpectrogramFormat(model_params)
        else:
            raise ValueError(f"Unknown sample format '{sample_format}'")
        
    @staticmethod
    @torch.no_grad()
    def create_new(model_params, unet_params, vae_params=None):
        
        #unet = UNetDualModel(**unet_params)
        unet = UNet(**unet_params)

        if vae_params is not None:
            vae = AutoencoderKLDual(**vae_params)
        else:
            vae = None

        return DualDiffusionPipeline(unet, vae, model_params=model_params)
    
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
        #unet = UNetDualModel.from_pretrained(unet_path,
        unet = UNet.from_pretrained(unet_path,
                                    torch_dtype=torch_dtype,
                                    device=device).requires_grad_(requires_grad).train(requires_grad)
        
        return DualDiffusionPipeline(unet, vae, model_params=model_params)
        
    @torch.no_grad()
    def __call__(
        self,
        steps: int = 100,
        seed: Union[int, torch.Generator]=None,
        loops: int = 0,
        batch_size: int = 1,
        length: int = 0,
        use_midpoint_integration: bool = True,
        use_perturbation: bool = True,
    ):
        if steps <= 0:
            raise ValueError(f"Steps must be > 0, got {steps}")
        if loops < 0:
            raise ValueError(f"Loops must be greater than or equal to 0, got {loops}")
        if length < 0:
            raise ValueError(f"Length must be greater than or equal to 0, got {length}")

        self.set_tiling_mode(loops > 0)

        if isinstance(seed, int):
            if seed == 0: seed = np.random.randint(100000,999999)
            generator = torch.Generator(device=self.device).manual_seed(seed)
        elif isinstance(seed, torch.Generator):
            generator = seed

        model_params = self.config["model_params"]
        
        sample_shape = self.format.get_sample_shape(bsz=batch_size, length=length)
        if getattr(self, "vae", None) is not None:    
            sample_shape = self.vae.get_latent_shape(sample_shape)
        print(f"Sample shape: {sample_shape}")

        sample = torch.randn(sample_shape, device=self.device,
                             dtype=self.unet.dtype,
                             generator=generator)
        sample -= sample.mean(dim=(1,2,3), keepdim=True)
        sample /= sample.square().mean(dim=(1,2,3), keepdim=True).sqrt()

        """
        from dual_diffusion_utils import load_audio
        crop_width = self.format.get_sample_crop_width(length=length)
        dataset_path = os.environ.get("DATASET_PATH", "./dataset/samples_hq")
        img2img_input_path = np.random.choice(os.listdir(dataset_path), 1, replace=False)[0]
        img2img_input_path = "Final Fantasy - Mystic Quest  [Mystic Quest Legend] - 07 Battle 1.flac"
        test_sample = load_audio(os.path.join(dataset_path, img2img_input_path), start=0, count=crop_width)
        test_sample = self.format.raw_to_sample(test_sample.unsqueeze(0).to(self.device))
        latents = self.vae.encode(test_sample.type(self.unet.dtype), return_dict=False)[0].mode()
        latents -= latents.mean()
        latents /= latents.std()

        img2img_strength = 0.85#2
        sample = slerp(sample, latents, 1 - img2img_strength)
        sample -= sample.mean(dim=(1,2,3), keepdim=True)
        sample /= sample.square().mean(dim=(1,2,3), keepdim=True).sqrt()
        print(img2img_input_path)
        """

        #v_schedule = (torch.linspace(0.5/steps, 1-0.5/steps, steps) * torch.pi).sin()# **2
        v_schedule = torch.ones(steps)

        v_schedule /= v_schedule.sum()
        t_schedule = (1 - v_schedule.cumsum(dim=0) + v_schedule[0]).tolist()
        v_schedule = v_schedule.tolist()

        #t_schedule = t_schedule[int(steps*(1-img2img_strength)+0.5):]
        #v_schedule = v_schedule[int(steps*(1-img2img_strength)+0.5):]

        for i, t in enumerate(self.progress_bar(t_schedule)):
            
            model_output, logvar = self.unet(sample, t, None, self.format, return_logvar=True)
            if use_perturbation:
                model_output += torch.randn_like(model_output) * (logvar/2).exp()
                model_output -= model_output.mean(dim=(1,2,3), keepdim=True)
                model_output /= model_output.square().mean(dim=(1,2,3), keepdim=True).sqrt()

            if use_midpoint_integration: # geodesic flow with v pred and midpoint euler integration

                sample_m = slerp(sample, model_output, v_schedule[i]/2)
                sample_m -= sample_m.mean(dim=(1,2,3), keepdim=True)
                sample_m /= sample_m.square().mean(dim=(1,2,3), keepdim=True).sqrt()
                
                model_output, logvar = self.unet(sample_m, t - v_schedule[i]/2, None, self.format, return_logvar=True)
                if use_perturbation:
                    model_output += torch.randn_like(model_output) * (logvar/2).exp()
                    model_output -= model_output.mean(dim=(1,2,3), keepdim=True)
                    model_output /= model_output.square().mean(dim=(1,2,3), keepdim=True).sqrt()

            sample = slerp(sample, model_output, v_schedule[i])
            sample -= sample.mean(dim=(1,2,3), keepdim=True)
            sample /= sample.square().mean(dim=(1,2,3), keepdim=True).sqrt()

        debug_path = os.environ.get("DEBUG_PATH", None)
        if debug_path is not None:
            print("Sample std: ", sample.std(dim=(1,2,3)).item())
            print("Sample mean: ", sample.mean(dim=(1,2,3)).item())
            save_raw(sample, os.path.join(debug_path, "debug_latents.raw"))
        
        if getattr(self, "vae", None) is not None:
            sample = sample * model_params["latent_std"] + model_params["latent_mean"]
            save_raw_img(sample, os.path.join(debug_path, "debug_latents.png"))
            sample = self.vae.decode(sample).sample
            save_raw(sample, os.path.join(debug_path, "debug_decoded_sample.raw"))
            save_raw_img(sample, os.path.join(debug_path, "debug_decoded_sample.png"))

        raw_sample = self.format.sample_to_raw(sample.float())
        if loops > 0: raw_sample = raw_sample.repeat(1, 1, loops+1)
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