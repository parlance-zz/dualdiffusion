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
from dual_diffusion_utils import mdct, imdct, save_raw, save_raw_img
from geodesic_flow import GeodesicFlow, normalize, slerp, get_cos_angle
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
    def get_positional_embedding(self, x, t_ranges, mode="linear", num_fourier_channels=0):

        mels = torch.linspace(self.mels_min, self.mels_max, x.shape[2] + 2, device=x.device)[1:-1]
        ln_freqs = DualSpectrogramFormat._mel_to_hz(mels, mel_scale=self.spectrogram_params.mel_scale_type).log2()

        if mode == "linear":
            emb_freq = ln_freqs.view(1, 1,-1, 1).repeat(x.shape[0], 1, 1, x.shape[3])
            emb_freq = (emb_freq - emb_freq.mean()) / emb_freq.std()

            if t_ranges is not None:
                t = torch.linspace(0, 1, x.shape[3], device=x.device).view(-1, 1)
                t = ((1 - t) * t_ranges[:, 0] + t * t_ranges[:, 1]).permute(1, 0).view(x.shape[0], 1, 1, x.shape[3])
                emb_time = t.repeat(1, 1, x.shape[2], 1)

                return torch.cat((emb_freq, emb_time), dim=1).to(x.dtype)
            else:
                return emb_freq.to(x.dtype)

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

        if vae is not None:
            target_snr = vae.get_target_snr()
        else:
            target_snr = model_params.get("target_snr", 1e4)

        self.geodesic_flow = GeodesicFlow(target_snr,
                                          model_params["diffusion_schedule"],
                                          model_params["diffusion_objective"])

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
        unet = UNet.from_pretrained(unet_path,
                                    torch_dtype=torch_dtype,
                                    device=device).requires_grad_(requires_grad).train(requires_grad)
        
        return DualDiffusionPipeline(unet, vae, model_params=model_params)
        
    @torch.no_grad()
    def __call__(
        self,
        steps: int = 120,
        seed: Union[int, torch.Generator]=None,
        loops: int = 0,
        batch_size: int = 1,
        length: int = 0,
        game_ids = None,
        cfg_scale: float = 5.,
        v_scale: float = 1.,
        use_midpoint_integration: bool = False,
        input_perturbation: float = 0.5,
        img2img_strength: float = 0.8,
        img2img_input: torch.Tensor = None,
    ):
        if steps <= 0:
            raise ValueError(f"Steps must be > 0, got {steps}")
        if loops < 0:
            raise ValueError(f"Loops must be greater than or equal to 0, got {loops}")
        if length < 0:
            raise ValueError(f"Length must be greater than or equal to 0, got {length}")

        debug_path = os.environ.get("DEBUG_PATH", None)
        
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

        sample = torch.randn(sample_shape, device=self.device, generator=generator)
        sample = normalize(sample, zero_mean=True)

        if img2img_input is not None:
            img2img_sample = self.format.raw_to_sample(img2img_input.unsqueeze(0).to(self.device).float())
            latents = self.vae.encode(img2img_sample.type(self.unet.dtype), return_dict=False)[0].mode()
            sample = self.geodesic_flow.add_noise(latents, sample, torch.tensor([img2img_strength], device=sample.device, dtype=sample.dtype))
            start_timestep = img2img_strength
        else:
            start_timestep = 1

        initial_noise = sample.clone()

        if game_ids is not None:
            labels = torch.tensor(game_ids, device=self.device, dtype=torch.long)
            #assert labels.shape[0] == batch_size
        else:
            labels = torch.randint(0, self.unet.label_dim, (batch_size,), device=self.device, generator=generator)

        if model_params["t_scale"] is None:
            t_ranges = None
        else:
            t_scale = model_params["t_scale"]
            t_ranges = torch.zeros((batch_size, 2), device=self.device)
            t_ranges[:, 1] = 1
            t_ranges = t_ranges * t_scale - t_scale/2

        t_schedule = torch.linspace(start_timestep, 0, steps+1)[:-1].tolist()

        debug_v_list = []
        debug_a_list = []
        debug_d_list = []
        debug_s_list = []
        debug_o_list = []

        cfg_model_output = torch.rand_like(sample)

        self.set_progress_bar_config(disable=True)
        for i, t in enumerate(self.progress_bar(t_schedule)):

            model_input = sample.to(self.unet.dtype)
            model_output = self.unet(model_input, t, labels, t_ranges, self.format)
            u_model_output = self.unet(model_input, t, None, t_ranges, self.format)

            last_cfg_model_output = cfg_model_output
            cfg_model_output = slerp(u_model_output, model_output, cfg_scale)

            if use_midpoint_integration:            
                raise NotImplementedError("Midpoint integration not implemented")
                sample_m = slerp(sample.float(), model_output, v_schedule[i]/2)
                sample_m -= sample_m.mean(dim=(1,2,3), keepdim=True)
                sample_m /= sample_m.square().mean(dim=(1,2,3), keepdim=True).sqrt()
                
                sample_m = sample_m.to(self.unet.dtype)
                model_output, logvar = self.unet(sample_m, (t - v_schedule[i]/2) * (timescale * torch.pi/2), labels, self.format, return_logvar=True)
                u_model_output = self.unet(sample_m, (t - v_schedule[i]/2) * (timescale * torch.pi/2), None, self.format)
                model_output = slerp(u_model_output.float(), model_output.float(), cfg_scale)

            debug_v_list.append(get_cos_angle(sample, cfg_model_output) / (torch.pi/2))
            debug_a_list.append(get_cos_angle(cfg_model_output, last_cfg_model_output) / (torch.pi/2))
            debug_d_list.append(get_cos_angle(initial_noise, sample) / (torch.pi/2))
            debug_s_list.append(cfg_model_output.std(dim=(1,2,3)))
            debug_o_list.append(cfg_model_output)

            print(f"step: {i:>{3}}/{steps:>{3}}",
                  f"v:{debug_v_list[-1][0].item():{8}f}",
                  f"a:{debug_a_list[-1][0].item():{8}f}",
                  f"d:{debug_d_list[-1][0].item():{8}f}",
                  f"s:{debug_s_list[-1][0].item():{8}f}")
            
            next_t = t_schedule[i+1] if i+1 < len(t_schedule) else 0
            sample = self.geodesic_flow.reverse_step(sample, cfg_model_output, v_scale, input_perturbation, t, next_t)

            save_raw_img(sample[0], os.path.join(debug_path, f"debug_sample_{i:03}.png"))
            save_raw_img(normalize(cfg_model_output[0], zero_mean=True), os.path.join(debug_path, f"debug_output_{i:03}.png"))
                        
        sample = sample.float()

        v_measured = torch.stack(debug_v_list, dim=0)
        a_measured = torch.stack(debug_a_list, dim=0)
        d_measured = torch.stack(debug_d_list, dim=0)
        s_measured = torch.stack(debug_s_list, dim=0)
        o_measured = torch.stack(debug_o_list, dim=0)

        print(f"Average v_measured: {v_measured.mean()}")
        print(f"Average a_measured: {a_measured.mean()}")
        print(f"Average s_measured: {s_measured.mean()}")
        print(f"Final distance: ", get_cos_angle(initial_noise, sample)[0].item() / (torch.pi/2))

        if debug_path is not None:
            save_raw(sample, os.path.join(debug_path, "debug_sampled_latents.raw"))
            save_raw_img(sample[0], os.path.join(debug_path, "debug_sampled_latents.png"))

            save_raw(v_measured, os.path.join(debug_path, "debug_v_measured.raw"))
            save_raw(a_measured, os.path.join(debug_path, "debug_a_measured.raw"))
            save_raw(d_measured, os.path.join(debug_path, "debug_d_measured.raw"))
            save_raw(s_measured, os.path.join(debug_path, "debug_s_measured.raw"))

            model_outputs = o_measured.view(steps, batch_size, -1)
            inner_products = (model_outputs.unsqueeze(0) * model_outputs.unsqueeze(1)).sum(dim=-1).permute(2, 0, 1)
            save_raw_img(inner_products[0], os.path.join(debug_path, "debug_o_inner_products.png"))
            
        if getattr(self, "vae", None) is not None:
            sample = self.vae.decode(sample.to(self.vae.dtype)).sample.float()
        
        if debug_path is not None:
            save_raw(sample, os.path.join(debug_path, "debug_decoded_sample.raw"))
            save_raw_img(sample[0], os.path.join(debug_path, "debug_decoded_sample.png"))

        raw_sample = self.format.sample_to_raw(sample)
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