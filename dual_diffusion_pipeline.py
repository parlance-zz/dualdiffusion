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
import cv2

from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from unet_edm2 import UNet
from autoencoder_kl_dual import AutoencoderKLDual
from autoencoder_kl_edm2 import AutoencoderKL_EDM2
from spectrogram import SpectrogramParams, SpectrogramConverter
from dual_diffusion_utils import mdct, imdct, save_raw, save_raw_img, multi_plot
from dual_diffusion_utils import normalize, slerp, get_cos_angle, load_safetensors
from loss import DualMultiscaleSpectralLoss, DualMultiscaleSpectralLoss2D

class DualMCLTFormat(torch.nn.Module):

    def __init__(self, model_params):
        super(DualMCLTFormat, self).__init__()

        self.model_params = model_params
        self.loss = DualMultiscaleSpectralLoss(model_params["vae_training_params"])

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
        self.loss = DualMultiscaleSpectralLoss2D(model_params["vae_training_params"])

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
    def __init__(self, unet, vae, model_params=None):
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
            
        self.format = DualDiffusionPipeline.get_sample_format(model_params)

        if vae is not None:
            target_snr = vae.get_target_snr()
        else:
            target_snr = model_params.get("target_snr", None)
        self.target_snr = target_snr

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
    def get_vae_class(model_params):
        vae_class = model_params.get("vae_class", None)
        if vae_class is None or vae_class == "AutoencoderKLDual":
            return AutoencoderKLDual
        elif vae_class == "AutoencoderKL_EDM2":
            return AutoencoderKL_EDM2
        else:
            raise ValueError(f"Unknown vae class '{vae_class}'")

    @staticmethod
    @torch.no_grad()
    def create_new(model_params, unet_params, vae_params=None):
        
        unet = UNet(**unet_params)
        unet.normalize_weights()
        
        vae_class = DualDiffusionPipeline.get_vae_class(model_params)
        if vae_params is not None:
            vae = vae_class(**vae_params)
            vae.normalize_weights()
        else:
            vae = None

        return DualDiffusionPipeline(unet, vae, model_params=model_params)
    
    @staticmethod
    @torch.no_grad()
    def from_pretrained(model_path,
                        torch_dtype=torch.float32,
                        device="cpu",
                        load_latest_checkpoints=False,
                        load_ema=False,
                        requires_grad=False):
        
        with open(os.path.join(model_path, "model_index.json"), "r") as f:
            model_index = json.load(f)
        model_params = model_index["model_params"]

        vae_class = DualDiffusionPipeline.get_vae_class(model_params)
        vae_path = os.path.join(model_path, "vae")
        if load_latest_checkpoints:
            vae_checkpoints = [f for f in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, f)) and f.startswith("vae_checkpoint")]
            if len(vae_checkpoints) > 0:
                vae_checkpoints = sorted(vae_checkpoints, key=lambda x: int(x.split("-")[1]))
                vae_path = os.path.join(model_path, vae_checkpoints[-1], "vae")
        vae = vae_class.from_pretrained(vae_path,
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
        
        if load_ema: # just load the first ema checkpoint found for now, need to integrate post-hoc combining multiple checkpoints ema later
            ema_model_path = os.path.join(model_path, unet_checkpoints[-1], "unet_ema")
            ema_checkpoints = [f for f in os.listdir(ema_model_path) if f.startswith("pf_ema") and f.endswith(".safetensors")]
            if len(ema_checkpoints) > 0:
                unet.ema_checkpoint = ema_checkpoints[0]
                ema_path = os.path.join(ema_model_path, ema_checkpoints[0])
                unet.state_dict().update(load_safetensors(ema_path))
            
        return DualDiffusionPipeline(unet, vae, model_params=model_params).to(device=device)

    # not sure why this is needed, DiffusionPipeline doesn't consider format to be a module, but it is
    @torch.no_grad()
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.format = self.format.to(*args, **kwargs)
        return self
    
    @torch.no_grad()
    def get_class_labels(self, labels):
        if isinstance(labels, torch.Tensor):
            if labels.ndim == 0: labels = labels.unsqueeze(0)
            return torch.nn.functional.one_hot(labels, num_classes=self.unet.label_dim).to(device=self.device, dtype=torch.float32)
        elif isinstance(labels, list):
            class_labels = torch.zeros((1, self.unet.label_dim))
            return class_labels.index_fill_(1, torch.tensor(labels, dtype=torch.long), 1).to(device=self.device)
        elif isinstance(labels, dict):
            class_ids, weights = torch.tensor(list(labels.keys()), dtype=torch.long), torch.tensor(list(labels.values()))
            return torch.zeros((1, self.unet.label_dim)).index_fill_(1, class_ids, weights).to(device=self.device)
        elif isinstance(labels, int):
            return torch.nn.functional.one_hot(torch.tensor([labels], dtype=torch.long), num_classes=self.unet.label_dim).to(device=self.device, dtype=torch.float32)
        else:
            raise ValueError(f"Unknown labels dtype '{type(labels)}'")

    @torch.no_grad()
    def __call__(
        self,
        steps: int = 120,
        seed: Union[int, torch.Generator]=None,
        batch_size: int = 1,
        length: int = 0,
        game_ids = None,
        cfg_scale: float = 5.,
        sigma_max = None,
        sigma_min = None,
        rho = 7,
        slerp_cfg: bool = False,
        use_midpoint_integration: bool = False,
        input_perturbation: float = 0.5,
        img2img_strength: float = 0.8,
        schedule: str = None,
        img2img_input: torch.Tensor = None,
        show_debug_plots: bool = False,
    ):
        if steps <= 0:
            raise ValueError(f"Steps must be > 0, got {steps}")
        if length < 0:
            raise ValueError(f"Length must be greater than or equal to 0, got {length}")
        
        if isinstance(seed, int):
            if seed == 0: seed = np.random.randint(100000, 999999)
            generator = torch.Generator(device=self.device).manual_seed(seed)
        elif isinstance(seed, torch.Generator):
            generator = seed

        debug_path = os.environ.get("DEBUG_PATH", None)
        model_params = self.config["model_params"]

        sample_shape = self.format.get_sample_shape(bsz=batch_size, length=length)
        if getattr(self, "vae", None) is not None:    
            sample_shape = self.vae.get_latent_shape(sample_shape)
        print(f"Sample shape: {sample_shape}")

        sigma_max = sigma_max or self.unet.sigma_max
        sigma_min = sigma_min or self.unet.sigma_min
        sigma_data = self.unet.sigma_data
        
        game_ids = game_ids or torch.randint(0, self.unet.label_dim, 1, device=self.device, generator=generator)
        class_labels = self.get_class_labels(game_ids)
        vae_class_embeddings = self.vae.get_class_embeddings(class_labels)
        unet_class_embeddings = self.unet.get_class_embeddings(class_labels)

        if img2img_input is not None:
            img2img_sample = self.format.raw_to_sample(img2img_input.unsqueeze(0).to(self.device).float())
            start_data = self.vae.encode(img2img_sample.type(self.unet.dtype), vae_class_embeddings, self.format).mode().float()
            start_data = normalize(start_data).float()
            start_timestep = img2img_strength
        else:
            start_data = 0
            start_timestep = 1

        if model_params["t_scale"] is None:
            t_ranges = None
        else:
            t_scale = model_params["t_scale"]
            t_ranges = torch.zeros((batch_size, 2), device=self.device)
            t_ranges[:, 1] = 1
            t_ranges = t_ranges * t_scale - t_scale/2

        def get_timestep_sigma(timesteps):
            return (sigma_max ** (1 / rho) + (1 - timesteps) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
  
        t_schedule = torch.linspace(start_timestep, 0, steps+1)
        sigma_schedule = get_timestep_sigma(t_schedule)

        noise = torch.randn(sample_shape, device=self.device, generator=generator)
        initial_noise = noise * sigma_max
        sample = noise * sigma_schedule[0] + start_data * sigma_data

        normalized_theta_schedule = (sigma_data / sigma_schedule).atan() / (torch.pi/2)
        sigma_schedule_list = sigma_schedule.tolist()

        cfg_fn = slerp if slerp_cfg else torch.lerp

        debug_v_list = []
        debug_a_list = []
        debug_d_list = []
        debug_s_list = []
        debug_m_list = []

        if show_debug_plots:
            cv2.namedWindow('sample / output', cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow('sample / output', int(sample.shape[3]*2), int(sample.shape[2]*2*2))
            cv2.moveWindow('sample / output', 0, 700)
            cv2.setWindowProperty('sample / output', cv2.WND_PROP_TOPMOST, 1)

        cfg_model_output = torch.rand_like(sample)

        i = 0
        self.set_progress_bar_config(disable=True)
        for sigma_curr, sigma_next in self.progress_bar(zip(sigma_schedule_list[:-1],
                                                            sigma_schedule_list[1:]),
                                                            total=len(sigma_schedule_list)-1):
            
            sigma = torch.tensor([sigma_curr], device=self.device)
            model_input = sample.to(self.unet.dtype)
            model_output = self.unet(model_input, sigma, unet_class_embeddings, t_ranges, self.format).float()
            u_model_output = self.unet(model_input, sigma, None, t_ranges, self.format).float()

            last_cfg_model_output = cfg_model_output
            cfg_model_output = cfg_fn(u_model_output, model_output, cfg_scale).float()
            
            if use_midpoint_integration:
                t_hat = sigma_next / sigma_curr
                sample_hat = (t_hat * sample + (1 - t_hat) * cfg_model_output)

                sigma_hat = torch.tensor([sigma_next], device=self.device)
                model_input_hat = sample_hat.to(self.unet.dtype)
                model_output_hat = self.unet(model_input_hat, sigma_hat, unet_class_embeddings, t_ranges, self.format).float()
                u_model_output_hat = self.unet(model_input_hat, sigma_hat, None, t_ranges, self.format).float()
                cfg_model_output_hat = cfg_fn(u_model_output_hat, model_output_hat, cfg_scale).float()
                
                cfg_model_output = (cfg_model_output + cfg_model_output_hat) / 2

            #if i <= 5:
            #    cfg_model_output /= cfg_model_output.square().mean(dim=(1,2,3), keepdim=True).sqrt().clip(min=0.23) / 0.23

            debug_v_list.append(get_cos_angle(sample, cfg_model_output) / (torch.pi/2))
            debug_a_list.append(get_cos_angle(cfg_model_output, last_cfg_model_output) / (torch.pi/2))
            debug_d_list.append(get_cos_angle(initial_noise, sample) / (torch.pi/2))
            debug_s_list.append(cfg_model_output.square().mean(dim=(1,2,3)).sqrt())
            debug_m_list.append(cfg_model_output.mean(dim=(1,2,3)))

            print(f"step: {i:>{3}}/{steps:>{3}}",
                  f"v:{debug_v_list[-1][0].item():{8}f}",
                  f"a:{debug_a_list[-1][0].item():{8}f}",
                  f"d:{debug_d_list[-1][0].item():{8}f}",
                  f"s:{debug_s_list[-1][0].item():{8}f}",
                  f"m:{debug_m_list[-1][0].item():{8}f}")
            
            t = sigma_next / sigma_curr
            sample = (t * sample + (1 - t) * cfg_model_output)

            sample0_img = save_raw_img(sample[0], os.path.join(debug_path, f"debug_sample_{i:03}.png"))
            output0_img = save_raw_img(cfg_model_output[0], os.path.join(debug_path, f"debug_output_{i:03}.png"))

            if show_debug_plots:
                cv2_img = cv2.vconcat([sample0_img, output0_img]).astype(np.float32)
                cv2_img = (cv2_img[:, :, :3] * cv2_img[:, :, 3:4] / 255).astype(np.uint8)
                cv2.imshow("sample / output", cv2_img)
                cv2.waitKey(1)

            i += 1
                        
        sample = sample.float() / sigma_data
        print("sample std:", sample.std().item(), " sample mean:", sample.mean().item())

        v_measured = torch.stack(debug_v_list, dim=0)
        a_measured = torch.stack(debug_a_list, dim=0); a_measured[0] = a_measured[1] # first value is irrelevant
        d_measured = torch.stack(debug_d_list, dim=0)
        s_measured = torch.stack(debug_s_list, dim=0)
        m_measured = torch.stack(debug_m_list, dim=0)

        print(f"Average v_measured: {v_measured.mean()}")
        print(f"Average a_measured: {a_measured.mean()}")
        print(f"Average s_measured: {s_measured.mean()}")
        print(f"Average m_measured: {m_measured.mean()}")
        print(f"Final distance: ", get_cos_angle(initial_noise, sample)[0].item() / (torch.pi/2), "  Final mean:", m_measured[-1, 0].item())

        sigma_schedule_error_logvar = self.unet.logvar_linear(self.unet.logvar_fourier(sigma_schedule.log().to(self.unet.device, self.unet.dtype)/4)).float()

        if debug_path is not None:
            save_raw(sample, os.path.join(debug_path, "debug_sampled_latents.raw"))
            save_raw_img(sample[0], os.path.join(debug_path, "debug_sampled_latents.png"))

            save_raw(v_measured, os.path.join(debug_path, "debug_v_measured.raw"))
            save_raw(a_measured, os.path.join(debug_path, "debug_a_measured.raw"))
            save_raw(d_measured, os.path.join(debug_path, "debug_d_measured.raw"))
            save_raw(s_measured, os.path.join(debug_path, "debug_s_measured.raw"))
            save_raw(m_measured, os.path.join(debug_path, "debug_m_measured.raw"))

            save_raw(sigma_schedule_error_logvar, os.path.join(debug_path, "debug_sigma_schedule_error_logvar.raw"))
            
        if getattr(self, "vae", None) is not None:
            sample = self.vae.decode(sample.to(self.vae.dtype), vae_class_embeddings, self.format).float()
        
        if debug_path is not None:
            save_raw(sample, os.path.join(debug_path, "debug_decoded_sample.raw"))
            save_raw_img(sample[0], os.path.join(debug_path, "debug_decoded_sample.png"))

        raw_sample = self.format.sample_to_raw(sample)

        if show_debug_plots:
            #multi_plot((v_measured, "normalized_velocity"),
            multi_plot((sigma_schedule, "sigma_schedule"),
                       (a_measured, "path_curvature"),
                       (s_measured, "output_norm"),
                       (m_measured, "output_mean"),
                       (d_measured, "normalized_distance"),
                       (sigma_schedule_error_logvar, "timestep_error_logvar"),
                       layout=(2, 3), figsize=(12, 5),
                       added_plots={4: (normalized_theta_schedule, "theta_schedule")})
            
            cv2.destroyAllWindows()
            
        return raw_sample