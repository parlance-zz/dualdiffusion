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
import json
from dataclasses import dataclass
from typing import Type, List, Optional

import numpy as np
import torch
import cv2

from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.models.modeling_utils import ModelMixin

from utils.dual_diffusion_utils import save_raw, save_raw_img, multi_plot
from utils.dual_diffusion_utils import normalize, slerp, get_cos_angle, load_safetensors

@dataclass
class DualDiffusionPipelineModule:
    module_name: str
    module_class: Type[ModelMixin]
    module_config: dict

@dataclass
class SamplingParams:
    steps: int                 = 100
    seed: Optional[int]        = None
    batch_size: int            = 1
    length: Optional[int]      = None
    cfg_scale: float           = 1.5
    sigma_max: Optional[float] = 200.
    sigma_min: Optional[float] = 0.03
    rho: float                 = 7.
    game_ids: dict
    generator: Optional[torch.Generator] = None
    use_midpoint_integration: bool       = True
    input_perturbation: Optional[float]  = None
    img2img_strength: Optional[float]    = None
    img2img_input: Optional[str]         = None
    schedule: Optional[str]              = "edm2"
    show_debug_plots: bool               = False

class DualDiffusionPipeline(DiffusionPipeline):

    @torch.no_grad()
    def __init__(self, **kwargs):
        super().__init__()
        self.register_modules(**kwargs)

    @staticmethod
    @torch.no_grad()
    def create_new(modules: List[DualDiffusionPipelineModule]):
        
        initialized_modules = {}
        for module in modules:
            initialized_modules[module.module_name] = module.module_class(**module.module_config)

            if hasattr(initialized_modules[module.module_name], "normalize_weights"):
                initialized_modules[module.module_name].normalize_weights()

        return DualDiffusionPipeline(**initialized_modules)
    
    @staticmethod
    @torch.no_grad()
    def from_pretrained(model_path,
                        torch_dtype=torch.float32,
                        device="cpu",
                        load_latest_checkpoints=False,
                        load_ema=None,
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
        
        if load_ema is not None:
            ema_model_path = os.path.join(model_path, unet_checkpoints[-1], "unet_ema", load_ema)
            if os.path.exists(ema_model_path):
                unet.load_state_dict(load_safetensors(ema_model_path))
                unet.normalize_weights()
            else:
                raise FileNotFoundError(f"EMA checkpoint '{load_ema}' not found in '{os.path.dirname(ema_model_path)}'")
            
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
            class_ids, weights = torch.tensor(list(labels.keys()), dtype=torch.long), torch.tensor(list(labels.values())).float()
            return torch.zeros((1, self.unet.label_dim)).scatter_(1, class_ids.unsqueeze(0), weights.unsqueeze(0)).to(device=self.device)
        elif isinstance(labels, int):
            return torch.nn.functional.one_hot(torch.tensor([labels], dtype=torch.long), num_classes=self.unet.label_dim).to(device=self.device, dtype=torch.float32)
        else:
            raise ValueError(f"Unknown labels dtype '{type(labels)}'")

    @torch.no_grad()
    def __call__(self, sampling_params: SamplingParams):

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

            debug_v_list.append(get_cos_angle(sample, cfg_model_output) / (torch.pi/2))
            debug_a_list.append(get_cos_angle(cfg_model_output, last_cfg_model_output) / (torch.pi/2))
            debug_d_list.append(get_cos_angle(initial_noise, sample) / (torch.pi/2))
            debug_s_list.append(cfg_model_output.square().mean(dim=(1,2,3)).sqrt())
            debug_m_list.append(cfg_model_output.mean(dim=(1,2,3)))

            print(f"step: {(i+1):>{3}}/{steps:>{3}}",
                  f"v:{debug_v_list[-1][0].item():{8}f}",
                  f"a:{debug_a_list[-1][0].item():{8}f}",
                  f"d:{debug_d_list[-1][0].item():{8}f}",
                  f"s:{debug_s_list[-1][0].item():{8}f}",
                  f"m:{debug_m_list[-1][0].item():{8}f}")
            
            t = sigma_next / sigma_curr if (i+1) < steps else 0
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