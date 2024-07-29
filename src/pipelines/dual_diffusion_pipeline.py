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

import utils.config as config

import os
import importlib
from dataclasses import dataclass
from typing import Type, List, Optional, Union

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
    sigma_max: Optional[float] = None
    sigma_min: Optional[float] = None
    schedule_params: Optional[dict]      = None
    game_ids: Optional[dict]             = None
    generator: Optional[torch.Generator] = None
    use_midpoint_integration: bool       = True
    add_sample_noise: Optional[float]    = None
    add_class_noise: Optional[float]     = None
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
    def create_new(modules: List[DualDiffusionPipelineModule]) -> "DualDiffusionPipeline":
        
        initialized_modules = {}
        for module in modules:
            initialized_modules[module.module_name] = module.module_class(**module.module_config)

            if hasattr(initialized_modules[module.module_name], "normalize_weights"):
                initialized_modules[module.module_name].normalize_weights()

        return DualDiffusionPipeline(**initialized_modules)
    
    @staticmethod
    @torch.no_grad()
    def from_pretrained(model_path: str,
                        torch_dtype: torch.dtype = torch.float32,
                        device: torch.device = "cpu",
                        load_latest_checkpoints: bool = False,
                        load_emas: Optional[dict] = None ) -> "DualDiffusionPipeline":
        
        model_index = config.load_json(os.path.join(model_path, "model_index.json"))
        model_modules = {}

        # load pipeline modules
        for module_name in model_index:
            if module_name.startswith("_") or not isinstance(model_index[module_name], list): continue

            module_package_name, module_class_name = model_index[module_name][0], model_index[module_name][1]
            module_package = importlib.import_module(module_package_name)
            module_class = getattr(module_package, module_class_name)
            
            module_path = os.path.join(model_path, module_name)
            if load_latest_checkpoints:

                module_checkpoints = []
                for path in os.listdir(model_path):
                    if os.path.isdir(path) and path.startswith(f"{module_name}_checkpoint"):
                        module_checkpoints.append(path)

                if len(module_checkpoints) > 0:
                    module_checkpoints = sorted(module_checkpoints, key=lambda x: int(x.split("-")[1]))
                    module_path = os.path.join(model_path, module_checkpoints[-1], module_name)

            model_modules[module_name] = module_class.from_pretrained(
                module_path, torch_dtype=torch_dtype, device=device).requires_grad_(False).train(False)
        
        # load and merge ema weights
        if load_emas is not None:
            for module_name in load_emas:
                ema_module_path = os.path.join(module_path, f"{module_name}_ema", load_emas[module_name])
                if os.path.exists(ema_module_path):
                    model_modules[module_name].load_state_dict(load_safetensors(ema_module_path))

                    if hasattr(model_modules[module_name], "normalize_weights"):
                        model_modules[module_name].normalize_weights()
                else:
                    raise FileNotFoundError(f"EMA checkpoint '{load_emas[module_name]}' not found in '{os.path.dirname(ema_module_path)}'")
        
        pipeline = DualDiffusionPipeline(**model_modules).to(device=device)

        # load optional dataset info
        dataset_info_path = os.path.join(model_path, "dataset_info.json")
        if os.path.isfile(dataset_info_path):
            pipeline.dataset_info = config.load_json(dataset_info_path)
        else:
            pipeline.dataset_info = None
        
        return pipeline
    
    @torch.no_grad()
    def get_class_label(self, label_name: Union[str, int]) -> int:

        if isinstance(label_name, int):
            return label_name
        elif isinstance(label_name, str):
            if self.dataset_info is None:
                raise ValueError("Unable to retrieve class label, pipeline.dataset_info not found")
            else:
                return self.dataset_info["games"][label_name]
        else:
            raise ValueError(f"Unknown label type '{type(label_name)}'")
        
    @torch.no_grad()
    def get_class_labels(self, labels: Union[int, torch.Tensor, list, dict],
                         module: str = "unet") -> torch.Tensor:

        label_dim = getattr(self, module).label_dim
        assert label_dim > 0, f"{module} label dim must be > 0, got {label_dim}"

        if isinstance(labels, int):
            class_labels = torch.tensor([labels])
        
        if isinstance(labels, torch.Tensor):
            if labels.ndim < 1: labels = labels.unsqueeze(0)
            class_labels = torch.nn.functional.one_hot(labels, num_classes=label_dim)
        
        elif isinstance(labels, list):
            class_labels = torch.zeros((1, label_dim))
            class_labels = class_labels.index_fill_(1, torch.tensor(labels, dtype=torch.long), 1)
        
        elif isinstance(labels, dict):
            _labels = {self.get_class_label(l): w for l, w in labels.items()}

            class_ids = torch.tensor(list(_labels.keys()), dtype=torch.long)
            weights = torch.tensor(list(_labels.values())).float()
            class_labels = torch.zeros((1, label_dim)).scatter_(1, class_ids.unsqueeze(0), weights.unsqueeze(0))
        else:
            raise ValueError(f"Unknown labels dtype '{type(labels)}'")
        
        return class_labels.to(device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def __call__(self, sampling_params: SamplingParams) -> torch.Tensor:
        
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

        if conditioning_perturbation > 0:
            perturbation = torch.randn((batch_size,) + unet_class_embeddings.shape[1:], generator=generator,
                                       device=unet_class_embeddings.device, dtype=unet_class_embeddings.dtype)
            unet_class_embeddings = mp_sum(unet_class_embeddings, perturbation, t=conditioning_perturbation)
            
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

        #sigma_schedule[0] = sigma_max
        #sigma_schedule = sigma_data / torch.linspace(np.arctan(sigma_data / sigma_max), np.arctan(sigma_data / sigma_min), steps+1).tan()
        #sigma_schedule = torch.linspace(np.log(sigma_max), np.log(sigma_min), steps+1).exp()
        #a = 1.7677669529663689
        #a = 0.5 ##3.125#np.exp(-0.54)#torch.pi#torch.pi#4.48
        #theta1 = np.arctan(a / sigma_max); theta0 = np.arctan(a / sigma_min)
        #theta = (1-t_schedule) * (theta0 - theta1) + theta1
        #sigma_schedule = theta.cos() / theta.sin() * a#sigma_data

        #from sigma_sampler import SigmaSampler
        #sigma_sampler = SigmaSampler(sigma_max=sigma_max, sigma_min=sigma_min, sigma_data=sigma_data, distribution="log_sech", dist_offset=-0.54)
        #sigma_schedule = sigma_sampler.sample(0, quantiles = t_schedule).flip(dims=(0,))

        #print("sigma_schedule:", sigma_schedule)

        noise = torch.randn(sample_shape, device=self.device, generator=generator)
        initial_noise = noise * sigma_max
        sample = noise * sigma_schedule[0] + start_data * sigma_data

        normalized_theta_schedule = (sigma_data / sigma_schedule).atan() / (torch.pi/2)
        sigma_schedule_list = sigma_schedule.tolist()

        ln_sigma_error = self.unet.logvar_linear(self.unet.logvar_fourier((sigma_schedule.to(self.unet.device).log()/4).to(self.unet.dtype))).float().flatten()
        #perturbation_gain = (-4 * ln_sigma_error).exp(); perturbation_gain /= perturbation_gain.amax()

        cfg_fn = slerp if slerp_cfg else torch.lerp
    
        debug_v_list = []
        debug_a_list = []
        debug_d_list = []
        debug_s_list = []
        debug_m_list = []

        if show_debug_plots:
            cv2.namedWindow('sample / output', cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow('sample / output', int(sample.shape[3]*3), int(sample.shape[2]*2*3))
            cv2.moveWindow('sample / output', 0, 700)
            cv2.setWindowProperty('sample / output', cv2.WND_PROP_TOPMOST, 1)

        cfg_model_output = torch.rand_like(sample)
        #x = torch.zeros_like(cfg_model_output)
        #sample = normalize(sample) * 2**0.5#***
        i = 0
        self.set_progress_bar_config(disable=True)
        for sigma_curr, sigma_next in self.progress_bar(zip(sigma_schedule_list[:-1],
                                                            sigma_schedule_list[1:]),
                                                            total=len(sigma_schedule_list)-1):

            if input_perturbation > 0:
                #p = input_perturbation * (1 - sigma_next/sigma_curr) * sigma_next
                #p = input_perturbation * (sigma_curr**2 - sigma_next**2)**0.5 #(perturb = 1.618)

                p = input_perturbation * sigma_curr # **** (input_perturbation=0.5)
                #if (p**2 + sigma_curr**2)**0.5 > sigma_max:
                #    p = sigma_max - sigma_curr

                #p = input_perturbation * sigma_curr * debug_a_list[-1].mean().item() * torch.pi if i > 1 else 0 # **** (input_perturbation=0.5)
                #p = input_perturbation * sigma_curr * min(debug_a_list[-1].mean().item() * 4*torch.pi, 1) if i > 1 else 0 # **** (input_perturbation=0.5)
                #p = input_perturbation * sigma_curr * min(ln_sigma_error[i].exp().item()**0.5, 1) if i > 1 else 0 # **** (input_perturbation=0.5)

                #t = sigma_next / sigma_curr if (i+1) < steps else 1
                #p = input_perturbation * sigma_curr * (1 - t) * np.sin(np.arctan(sigma_data / sigma_curr))

                #t = (np.arctan(sigma_data / sigma_curr) - get_cos_angle(initial_noise, sample)).clip(min=0)
                #p = input_perturbation * sigma_curr * np.sin(t[0].item())

                sample += torch.randn(sample.shape, generator=generator, device=sample.device, dtype=sample.dtype) * p
                sigma_curr = (sigma_curr**2 + p**2)**0.5

                """
                sigma_next = 1e-2
                if i > 0:
                    p = sigma_curr
                    sample += torch.randn(sample.shape, generator=generator, device=sample.device, dtype=sample.dtype) * p
                """
                if i == 0:
                    print(sample.std().item(), sigma_curr)

            sigma = torch.tensor([sigma_curr], device=self.device)
            model_input = sample.to(self.unet.dtype)
            #print(model_input.std().item(), sigma_curr)
            
            model_output = self.unet(model_input, sigma, unet_class_embeddings, t_ranges, self.format).float()
            u_model_output = self.unet(model_input, sigma, None, t_ranges, self.format).float()

            last_cfg_model_output = cfg_model_output
            cfg_model_output = cfg_fn(u_model_output, model_output, cfg_scale).float()
    
            #last_x = x
            #x = self.unet.last_x
                
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
            #debug_a_list.append(get_cos_angle(x, last_x) / (torch.pi/2))
            debug_d_list.append(get_cos_angle(initial_noise, sample) / (torch.pi/2))
            debug_s_list.append(cfg_model_output.square().mean(dim=(1,2,3)).sqrt())
            debug_m_list.append(cfg_model_output.mean(dim=(1,2,3)))

            #print(f"step: {(i+1):>{3}}/{steps:>{3}}",
            #      f"v:{debug_v_list[-1][0].item():{8}f}",
            #      f"a:{debug_a_list[-1][0].item():{8}f}",
            #      f"d:{debug_d_list[-1][0].item():{8}f}",
            #      f"s:{debug_s_list[-1][0].item():{8}f}",
            #      f"m:{debug_m_list[-1][0].item():{8}f}")
            
            t = sigma_next / sigma_curr if (i+1) < steps else 0
            #sample = (t * sample + (1 - t) * cfg_model_output)
            alpha = (sigma_next**2 + sigma_data**2)**0.5
            sample = (t * sample + (1 - t) * cfg_model_output)#.clip(min=-3*alpha, max=3*alpha)

            sample0 = sample[0]
            output0 = cfg_model_output[0]
            #output0 = self.unet.last_x[0]

            sample0_img = save_raw_img(sample0, os.path.join(debug_path, f"debug_sample_{i:03}.png"), no_save=True)
            output0_img = save_raw_img(output0, os.path.join(debug_path, f"debug_output_{i:03}.png"), no_save=True)

            if show_debug_plots:
                cv2_img = cv2.vconcat([sample0_img, output0_img]).astype(np.float32)
                #cv2_img = output0_img.astype(np.float32); cv2.resizeWindow('sample / output', int(output0.shape[2]*2), int(output0.shape[1]*2*2))
                cv2_img = (cv2_img[:, :, :3] * cv2_img[:, :, 3:4] / 255).astype(np.uint8)
                cv2.imshow("sample / output", cv2_img)
                cv2.waitKey(1)

            i += 1
                        
        #sample = sample.float() / sigma_data
        sample = normalize(sample).float() * sigma_data#***
        print("sample std:", sample.std().item(), " sample mean:", sample.mean().item())

        v_measured = torch.stack(debug_v_list, dim=0)
        a_measured = torch.stack(debug_a_list, dim=0); a_measured[0] = a_measured[1] # first value is irrelevant
        d_measured = torch.stack(debug_d_list, dim=0)
        s_measured = torch.stack(debug_s_list, dim=0)
        m_measured = torch.stack(debug_m_list, dim=0)

        #print(f"Average v_measured: {v_measured.mean()}")
        #print(f"Average a_measured: {a_measured.mean()}")
        #print(f"Average s_measured: {s_measured.mean()}")
        #print(f"Average m_measured: {m_measured.mean()}")
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
            multi_plot((sigma_schedule.log(), "ln_sigma_schedule"),
                       (a_measured, "path_curvature"),
                       (s_measured, "output_norm"),
                       (m_measured, "output_mean"),
                       (d_measured, "normalized_distance"),
                       (sigma_schedule_error_logvar, "timestep_error_logvar"),
                       layout=(2, 3), figsize=(12, 5),
                       added_plots={4: (normalized_theta_schedule, "theta_schedule")})
            
            cv2.destroyAllWindows()
            
        return raw_sample