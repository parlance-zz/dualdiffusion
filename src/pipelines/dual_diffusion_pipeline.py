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
from typing import Optional, Union

import numpy as np
import torch

from modules.module import DualDiffusionModule
from modules.mp_tools import mp_sum
from utils.dual_diffusion_utils import (
    open_img_window, close_img_window, show_img, tensor_to_img, save_img,
    multi_plot, normalize, get_cos_angle, load_safetensors
)
from sampling.schedule import SamplingSchedule

@dataclass
class SampleParams:
    steps: int                  = 100
    seed: Optional[int]         = None
    batch_size: int             = 1
    num_batches: int            = 1
    length: Optional[int]       = None
    cfg_scale: float            = 1.5
    sigma_max: Optional[float]  = None
    sigma_min: Optional[float]  = None
    sigma_data: Optional[float] = None
    schedule: Optional[str]               = "edm2"
    schedule_params: Optional[dict]       = None
    game_ids: Optional[dict]              = None
    generator: Optional[torch.Generator]  = None
    use_midpoint_integration: bool        = True
    input_perturbation: Optional[float]   = None
    conditioning_perturbation: Optional[float] = None
    img2img_strength: Optional[float]          = None
    img2img_input: Optional[Union[str, torch.Tensor]] = None
    num_fgla_iterations: Optional[int]                = None
    

class DualDiffusionPipeline(torch.nn.Module):

    def __init__(self, pipeline_modules: dict[str, DualDiffusionModule]) -> None:
        super().__init__()

        for module_name, module in pipeline_modules.items():
            if not isinstance(module, DualDiffusionModule):
                raise ValueError(f"Module '{module_name}' must be an instance of DualDiffusionModule")
            
            self.add_module(module_name, module)

        self.dtype = torch.get_default_dtype()
        self.device = torch.device("cpu")

    def to(self, device: Optional[torch.device] = None,
           dtype: Optional[torch.dtype] = None, **kwargs) -> "DualDiffusionPipeline":
        super().to(device=device, dtype=dtype, **kwargs)

        self.dtype = dtype or self.dtype
        self.device = device or self.device
        return self
    
    def half(self) -> "DualDiffusionModule":
        return self.to(dtype=torch.bfloat16)
    
    @staticmethod
    def from_pretrained(model_path: str,
                        torch_dtype: torch.dtype = torch.float32,
                        device: Optional[torch.device] = None,
                        load_latest_checkpoints: bool = False,
                        load_emas: Optional[dict[str, str]] = None ) -> "DualDiffusionPipeline":
        
        model_index = config.load_json(os.path.join(model_path, "model_index.json"))
        model_modules: dict[str, DualDiffusionModule] = {}
        load_emas = load_emas or {}
        model_path_contents = os.listdir(model_path)

        # load pipeline modules
        for module_name, module_import_dict in model_index["modules"].items():
            
            module_package = importlib.import_module(module_import_dict["package"])
            module_class: type[DualDiffusionModule] = getattr(module_package, module_import_dict["class"])
            
            module_path = os.path.join(model_path, module_name)
            if load_latest_checkpoints == True:
                
                module_checkpoints: list[str] = []
                for path in model_path_contents:
                    if os.path.isdir(os.path.join(model_path, path)) and path.startswith(f"{module_name}_checkpoint"):
                        module_checkpoints.append(path)

                if len(module_checkpoints) > 0:
                    module_checkpoints = sorted(module_checkpoints, key=lambda x: int(x.split("-")[1]))
                    module_path = os.path.join(model_path, module_checkpoints[-1], module_name)

            model_modules[module_name] = module_class.from_pretrained(
                module_path, torch_dtype=torch_dtype, device=device, load_config_only=module_name in load_emas)
            
            if module_name in load_emas: # load and merge ema weights
                ema_module_path = os.path.join(module_path, load_emas[module_name])
                if os.path.isfile(ema_module_path):
                    model_modules[module_name].load_state_dict(load_safetensors(ema_module_path))

                    if hasattr(model_modules[module_name], "normalize_weights"):
                        model_modules[module_name].normalize_weights()
                else:
                    raise FileNotFoundError(f"EMA checkpoint '{load_emas[module_name]}' not found in '{os.path.dirname(ema_module_path)}'")
        
        pipeline = DualDiffusionPipeline(model_modules).to(device=device)

        # load optional dataset info
        dataset_info_path = os.path.join(model_path, "dataset_info.json")
        if os.path.isfile(dataset_info_path):
            pipeline.dataset_info = config.load_json(dataset_info_path)
            pipeline.dataset_game_ids = pipeline.dataset_info["game_id"]
            pipeline.dataset_game_names = {value: key for key, value in pipeline.dataset_info["game_id"].items()}
        else:
            pipeline.dataset_info = None
            pipeline.dataset_game_names = None
        
        return pipeline
    
    def save_pretrained(self, model_path: str, subfolder: Optional[str] = None) -> None:
        
        if subfolder is not None:
            model_path = os.path.join(model_path, subfolder)
        os.makedirs(model_path, exist_ok=True)
        
        model_modules: dict[str, dict[str, str]] = {}
        for module_name, module in self.named_children():
            if not isinstance(module, DualDiffusionModule):
                continue
            
            module_import_dict = {
                "package": module.__class__.__module__,
                "class": module.__class__.__name__
            }
            model_modules[module_name] = module_import_dict
            module.save_pretrained(model_path, subfolder=module_name)

        model_index = {"modules": model_modules}
        config.save_json(model_index, os.path.join(model_path, "model_index.json"))

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
    def get_class_labels(self, labels: Union[int, torch.Tensor, list[int], dict[str, float]],
                         module: str = "unet") -> torch.Tensor:

        label_dim = getattr(self, module).config.label_dim
        assert label_dim > 0, f"{module} label dim must be > 0, got {label_dim}"
        
        if isinstance(labels, int):
            labels = torch.tensor([labels])
        
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

    def get_latent_shape(self, sample_shape: Union[torch.Size, tuple[int, int, int, int]]) -> torch.Size:
        latent_shape = self.vae.get_latent_shape(sample_shape)
        return self.unet.get_latent_shape(latent_shape)
    
    def get_sample_shape(self, bsz: int = 1, length: Optional[int] = None) -> tuple:
        sample_shape = self.format.get_sample_shape(bsz=bsz, length=length)
        latent_shape = self.get_latent_shape(sample_shape)
        return self.vae.get_sample_shape(latent_shape)
    
    @torch.inference_mode()
    def __call__(self, params: SampleParams) -> torch.Tensor:
        
        params.seed = params.seed or np.random.randint(100000, 999999)
        params.generator = params.generator or torch.Generator(
            device=self.device).manual_seed(params.seed)

        sample_shape = self.get_sample_shape(bsz=params.batch_size, length=params.length)
        latent_shape = self.get_latent_shape(sample_shape)
        print(f"Sample shape: {sample_shape} Latent shape: {latent_shape}")

        params.game_ids = params.game_ids or torch.randint(0, self.unet.label_dim, 1,
                                            device=self.device, generator=params.generator)
        vae_class_labels = self.get_class_labels(params.game_ids, module="vae")
        vae_class_embeddings = self.vae.get_class_embeddings(vae_class_labels)
        unet_class_labels = self.get_class_labels(params.game_ids, module="unet")
        unet_class_embeddings = self.unet.get_class_embeddings(unet_class_labels)

        if params.add_class_noise > 0:
            class_noise = torch.randn((params.batch_size,) + unet_class_embeddings.shape[1:],
                generator=params.generator, device=unet_class_embeddings.device, dtype=unet_class_embeddings.dtype)
            unet_class_embeddings = mp_sum(unet_class_embeddings, class_noise, t=params.add_class_noise)
            
        if params.img2img_input is not None:
            if params.img2img_input.ndim < 2: params.img2img_input = params.img2img_input.unsqueeze(0)
            img2img_sample = self.format.raw_to_sample(params.img2img_input.to(self.device).float())

            start_data = self.vae.encode(
                img2img_sample.type(self.unet.dtype), vae_class_embeddings, self.format).mode().float()
            start_data = normalize(start_data).float()
            start_timestep = params.img2img_strength
        else:
            start_data = 0
            start_timestep = 1

        if model_params["t_scale"] is None:
            t_ranges = None
        else:
            t_scale = model_params["t_scale"]
            t_ranges = torch.zeros((params.batch_size, 2), device=self.device)
            t_ranges[:, 1] = 1
            t_ranges = t_ranges * t_scale - t_scale/2

        params.sigma_max  = params.sigma_max  or self.unet.sigma_max
        params.sigma_min  = params.sigma_min  or self.unet.sigma_min
        params.sigma_data = params.sigma_data or self.unet.sigma_data
        sigma_schedule_params = {
            "sigma_max": params.sigma_max,
            "sigma_min": params.sigma_min,
            **params.schedule_params,
        }
        sigma_schedule = SamplingSchedule.get_schedule(
            params.schedule, params.steps + 1, start_timestep, **sigma_schedule_params)
        sigma_schedule_list = sigma_schedule.tolist()

        start_noise = torch.randn(latent_shape, device=self.device, generator=params.generator)
        sample = start_noise * sigma_schedule[0] + start_data * params.sigma_data

        ln_sigma_error = self.unet.logvar_linear(
            self.unet.logvar_fourier((sigma_schedule.to(self.unet.device).log()/4).to(self.unet.dtype))).float().flatten()

        debug_v_list = []
        debug_a_list = []
        debug_d_list = []
        debug_s_list = []
        debug_m_list = []

        if params.show_debug_plots:
            window_h, window_w = int(latent_shape.shape[2]*3 * 2), int(latent_shape.shape[3]*3)
            open_img_window("x(t) / x(0)", width=window_w, height=window_h, topmost=True)

        cfg_model_output = torch.zeros_like(sample)

        i = 0
        for sigma_curr, sigma_next in self.progress_bar(zip(sigma_schedule_list[:-1],
                                                            sigma_schedule_list[1:]),
                                                            total=len(sigma_schedule_list)-1):

            if params.add_sample_noise > 0:
                added_noise_sigma = params.add_sample_noise * sigma_curr
                sample += torch.randn(sample.shape, generator=params.generator,
                                      device=sample.device, dtype=sample.dtype) * added_noise_sigma
                sigma_curr = (sigma_curr**2 + added_noise_sigma**2)**0.5

            sigma = torch.tensor([sigma_curr], device=self.device)
            model_input = sample.to(self.unet.dtype)
            
            model_output = self.unet(model_input, sigma, unet_class_embeddings, t_ranges, self.format).float()
            u_model_output = self.unet(model_input, sigma, None, t_ranges, self.format).float()

            last_cfg_model_output = cfg_model_output
            cfg_model_output = torch.lerp(u_model_output, model_output, params.cfg_scale).float()
    
            #last_x = x
            #x = self.unet.last_x
                
            if params.use_midpoint_integration:
                t_hat = sigma_next / sigma_curr
                sample_hat = (t_hat * sample + (1 - t_hat) * cfg_model_output)

                sigma_hat = torch.tensor([sigma_next], device=self.device)
                model_input_hat = sample_hat.to(self.unet.dtype)
                model_output_hat = self.unet(model_input_hat, sigma_hat, unet_class_embeddings, t_ranges, self.format).float()
                u_model_output_hat = self.unet(model_input_hat, sigma_hat, None, t_ranges, self.format).float()
                cfg_model_output_hat = torch.lerp(u_model_output_hat, model_output_hat, params.cfg_scale).float()
                
                cfg_model_output = (cfg_model_output + cfg_model_output_hat) / 2

            debug_v_list.append(get_cos_angle(sample, cfg_model_output) / (torch.pi/2))
            debug_a_list.append(get_cos_angle(cfg_model_output, last_cfg_model_output) / (torch.pi/2))
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