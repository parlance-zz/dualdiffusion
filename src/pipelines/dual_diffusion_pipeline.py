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
    multi_plot, normalize, get_cos_angle, load_safetensors, torch_dtype
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
    
@dataclass
class ModuleInventory:
    name: str
    checkpoints: list[str]
    emas: dict[str, list[str]]

class DualDiffusionPipeline(torch.nn.Module):

    def __init__(self, pipeline_modules: dict[str, DualDiffusionModule]) -> None:
        super().__init__()

        for module_name, module in pipeline_modules.items():
            if not isinstance(module, DualDiffusionModule):
                raise ValueError(f"Module '{module_name}' must be an instance of DualDiffusionModule")
            
            self.add_module(module_name, module)

    def to(self, device: Optional[Union[dict[str, torch.device], torch.device]] = None,
                 dtype:  Optional[Union[dict[str, torch.dtype],  torch.dtype]]  = None,
                 **kwargs) -> "DualDiffusionPipeline":
        
        if device is not None:
            if isinstance(device, dict):
                for module_name, device in device.items():
                    getattr(self, module_name).to(device=device)
            else:
                for module in self.children():
                    module.to(device=device)

        if dtype is not None:
            if isinstance(dtype, dict):
                for module_name, module_dtype in dtype.items():
                    getattr(self, module_name).to(dtype=torch_dtype(module_dtype))
            else:
                for module in self.children():
                    module.to(dtype=torch_dtype(dtype))

        if len(kwargs) > 0:
            for module in self.children():
                module.to(**kwargs)
        
        return self
    
    def half(self) -> "DualDiffusionModule":
        for module in self.children():
            module.to(dtype=torch.bfloat16)
        return self
    
    def compile(self, compile_options: dict) -> None:
        # ugly way to check if compile_options is a per-module dict
        if (all(isinstance(value, dict) for value in compile_options.values()) and
                    not any(key=="options" for key in compile_options.keys())):
            for module_name, module in self.named_children():
                if module_name in compile_options:
                    module.compile(**compile_options[module_name])
        else:
            for module_name, module in self.named_children():
                module.compile(**compile_options)
                
    @staticmethod
    def get_model_module_inventory(model_path: str) -> dict[str, ModuleInventory]:
        
        model_index = config.load_json(os.path.join(model_path, "model_index.json"))
        model_inventory: dict[str, ModuleInventory] = {}

        def get_ema_list(module_path: str) -> list[str]:
            ema_list = []
            for path in os.listdir(module_path):
                if os.path.isfile(os.path.join(module_path, path)) and path.startswith("pf_ema"):
                    ema_list.append(path)
            return sorted(ema_list)
        
        for module_name, _ in model_index["modules"].items():
            module_inventory = ModuleInventory(module_name, [], {})
            
            # get and sort module checkpoints
            for path in os.listdir(model_path):
                if os.path.isdir(os.path.join(model_path, path)):
                    if path.startswith(f"{module_name}_checkpoint"):
                        module_inventory.checkpoints.append(path)

            module_inventory.checkpoints = sorted(module_inventory.checkpoints, key=lambda x: int(x.split("-")[1]))

            # get ema list for each checkpoint
            module_inventory.emas[""] = get_ema_list(os.path.join(model_path, module_name))
            for checkpoint in module_inventory.checkpoints:
                module_inventory.emas[checkpoint] = get_ema_list(os.path.join(model_path, checkpoint, module_name))

            model_inventory[module_name] = module_inventory

        return model_inventory

    @staticmethod
    def get_model_module_classes(model_path: str) -> dict[str, type[DualDiffusionModule]]:
        
        model_index = config.load_json(os.path.join(model_path, "model_index.json"))
        model_module_classes: dict[str, type[DualDiffusionModule]] = {}
        
        for module_name, module_import_dict in model_index["modules"].items():
            module_package = importlib.import_module(module_import_dict["package"])
            module_class = getattr(module_package, module_import_dict["class"])
            model_module_classes[module_name] = module_class
        
        return model_module_classes
    
    @staticmethod
    def from_pretrained(model_path: str,
                        torch_dtype: Union[dict[str, torch.dtype], torch.dtype] = torch.float32,
                        device: Optional[Union[dict[str, torch.device], torch.device]] = None,
                        load_checkpoints: Optional[Union[dict[str, str], bool]] = False,
                        load_emas: Optional[Union[dict[str, str], bool]] = False,
                        compile_options: Optional[Union[dict[str, dict], dict]] = None) -> "DualDiffusionPipeline":
        
        model_module_classes = DualDiffusionPipeline.get_model_module_classes(model_path)
        model_inventory = DualDiffusionPipeline.get_model_module_inventory(model_path)

        load_checkpoints = load_checkpoints or False
        if isinstance(load_checkpoints, bool):
            load_checkpoints = {}
            if load_checkpoints == True:
                for module_name, module_inventory in model_inventory.items():
                    if len(module_inventory.checkpoints) > 0:
                        load_checkpoints[module_name] = module_inventory.checkpoints[-1]

        load_emas = load_emas or False
        if isinstance(load_emas, bool):
            load_emas = {}
            if load_emas == True:
                for module_name, module_inventory in model_inventory.items():
                    module_checkpoint = load_checkpoints.get(module_name, "")
                    if len(module_inventory.emas[module_checkpoint]) > 0:
                        load_emas[module_name] = module_inventory.emas[module_checkpoint][-1]

        # load pipeline modules
        model_modules: dict[str, DualDiffusionModule] = {}
        for module_name, module_class in model_module_classes.items():

            # load module weights / checkpoint
            module_checkpoint = load_checkpoints.get(module_name, "")
            module_path = os.path.join(model_path, module_checkpoint, module_name)
            model_modules[module_name] = module_class.from_pretrained(
                module_path, device=device, load_config_only=module_name in load_emas)
            
            # load and merge ema weights
            if module_name in load_emas:
                ema_module_path = os.path.join(module_path, load_emas[module_name])                
                model_modules[module_name].load_ema(ema_module_path)
        
        pipeline = DualDiffusionPipeline(model_modules).to(device=device, dtype=torch_dtype)

        if compile_options is not None:
            pipeline.compile(compile_options)

        # load dataset info
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
                         module_name: str = "unet") -> torch.Tensor:

        module: DualDiffusionModule = getattr(self, module_name)
        label_dim = module.config.label_dim
        assert label_dim > 0, f"{module_name} label dim must be > 0, got {label_dim}"
        
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
        
        return class_labels.to(device=module.device, dtype=module.dtype)

    def get_latent_shape(self, sample_shape: Union[torch.Size, tuple[int, int, int, int]]) -> torch.Size:
        latent_shape = self.vae.get_latent_shape(sample_shape)
        return self.unet.get_latent_shape(latent_shape)
    
    def get_sample_shape(self, bsz: int = 1, length: Optional[int] = None) -> tuple:
        sample_shape = self.format.get_sample_shape(bsz=bsz, length=length)
        latent_shape = self.get_latent_shape(sample_shape)
        return self.vae.get_sample_shape(latent_shape)
    
    @torch.inference_mode()
    def __call__(self, params: SampleParams) -> torch.Tensor:
        
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

        temperature_scale = max(min(temperature_scale, 1), 0)

        game_ids = game_ids or torch.randint(0, self.unet.label_dim, 1, device=self.device, generator=generator)
        original_game_ids = {**game_ids}
        class_labels = self.get_class_labels(game_ids)
        vae_class_embeddings = self.vae.get_class_embeddings(class_labels)
        unet_class_embeddings = self.unet.get_class_embeddings(class_labels)
        
        #unet_class_embeddings = normalize(unet_class_embeddings).float() * 1.17
        """
        unet_class_embeddings = []
        for g in game_ids:
            l = self.unet.get_class_embeddings(self.get_class_labels(g))
            unet_class_embeddings.append(l)
        unet_class_embeddings = torch.cat(unet_class_embeddings, dim=0)
        #mask = (torch.randn(unet_class_embeddings.shape, generator=generator, device=unet_class_embeddings.device)*0.5).sigmoid()
        mask = (torch.rand(unet_class_embeddings.shape, generator=generator, device=unet_class_embeddings.device) > 0.5).float()
        print(mask)
        unet_class_embeddings = normalize(unet_class_embeddings[0:1].repeat(batch_size,1) * mask**0.5 + unet_class_embeddings[1:2].repeat(batch_size,1) * (1-mask)**0.5).to(dtype=self.unet.dtype)
        print(unet_class_embeddings.shape, unet_class_embeddings.dtype, unet_class_embeddings.std())
        """
        print("unet_class_embeddings mean:", unet_class_embeddings.mean().item(), unet_class_embeddings.dtype, "std:", unet_class_embeddings.std().item())
        
        #conditioning_perturbation = max(1.08 - unet_class_embeddings.std().item(), 0)**0.5
        #print("conditioning_perturbation", conditioning_perturbation)

        #if static_conditioning_perturbation > 0:
        #    perturbation = torch.randn((batch_size,) + unet_class_embeddings.shape[1:], generator=generator,
        #                               device=unet_class_embeddings.device, dtype=unet_class_embeddings.dtype)
        #    unet_class_embeddings = mp_sum(unet_class_embeddings, perturbation, t=static_conditioning_perturbation)

        if img2img_input is not None:
            img2img_sample = self.format.raw_to_sample(img2img_input.unsqueeze(0).to(self.device).float())
            start_data = self.vae.encode(img2img_sample.type(self.unet.dtype), vae_class_embeddings, self.format).mode().float()
            start_data = normalize(start_data).float()
            #start_timestep = img2img_strength
            start_timestep = 1
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

        #sigma_schedule = (torch.rand(steps+1, device=self.device, generator=generator) * (np.log(sigma_max) - np.log(sigma_min)) + np.log(sigma_min)).exp()
        #sigma_schedule = sigma_schedule.cumsum(dim=0).flip(dims=(0,))
        #sigma_schedule /= sigma_schedule.amax() / sigma_max
        #sigma_schedule -= sigma_schedule.amin() - sigma_min
        
        #sigma_schedule -= sigma_schedule.amin() - sigma_min
        #sigma_schedule /= sigma_schedule.amax() / sigma_max

        #sigma_schedule[0] = sigma_max
        #sigma_schedule = sigma_data / torch.linspace(np.arctan(sigma_data / sigma_max), np.arctan(sigma_data / sigma_min), steps+1).tan()
        #sigma_schedule = torch.linspace(np.log(sigma_max), np.log(sigma_min), steps+1).exp()
        #a = 1#sigma_min#np.e**0.5
        #a = 0.5 ##3.125#np.exp(-0.54)#torch.pi#torch.pi#4.48
        theta1 = np.arctan(1 / sigma_max); theta0 = np.arctan(1 / sigma_min)
        #theta1 = np.arctan(sigma_min/sigma_max); theta0 = np.arctan(1)
        theta = (1-t_schedule) * (theta0 - theta1) + theta1
        #sigma_schedule = theta.cos() / theta.sin() #* sigma_min

        ln_sigma = torch.linspace(np.log(sigma_min), np.log(sigma_max), steps, device=self.unet.device, dtype=self.unet.dtype)
        ln_sigma_error = self.unet.logvar_linear(self.unet.logvar_fourier(ln_sigma/4)).float().flatten()
        batch_distribution_pdf = ((-6 * ln_sigma_error) + 0).exp()
        from sigma_sampler import SigmaSampler
        sigma_sampler = SigmaSampler(sigma_max=sigma_max, sigma_min=sigma_min, sigma_data=sigma_data, distribution="ln_data", dist_offset=0., dist_scale=1., distribution_pdf=batch_distribution_pdf)
        #sigma_schedule = sigma_sampler.sample(0, quantiles = t_schedule)#.flip(dims=(0,))

        #sigma_schedule = sigma_min/( (1 - t_schedule) +sigma_min / sigma_max)

        print("sigma_schedule:", sigma_schedule[0].item(), "-", sigma_schedule[-1].item(), sigma_schedule[::10])
        #exit()
        
        noise = torch.randn(sample_shape, device=self.device, generator=generator)
        initial_noise = noise * sigma_max
        sample = noise * sigma_schedule[0] + start_data * sigma_data

        normalized_theta_schedule = (sigma_data / sigma_schedule).atan() / (torch.pi/2)
        sigma_schedule_list = sigma_schedule.tolist()

        
        sigma_schedule_error_logvar = self.unet.logvar_linear(self.unet.logvar_fourier(sigma_schedule.log().to(self.unet.device, self.unet.dtype)/4)).float().cpu()

        #perturbation_gain = (-4 * ln_sigma_error).exp(); perturbation_gain /= perturbation_gain.amax()

        cfg_fn = slerp if slerp_cfg else torch.lerp
    
        debug_v_list = []
        debug_a_list = []
        debug_d_list = []
        debug_s_list = []
        debug_m_list = []
        debug_ip_list = []

        if show_debug_plots:
            window_size_x = sample.shape[3]*3
            window_size_y = sample.shape[2]*3 * (2 + int(img2img_input is not None))
            cv2.namedWindow('sample / output', cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow('sample / output', window_size_x, window_size_y)
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
            """
            if input_perturbation > 0:

                p = input_perturbation * sigma_curr

                added_noise = torch.randn(sample.shape, generator=generator, device=sample.device, dtype=sample.dtype) * p
                sample += added_noise
                initial_noise += added_noise
                sigma_curr = (sigma_curr**2 + p**2)**0.5
                if i == 0: print("effective sigma_max:", sigma_curr)
            """

            """
            old_sigma_next = sigma_next
            sigma_next = max(sigma_next / (sigma_curr / sigma_next), 0)
            """

            old_sigma_next = sigma_next
            #effective_input_perturbation = (sigma_schedule_error_logvar[i]/4).exp().item() * input_perturbation
            effective_input_perturbation = input_perturbation
            debug_ip_list.append(effective_input_perturbation)
            
            sigma_next *= (1 - (max(min(effective_input_perturbation, 1), 0)))

            sigma = torch.tensor([sigma_curr], device=self.device)
            model_input = sample.to(self.unet.dtype)
            #print(model_input.std().item(), sigma_curr)
            
            u_model_output = self.unet(model_input, sigma, None, t_ranges, self.format).float()
            #model_output = self.unet(model_input, sigma, unet_class_embeddings, t_ranges, self.format).float()
            
            if static_conditioning_perturbation > 0:
                p_unet_class_embeddings = mp_sum(unet_class_embeddings, -self.unet.u_class_embeddings, static_conditioning_perturbation)
                #p_unet_class_embeddings = unet_class_embeddings - self.unet.u_class_embeddings * static_conditioning_perturbation
                #p_unet_class_embeddings = unet_class_embeddings - self.unet.u_class_embeddings * static_conditioning_perturbation
                #p_unet_class_embeddings -= p_unet_class_embeddings.mean()
                #a = p_unet_class_embeddings.mean()
                #p_unet_class_embeddings -= a
                #p_unet_class_embeddings /= p_unet_class_embeddings.std() / 1.7
                #p_unet_class_embeddings += a
                #p_unet_class_embeddings /=  1.382
            else:
                p_unet_class_embeddings = unet_class_embeddings

            if i == 0:
                print("p_unet_class_embeddings mean:", p_unet_class_embeddings.mean().item(), "std:", p_unet_class_embeddings.std())

            if dynamic_conditioning_perturbation > 0:
                perturbation = torch.randn((batch_size,) + p_unet_class_embeddings.shape[1:], generator=generator,
                                           device=p_unet_class_embeddings.device, dtype=p_unet_class_embeddings.dtype)
                p_unet_class_embeddings = mp_sum(p_unet_class_embeddings, perturbation, dynamic_conditioning_perturbation)#*1.35
                print("p_unet_class_embeddings mean:", p_unet_class_embeddings.mean().item(), "std:", p_unet_class_embeddings.std())
                #p_unet_class_embeddings = p_unet_class_embeddings + perturbation * dynamic_conditioning_perturbation
                #for game_id in game_ids:
                #    game_ids[game_id] = original_game_ids[game_id] * np.random.uniform()**dynamic_conditioning_perturbation
                #class_labels = self.get_class_labels(game_ids)
                #unet_class_embeddings = self.unet.get_class_embeddings(class_labels)
                #p_unet_class_embeddings = unet_class_embeddings

            #p_unet_class_embeddings = normalize(p_unet_class_embeddings).to(dtype=p_unet_class_embeddings.dtype)
            model_output = self.unet(model_input, sigma, p_unet_class_embeddings, t_ranges, self.format).float()
            #model_output = self.unet(model_input, sigma, self.unet.u_class_embeddings.lerp(unet_class_embeddings, conditioning_perturbation+1), t_ranges, self.format).float()
            #model_output = self.unet(model_input, sigma, unet_class_embeddings - self.unet.u_class_embeddings * conditioning_perturbation, t_ranges, self.format).float()
            
            #model_output = self.unet(model_input, sigma, unet_class_embeddings_a, unet_class_embeddings_b, t_ranges, self.format).float()
            #u_model_output = self.unet(model_input, sigma, None, None, t_ranges, self.format).float()

            last_cfg_model_output = cfg_model_output
            cfg_model_output = cfg_fn(u_model_output, model_output, cfg_scale).float()

            """
            if i == 0:
                for game_id1 in game_ids.keys():
                    for game_id2 in game_ids.keys():
                        if game_id1 != game_id2:
                            emb1 = self.unet.get_class_embeddings(self.get_class_labels([game_id1]))
                            emb2 = self.unet.get_class_embeddings(self.get_class_labels([game_id2]))
                            print(game_id1, game_id2, get_cos_angle(emb1, emb2)/ (torch.pi/2))
                            print(game_id1, "u", get_cos_angle(emb1, -self.unet.u_class_embeddings.unsqueeze(0))/ (torch.pi/2))
                            print(game_id1, "m", get_cos_angle(emb1, mp_sum(unet_class_embeddings, -self.unet.u_class_embeddings, conditioning_perturbation))/ (torch.pi/2))
            """
            #last_x = x
            #x = self.unet.last_x
                
            if use_midpoint_integration:
                #sigma_hat = (sigma_next * sigma_curr)**0.5
                #sigma_hat = sigma_min #* 0.9 #************************************ best as of aug 29/2024
                #sigma_hat = old_sigma_next
                #sigma_hat = max(sigma_next, sigma_min) # !!!
                sigma_hat = max(sigma_next, sigma_min)
                #sigma_hat = np.e**0.55
                #sigma_hat = max(sigma_next, sigma_min)
                #sigma_hat = (sigma_next * sigma_min)**0.5
                t_hat = sigma_hat / sigma_curr
                #t_hat = sigma_next / sigma_curr
                #t_hat = max(sigma_next, 0.03) / sigma_curr
                #t_hat = (old_sigma_next*0.5) / sigma_curr
                #t_hat = sigma_min / sigma_curr
                #t_hat = (sigma_next * sigma_curr)**0.5 / sigma_curr
                #t_hat = sigma_curr**0.5 / sigma_curr #best
                #t_hat = 0.5 #best
                #t_hat = (sigma_curr * sigma_next)**0.5 / sigma_curr
                #t_hat = (sigma_curr * old_sigma_next)**0.5 / sigma_curr

                sample_hat = (t_hat * sample + (1 - t_hat) * cfg_model_output)
                #sample_hat = sample
                sigma_hat = torch.tensor([t_hat * sigma_curr], device=self.device)

                """
                if static_conditioning_perturbation > 0:
                    p_unet_class_embeddings = mp_sum(unet_class_embeddings, -self.unet.u_class_embeddings, static_conditioning_perturbation)
                else:
                    p_unet_class_embeddings = unet_class_embeddings
                if dynamic_conditioning_perturbation > 0:
                    perturbation = torch.randn((batch_size,) + p_unet_class_embeddings.shape[1:], generator=generator,
                                            device=p_unet_class_embeddings.device, dtype=p_unet_class_embeddings.dtype)
                    p_unet_class_embeddings = mp_sum(p_unet_class_embeddings, perturbation, dynamic_conditioning_perturbation)
                """

                #p = max(sigma_curr**2 - old_sigma_next**2, 0)**0.5
                #added_noise = torch.randn(sample.shape, generator=generator, device=sample.device, dtype=sample.dtype)
                #sample_hat = sample_hat + added_noise * p
                #sigma_hat = torch.tensor([(old_sigma_next**2 + p**2)**0.5], device=self.device)

                #sigma_hat = torch.tensor([sigma_next], device=self.device)
                model_input_hat = sample_hat.to(self.unet.dtype)
                u_model_output_hat = self.unet(model_input_hat, sigma_hat, None, t_ranges, self.format).float()
                #model_output_hat = self.unet(model_input_hat, sigma_hat, unet_class_embeddings, t_ranges, self.format).float()
                model_output_hat = self.unet(model_input_hat, sigma_hat, p_unet_class_embeddings, t_ranges, self.format).float()
                #model_output_hat = self.unet(model_input_hat, sigma_hat, self.unet.u_class_embeddings.lerp(unet_class_embeddings, conditioning_perturbation+1), t_ranges, self.format).float()
                #model_output_hat = self.unet(model_input_hat, sigma_hat, unet_class_embeddings - self.unet.u_class_embeddings * conditioning_perturbation, t_ranges, self.format).float()
                
                cfg_model_output_hat = cfg_fn(u_model_output_hat, model_output_hat, cfg_scale).float()
                
                #print(torch.linalg.vector_norm(cfg_model_output).item(), torch.linalg.vector_norm(cfg_model_output_hat).item())
                cfg_model_output = (cfg_model_output + cfg_model_output_hat) / 2
                #cfg_model_output = (cfg_model_output + normalize(cfg_model_output_hat) * torch.linalg.vector_norm(cfg_model_output, dim=(1,2,3), keepdim=True)) / 2
                

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

            original_cfg_model_output = cfg_model_output
            if isinstance(start_data, torch.Tensor):
                if (i+1) < steps:
                    #print("cfg_model_output std:", cfg_model_output.std().item(), " start_data std:", start_data.std().item())
                    #t = sigma_next / sigma_curr if (i+1) < steps else 0
                    cfg_model_output = mp_sum(start_data * cfg_model_output.std(dim=(1,2,3), keepdim=True),
                                              cfg_model_output, img2img_strength)
                    #print("cfg_model_output std:", cfg_model_output.std().item())
                    #cfg_model_output = mp_sum(start_data, cfg_model_output, old_sigma_next / (old_sigma_next**2 + 1)**0.5)
                    #t = sigma_next / sigma_curr if (i+1) < steps else 0
                    #cfg_model_output = start_data * t + cfg_model_output * (1 - t)
                original_cfg_model_output = torch.cat((start_data, original_cfg_model_output[0:1]), dim=2)
            
            t = sigma_next / sigma_curr if (i+1) < steps else 0
            #t /= 0.9
            #t *= 0
            #t *= 0.5#0.786#0.618#0.99
            #old_sample = sample
            sample = (t * sample + (1 - t) * cfg_model_output)
            
            #sample = normalize(sample).float() * (sigma_next**2 + sigma_data**2)**0.5
            if temperature_scale < 1:
                ideal_norm = (sigma_next**2 + sigma_data**2)**0.5
                measured_norm = sample.square().mean(dim=(1,2,3), keepdim=True).sqrt()
                sample = sample / (measured_norm / ideal_norm)**(1 - temperature_scale)

            #"""
            if i+1 < steps:
                p = max(old_sigma_next**2 - sigma_next**2, 0)**0.5 #* input_perturbation #* 0.94 # * 0.96#0.95
                added_noise = torch.randn(sample.shape, generator=generator, device=sample.device, dtype=sample.dtype)
                #diff = (normalize(old_sample) - normalize(cfg_model_output)).abs().pow(0.25).float()
                #added_noise *= diff
                #added_noise /= added_noise.std()

                #if isinstance(start_data, torch.Tensor):
                #    added_noise = mp_sum(start_data, added_noise, img2img_strength)
                sample += added_noise * p
                #print(p / old_sigma_next)
            #"""

            sample0 = sample[0]
            output0 = original_cfg_model_output[0]
            #output0 = self.unet.last_x[0]

            sample0_img = save_raw_img(sample0, os.path.join(debug_path, f"debug_sample_{i:03}.png"), no_save=True)
            output0_img = save_raw_img(output0, os.path.join(debug_path, f"debug_output_{i:03}.png"), no_save=True)

            if show_debug_plots:
                cv2_img = cv2.vconcat([sample0_img, output0_img]).astype(np.float32)
                #cv2_img = output0_img.astype(np.float32); cv2.resizeWindow('sample / output', int(output0.shape[2]*2), int(output0.shape[1]*2*2))
                cv2_img = (cv2_img[:, :, :3] * cv2_img[:, :, 3:4] / 255).astype(np.uint8)
                cv2.imshow("sample / output", cv2_img)
                #save_img(cv2_img, os.path.join(debug_path, f"debug_sample_{i:03}.png"))
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
        #effective_input_perturbation = (sigma_schedule_error_logvar/4).exp()
        debug_input_perturbation = torch.tensor(debug_ip_list)

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
                       (normalized_theta_schedule, "normalized_distance"),
                       (debug_input_perturbation, "input_perturbation"),
                       #(sigma_schedule_error_logvar, "sigma_schedule_error_logvar"),
                       layout=(2, 3), figsize=(12, 5),
                       added_plots={4: (normalized_theta_schedule, "theta_schedule")})
            
            cv2.destroyAllWindows()
            
        return raw_sample 