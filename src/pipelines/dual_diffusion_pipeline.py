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
import asyncio
from dataclasses import dataclass
from typing import Callable, Optional, Union, Any
from datetime import datetime

import numpy as np
import torch
import torchaudio
from tqdm.auto import tqdm

from modules.module import DualDiffusionModule
from modules.unets.unet import DualDiffusionUNet
from utils.dual_diffusion_utils import (
    normalize, load_safetensors, torch_dtype, load_audio, get_cos_angle
)
from sampling.schedule import SamplingSchedule

@dataclass
class SampleParams:
    seed: Optional[int]         = None
    num_steps: int              = 100
    batch_size: int             = 1
    length: Optional[int]       = None
    cfg_scale: float            = 1.5
    sigma_max: Optional[float]  = None
    sigma_min: Optional[float]  = None
    sigma_data: Optional[float] = None
    rho: float                  = 7.
    schedule: Optional[str]               = "edm2"
    #schedule_kwargs: Optional[dict]       = None
    prompt: Optional[dict]                = None
    generator: Optional[torch.Generator]  = None
    use_heun: bool                        = True
    input_perturbation: float             = 1.
    conditioning_perturbation: float      = 0.
    num_fgla_iters: int                   = 250
    img2img_strength: float               = 0.5
    input_audio: Optional[Union[str, torch.Tensor]] = None
    input_audio_pre_encoded: bool                   = False
    inpainting_mask: Optional[torch.Tensor]         = None

    def sanitize(self) -> "SampleParams":
        self.seed = int(self.seed) if self.seed is not None else None
        self.length = int(self.length) if self.length is not None else None
        self.num_steps = int(self.num_steps)
        self.batch_size = int(self.batch_size)
        self.num_fgla_iters = int(self.num_fgla_iters)
        # todo: additional sanitization (clip values like input perturb, etc.)
        return self

    def get_metadata(self) -> dict[str, Any]:
        metadata = self.__dict__.copy()

        if metadata["input_audio"] is not None and (not isinstance(metadata["input_audio"], str)):
            metadata["input_audio"] = True
        if metadata["inpainting_mask"] is not None:
            metadata["inpainting_mask"] = True
        if metadata["generator"] is not None:
            metadata["generator"] = True

        metadata["timestamp"] = datetime.now().strftime(r"%m/%d/%Y %I:%M:%S %p")
        return {str(key): str(value) for key, value in metadata.items()}
    
    def get_label(self, pipeline: "DualDiffusionPipeline") -> str:
        
        module_name = "unet"
        #if self.inpainting_mask is not None: # I prefer having the step label from the main unet for now
        #    if getattr(pipeline, "unet_inpainting", None) is not None:
        #        module_name = "unet_inpainting"

        last_global_step = pipeline.model_metadata["last_global_step"]["module_name"]
        if module_name in pipeline.model_metadata["load_emas"]:
            ema = pipeline.model_metadata["load_emas"][module_name].replace("pf_ema_std-", "").replace(".safetensors", "")
        else:
            ema = None
        top_game_name = sorted(self.prompt.items(), key=lambda x:x[1])[-1][0]
        top_game_id = pipeline.dataset_game_ids[top_game_name]

        label = f"step_{last_global_step}_{int(self.num_steps)}_{'ema'+ema+'_' if ema else ''}cfg{self.cfg_scale}"
        label += f"_sgm{self.sigma_max}-{self.sigma_min}_ip{self.input_perturbation}_r{self.rho}_g{top_game_id}_s{int(self.seed)}"
        
        return label

@dataclass
class SampleOutput:
    raw_sample: torch.Tensor
    spectrogram: torch.Tensor
    params: SampleParams
    debug_info: dict[str, Any]
    latents: Optional[torch.Tensor] = None

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
            if load_checkpoints == True:
                load_checkpoints = {}
                for module_name, module_inventory in model_inventory.items():
                    if len(module_inventory.checkpoints) > 0:
                        load_checkpoints[module_name] = module_inventory.checkpoints[-1]
            else:
                load_checkpoints = {}

        load_emas = load_emas or False
        if isinstance(load_emas, bool):
            if load_emas == True:
                load_emas = {}
                for module_name, module_inventory in model_inventory.items():
                    module_checkpoint = load_checkpoints.get(module_name, "")
                    if len(module_inventory.emas[module_checkpoint]) > 0:
                        load_emas[module_name] = module_inventory.emas[module_checkpoint][-1]
            else:
                load_emas = {}

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

        model_metadata: dict[str, any] = {
            "model_path": model_path,
            "model_module_classes": {module_name: str(module_class)
                                     for module_name, module_class in model_module_classes.items()},
            "torch_dtype": torch_dtype,
            "load_checkpoints": load_checkpoints,
            "load_emas": load_emas,
            "compile_options": compile_options,
            "last_global_step": {module_name: module.config.last_global_step
                                 for module_name, module in pipeline.named_children()}
        }
        pipeline.model_metadata = model_metadata

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
                return self.dataset_game_ids[label_name]
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
    async def __call__(self, params: SampleParams,
            progress_callback: Optional[Callable[[str, Union[float, torch.Tensor]], Any]] = None) -> torch.Tensor:
        
        debug_info = {}
        params = SampleParams(**params.__dict__).sanitize() # todo: this should be properly deepcopied because of tensor params
        
        # automatically substitute dedicated inpainting unet if it exists and we have an inpainting mask
        unet: DualDiffusionUNet = self.unet
        if params.inpainting_mask is not None:
            inpainting_unet: DualDiffusionUNet = getattr(self, "unet_inpainting", None)
            unet = inpainting_unet or unet

        params.seed = params.seed or int(np.random.randint(100000, 999999))
        params.length = params.length or self.format.config.sample_raw_length
        params.sigma_max = params.sigma_max or unet.config.sigma_max
        params.sigma_min = params.sigma_min or unet.config.sigma_min
        params.sigma_data = params.sigma_data or unet.config.sigma_data
        #params.schedule_kwargs = params.schedule_kwargs or {}
        params.prompt = params.prompt or {}
        
        if isinstance(params.input_audio, str):
            if params.input_audio_pre_encoded == True:
                input_audio = load_safetensors(params.input_audio)["latents"][0:1]
            else:
                input_sample_shape = self.get_sample_shape(bsz=1, length=params.length)
                input_audio_sample_rate, input_audio = load_audio(
                    params.input_audio, count=self.format.get_audio_shape(input_sample_shape), return_sample_rate=True)
                if input_audio_sample_rate != self.format.config.sample_rate:
                    input_audio = torchaudio.functional.resample(
                        input_audio, input_audio_sample_rate, self.format.config.sample_rate)
        else:
            input_audio = params.input_audio

        self.format.config.num_fgla_iters = params.num_fgla_iters
        generator = params.generator or torch.Generator(
            device=unet.device).manual_seed(params.seed)

        sample_shape = self.get_sample_shape(bsz=params.batch_size, length=params.length)
        if getattr(self, "vae", None) is not None:
            latent_diffusion = True
            sample_shape = self.get_latent_shape(sample_shape)
        else:
            latent_diffusion = False
        debug_info["sample_shape"] = tuple(sample_shape)
        debug_info["latent_diffusion"] = latent_diffusion

        vae_class_embeddings = self.vae.get_class_embeddings(self.get_class_labels(params.prompt, module_name="vae"))
        conditioning_mask = torch.cat((torch.ones(params.batch_size), torch.zeros(params.batch_size)))
        unet_class_embeddings = unet.get_class_embeddings(
            self.get_class_labels(params.prompt, module_name="unet"), conditioning_mask)
        debug_info["unet_class_embeddings mean"] = unet_class_embeddings.mean().item()
        debug_info["unet_class_embeddings std"] = unet_class_embeddings.std().item()

        if input_audio is not None:
            if params.input_audio_pre_encoded == True:
                input_audio_sample = input_audio
            else:
                while input_audio.ndim < 3: input_audio.unsqueeze_(0)
                input_audio_sample = self.format.raw_to_sample(input_audio.float().to(self.format.device))
                if latent_diffusion:
                    input_audio_sample = self.vae.encode(
                        input_audio_sample.to(device=self.vae.device, dtype=self.vae.dtype),
                        vae_class_embeddings, self.format).mode().float()[..., :688] #***
                    await asyncio.sleep(0)
        else:
            input_audio_sample = torch.zeros(sample_shape, device=unet.device)

        if params.inpainting_mask is not None:
            while params.inpainting_mask.ndim < input_audio_sample.ndim: params.inpainting_mask.unsqueeze_(0)
            params.inpainting_mask = (params.inpainting_mask.to(unet.device) > 0.5).float()
            ref_sample = torch.cat([input_audio_sample * (1 - params.inpainting_mask), params.inpainting_mask], dim=1)
        else:
            ref_sample = torch.cat((torch.zeros_like(input_audio_sample),
                                    torch.ones_like(input_audio_sample[:, :1])), dim=1)
        input_ref_sample = ref_sample.to(unet.dtype).repeat(2, 1, 1, 1)

        if unet.config.use_t_ranges == True:
            raise NotImplementedError("sampling with unet.config.use_t_ranges=True not implemented")
        else:
            t_ranges = None

        start_timestep = 1
        sigma_schedule = SamplingSchedule.get_schedule(params.schedule,
            params.num_steps, start_timestep, device=unet.device,
            sigma_max=params.sigma_max, sigma_min=params.sigma_min, rho=params.rho)#**params.schedule_kwargs)
        sigma_schedule_list = sigma_schedule.tolist()
        debug_info["sigma_schedule"] = sigma_schedule_list

        noise = torch.randn(sample_shape, device=unet.device, generator=generator)
        sample = noise * sigma_schedule[0] + input_audio_sample * params.sigma_data

        progress_bar = tqdm(total=params.num_steps)
        for i, (sigma_curr, sigma_next) in enumerate(zip(sigma_schedule_list[:-1], sigma_schedule_list[1:])):

            old_sigma_next = sigma_next
            #effective_input_perturbation = (sigma_schedule_error_logvar[i]/4).exp().item() * input_perturbation
            #effective_input_perturbation = params.input_perturbation
            effective_input_perturbation = params.input_perturbation * (1 - (i/params.num_steps)*(1 - i/params.num_steps))#**2 #***
            sigma_next *= (1 - (max(min(effective_input_perturbation, 1), 0)))

            input_sigma = torch.tensor([sigma_curr], device=unet.device)
            input_sample = sample.to(unet.dtype).repeat(2, 1, 1, 1)
            #if params.static_conditioning_perturbation > 0:
            #    p_unet_class_embeddings = mp_sum(unet_class_embeddings, -unet.u_class_embeddings, static_conditioning_perturbation)
            #else:
            #    p_unet_class_embeddings = unet_class_embeddings

            #if dynamic_conditioning_perturbation > 0:
            #    perturbation = torch.randn((batch_size,) + p_unet_class_embeddings.shape[1:], generator=generator,
            #                               device=p_unet_class_embeddings.device, dtype=p_unet_class_embeddings.dtype)
            #    p_unet_class_embeddings = mp_sum(p_unet_class_embeddings, perturbation, dynamic_conditioning_perturbation)#*1.35
            #    print("p_unet_class_embeddings mean:", p_unet_class_embeddings.mean().item(), "std:", p_unet_class_embeddings.std())

            model_output = unet(input_sample, input_sigma, self.format, unet_class_embeddings, t_ranges, input_ref_sample).float()
            cfg_model_output = model_output[params.batch_size:].lerp(model_output[:params.batch_size], params.cfg_scale)
                
            if params.use_heun:
                sigma_hat = max(sigma_next, params.sigma_min)
                t_hat = sigma_hat / sigma_curr

                input_sample_hat = (t_hat * sample + (1 - t_hat) * cfg_model_output).to(unet.dtype).repeat(2, 1, 1, 1)
                #input_sample_hat = normalize(input_sample_hat).to(unet.dtype) * (sigma_next**2 + params.sigma_data**2)**0.5 #***
                input_sigma_hat = torch.tensor([t_hat * sigma_curr], device=unet.device)

                model_output_hat = unet(input_sample_hat, input_sigma_hat, self.format, unet_class_embeddings, t_ranges, input_ref_sample).float()
                cfg_model_output_hat = model_output_hat[params.batch_size:].lerp(model_output_hat[:params.batch_size], params.cfg_scale)
                cfg_model_output = (cfg_model_output + cfg_model_output_hat) / 2
            
            t = sigma_next / sigma_curr if (i+1) < params.num_steps else 0
            sample = (t * sample + (1 - t) * cfg_model_output)
            
            #sample = normalize(sample).float() * (sigma_next**2 + params.sigma_data**2)**0.5 #***
            #if params.temperature_scale < 1:
            #    ideal_norm = (sigma_next**2 + sigma_data**2)**0.5
            #    measured_norm = sample.square().mean(dim=(1,2,3), keepdim=True).sqrt()
            #    sample = sample / (measured_norm / ideal_norm)**(1 - temperature_scale)

            if progress_callback is not None:
                progress_callback("latents", cfg_model_output.cpu())

            if i+1 < params.num_steps:
                p = max(old_sigma_next**2 - sigma_next**2, 0)**0.5
                added_noise = torch.randn(sample.shape,
                    generator=generator, device=sample.device, dtype=sample.dtype)

                sample += added_noise * p

            if progress_callback is not None:
                if progress_callback("sampling", (i+1) / params.num_steps) == False:
                    progress_bar.close()
                    return None
            progress_bar.update(1)
            await asyncio.sleep(0.0083)
        
        progress_bar.close()

        debug_info["final_sample_mean"] = sample.mean().item()
        debug_info["final_sample_std"] = sample.std().item()
        
        #sample = sample.float() / sigma_data
        sample = normalize(sample).float() * params.sigma_data

        if latent_diffusion == True:
            latents = sample
            spectrogram = self.vae.decode(sample.to(self.vae.dtype), vae_class_embeddings, self.format).float()
            await asyncio.sleep(0)
        else:
            latents = None
            spectrogram = sample
        raw_sample = await self.format.sample_to_raw(spectrogram)
        
        return SampleOutput(raw_sample, spectrogram, params, debug_info, latents)