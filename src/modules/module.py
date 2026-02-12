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

from utils import config

import os
import inspect
from typing import Optional, Type, Union, Literal
from abc import ABC
from dataclasses import dataclass
from logging import getLogger

import torch

from utils.dual_diffusion_utils import load_safetensors, save_safetensors, torch_dtype, TF32_Disabled
from training.ema import reconstruct_phema


@dataclass
class DualDiffusionModuleConfig(ABC):
    last_global_step: int = 0

class DualDiffusionModule(torch.nn.Module, ABC):
    
    config_class: Optional[Type[DualDiffusionModuleConfig]] = None
    module_name: Optional[str] = None
    has_trainable_parameters: bool = True
    supports_half_precision: bool = True
    supports_channels_last: Union[bool, Literal["3d"]] = True
    supports_compile: bool = True

    def __init__(self):
        super().__init__()
        
        self.dtype = torch.get_default_dtype()
        self.device = torch.device("cpu")
        self.memory_format = torch.contiguous_format
        self.module_path = None
        
    @classmethod
    @torch.no_grad()
    def from_pretrained(cls: Type["DualDiffusionModule"],
                        module_path: str,
                        subfolder: Optional[str] = None,
                        torch_dtype: Optional[torch.dtype] = None,
                        device: Optional[torch.device] = None,
                        load_config_only: bool = False
                        ) -> "DualDiffusionModule":
        
        if subfolder is not None:
            module_path = os.path.join(module_path, subfolder)
        
        config_class = cls.config_class or inspect.signature(cls.__init__).parameters["config"].annotation
        #module_name = cls.module_name or os.path.basename(module_path)
        module_name = os.path.basename(module_path)

        module_config_file_path = os.path.join(module_path, f"{module_name}.json")
        module_config = config.load_config(config_class, module_config_file_path)
        module = cls(module_config).requires_grad_(False).train(False)
        
        if (not load_config_only) and cls.has_trainable_parameters:
            module.load_state_dict(load_safetensors(os.path.join(module_path, f"{module_name}.safetensors")))
        
        module.module_path = module_path
        return module.to(dtype=torch_dtype, device=device)
    
    @torch.no_grad()
    def save_pretrained(self, module_path: str, subfolder: Optional[str] = None,
                        save_config_only: bool = False) -> None:
        
        if subfolder is not None:
            module_path = os.path.join(module_path, subfolder)
        os.makedirs(module_path, exist_ok=True)
        
        #module_name = type(self).module_name or os.path.basename(module_path)
        module_name = os.path.basename(module_path)

        config.save_config(self.config, os.path.join(module_path, f"{module_name}.json"))
        if type(self).has_trainable_parameters and save_config_only == False:
            save_safetensors(self.state_dict(), os.path.join(module_path, f"{module_name}.safetensors"))

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None,
           memory_format: Optional[torch.memory_format] = None,**kwargs) -> "DualDiffusionModule":
        
        if device is not None:
            device = torch.device(device)

        if dtype is not None:
            dtype = torch_dtype(dtype)
            if dtype is torch.float16 or dtype is torch.bfloat16:
                if type(self).supports_half_precision == False:
                    dtype = torch.float32

        if memory_format == torch.channels_last:
            if type(self).supports_channels_last == False:
                memory_format = None
            elif type(self).supports_channels_last == "3d":
                memory_format = torch.channels_last_3d

        super().to(device=device, dtype=dtype, memory_format=memory_format, **kwargs)

        self.dtype = dtype or self.dtype
        self.device = device or self.device
        self.memory_format = memory_format or self.memory_format

        return self

    def double(self) -> "DualDiffusionModule":
        return self.to(dtype=torch.float64)
        
    def float(self) -> "DualDiffusionModule":
        return self.to(dtype=torch.float32)
        
    def half(self) -> "DualDiffusionModule":
        return self.to(dtype=torch.bfloat16)
        
    def type(self, dtype: torch.dtype) -> "DualDiffusionModule":
        return self.to(dtype=dtype)

    def cpu(self, **kwargs) -> "DualDiffusionModule":
        return self.to(device="cpu", **kwargs)
    
    def cuda(self, device: Optional[int] = None) -> "DualDiffusionModule":
        return self.to(device="cuda" if device is None else f"cuda:{device}")

    def compile(self, **kwargs) -> None:
        if type(self).supports_compile == True:
            self.forward = torch.compile(self.forward, **kwargs)
            # this is disabled because it can cause weights to diverge by small amounts in distributed training
            #if hasattr(self, "normalize_weights") and self.training == True:
            #    self.normalize_weights = torch.compile(self.normalize_weights, **kwargs)

    @torch.no_grad()
    @TF32_Disabled()
    def load_ema(self, ema_path: str, phema_path: Optional[str] = None) -> None:
        if os.path.isfile(ema_path) == False:
            
            def extract_number_from_string(s):
                num = next((s[i:] for i, c in enumerate(s) if c.isdigit() or c in "+-."), None)
                str = "".join(c for c in num if c.isdigit() or c in "+-.") if num else None
                return float(str.rstrip("+-."))

            if os.path.basename(ema_path).split("_")[0] == "phema":

                phema_std = extract_number_from_string(os.path.basename(ema_path).split("_")[1])
                self.load_state_dict(reconstruct_phema(phema_std, phema_path))

                try: # attempt to save a cached copy of the computed post-hoc ema reconstruction
                    save_safetensors(self.state_dict(), ema_path)
                except Exception as e: # todo: there should probably be a quiet option for this
                    getLogger().warning(f"Warning: Could not save reconstructed phema at {ema_path}")
            else:
                raise FileNotFoundError(f"Error: Could not find ema file '{ema_path}'")
        else:
            self.load_state_dict(load_safetensors(ema_path))
        self.normalize_weights()

    @torch.no_grad()
    @TF32_Disabled()
    def blend_weights(self, other: "DualDiffusionModule", t: float = 0.5) -> None:
        for ((param_name, param), (other_param_name, other_param)) in zip(self.named_parameters(), other.named_parameters()):
            if param.data.shape != other_param.data.shape:
                raise ValueError(f"Cannot blend parameters with different shapes: {param_name} {param.data.shape} != {other_param_name} {other_param.data.shape}")
            param.data.lerp_(other_param.data, t)
        self.normalize_weights()

    @torch.no_grad()
    def normalize_weights(self) -> None:
        if type(self).has_trainable_parameters == False: return
        with TF32_Disabled():
            for module in self.modules():
                if hasattr(module, "normalize_weights") and module is not self:
                    module.normalize_weights()