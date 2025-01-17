from utils import config

import os
import inspect
from typing import Optional, Type, Union, Literal
from abc import ABC
from dataclasses import dataclass, fields
from logging import getLogger

import torch

from utils.dual_diffusion_utils import load_safetensors, save_safetensors, torch_dtype, TF32_Disabled

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

        logger = getLogger()
        module_config_dict = {}
        module_config_fields = [field.name for field in fields(config_class)]
        module_config_file_path = os.path.join(module_path, f"{module_name}.json")
        module_config_file_dict = config.load_json(module_config_file_path)
        for field, value in module_config_file_dict.items():
            if field not in module_config_fields:
                logger.warning(f"Warning: field '{field}' not found in module config dataclass '{config_class.__name__}', ignoring...")
            else:
                module_config_dict[field] = value
        module_config = config_class(**module_config_dict)
        for field in module_config_fields:
            if field not in module_config_file_dict:
                logger.warning(f"Warning: field '{field}' not found in config file '{module_config_file_path}',"
                                f" initializing with default value '{getattr(module_config, field)}'")

        module = cls(module_config).requires_grad_(False).train(False)
        
        if (not load_config_only) and cls.has_trainable_parameters:
            module.load_state_dict(load_safetensors(os.path.join(module_path, f"{module_name}.safetensors")))
        
        module.module_path = module_path
        return module.to(dtype=torch_dtype, device=device)
    
    @torch.no_grad()
    def save_pretrained(self, module_path: str, subfolder: Optional[str] = None) -> None:
        
        if subfolder is not None:
            module_path = os.path.join(module_path, subfolder)
        os.makedirs(module_path, exist_ok=True)
        
        #module_name = type(self).module_name or os.path.basename(module_path)
        module_name = os.path.basename(module_path)

        config.save_json(self.config.__dict__, os.path.join(module_path, f"{module_name}.json"))
        if type(self).has_trainable_parameters:
            save_safetensors(self.state_dict(), os.path.join(module_path, f"{module_name}.safetensors"))

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None,
           memory_format: Optional[torch.memory_format] = None,**kwargs) -> "DualDiffusionModule":
        
        if dtype is not None:
            if type(self).supports_half_precision == True:
                dtype = torch_dtype(dtype)
            else:
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
    
    def half(self) -> "DualDiffusionModule":
        if type(self).supports_half_precision == True:
            self.dtype = torch.bfloat16
            return self.to(dtype=torch.bfloat16)
        else:
            return self
    
    def compile(self, **kwargs) -> None:
        if type(self).supports_compile == True:
            self.forward = torch.compile(self.forward, **kwargs)
            if hasattr(self, "normalize_weights") and self.training == True:
                self.normalize_weights = torch.compile(self.normalize_weights, **kwargs)

    @torch.no_grad()
    def load_ema(self, ema_path: str) -> None:
        with TF32_Disabled():
            self.load_state_dict(load_safetensors(ema_path))
            self.normalize_weights()

    @torch.no_grad()
    def blend_weights(self, other: "DualDiffusionModule", t: float = 0.5) -> None:
        with TF32_Disabled():
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
                if hasattr(module, "normalize_weights") and module != self:
                    module.normalize_weights()