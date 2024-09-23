from utils import config

import os
import inspect
from typing import Optional, Type
from abc import ABC
from dataclasses import dataclass

import torch

from utils.dual_diffusion_utils import load_safetensors, save_safetensors, torch_dtype

@dataclass
class DualDiffusionModuleConfig(ABC):
    last_global_step: int = 0

class DualDiffusionModule(torch.nn.Module, ABC):
    
    config_class: Optional[Type[DualDiffusionModuleConfig]] = None
    module_name: Optional[str] = None
    has_trainable_parameters: bool = True
    
    def __init__(self):
        super().__init__()
        
        self.dtype = torch.get_default_dtype()
        self.device = torch.device("cpu")

    @classmethod
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
        module_name = cls.module_name or os.path.basename(module_path)
        module_config = config_class(**config.load_json(os.path.join(module_path, f"{module_name}.json")))

        module = cls(module_config).requires_grad_(False).train(False)
        
        if (not load_config_only) and cls.has_trainable_parameters:
            module.load_state_dict(load_safetensors(os.path.join(module_path, f"{module_name}.safetensors")))
        
        return module.to(dtype=torch_dtype, device=device)
    
    def save_pretrained(self, module_path: str, subfolder: Optional[str] = None) -> None:
        
        if subfolder is not None:
            module_path = os.path.join(module_path, subfolder)
        os.makedirs(module_path, exist_ok=True)
        
        module_name = type(self).module_name or os.path.basename(module_path)

        config.save_json(self.config.__dict__, os.path.join(module_path, f"{module_name}.json"))
        if type(self).has_trainable_parameters:
            save_safetensors(self.state_dict(), os.path.join(module_path, f"{module_name}.safetensors"))

    def to(self, device: Optional[torch.device] = None,
           dtype: Optional[torch.dtype] = None, **kwargs) -> "DualDiffusionModule":
        
        if dtype is not None: dtype = torch_dtype(dtype)
        super().to(device=device, dtype=dtype, **kwargs)

        self.dtype = dtype or self.dtype
        self.device = device or self.device
        return self
    
    def half(self) -> "DualDiffusionModule":
        return self.to(dtype=torch.bfloat16)
    
    def compile(self, compile_options: dict) -> None:
        self.forward = torch.compile(self.forward, **compile_options)

    def load_ema(self, ema_path: str) -> None:
        self.load_state_dict(load_safetensors(ema_path))
        self.normalize_weights()

    @torch.no_grad()
    def normalize_weights(self) -> None:
        if self.has_trainable_parameters == False: return
        for module in self.modules():
            if hasattr(module, "normalize_weights") and module != self:
                module.normalize_weights()