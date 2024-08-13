from utils import config

import os
import inspect
from typing import Optional, Type
from abc import ABC
from dataclasses import dataclass

import torch

from utils.dual_diffusion_utils import load_safetensors, save_safetensors

@dataclass
class DualDiffusionModuleConfig(ABC):
    last_global_step: int = 0

class DualDiffusionModule(torch.nn.Module, ABC):
    
    config_class: Optional[Type[DualDiffusionModuleConfig]] = None
    module_name: Optional[str] = None
    has_trainable_parameters: bool = True
        
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

        config.save_json(self.config.asdict(), os.path.join(module_path, f"{module_name}.json"))
        if type(self).has_trainable_parameters:
            save_safetensors(self.state_dict(), os.path.join(module_path, f"{module_name}.safetensors"))

    @torch.no_grad()
    def normalize_weights(self) -> None:
        for module in self.modules():
            if hasattr(module, "normalize_weights"):
                module.normalize_weights()