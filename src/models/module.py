from utils import config

import os
from typing import Optional, Type
from abc import ABC, abstractmethod

import torch


class DualDiffusionModule(torch.nn.Module, ABC):
    
    @classmethod
    @torch.no_grad()
    def from_pretrained(cls: Type["DualDiffusionModule"],
                        module_path: str,
                        subfolder: Optional[str] = None,
                        torch_dtype: Optional[torch.dtype] = None,
                        device: Optional[torch.device] = None
                        ) -> "DualDiffusionModule":
        
        if subfolder is not None:
            module_path = os.path.join(module_path, subfolder)



        # load pipeline modules
        for module_name, module_import_dict in model_index["modules"].items():
            
            module_package = importlib.import_module(module_import_dict["package"])
            module_class = getattr(module_package, module_import_dict["class"])
            
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
    
    def save_pretrained(self, model_path: str, subfolder: Optional[str] = None) -> None:
        
        if subfolder is not None:
            model_path = os.path.join(model_path, subfolder)
        os.makedirs(model_path, exist_ok=True)
        
        model_modules = {}
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
