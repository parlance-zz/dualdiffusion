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
from copy import deepcopy
from typing import Optional

import torch

from modules.module import DualDiffusionModule
from utils.dual_diffusion_utils import TF32_Disabled, save_safetensors, load_safetensors


def get_ema_list(module_path: str) -> tuple[list[str], list[float]]:
    ema_list = []; ema_betas = []
    for path in os.listdir(module_path):
        if os.path.isfile(os.path.join(module_path, path)):
            if path.startswith("ema_") and path.endswith(".safetensors"):
                ema_betas.append(float(path.replace("ema_", "").replace(".safetensors", "")))
                ema_list.append(path)
    
    # sort both lists ascending by beta values
    ema_list = [x for _, x in sorted(zip(ema_betas, ema_list))]
    ema_betas = sorted(ema_betas)

    return ema_list, ema_betas

# todo: if device is cpu (offloading) try pin_memory() and non_blocking=True
#       for improved performance

class EMA_Manager:

    @torch.no_grad()
    def __init__(self, module: torch.nn.Module, betas: list[float],
                 device: Optional[torch.device] = None) -> None:

        self.module = module
        self.betas = betas
        self.device = device

        self.emas: list[DualDiffusionModule] = [deepcopy(module).to(device) for _ in betas]

    @torch.no_grad()
    def reset(self) -> None:
        for ema in self.emas:
            torch._foreach_copy_(tuple(ema.parameters()), tuple(self.module.parameters()))

    @torch.no_grad()
    def update(self, cur_nimg: int, batch_size: int) -> None:
        with torch.amp.autocast("cuda", enabled=False), TF32_Disabled():
            net_parameters = tuple(self.module.parameters())
            for beta, ema in zip(self.betas, self.emas):
                torch._foreach_lerp_(tuple(ema.parameters()), net_parameters, 1 - beta)

    @torch.no_grad()
    def feedback(self, cur_nimg: int, batch_size: int, beta: float) -> None:
        with torch.amp.autocast("cuda", enabled=False), TF32_Disabled():
            net_parameters = tuple(self.module.parameters())
            ema_parameters = tuple(self.emas[0].parameters())
            torch._foreach_lerp_(net_parameters, ema_parameters, 1 - beta)

    @torch.no_grad()
    def save(self, save_directory: str, subfolder: Optional[str] = None) -> None:

        if subfolder is not None:
            save_directory = os.path.join(save_directory, subfolder)
        os.makedirs(save_directory, exist_ok=True)

        for beta, ema in zip(self.betas, self.emas):
            ema_save_path = os.path.join(save_directory, f"ema_{beta}.safetensors")
            save_safetensors(ema.state_dict(), ema_save_path)

    @torch.no_grad()
    def load(self, ema_path: str, subfolder: Optional[str] = None,
             target_module: Optional[torch.nn.Module] = None) -> list[str]:
        
        if subfolder is not None:
            ema_path = os.path.join(ema_path, subfolder)
            
        self.module = target_module or self.module

        load_errors = []
        for beta, ema in zip(self.betas, self.emas):
            ema_load_path = os.path.join(ema_path, f"ema_{beta}.safetensors")
            if os.path.isfile(ema_load_path):
                ema.load_state_dict(load_safetensors(ema_load_path))
            else:
                error_str = f"Could not find EMA weights for beta={beta} at {ema_load_path}"
                if target_module is not None:
                    torch._foreach_copy_(tuple(ema.parameters()), tuple(target_module.parameters()))
                    load_errors.append(error_str)
                else:
                    raise FileNotFoundError(error_str)
                
        return load_errors
    
