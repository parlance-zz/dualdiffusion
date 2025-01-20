# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Routines for post-hoc EMA and power function EMA proposed in the paper
"Analyzing and Improving the Training Dynamics of Diffusion Models"."""

# Modifications under MIT License
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

from __future__ import annotations # workaround for circular import, needed for type annotation only

import os
from dataclasses import dataclass
from copy import deepcopy
from typing import Optional, Any

import torch
import numpy as np

from modules.module import DualDiffusionModule
from utils.dual_diffusion_utils import TF32_Disabled, save_safetensors, load_safetensors

# workaround for circular import, needed for type annotation only
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from training.trainer import DualDiffusionTrainer

#----------------------------------------------------------------------------
# beginning of nvidia code
#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
# Convert power function exponent to relative standard deviation
# according to Equation 123.

def exp_to_std(exp: np.ndarray) -> np.ndarray:
    exp = np.float64(exp)
    std = np.sqrt((exp + 1) / (exp + 2) ** 2 / (exp + 3))
    return std

#----------------------------------------------------------------------------
# Convert relative standard deviation to power function exponent
# according to Equation 126 and Algorithm 2.

def std_to_exp(std: np.ndarray) -> np.ndarray:
    std = np.float64(std)
    tmp = std.flatten() ** -2
    exp = [np.roots([1, 7, 16 - t, 12 - t]).real.max() for t in tmp]
    exp = np.float64(exp).reshape(std.shape)
    return exp

#----------------------------------------------------------------------------
# Construct response functions for the given EMA profiles
# according to Equations 121 and 108.

def power_function_response(ofs: np.ndarray, std: np.ndarray,
                            len: int, axis: int = 0) -> np.ndarray:
    ofs, std = np.broadcast_arrays(ofs, std)
    ofs = np.stack([np.float64(ofs)], axis=axis)
    exp = np.stack([std_to_exp(std)], axis=axis)
    s = [1] * exp.ndim
    s[axis] = -1
    t = np.arange(len).reshape(s)
    resp = np.where(t <= ofs, (t / ofs) ** exp, 0) / ofs * (exp + 1)
    resp = resp / np.sum(resp, axis=axis, keepdims=True)
    return resp

#----------------------------------------------------------------------------
# Compute inner products between the given pairs of EMA profiles
# according to Equation 151 and Algorithm 3.

def power_function_correlation(a_ofs: np.ndarray, a_std: np.ndarray,
                               b_ofs: np.ndarray, b_std: np.ndarray) -> np.ndarray:
    a_exp = std_to_exp(a_std)
    b_exp = std_to_exp(b_std)
    t_ratio = a_ofs / b_ofs
    t_exp = np.where(a_ofs < b_ofs, b_exp, -a_exp)
    t_max = np.maximum(a_ofs, b_ofs)
    num = (a_exp + 1) * (b_exp + 1) * t_ratio ** t_exp
    den = (a_exp + b_exp + 1) * t_max
    return num / den

#----------------------------------------------------------------------------
# Calculate beta for tracking a given EMA profile during training
# according to Equation 127.

def power_function_beta(std: np.ndarray, t_next: int, t_delta: int) -> np.ndarray:
    beta = (1 - t_delta / t_next) ** (std_to_exp(std) + 1)
    return beta

#----------------------------------------------------------------------------
# Solve the coefficients for post-hoc EMA reconstruction
# according to Algorithm 3.

def solve_posthoc_coefficients(in_ofs: np.ndarray, in_std: np.ndarray,
                               out_ofs: np.ndarray, out_std: np.ndarray) -> np.ndarray:
    in_ofs, in_std = np.broadcast_arrays(in_ofs, in_std)
    out_ofs, out_std = np.broadcast_arrays(out_ofs, out_std)
    rv = lambda x: np.float64(x).reshape(-1, 1)
    cv = lambda x: np.float64(x).reshape(1, -1)
    A = power_function_correlation(rv(in_ofs), rv(in_std), cv(in_ofs), cv(in_std))
    B = power_function_correlation(rv(in_ofs), rv(in_std), cv(out_ofs), cv(out_std))
    X = np.linalg.solve(A, B)
    X = X / np.sum(X, axis=0)
    return X

#----------------------------------------------------------------------------
# end of nvidia code
#----------------------------------------------------------------------------

# get all emas in dir as dict with ema names as keys and filenames as values
def find_emas_in_dir(module_path: str) -> dict[str, str]:
    ema_dict = {}
    for path in reversed(sorted(os.listdir(module_path))):
        if os.path.isfile(os.path.join(module_path, path)):
            if path.startswith("ema_") and path.endswith(".safetensors"):
                ema_name = path[len("ema_"):-len(".safetensors")]
                ema_dict[ema_name] = path

    return ema_dict

@dataclass
class EMA_Config:
    name: str
    cpu_offload: bool                     = False # enable to save vram by keeping ema in system memory
    include_in_validation: bool           = True  # enable to calculate validation loss for this ema
    num_switch_ema_epochs: Optional[int]  = None  # if set, loads ema weights into train weights every n epochs
    beta: Optional[float]                 = None  # beta for classic ema (std must be None)
    std: Optional[float]                  = None  # std for power function ema (beta must be None)
    num_warmup_steps: Optional[int]       = None  # if set, number of steps to scale effective beta from 0 to beta
    num_archive_steps: Optional[int]      = None  # if set, saves ema weights in bf16 every n steps to a separate path for post-hoc reconstruction
    feedback_beta: Optional[float]        = None  # if set, ema this ema's weights back into training weights with this beta

    def __post_init__(self):
        if self.beta is not None and self.std is not None:
            raise ValueError(f"Cannot specify both beta ({self.beta}) and std ({self.std}) in EMA_Config for ema_{self.name}")
        if self.beta is None and self.std is None:
            raise ValueError(f"Must specify either beta or std in EMA_Config for ema_{self.name}")
        if self.beta is not None and (self.beta < 0 or self.beta >= 1):
            raise ValueError(f"Invalid beta value ({self.beta}) in EMA_Config for ema_{self.name}")
        if self.std is not None and self.std < 0:
            raise ValueError(f"Invalid std value ({self.std}) in EMA_Config for ema_{self.name}")
        if self.num_warmup_steps is not None and self.num_warmup_steps < 0:
            raise ValueError(f"Invalid warmup_steps value ({self.num_warmup_steps}) in EMA_Config for ema_{self.name}")
        if self.feedback_beta is not None and (self.feedback_beta < 0 or self.feedback_beta >= 1):
            raise ValueError(f"Invalid feedback_beta value ({self.feedback_beta}) in EMA_Config for ema_{self.name}")
        if self.num_switch_ema_epochs is not None and self.num_switch_ema_epochs <= 0:
            raise ValueError(f"Invalid switch_ema_epochs value ({self.num_switch_ema_epochs}) in EMA_Config for ema_{self.name}")
        if self.std is not None and self.num_warmup_steps is not None and self.num_warmup_steps > 0:
            raise ValueError(f"Cannot use power function ema (std: {self.std}) with warmup (warmup_steps: {self.num_warmup_steps}) in EMA_Config for ema_{self.name}")
        if len(self.name) == 0:
            raise ValueError("EMA name cannot be empty in EMA_Config")

class EMA_Manager:

    @torch.no_grad()
    def __init__(self, ema_configs: dict[str, dict[str, Any]], trainer: DualDiffusionTrainer) -> None:

        self.trainer: DualDiffusionTrainer = trainer
        self.module: DualDiffusionModule = trainer.module
        self.ema_modules: dict[str, DualDiffusionModule] = {}
        self.ema_configs: dict[str, EMA_Config] = {}
        self.switch_ema_name = None

        # initialize ema_modules for each config from module
        for name, config in ema_configs.items():
            if config.get("name", None) is not None:
                raise ValueError(f"Found unknown attribute 'name' in EMA_Config for ema_{name}")
            ema_config = EMA_Config(name, **config)
            self.ema_configs[name] = ema_config

            if ema_config.num_switch_ema_epochs is not None:
                if self.switch_ema_name is None:
                    self.switch_ema_name = name
                else:
                    raise ValueError("Only one EMA can be designated as the switch EMA")
                
            ema_device = "cpu" if ema_config.cpu_offload == True else self.trainer.accelerator.device
            self.ema_modules[name] = deepcopy(self.module).to(ema_device)

            # may increase performance when using CPU offload
            if ema_device == "cpu":
                for param in self.ema_modules[name].parameters():
                    param.pin_memory()

    def get_validation_emas(self) -> list[str]:
        return [name for name, config in self.ema_configs.items() if config.include_in_validation == True]

    @torch.no_grad() # reset all emas to the current state of the module
    def reset(self) -> None:
        for ema_module in self.ema_modules.values():
            torch._foreach_copy_(tuple(ema_module.parameters()), tuple(self.module.parameters()))

    @torch.no_grad() # update/step all emas with feedback (if any)
    def update(self) -> None:

        with torch.amp.autocast("cuda", enabled=False), TF32_Disabled():
            net_parameters = tuple(self.module.parameters())
            for name, config in self.ema_configs.items():
                
                # if using power function ema, calculate the effective beta
                beta = config.beta or float(power_function_beta(std=config.std,
                    t_next=self.trainer.persistent_state.total_samples_processed + self.trainer.total_batch_size,
                        t_delta=self.trainer.total_batch_size))
                if config.num_warmup_steps is not None and config.num_warmup_steps > 0:
                    beta *= min(self.trainer.global_step / config.num_warmup_steps, 1)

                ema_parameters = tuple(self.ema_modules[name].parameters())
                torch._foreach_lerp_(ema_parameters, net_parameters, 1 - beta)

                if config.feedback_beta is not None:
                    torch._foreach_lerp_(net_parameters, ema_parameters, 1 - config.feedback_beta)

                if self.trainer.accelerator.is_main_process == True:
                    if config.num_archive_steps is not None and config.num_archive_steps > 0:
                        if self.trainer.global_step % config.num_archive_steps == 0 and self.trainer.global_step > 0:
                            self.save_ema(name, self.trainer.config.model_path,
                                subfolder=f"{self.trainer.config.module_name}_ema_archive", archive=True)

    @torch.no_grad() # if enabled, load switch ema weights into train weights if it isn't warming up
    def switch_ema(self) -> Optional[str]:

        if self.switch_ema_name is not None:
            if self.trainer.global_step >= self.ema_configs[self.switch_ema_name].num_warmup_steps:
                if self.trainer.epoch % self.ema_configs[self.switch_ema_name].num_switch_ema_epochs == 0:
                    self.module.load_state_dict(self.ema_modules[self.switch_ema_name].state_dict())
                    self.module.normalize_weights()
                    return self.switch_ema_name # return the name of the ema if we switched
        
        return None # otherwise return None

    @torch.no_grad() # save all registered emas
    def save(self, save_directory: str, subfolder: Optional[str] = None) -> None:
        for ema_name in self.ema_configs:
            self.save_ema(ema_name, save_directory, subfolder=subfolder)
            
    @torch.no_grad() # save a specific ema to disk with corresponding config as metadata
    def save_ema(self, ema_name: str, save_directory: str, subfolder: Optional[str] = None, archive: bool = False) -> None:

        if subfolder is not None:
            save_directory = os.path.join(save_directory, subfolder)
        os.makedirs(save_directory, exist_ok=True)

        ema_metadata = {k: str(v) for k, v in self.ema_configs[ema_name].__dict__.items()}
        ema_metadata["global_step"] = str(self.trainer.global_step)
        ema_metadata["total_samples_processed"] = str(self.trainer.persistent_state.total_samples_processed)
        
        if archive == True:
            ema_save_path = os.path.join(save_directory, f"{self.trainer.global_step}_ema_{ema_name}.safetensors")
            state_dict = {k: v.to("cpu", dtype=torch.bfloat16) for k, v in self.ema_modules[ema_name].state_dict().items()}
        else:
            ema_save_path = os.path.join(save_directory, f"ema_{ema_name}.safetensors")
            state_dict = self.ema_modules[ema_name].state_dict()

        save_safetensors(state_dict, ema_save_path, metadata=ema_metadata)

    @torch.no_grad() # load all emas registered in ema_configs from specified path
    def load(self, ema_path: str, subfolder: Optional[str] = None,
             target_module: Optional[torch.nn.Module] = None) -> list[str]:
        
        if subfolder is not None:
            ema_path = os.path.join(ema_path, subfolder)
        
        # target_module is used to initialize any ema_modules that are not found in the load path
        self.module = target_module or self.module
        
        # load emas that are registered in ema_configs, if they can't be found return an error
        load_errors = []
        for name in self.ema_configs:
            ema_load_path = os.path.join(ema_path, f"ema_{name}.safetensors")
            if os.path.isfile(ema_load_path):
                self.ema_modules[name].load_state_dict(load_safetensors(ema_load_path))
            else:
                error_str = f"Could not find EMA weights for {name} at {ema_load_path} - will init from train weights"
                if target_module is not None:
                    torch._foreach_copy_(tuple(self.ema_modules[name].parameters()), tuple(target_module.parameters()))
                    load_errors.append(error_str)
                else: # error is fatal if no target_module is provided to initialize from
                    raise FileNotFoundError(error_str)
        
        # if there are any emas in the load path that aren't registered in ema_configs return an error
        emas_in_path = find_emas_in_dir(ema_path)
        for name in emas_in_path:
            if name not in self.ema_configs:
                error_str = f"Found EMA weights {os.path.join(ema_path, emas_in_path[name])} but no corresponding EMA_Config"
                error_str += " - this ema will be discarded"
                load_errors.append(error_str)

        return load_errors