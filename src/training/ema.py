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

import os
import copy
from typing import Optional

import torch
import numpy as np

from utils.dual_diffusion_utils import save_safetensors, load_safetensors

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
# Class for tracking power function EMA during the training.

class PowerFunctionEMA:
    
    def __init__(self, module: torch.nn.Module, stds: list[float],
                 device: Optional[torch.device] = None) -> None:

        self.module = module
        self.stds = stds
        self.device = device

        self.emas = [copy.deepcopy(module).to(device) for _ in stds]

    @torch.no_grad()
    def reset(self) -> None:
        for ema in self.emas:
            torch._foreach_copy_(ema.parameters(), self.module.parameters())

    @torch.no_grad()
    def update(self, cur_nimg: int, batch_size: int) -> list[tuple[float, float]]:
        betas = []
        net_parameters = tuple(self.module.parameters())
        for std, ema in zip(self.stds, self.emas):
            beta = float(power_function_beta(std=std, t_next=cur_nimg, t_delta=batch_size))
            torch._foreach_lerp_(tuple(ema.parameters()), net_parameters, 1 - beta)
            betas.append(beta)
        return zip(self.stds, betas)

    def save(self, save_directory: str, subfolder: Optional[str] = None) -> None:

        if subfolder is not None:
            save_directory = os.path.join(save_directory, subfolder)
        os.makedirs(save_directory, exist_ok=True)

        for std, ema in zip(self.stds, self.emas):
            ema_save_path = os.path.join(save_directory, f"pf_ema_std-{std:.3f}.safetensors")
            save_safetensors(ema.state_dict(), ema_save_path)

    @torch.no_grad()
    def load(self, ema_path: str, subfolder: Optional[str] = None,
             target_module: Optional[torch.nn.Module] = None) -> list[str]:
        
        if subfolder is not None:
            ema_path = os.path.join(ema_path, subfolder)
            
        self.module = target_module or self.module

        load_errors = []
        for std, ema in zip(self.stds, self.emas):
            ema_load_path = os.path.join(ema_path, f"pf_ema_std-{std:.3f}.safetensors")
            if os.path.isfile(ema_load_path):
                ema.load_state_dict(load_safetensors(ema_load_path))
            else:
                error_str = f"Could not find PowerFunctionEMA model for std={std:.3f} at {ema_load_path}"
                if target_module is not None:
                    torch._foreach_copy_(tuple(ema.parameters()), tuple(target_module.parameters()))
                    load_errors.append(error_str)
                else:
                    raise FileNotFoundError(error_str)
        return load_errors