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

from typing import Optional, Any
import inspect

import torch
import numpy as np


class SamplingSchedule:

    @staticmethod
    @torch.inference_mode()
    def get_schedule(name: str, steps: int, t_start: float = 1., device: Optional[torch.device] = None, **kwargs) -> torch.Tensor:
        schedule_fn = getattr(SamplingSchedule, f"schedule_{name}")
        t = torch.linspace(t_start, 0, int(steps) + 1, device=device)
        return schedule_fn(t, **kwargs)
    
    @staticmethod
    def get_schedule_params(name: str) -> dict[str, type[Any]]:
        params = {param_name: param_type.annotation
            for param_name, param_type in inspect.signature(getattr(SamplingSchedule, f"schedule_{name}")).parameters.items()}
        if "t" in params: del params["t"]
        if "_" in params: del params["_"]
        if "sigma_max" in params: del params["sigma_max"]
        if "sigma_min" in params: del params["sigma_min"]
        return params
    
    @classmethod
    def get_schedules_list(cls) -> list[str]:
        schedules = []
        for attr in dir(cls):
            if callable(getattr(cls, attr)) and attr.startswith("schedule_"):
                schedules.append(attr.removeprefix("schedule_"))
        return schedules

    @staticmethod
    def schedule_edm2(t: torch.Tensor, sigma_max: float, sigma_min: float, rho: float = 7., **_) -> torch.Tensor:
        return (sigma_max ** (1 / rho) + (1 - t) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        
    @staticmethod
    def schedule_ln_linear(t: torch.Tensor, sigma_max: float, sigma_min: float, **_) -> torch.Tensor:
        return (np.log(sigma_min) + (np.log(sigma_max) - np.log(sigma_min)) * t).exp()
    
    @staticmethod
    def schedule_linear(t: torch.Tensor, sigma_max: float, sigma_min: float, rho: float = 1., **_) -> torch.Tensor:
        t = (sigma_max**(1/rho) - sigma_min**(1/rho)) * t + sigma_min**(1/rho)
        return t**rho
    
    @staticmethod
    def schedule_cos(t: torch.Tensor, sigma_max: float, sigma_min: float, rho: float = 1., **_) -> torch.Tensor:
        theta_max = np.pi/2 - np.arctan(sigma_max / rho)
        theta_min = np.pi/2 - np.arctan(sigma_min / rho)
        theta = (1-t) * (theta_min - theta_max) + theta_max
        return theta.cos() / theta.sin() * rho


if __name__ == "__main__":
    schedule_list = SamplingSchedule.get_schedules_list()
    for schedule in schedule_list:
        print(schedule, ":", SamplingSchedule.get_schedule_params(schedule))