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

class SamplingSchedule:

    @staticmethod
    @torch.no_grad()
    def get_schedule(name: str, steps: int, t_start: float = 1., device: Optional[torch.device] = None, **kwargs) -> torch.Tensor:
        schedule_fn = getattr(SamplingSchedule, name)
        t = torch.linspace(t_start, 0, int(steps) + 1, device=device)
        return schedule_fn(t, **kwargs)
    
    @staticmethod
    def get_schedule_params(name: str) -> dict[str, type[Any]]:
        params = {param_name: param_type.annotation
            for param_name, param_type in inspect.signature(getattr(SamplingSchedule, name)).parameters.items()}
        if "t" in params: del params["t"]
        if "_" in params: del params["_"]
        if "sigma_max" in params: del params["sigma_max"]
        if "sigma_min" in params: del params["sigma_min"]
        return params
    
    @classmethod
    def get_schedules_list(cls) -> list[str]:
        schedules = []
        for name in dir(cls):
            if (callable(getattr(cls, name))
                and not name.startswith("__")
                and name not in ["get_schedule", "get_schedules_list", "get_schedule_params"]):

                schedules.append(name)
        return schedules
    
    @staticmethod
    @torch.no_grad()
    def linear(t: torch.Tensor, sigma_max: float, sigma_min: float, **_) -> torch.Tensor:
        return sigma_min + (sigma_max - sigma_min) * t
    
    @staticmethod
    @torch.no_grad()
    def edm2(t: torch.Tensor, sigma_max: float, sigma_min: float, rho: float = 7., **_) -> torch.Tensor:
        return (sigma_max ** (1 / rho) + (1 - t) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    
    @staticmethod
    @torch.no_grad()
    def scale_invariant(t: torch.Tensor, sigma_max: float, sigma_min: float, rho: float = 1., **_) -> torch.Tensor:
        return sigma_min / ((1 - t)**rho + sigma_min / sigma_max)
    

if __name__ == "__main__":
    schedule_list = SamplingSchedule.get_schedules_list()
    for schedule in schedule_list:
        print(schedule, ":", SamplingSchedule.get_schedule_params(schedule))