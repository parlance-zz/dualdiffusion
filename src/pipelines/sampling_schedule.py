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

import torch

class SamplingSchedule:

    @staticmethod
    @torch.no_grad()
    def get_schedule(name: str, steps: int, device: torch.device = "cpu", **kwargs) -> torch.Tensor:
        schedule_fn = getattr(SamplingSchedule, name)
        t = torch.linspace(1, 0, steps, device=device)
        return schedule_fn(t, **kwargs)
    
    @staticmethod
    @torch.no_grad()
    def linear(t: torch.Tensor, sigma_max: float, sigma_min: float) -> torch.Tensor:
        return sigma_min + (sigma_max - sigma_min) * t
    
    @staticmethod
    @torch.no_grad()
    def edm2(t: torch.Tensor, sigma_max: float, sigma_min: float, rho: float = 7.) -> torch.Tensor:
        return (sigma_max ** (1 / rho) + (1 - t) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho