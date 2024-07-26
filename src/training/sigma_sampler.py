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

from dataclasses import dataclass
from typing import Optional, Literal
import torch
import numpy as np

@dataclass
class SigmaSamplerConfig:

    sigma_max: float  = 200.
    sigma_min: float  = 0.03
    sigma_data: float = 1.
    distribution: Literal["ln_normal", "ln_sech", "ln_sech^2", "ln_linear", "ln_pdf"] = "ln_sech"
    dist_scale: float  = 1.
    dist_offset: float = 0.1
    dist_pdf: Optional[torch.Tensor] = None
    use_stratified_sigma_sampling: bool = True
    sigma_pdf_resolution: Optional[int] = 127

    @property
    def ln_sigma_min(self) -> float:
        return np.log(self.sigma_min)
    
    @property
    def ln_sigma_max(self) -> float:
        return np.log(self.sigma_max)
    
class SigmaSampler():

    def __init__(self, sigma_sampler_config: SigmaSamplerConfig) -> None:
        self.config = sigma_sampler_config

        if self.config.distribution not in ["ln_normal", "ln_sech", "ln_sech^2", "ln_linear", "ln_pdf"]:
            raise ValueError(f"Invalid distribution: {self.config.distribution}")
            
        if self.config.distribution == "ln_normal":
            self.sample_fn = self.sample_ln_normal
        elif self.config.distribution == "ln_sech":
            self.sample_fn = self.sample_ln_sech
        elif self.config.distribution == "ln_sech^2":
            self.sample_fn = self.sample_ln_sech2
        elif self.config.distribution == "ln_linear":
            self.sample_fn = self.sample_ln_linear
        elif self.config.distribution == "ln_pdf":
            
            if self.config.dist_pdf is None:
                self.config.dist_pdf = torch.ones(self.config.sigma_pdf_resolution)
            self.dist_pdf = self.config.dist_pdf / self.config.dist_pdf.sum()
            self.dist_cdf = torch.cat((torch.tensor([0.], device=self.dist_pdf.device), self.dist_pdf.cumsum(dim=0)))

            self.sample_fn = self.sample_ln_pdf

    def _sample_uniform_stratified(self, n_samples: int) -> torch.Tensor:
        return (torch.arange(n_samples) + 0.5) / n_samples + (torch.rand(1) - 0.5) / n_samples
    
    def sample(self, n_samples: int, device: Optional[torch.device]=None) -> torch.Tensor:
        quantiles = self._sample_uniform_stratified(n_samples) if self.config.use_stratified_sigma_sampling else None
        return self.sample_fn(n_samples, quantiles).to(device)

    def sample_ln_normal(self, n_samples: Optional[int]=None, quantiles: Optional[torch.Tensor]=None) -> torch.Tensor:
        quantiles = quantiles or torch.rand(n_samples)
        
        ln_sigma = self.config.dist_offset + (self.config.dist_scale * 2**0.5) * (quantiles * 2 - 1).erfinv().clip(min=-5, max=5)
        ln_sigma[ln_sigma < self.config.ln_sigma_min] += self.config.ln_sigma_max - self.config.ln_sigma_min
        ln_sigma[ln_sigma > self.config.ln_sigma_max] -= self.config.ln_sigma_max - self.config.ln_sigma_min
        return ln_sigma.exp().clip(self.config.sigma_min, self.config.sigma_max)
    
    def sample_ln_sech(self, n_samples: Optional[int]=None, quantiles: Optional[torch.Tensor]=None) -> torch.Tensor:
        quantiles = quantiles or torch.rand(n_samples)

        theta_min = np.arctan(1 / self.config.sigma_max * np.exp(self.config.dist_offset))
        theta_max = np.arctan(1 / self.config.sigma_min * np.exp(self.config.dist_offset))

        theta = quantiles * (theta_max - theta_min) + theta_min
        ln_sigma = (1 / theta.tan()).log() * self.config.dist_scale + self.config.dist_offset
        return ln_sigma.exp().clip(min=self.config.sigma_min, max=self.config.sigma_max)
    
    def sample_ln_sech2(self, n_samples: Optional[int]=None, quantiles: Optional[torch.Tensor]=None) -> torch.Tensor:
        quantiles = quantiles or torch.rand(n_samples)

        low = np.tanh(self.config.ln_sigma_min); high = np.tanh(self.config.ln_sigma_max)
        ln_sigma = (quantiles * (high - low) + low).atanh() * self.config.dist_scale + self.config.dist_offset
        ln_sigma[ln_sigma < self.config.ln_sigma_min] += self.config.ln_sigma_max - self.config.ln_sigma_min
        ln_sigma[ln_sigma > self.config.ln_sigma_max] -= self.config.ln_sigma_max - self.config.ln_sigma_min
        return ln_sigma.exp().clip(self.config.sigma_min, self.config.sigma_max)
    
    def sample_ln_linear(self, n_samples: Optional[int]=None, quantiles: Optional[torch.Tensor]=None) -> torch.Tensor:
        quantiles = quantiles or torch.rand(n_samples)
        
        ln_sigma = quantiles * (self.config.ln_sigma_max - self.config.ln_sigma_min) + self.config.ln_sigma_min
        return ln_sigma.exp().clip(self.config.sigma_min, self.config.sigma_max)
    
    def update_pdf(self, pdf: torch.Tensor) -> None:
        self.dist_pdf = pdf / pdf.sum()
        self.dist_cdf[1:] = self.dist_pdf.cumsum(dim=0)

    def _sample_pdf(self, quantiles: torch.Tensor) -> torch.Tensor:
        quantiles = quantiles.to(self.dist_cdf.device)

        sample_indices = torch.searchsorted(self.dist_cdf, quantiles, out_int32=True).clip(max=self.dist_cdf.shape[0]-2)
        left_bin_values = self.dist_cdf[sample_indices]
        right_bin_values = self.dist_cdf[sample_indices+1]
        t = (quantiles - left_bin_values) / (right_bin_values - left_bin_values)

        return (sample_indices + t) / (self.dist_cdf.shape[0]-1)

    def sample_ln_pdf(self, n_samples: Optional[int]=None, quantiles: Optional[torch.Tensor]=None) -> torch.Tensor:
        quantiles = quantiles or torch.rand(n_samples)

        ln_sigma = self._sample_pdf(quantiles) * (self.config.ln_sigma_max - self.config.ln_sigma_min) + self.config.ln_sigma_min
        return ln_sigma.exp().clip(min=self.config.sigma_min, max=self.config.sigma_max)