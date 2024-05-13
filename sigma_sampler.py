import torch
import numpy as np

from dual_diffusion_utils import load_safetensors

class SigmaSampler():

    def __init__(self,
                 sigma_max = 80.,
                 sigma_min = 0.002,
                 sigma_data = 0.5,
                 distribution="log_normal",
                 dist_scale = 1.,
                 dist_offset = -0.4,
                 distribution_file=None):
        
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.sigma_data = sigma_data
        self.distribution = distribution
        self.dist_scale = dist_scale
        self.dist_offset = dist_offset

        if distribution_file is not None:
            self.distribution = "ln_data"
            dist_statistics = load_safetensors(distribution_file)

            self.dist_pdf = dist_statistics["ln_sigma_pdf"]
            self.dist_cdf = self.dist_pdf.cumsum(dim=0)

            self.sigma_ln_min = dist_statistics["ln_sigma_range"][0]
            self.sigma_ln_max = dist_statistics["ln_sigma_range"][1]
            self.sigma_min = self.sigma_ln_min.exp()
            self.sigma_max = self.sigma_ln_max.exp()
            self.dist_scale = None
            self.dist_offset = None
        else:
            self.sigma_ln_min = np.log(sigma_min)
            self.sigma_ln_max = np.log(sigma_max)
            
            if distribution == "ln_data":
                raise ValueError("distribution_file is required for ln_data distribution")
            
        if distribution not in ["log_normal", "log_sech^2", "ln_data"]:
            raise ValueError(f"Invalid distribution: {distribution}")

    def sample(self, n_samples, quantiles=None):
        
        if self.distribution == "log_normal":
            return self.sample_log_normal(n_samples, quantiles)
        elif self.distribution == "log_sech^2":
            return self.sample_log_sech2(n_samples, quantiles)
        elif self.distribution == "ln_data":
            return self.sample_ln_data(n_samples, quantiles)
    
    def sample_log_normal(self, n_samples, quantiles=None):
        if quantiles is None:
            return (torch.randn(n_samples) * self.dist_scale + self.dist_offset).exp().clip(self.sigma_min, self.sigma_max)
        ln_sigma = self.dist_offset + (self.dist_scale * 2**0.5) * (quantiles * 2 - 1).erfinv().clip(min=-5, max=5)
        return ln_sigma.exp().clip(self.sigma_min, self.sigma_max)
    
    def sample_log_sech2(self, n_samples, quantiles=None):
        if quantiles is None:
            quantiles = torch.rand(n_samples)
        low = np.tanh(self.sigma_ln_min); high = np.tanh(self.sigma_ln_max)
        ln_sigma = (quantiles * (high - low) + low).atanh() * self.dist_scale + self.dist_offset
        return ln_sigma.exp().clip(self.sigma_min, self.sigma_max)
        
    def sample_ln_data(self, n_samples, quantiles=None):
        if quantiles is None:
            quantiles = torch.rand(n_samples)
        ln_sigma = torch.quantile(self.dist_cdf, quantiles) * (self.sigma_ln_max - self.sigma_ln_min) + self.sigma_ln_min
        return ln_sigma.exp().clip(self.sigma_min, self.sigma_max)