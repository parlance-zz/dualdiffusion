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


if __name__ == "__main__":

    import utils.config as config
    import os
    from pipelines.dual_diffusion_pipeline import DualDiffusionPipeline
    from utils.dual_diffusion_utils import multi_plot

    reference_model_name = "edm2_vae_test7_12"
    target_snr = 32.
    sigma_max = 200.
    sigma_data = 1.
    sigma_min = sigma_data / target_snr

    training_batch_size = 384
    batch_distribution = "ln_pdf"
    batch_dist_scale = 1.
    batch_dist_offset = 0.
    batch_stratified_sampling = True

    reference_batch_size = 384
    reference_distribution = "log_sech"
    reference_dist_scale = 1.
    reference_dist_offset = 0.1
    reference_stratified_sampling = True

    n_iter = 10000
    n_histo_bins = 200
    use_y_log_scale = False

    # ***********************************

    batch_distribution_pdf = None
    reference_distribution_pdf = None

    if batch_distribution == "ln_pdf" or reference_distribution == "ln_pdf":

        model_path = os.path.join(config.MODEL_PATH, reference_model_name)
        print(f"Loading DualDiffusion model from '{model_path}'...")
        pipeline = DualDiffusionPipeline.from_pretrained(model_path, load_latest_checkpoints=True)

        ln_sigma = torch.linspace(np.log(sigma_min), np.log(sigma_max), n_histo_bins)
        ln_sigma_error = pipeline.unet.logvar_linear(pipeline.unet.logvar_fourier(ln_sigma/4)).float().flatten()
    
        if batch_distribution == "ln_pdf":
            batch_distribution_pdf = (-ln_sigma_error / batch_dist_scale).exp()
        if reference_distribution == "ln_pdf":
            reference_distribution_pdf = (-ln_sigma_error / reference_dist_scale).exp()

    batch_sampler_config = SigmaSamplerConfig(
        sigma_max=sigma_max,
        sigma_min=sigma_min,
        sigma_data=sigma_data,
        distribution=batch_distribution,
        dist_scale=batch_dist_scale,
        dist_offset=batch_dist_offset,
        dist_pdf=batch_distribution_pdf,
        use_stratified_sigma_sampling=batch_stratified_sampling,
    )
    batch_sampler = SigmaSampler(batch_sampler_config)

    reference_sampler_config = SigmaSamplerConfig(
        sigma_max=sigma_max,
        sigma_min=sigma_min,
        sigma_data=sigma_data,
        distribution=reference_distribution,
        dist_scale=reference_dist_scale,
        dist_offset=reference_dist_offset,
        dist_pdf=reference_distribution_pdf,
        use_stratified_sigma_sampling=reference_stratified_sampling,
    )
    reference_sampler = SigmaSampler(reference_sampler_config)

    if batch_sampler.config.distribution == "ln_pdf":
        multi_plot((batch_sampler.dist_pdf, "batch_distribution_pdf"),
                   (batch_sampler.dist_cdf, "batch_distribution_cdf"),)
        
    avg_batch_mean = avg_batch_min = avg_batch_max = 0
    avg_reference_mean = avg_reference_min = avg_reference_max = 0

    batch_sigma_histo = torch.zeros(n_histo_bins)
    reference_sigma_histo = torch.zeros(n_histo_bins)

    batch_example = torch.ones(n_histo_bins * 8)
    reference_example = torch.ones(n_histo_bins * 8)

    for i in range(n_iter):
        
        batch_ln_sigma = batch_sampler.sample(training_batch_size).log()
        reference_ln_sigma = reference_sampler.sample(reference_batch_size).log()

        if i == 0:
            print(f"batch example sigma: {batch_ln_sigma.exp()}")
            print(f"reference example sigma: {reference_ln_sigma.exp()}")

            batch_example_idx = (batch_ln_sigma - np.log(sigma_min)) / (np.log(sigma_max) - np.log(sigma_min)) * (batch_example.shape[0]-1)
            batch_example.scatter_add_(0, batch_example_idx.long(), torch.ones(training_batch_size))
            reference_example_idx = (reference_ln_sigma - np.log(sigma_min)) / (np.log(sigma_max) - np.log(sigma_min)) * (reference_example.shape[0]-1)
            reference_example.scatter_add_(0, reference_example_idx.long(), torch.ones(reference_batch_size))

        avg_batch_mean += batch_ln_sigma.mean().item()
        avg_batch_min  += batch_ln_sigma.amin().item()
        avg_batch_max  += batch_ln_sigma.amax().item()
        avg_reference_mean += reference_ln_sigma.mean().item()
        avg_reference_min  += reference_ln_sigma.amin().item()
        avg_reference_max  += reference_ln_sigma.amax().item()

        batch_sigma_histo += batch_ln_sigma.histc(bins=n_histo_bins, min=np.log(sigma_min), max=np.log(sigma_max)) / (training_batch_size * n_iter)
        reference_sigma_histo += reference_ln_sigma.histc(bins=n_histo_bins, min=np.log(sigma_min), max=np.log(sigma_max)) / (reference_batch_size * n_iter)

    avg_batch_mean = np.exp(avg_batch_mean / n_iter)
    avg_batch_min = np.exp(avg_batch_min / n_iter)
    avg_batch_max = np.exp(avg_batch_max / n_iter)
    avg_reference_mean = np.exp(avg_reference_mean / n_iter)
    avg_reference_min = np.exp(avg_reference_min / n_iter)
    avg_reference_max = np.exp(avg_reference_max / n_iter)

    print(f"avg batch     mean: {avg_batch_mean:{5}f}, min: {avg_batch_min:{5}f}, max: {avg_batch_max:{5}f}")
    print(f"avg reference mean: {avg_reference_mean:{5}f}, min: {avg_reference_min:{5}f}, max: {avg_reference_max:{5}f}")

    batch_center = ((batch_sigma_histo**2 * torch.linspace(np.log(sigma_min), np.log(sigma_max), n_histo_bins)).sum() / (batch_sigma_histo**2).sum()).item()
    reference_center = ((reference_sigma_histo**2 * torch.linspace(np.log(sigma_min), np.log(sigma_max), n_histo_bins)).sum() / (reference_sigma_histo**2).sum()).item()
    print(f"batch center: {batch_center:{5}f}, reference center: {reference_center:{5}f}")

    multi_plot((batch_sigma_histo, "batch sigma"), (batch_example, "batch example"), (reference_example, "reference example"),
            added_plots={0: (reference_sigma_histo, "reference_sigma")},
            y_log_scale=use_y_log_scale, x_axis_range=(np.log(sigma_min), np.log(sigma_max)))