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
import numpy as np

class SigmaSampler():

    def __init__(self,
                 sigma_max = 80.,
                 sigma_min = 0.002,
                 sigma_data = 0.5,
                 distribution="log_normal",
                 dist_scale = 1.,
                 dist_offset = -0.4,
                 distribution_pdf=None):
        
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.sigma_data = sigma_data
        self.distribution = distribution
        self.dist_scale = dist_scale
        self.dist_offset = dist_offset
        self.sigma_ln_min = np.log(sigma_min)
        self.sigma_ln_max = np.log(sigma_max)
    
        if distribution_pdf is not None:
            self.distribution = "ln_data"
            self.dist_pdf = distribution_pdf / distribution_pdf.sum()
            self.dist_cdf = torch.cat((torch.tensor([0.], device=self.dist_pdf.device), self.dist_pdf.cumsum(dim=0)))

            self.dist_scale = None
            self.dist_offset = None
        else:
            if distribution == "ln_data":
                raise ValueError("distribution_pdf is required for ln_data distribution")
            
        if distribution not in ["log_normal", "log_sech", "log_sech^2", "ln_data", "linear"]:
            raise ValueError(f"Invalid distribution: {distribution}")
        
        if sigma_min <= 0:
            raise ValueError(f"sigma_min must be greater than 0: {sigma_min}")

    def sample_uniform_stratified(self, n_samples):
        return (torch.arange(n_samples) + 0.5) / n_samples + (torch.rand(1) - 0.5) / n_samples
    
    def sample(self, n_samples, quantiles=None):
        
        if self.distribution == "log_normal":
            return self.sample_log_normal(n_samples, quantiles=quantiles)
        elif self.distribution == "log_sech":
            return self.sample_log_sech(n_samples, quantiles=quantiles)
        elif self.distribution == "log_sech^2":
            return self.sample_log_sech2(n_samples, quantiles=quantiles)
        elif self.distribution == "ln_data":
            return self.sample_log_data(n_samples, quantiles=quantiles)
        elif self.distribution == "linear":
            return self.sample_linear(n_samples, quantiles=quantiles)
    
    def sample_log_normal(self, n_samples, quantiles=None):
        if quantiles is None:
            return (torch.randn(n_samples) * self.dist_scale + self.dist_offset).exp().clip(self.sigma_min, self.sigma_max)
        
        ln_sigma = self.dist_offset + (self.dist_scale * 2**0.5) * (quantiles * 2 - 1).erfinv().clip(min=-5, max=5)
        return ln_sigma.exp().clip(self.sigma_min, self.sigma_max)
    
    def sample_log_sech(self, n_samples, quantiles=None):
        if quantiles is None:
            quantiles = torch.rand(n_samples)

        theta_min = np.arctan(self.sigma_data / self.sigma_max)
        theta_max = np.arctan(self.sigma_data / self.sigma_min)

        theta = quantiles * (theta_max - theta_min) + theta_min
        ln_sigma = (1 / theta.tan()).log() * self.dist_scale + self.dist_offset
        return ln_sigma.exp().clip(min=self.sigma_min, max=self.sigma_max)
    
    def sample_log_sech2(self, n_samples, quantiles=None):
        if quantiles is None:
            quantiles = torch.rand(n_samples)

        low = np.tanh(self.sigma_ln_min); high = np.tanh(self.sigma_ln_max)
        ln_sigma = (quantiles * (high - low) + low).atanh() * self.dist_scale + self.dist_offset
        return ln_sigma.exp().clip(self.sigma_min, self.sigma_max)
    
    def sample_linear(self, n_samples, quantiles=None):
        if quantiles is None:
            quantiles = torch.rand(n_samples)
        
        ln_sigma = quantiles * (self.sigma_ln_max - self.sigma_ln_min) + self.sigma_ln_min
        return ln_sigma.exp().clip(self.sigma_min, self.sigma_max)
    
    def update_pdf(self, pdf):
        self.dist_pdf = pdf / pdf.sum()
        self.dist_cdf[1:] = self.dist_pdf.cumsum(dim=0)

    def _sample_pdf(self, quantiles):
        quantiles = quantiles.to(self.dist_cdf.device)
        sample_indices = torch.searchsorted(self.dist_cdf, quantiles, out_int32=True).clip(max=self.dist_cdf.shape[0]-2)
        left_bin_values = self.dist_cdf[sample_indices]
        right_bin_values = self.dist_cdf[sample_indices+1]
        t = (quantiles - left_bin_values) / (right_bin_values - left_bin_values)
        return (sample_indices + t) / (self.dist_cdf.shape[0]-1)

    def sample_log_data(self, n_samples=None, quantiles=None):
        if quantiles is None:
            quantiles = torch.rand(n_samples)

        ln_sigma = self._sample_pdf(quantiles) * (self.sigma_ln_max - self.sigma_ln_min) + self.sigma_ln_min
        return ln_sigma.exp().clip(min=self.sigma_min, max=self.sigma_max)


if __name__ == "__main__":

    from dual_diffusion_pipeline import DualDiffusionPipeline
    from dual_diffusion_utils import multi_plot
    from dotenv import load_dotenv
    import os

    load_dotenv(override=True)

    reference_model_name = "edm2_vae_test7_4"
    target_snr = 32
    sigma_max = 125 #80
    sigma_data = 0.5
    sigma_min = sigma_data / target_snr #0.002

    training_batch_size = 120
    batch_distribution = "ln_data"
    batch_dist_scale = 2.2
    batch_dist_offset = 0
    batch_stratified_sampling = True
    batch_distribution_pdf = None

    reference_batch_size = 60
    reference_distribution = "ln_data"
    reference_dist_scale = 2
    reference_dist_offset = 0
    #reference_distribution = "log_normal"
    #reference_dist_scale = 1
    #reference_dist_offset = -0.4
    #reference_distribution = "log_sech"
    #reference_dist_scale = 1
    #reference_dist_offset = -0.54
    reference_stratified_sampling = True
    reference_distribution_pdf = None

    n_iter = 10000
    n_histo_bins = 200
    use_y_log_scale = True

    if batch_distribution == "ln_data" or reference_distribution == "ln_data":
        model_path = os.path.join(os.environ.get("MODEL_PATH", "./"), reference_model_name)
        print(f"Loading DualDiffusion model from '{model_path}'...")
        pipeline = DualDiffusionPipeline.from_pretrained(model_path, load_latest_checkpoints=True)
        ln_sigma = torch.linspace(np.log(sigma_min), np.log(sigma_max), n_histo_bins)
        ln_sigma_error = pipeline.unet.logvar_linear(pipeline.unet.logvar_fourier(ln_sigma/4)).float().flatten()
    
    if batch_distribution == "ln_data":
        batch_distribution_pdf = ((-batch_dist_scale * ln_sigma_error) + batch_dist_offset).exp()

    if reference_distribution == "ln_data":
        reference_distribution_pdf = ((-reference_dist_scale * ln_sigma_error) + reference_dist_offset).exp()

    batch_sampler = SigmaSampler(sigma_max=sigma_max,
                                 sigma_min=sigma_min,
                                 sigma_data=sigma_data,
                                 distribution=batch_distribution,
                                 dist_scale=batch_dist_scale,
                                 dist_offset=batch_dist_offset,
                                 distribution_pdf=batch_distribution_pdf)

    reference_sampler = SigmaSampler(sigma_max=sigma_max,
                                     sigma_min=sigma_min,
                                     sigma_data=sigma_data,
                                     distribution=reference_distribution,
                                     dist_scale=reference_dist_scale,
                                     dist_offset=reference_dist_offset,
                                     distribution_pdf=reference_distribution_pdf)

    if batch_sampler.distribution == "ln_data":
        multi_plot((batch_sampler.dist_pdf, "batch_distribution_pdf"),
                   (batch_sampler.dist_cdf, "batch_distribution_cdf"),)
        
    avg_batch_mean = avg_batch_min = avg_batch_max = 0
    avg_reference_mean = avg_reference_min = avg_reference_max = 0

    batch_sigma_histo = torch.zeros(n_histo_bins)
    reference_sigma_histo = torch.zeros(n_histo_bins)

    batch_example = torch.ones(n_histo_bins * 4)
    reference_example = torch.ones(n_histo_bins * 4)

    for i in range(n_iter):
        
        batch_quantiles = batch_sampler.sample_uniform_stratified(training_batch_size) if batch_stratified_sampling else None
        batch_ln_sigma = batch_sampler.sample(training_batch_size, quantiles=batch_quantiles).log()

        reference_quantiles = reference_sampler.sample_uniform_stratified(reference_batch_size) if reference_stratified_sampling else None
        reference_ln_sigma = reference_sampler.sample(reference_batch_size, quantiles=reference_quantiles).log()

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

    multi_plot((batch_sigma_histo, "batch sigma"), (batch_example, "batch example"), (reference_example, "reference example"),
            added_plots={0: (reference_sigma_histo, "reference_sigma")},
            y_log_scale=use_y_log_scale, x_axis_range=(np.log(sigma_min), np.log(sigma_max)))