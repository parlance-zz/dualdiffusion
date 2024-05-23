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

from dual_diffusion_pipeline import DualDiffusionPipeline

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

        if distribution_pdf is not None:
            self.distribution = "ln_data"
            self.dist_pdf = distribution_pdf / distribution_pdf.sum()
            self.dist_cdf = torch.cat((torch.tensor([0.], device=self.dist_pdf.device), self.dist_pdf.cumsum(dim=0)))

            self.sigma_ln_min = np.log(sigma_min)
            self.sigma_ln_max = np.log(sigma_max)
            self.sigma_min = np.exp(self.sigma_ln_min)
            self.sigma_max = np.exp(self.sigma_ln_max)
            self.dist_scale = None
            self.dist_offset = None
            
        else:
            self.sigma_ln_min = np.log(sigma_min)
            self.sigma_ln_max = np.log(sigma_max)

            if distribution == "ln_data":
                raise ValueError("distribution_pdf is required for ln_data distribution")
            
        if distribution not in ["log_normal", "log_sech^2", "ln_data"]:
            raise ValueError(f"Invalid distribution: {distribution}")
        
        if sigma_min <= 0:
            raise ValueError(f"sigma_min must be greater than 0: {sigma_min}")

    def sample_stratified(self, n_samples):
        return (torch.arange(n_samples) + 0.5) / n_samples + (torch.rand(1) - 0.5) / n_samples
    
    def sample(self, n_samples, stratified_sampling=False):
        
        if self.distribution == "log_normal":
            return self.sample_log_normal(n_samples, stratified_sampling)
        elif self.distribution == "log_sech^2":
            return self.sample_log_sech2(n_samples, stratified_sampling)
        elif self.distribution == "ln_data":
            return self.sample_log_data(n_samples, stratified_sampling)
    
    def sample_log_normal(self, n_samples, stratified_sampling=False):
        if stratified_sampling is False:
            return (torch.randn(n_samples) * self.dist_scale + self.dist_offset).exp().clip(self.sigma_min, self.sigma_max)
        
        quantiles = self.sample_stratified(n_samples)
        ln_sigma = self.dist_offset + (self.dist_scale * 2**0.5) * (quantiles * 2 - 1).erfinv().clip(min=-5, max=5)
        return ln_sigma.exp().clip(self.sigma_min, self.sigma_max)
    
    def sample_log_sech2(self, n_samples, stratified_sampling=False):
        if stratified_sampling is False:
            quantiles = torch.rand(n_samples)
        else:
            quantiles = self.sample_stratified(n_samples)

        low = np.tanh(self.sigma_ln_min); high = np.tanh(self.sigma_ln_max)
        ln_sigma = (quantiles * (high - low) + low).atanh() * self.dist_scale + self.dist_offset
        return ln_sigma.exp().clip(self.sigma_min, self.sigma_max)
    
    def update_pdf(self, pdf):
        self.dist_pdf = pdf / pdf.sum()
        self.dist_cdf[1:] = self.dist_pdf.cumsum(dim=0)

    def _sample_pdf(self, quantiles):
        sample_indices = torch.searchsorted(self.dist_cdf[1:], quantiles, out_int32=True)
        left_bin_values = self.dist_cdf[sample_indices]
        right_bin_values = self.dist_cdf[sample_indices+1]
        t = (quantiles - left_bin_values) / (right_bin_values - left_bin_values)
        return (sample_indices + t) / self.dist_cdf.shape[0]

    def sample_log_data(self, n_samples, stratified_sampling=False, quantiles=None):
        if stratified_sampling is False:
            quantiles = quantiles or torch.rand(n_samples)
        else:
            quantiles = quantiles or self.sample_stratified(n_samples)

        ln_sigma = self._sample_pdf(quantiles) * (self.sigma_ln_max - self.sigma_ln_min) + self.sigma_ln_min
        return ln_sigma.exp().clip(min=self.sigma_min, max=self.sigma_max)


if __name__ == "__main__":

    from dual_diffusion_utils import multi_plot
    from dotenv import load_dotenv
    import os

    load_dotenv(override=True)

    reference_model_name = "edm2_vae_test7_4"
    target_snr = 32
    sigma_max = 80
    sigma_data = 0.5
    sigma_min = 0.002

    training_batch_size = 30
    #batch_distribution = "log_sech^2"
    batch_distribution = "ln_data"
    batch_dist_scale = 1
    batch_dist_offset = -0.4
    batch_stratified_sampling = True
    batch_distribution_pdf = None

    reference_batch_size = 2048
    reference_distribution = "log_normal"
    reference_dist_scale = 1
    reference_dist_offset = -0.4
    reference_stratified_sampling = False
    reference_distribution_pdf = None

    n_iter = 10000
    n_histo_bins = 200
    use_y_log_scale = False

    if batch_distribution == "ln_data":
        model_path = os.path.join(os.environ.get("MODEL_PATH", "./"), reference_model_name)
        print(f"Loading DualDiffusion model from '{model_path}'...")
        pipeline = DualDiffusionPipeline.from_pretrained(model_path, load_latest_checkpoints=True)
        ln_sigma = torch.linspace(np.log(sigma_min), np.log(sigma_max), n_histo_bins)
        ln_sigma_error = pipeline.unet.logvar_linear(pipeline.unet.logvar_fourier(ln_sigma/4)).float().flatten()
        batch_distribution_pdf = (-torch.e * ln_sigma_error).exp()

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
        multi_plot((batch_sampler.dist_pdf, "ln_sigma_pdf"),
                   (batch_sampler.dist_cdf, "ln_sigma_cdf"),)
        
    avg_batch_mean = avg_batch_min = avg_batch_max = 0
    avg_reference_mean = avg_reference_min = avg_reference_max = 0

    batch_sigma_histo = torch.zeros(n_histo_bins)
    reference_sigma_histo = torch.zeros(n_histo_bins)

    for i in range(n_iter):

        batch_ln_sigma = batch_sampler.sample(training_batch_size, batch_stratified_sampling).log()
        reference_ln_sigma = reference_sampler.sample(reference_batch_size, reference_stratified_sampling).log()

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

    multi_plot((batch_sigma_histo, "batch sigma"),
            added_plots={0: (reference_sigma_histo, "reference_sigma")},
            y_log_scale=use_y_log_scale, x_axis_range=(np.log(sigma_min), np.log(sigma_max)))