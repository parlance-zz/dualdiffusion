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
        
        if sigma_min <= 0:
            raise ValueError(f"sigma_min must be greater than 0: {sigma_min}")

    def sample_stratified_quantiles(self, n_samples):
        return (torch.arange(n_samples) + 0.5) / n_samples + (torch.rand(1) - 0.5) / n_samples
    
    def sample(self, n_samples, stratified_sampling=False):
        
        if self.distribution == "log_normal":
            return self.sample_log_normal(n_samples, stratified_sampling)
        elif self.distribution == "log_sech^2":
            return self.sample_log_sech2(n_samples, stratified_sampling)
        elif self.distribution == "ln_data":
            return self.sample_ln_data(n_samples, stratified_sampling)
    
    def sample_log_normal(self, n_samples, stratified_sampling=False):
        if stratified_sampling is False:
            return (torch.randn(n_samples) * self.dist_scale + self.dist_offset).exp().clip(self.sigma_min, self.sigma_max)
        
        quantiles = self.sample_stratified_quantiles(n_samples)
        ln_sigma = self.dist_offset + (self.dist_scale * 2**0.5) * (quantiles * 2 - 1).erfinv().clip(min=-5, max=5)
        return ln_sigma.exp().clip(self.sigma_min, self.sigma_max)
    
    def sample_log_sech2(self, n_samples, stratified_sampling=False):
        if stratified_sampling is False:
            quantiles = torch.rand(n_samples)
        else:
            quantiles = self.sample_stratified_quantiles(n_samples)

        low = np.tanh(self.sigma_ln_min); high = np.tanh(self.sigma_ln_max)
        ln_sigma = (quantiles * (high - low) + low).atanh() * self.dist_scale + self.dist_offset
        return ln_sigma.exp().clip(self.sigma_min, self.sigma_max)
        
    def sample_ln_data(self, n_samples, stratified_sampling=False):
        if stratified_sampling is False:
            quantiles = torch.rand(n_samples)
        else:
            quantiles = self.sample_stratified_quantiles(n_samples)

        ln_sigma = torch.quantile(self.dist_cdf, quantiles) * (self.sigma_ln_max - self.sigma_ln_min) + self.sigma_ln_min
        return ln_sigma.exp().clip(self.sigma_min, self.sigma_max)
    

if __name__ == "__main__":

    from dual_diffusion_utils import multi_plot
    from dotenv import load_dotenv
    import os

    load_dotenv(override=True)
    dataset_path = os.environ.get("LATENTS_DATASET_PATH", "./dataset/latents")

    target_snr = 32
    sigma_max = 80
    sigma_data = 0.5
    sigma_min = 0.002

    training_batch_size = 91
    batch_distribution = "log_sech^2"
    batch_dist_scale = 1.22
    batch_dist_offset = -0.5
    batch_stratified_sampling = True
    batch_distribution_file = None
    #batch_distribution_file = os.path.join(dataset_path, "statistics.safetensors")

    reference_batch_size = 2048
    reference_distribution = "log_normal"
    reference_dist_scale = 1
    reference_dist_offset = -0.4
    reference_stratified_sampling = False
    reference_distribution_file = None

    n_iter = 10000
    n_histo_bins = 5000
    use_y_log_scale = True

    batch_sampler = SigmaSampler(sigma_max=sigma_max,
                                 sigma_min=sigma_min,
                                 sigma_data=sigma_data,
                                 distribution=batch_distribution,
                                 dist_scale=batch_dist_scale,
                                 dist_offset=batch_dist_offset,
                                 distribution_file=batch_distribution_file)

    reference_sampler = SigmaSampler(sigma_max=sigma_max,
                                     sigma_min=sigma_min,
                                     sigma_data=sigma_data,
                                     distribution=reference_distribution,
                                     dist_scale=reference_dist_scale,
                                     dist_offset=reference_dist_offset,
                                     distribution_file=reference_distribution_file)

    avg_batch_mean = avg_batch_min = avg_batch_max = 0
    avg_reference_mean = avg_reference_min = avg_reference_max = 0

    batch_sigma_histo = torch.zeros(n_histo_bins)
    reference_sigma_histo = torch.zeros(n_histo_bins)

    for i in range(n_iter):

        batch_sigma = batch_sampler.sample(training_batch_size, batch_stratified_sampling)
        reference_sigma = reference_sampler.sample(reference_batch_size, reference_stratified_sampling)

        avg_batch_mean += batch_sigma.mean().item()
        avg_batch_min  += batch_sigma.amin().item()
        avg_batch_max  += batch_sigma.amax().item()
        avg_reference_mean += reference_sigma.mean().item()
        avg_reference_min  += reference_sigma.amin().item()
        avg_reference_max  += reference_sigma.amax().item()

        batch_sigma_histo += batch_sigma.histc(bins=n_histo_bins, min=sigma_min, max=sigma_max) / (training_batch_size * n_iter)
        reference_sigma_histo += reference_sigma.histc(bins=n_histo_bins, min=sigma_min, max=sigma_max) / (reference_batch_size * n_iter)

    print(f"avg batch     mean: {avg_batch_mean / n_iter:{5}f}, min: {avg_batch_min / n_iter:{5}f}, max: {avg_batch_max / n_iter:{5}f}")
    print(f"avg reference mean: {avg_reference_mean / n_iter:{5}f}, min: {avg_reference_min / n_iter:{5}f}, max: {avg_reference_max / n_iter:{5}f}")

    multi_plot((batch_sigma_histo, "batch sigma"),
            added_plots={0: (reference_sigma_histo, "reference_sigma")},
            x_log_scale=True, y_log_scale=use_y_log_scale, x_axis_range=(sigma_min, sigma_max))