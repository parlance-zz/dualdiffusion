import utils.config as config

import os

import numpy as np
import torch

from training.sigma_sampler import SigmaSampler, SigmaSamplerConfig
from pipelines.dual_diffusion_pipeline import DualDiffusionPipeline
from utils.dual_diffusion_utils import multi_plot

if __name__ == "__main__":

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