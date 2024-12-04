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

import utils.config as config

import os

import numpy as np
import torch

from training.sigma_sampler import SigmaSampler, SigmaSamplerConfig
from pipelines.dual_diffusion_pipeline import DualDiffusionPipeline
from utils.dual_diffusion_utils import multi_plot, init_cuda

@torch.inference_mode()
def sigma_sampler_test():

    test_params = config.load_json(os.path.join(config.CONFIG_PATH, "tests", "sigma_sampler.json"))

    reference_model_name = test_params["reference_model_name"]
    model_load_options = test_params["model_load_options"]
    reference_model_class_id = test_params["reference_model_class_id"]
    use_unconditional_model_class_id = test_params["use_unconditional_model_class_id"]

    sigma_max = test_params["sigma_max"]
    sigma_min = test_params["sigma_min"]
    sigma_data = test_params["sigma_data"]

    training_batch_size = test_params["training_batch_size"]
    batch_distribution = test_params["batch_distribution"]
    batch_dist_scale = test_params["batch_dist_scale"]
    batch_dist_offset = test_params["batch_dist_offset"]
    batch_stratified_sampling = test_params["batch_stratified_sampling"]

    reference_batch_size = test_params["reference_batch_size"]
    reference_distribution = test_params["reference_distribution"]
    reference_dist_scale = test_params["reference_dist_scale"]
    reference_dist_offset = test_params["reference_dist_offset"]
    reference_stratified_sampling = test_params["reference_stratified_sampling"]

    n_iter = test_params["n_iter"]
    n_histo_bins = test_params["n_histo_bins"]
    use_y_log_scale = test_params["use_y_log_scale"]

    batch_distribution_pdf = None
    reference_distribution_pdf = None

    if batch_distribution == "ln_pdf" or reference_distribution == "ln_pdf":

        model_path = os.path.join(config.MODELS_PATH, reference_model_name)
        print(f"Loading DualDiffusion model from '{model_path}'...")
        pipeline = DualDiffusionPipeline.from_pretrained(model_path, **model_load_options)

        class_labels = pipeline.get_class_labels([reference_model_class_id])
        conditioning_mask = torch.ones((1,), device=pipeline.unet.device) * float(not use_unconditional_model_class_id)
        unet_class_embeddings = pipeline.unet.get_class_embeddings(class_labels, conditioning_mask).repeat(n_histo_bins, 1)
        sigma = torch.linspace(np.log(sigma_min), np.log(sigma_max), n_histo_bins).exp()
        sigma_error = pipeline.unet.get_sigma_loss_logvar(sigma, unet_class_embeddings).float().flatten()
    
        if batch_distribution == "ln_pdf":
            batch_distribution_pdf = (-sigma_error * batch_dist_scale).exp()
        if reference_distribution == "ln_pdf":
            reference_distribution_pdf = (-sigma_error * reference_dist_scale).exp()

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
                   (batch_sampler.dist_cdf, "batch_distribution_cdf"),
                   y_axis_range=(0, None))
        
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

    #batch_center = ((batch_sigma_histo**2 * torch.linspace(np.log(sigma_min), np.log(sigma_max), n_histo_bins)).sum() / (batch_sigma_histo**2).sum()).item()
    batch_center = batch_sigma_histo.argmax() / n_histo_bins * (np.log(sigma_max) - np.log(sigma_min)) + np.log(sigma_min)
    #reference_center = ((reference_sigma_histo**2 * torch.linspace(np.log(sigma_min), np.log(sigma_max), n_histo_bins)).sum() / (reference_sigma_histo**2).sum()).item()
    reference_center = reference_sigma_histo.argmax() / n_histo_bins * (np.log(sigma_max) - np.log(sigma_min)) + np.log(sigma_min)
    print(f"batch center: {batch_center:{5}f}, reference center: {reference_center:{5}f}")

    multi_plot((batch_sigma_histo, "batch sigma"), (batch_example, "batch example"), (reference_example, "reference example"),
            added_plots={0: (reference_sigma_histo, "reference_sigma")},
            y_log_scale=use_y_log_scale, x_axis_range=(np.log(sigma_min), np.log(sigma_max)))
    
if __name__ == "__main__":

    init_cuda()
    sigma_sampler_test()
    