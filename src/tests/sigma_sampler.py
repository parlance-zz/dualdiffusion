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

from typing import Optional
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from training.sigma_sampler import SigmaSampler, SigmaSamplerConfig
from pipelines.dual_diffusion_pipeline import DualDiffusionPipeline
from utils.dual_diffusion_utils import init_cuda


def multi_plot(*args, layout: Optional[tuple[int, int]] = None,
               figsize: Optional[tuple[int, int]] = None,
               added_plots: Optional[dict] = None,
               x_log_scale: bool = False,
               y_log_scale: bool = False,
               x_axis_range: Optional[tuple] = None,
               y_axis_range: Optional[tuple] = None,
               save_path: Optional[str] = None) -> None:
    
    layout = layout or (len(args), 1)
    axes = np.atleast_2d(plt.subplots(layout[0],
                                      layout[1],
                                      figsize=figsize)[1])

    for i, axis in enumerate(axes.flatten()):

        if i < len(args):

            y_values = args[i][0].detach().float().resolve_conj().cpu().numpy()
            if x_axis_range is not None:
                x_values = np.linspace(x_axis_range[0],
                                       x_axis_range[1],
                                       y_values.shape[-1])
            else:
                x_values = np.arange(y_values.shape[0])
            axis.plot(x_values, y_values, label=args[i][1])
            if y_axis_range is not None:
                axis.set_ylim(ymin=y_axis_range[0], ymax=y_axis_range[1])

            if added_plots is not None:
                added_plot = added_plots.get(i, None)
                if added_plot is not None:
                    y_values = added_plot[0].detach().float().resolve_conj().cpu().numpy()
                    if x_axis_range is not None:
                        x_values = np.linspace(x_axis_range[0],
                                               x_axis_range[1],
                                               y_values.shape[-1])
                    else:
                        x_values = np.arange(y_values.shape[0])
                    axis.plot(x_values, y_values, label=added_plot[1])
            
            axis.legend()
            
            if x_log_scale: axis.set_xscale("log")
            if y_log_scale: axis.set_yscale("log")
        else:
            axis.axis("off")

    figsize = plt.gcf().get_size_inches()
    plt.subplots_adjust(left=0.6/figsize[0],
                        bottom=0.25/figsize[1],
                        right=1-0.1/figsize[0],
                        top=1-0.1/figsize[1],
                        wspace=1.8/figsize[0],
                        hspace=1/figsize[1])
    
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)

    plt.show()

@torch.inference_mode()
def sigma_sampler_test():

    test_params = config.load_json(os.path.join(config.CONFIG_PATH, "tests", "sigma_sampler.json"))
    output_path = os.path.join(config.DEBUG_PATH, "sigma_sampler")

    reference_model_name = test_params["reference_model_name"]
    model_load_options = test_params["model_load_options"]

    sigma_max = test_params["sigma_max"]
    sigma_min = test_params["sigma_min"]
    sigma_data = test_params["sigma_data"]

    training_batch_size = test_params["training_batch_size"]
    batch_distribution = test_params["batch_distribution"]
    batch_dist_scale = test_params["batch_dist_scale"]
    batch_dist_offset = test_params["batch_dist_offset"]
    batch_stratified_sampling = test_params["batch_stratified_sampling"]
    batch_pdf_sanitization = test_params["batch_pdf_sanitization"]
    batch_sigma_pdf_offset = test_params["batch_sigma_pdf_offset"]
    batch_sigma_pdf_min = test_params["batch_sigma_pdf_min"]

    reference_batch_size = test_params["reference_batch_size"]
    reference_distribution = test_params["reference_distribution"]
    reference_dist_scale = test_params["reference_dist_scale"]
    reference_dist_offset = test_params["reference_dist_offset"]
    reference_stratified_sampling = test_params["reference_stratified_sampling"]
    reference_pdf_sanitization = test_params["reference_pdf_sanitization"]
    reference_sigma_pdf_offset = test_params["reference_sigma_pdf_offset"]
    reference_sigma_pdf_min = test_params["reference_sigma_pdf_min"]

    n_iter = test_params["n_iter"]
    n_histo_bins = test_params["n_histo_bins"]
    use_y_log_scale = test_params["use_y_log_scale"]

    if batch_distribution == "ln_pdf" or reference_distribution == "ln_pdf":
        model_path = os.path.join(config.MODELS_PATH, reference_model_name)
        print(f"Loading DualDiffusion model from '{model_path}'...")
        pipeline = DualDiffusionPipeline.from_pretrained(model_path, **model_load_options)
        module = getattr(pipeline, test_params["reference_module"])
        output_path = os.path.join(output_path, f"{reference_model_name}_step_{module.config.last_global_step}")
    else:
        module = None

    batch_sampler_config = SigmaSamplerConfig(
        sigma_max=sigma_max,
        sigma_min=sigma_min,
        sigma_data=sigma_data,
        distribution=batch_distribution,
        dist_scale=batch_dist_scale,
        dist_offset=batch_dist_offset,
        use_stratified_sigma_sampling=batch_stratified_sampling,
        sigma_pdf_sanitization=batch_pdf_sanitization,
        sigma_pdf_offset=batch_sigma_pdf_offset,
        sigma_pdf_min=batch_sigma_pdf_min,
        sigma_pdf_warmup_steps=0
    )
    batch_sampler = SigmaSampler(batch_sampler_config)
    try: batch_sampler.update_pdf_from_logvar(module, 9999999)
    except: pass

    reference_sampler_config = SigmaSamplerConfig(
        sigma_max=sigma_max,
        sigma_min=sigma_min,
        sigma_data=sigma_data,
        distribution=reference_distribution,
        dist_scale=reference_dist_scale,
        dist_offset=reference_dist_offset,
        use_stratified_sigma_sampling=reference_stratified_sampling,
        sigma_pdf_sanitization=reference_pdf_sanitization,
        sigma_pdf_offset=reference_sigma_pdf_offset,
        sigma_pdf_min=reference_sigma_pdf_min,
        sigma_pdf_warmup_steps=0
    )
    reference_sampler = SigmaSampler(reference_sampler_config)
    try: reference_sampler.update_pdf_from_logvar(module, 9999999)
    except: pass

    if batch_sampler.config.distribution == "ln_pdf":
        multi_plot((batch_sampler.dist_pdf, "batch_distribution_pdf"),
                   (batch_sampler.dist_cdf, "batch_distribution_cdf"),
                   y_axis_range=(0, None),
                   save_path=os.path.join(output_path, "batch_pdf_cdf.png"))

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
            y_log_scale=use_y_log_scale, x_axis_range=(np.log(sigma_min), np.log(sigma_max)),
            save_path=os.path.join(output_path, "batch_ref_sigma_histo.png"))
    
if __name__ == "__main__":

    init_cuda()
    sigma_sampler_test()
    