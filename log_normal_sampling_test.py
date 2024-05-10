import torch
import numpy as np

from dual_diffusion_utils import get_expected_max_normal, multi_plot

total_batch_size = 132
reference_batch_size = 2048

target_snr = 32
sigma_max = 80
sigma_min = 0.5 / target_snr

P_std = 1.
P_mean = -0.4
mean_correction_amount = 1 # (for non-theta formulation, best match overall with 1)
use_stratified_sampling = True
use_theta_formulation = True#False

n_iter = 1000
use_y_log_scale = False

print("total_batch_size:", total_batch_size, "reference_batch_size:", reference_batch_size)
print("")

reference_expected_max = get_expected_max_normal(reference_batch_size)
batch_expected_max = get_expected_max_normal(total_batch_size)
print("reference_expected_max:", reference_expected_max)
print("batch_expected_max:    ", batch_expected_max)
print("")

std_correction_factor = reference_expected_max / batch_expected_max
P_corrected_std = P_std * std_correction_factor

mean_correction_offset = P_std**2 / 2 - P_corrected_std**2 / 2
P_corrected_mean = (P_mean + mean_correction_offset) * mean_correction_amount + P_mean * (1 - mean_correction_amount)

print("std_correction_factor:", std_correction_factor)
print("mean_correction_offset:", mean_correction_offset)
print("")
print("P_std:", P_std, "P_corrected_std:", P_corrected_std)
print("P_mean:", P_mean, "P_corrected_mean:", P_corrected_mean)
print("")

mean_error = min_error = max_error = 0
avg_batch_mean = avg_batch_min = avg_batch_max = 0
avg_reference_mean = avg_reference_min = avg_reference_max = 0

batch_sigma_histo = torch.zeros(100)
reference_sigma_histo = torch.zeros(100)
theta_min = np.arctan(1/sigma_max); theta_max = np.arctan(1/sigma_min)

for i in range(n_iter):

    if use_stratified_sampling:

        batch_normal = (torch.arange(total_batch_size) + 0.5) / total_batch_size
        batch_normal += (torch.rand(1) - 0.5) / total_batch_size

        if use_theta_formulation:
            batch_normal = batch_normal * (theta_max - theta_min) + theta_min
        else:
            batch_normal = P_corrected_mean + (P_corrected_std * (2 ** 0.5)) * (batch_normal * 2 - 1).erfinv().clip(min=-5, max=5)
    else:
        if use_theta_formulation:
            batch_normal = torch.rand(total_batch_size) * (theta_max - theta_min) + theta_min
        else:
            batch_normal = torch.randn(total_batch_size) * P_corrected_std + P_corrected_mean

    if use_theta_formulation:
        batch_sigma = 1/batch_normal.tan()
    else:
        batch_sigma = batch_normal.exp().clip(min=sigma_min, max=sigma_max)

    reference_normal = torch.randn(reference_batch_size)
    reference_sigma = (reference_normal * P_std + P_mean).exp().clip(min=sigma_min, max=sigma_max)

    mean_error += (batch_sigma.mean() - reference_sigma.mean()).item()
    min_error  += (batch_sigma.amin() - reference_sigma.amin()).item()
    max_error  += (batch_sigma.amax() - reference_sigma.amax()).item()

    avg_batch_mean += batch_sigma.mean().item()
    avg_batch_min  += batch_sigma.amin().item()
    avg_batch_max  += batch_sigma.amax().item()
    avg_reference_mean += reference_sigma.mean().item()
    avg_reference_min  += reference_sigma.amin().item()
    avg_reference_max  += reference_sigma.amax().item()

    batch_sigma_histo += batch_sigma.histc(bins=100, min=sigma_min, max=sigma_max)
    reference_sigma_histo += reference_sigma.histc(bins=100, min=sigma_min, max=sigma_max)

print(f"mean error: {mean_error / n_iter:{5}f}")
print(f"min  error: {min_error / n_iter:{5}f}")
print(f"max  error: {max_error / n_iter:{5}f}")

print("")
print(f"avg batch     mean: {avg_batch_mean / n_iter:{5}f}, min: {avg_batch_min / n_iter:{5}f}, max: {avg_batch_max / n_iter:{5}f}")
print(f"avg reference mean: {avg_reference_mean / n_iter:{5}f}, min: {avg_reference_min / n_iter:{5}f}, max: {avg_reference_max / n_iter:{5}f}")

batch_sigma_histo /= total_batch_size
reference_sigma_histo /= reference_batch_size
multi_plot((batch_sigma_histo, "batch sigma"), (reference_sigma_histo, "reference sigma"), x_log_scale=True, y_log_scale=use_y_log_scale, x_axis_range=(sigma_min, sigma_max))