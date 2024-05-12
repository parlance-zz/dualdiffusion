import torch

from dual_diffusion_utils import multi_plot

total_batch_size = 180
reference_batch_size = 2048

target_snr = 32
sigma_max = 80
sigma_data = 0.5
sigma_min = 0.002

P_batch_std = 1
P_batch_mean = -0.4

P_reference_std = 1.
P_reference_mean = -0.4

use_stratified_sampling = True#False

n_iter = 10000
n_histo_bins = 5000
use_y_log_scale = True

print("total_batch_size:", total_batch_size, "reference_batch_size:", reference_batch_size)
print("P_batch_std:", P_batch_std, "P_batch_mean:", P_batch_mean)
print("P_reference_std:", P_reference_std, "P_reference_mean:", P_reference_mean)
print("")

avg_batch_mean = avg_batch_min = avg_batch_max = 0
avg_reference_mean = avg_reference_min = avg_reference_max = 0

batch_sigma_histo = torch.zeros(n_histo_bins)
reference_sigma_histo = torch.zeros(n_histo_bins)

for i in range(n_iter):

    if use_stratified_sampling:
        batch_uniform = (torch.arange(total_batch_size) + 0.5) / total_batch_size
        batch_uniform += (torch.rand(1) - 0.5) / total_batch_size
        batch_normal = P_batch_mean + (P_batch_std * (2 ** 0.5)) * (batch_uniform * 2 - 1).erfinv().clip(min=-5, max=5)
    else:
        batch_normal = torch.randn(total_batch_size) * P_batch_std + P_batch_mean
    batch_sigma = batch_normal.exp().clip(min=sigma_min, max=sigma_max)

    reference_normal = torch.randn(reference_batch_size) * P_reference_std + P_reference_mean
    reference_sigma = reference_normal.exp().clip(min=sigma_min, max=sigma_max)

    avg_batch_mean += batch_sigma.mean().item()
    avg_batch_min  += batch_sigma.amin().item()
    avg_batch_max  += batch_sigma.amax().item()
    avg_reference_mean += reference_sigma.mean().item()
    avg_reference_min  += reference_sigma.amin().item()
    avg_reference_max  += reference_sigma.amax().item()

    batch_sigma_histo += batch_sigma.histc(bins=n_histo_bins, min=sigma_min, max=sigma_max) / (total_batch_size * n_iter)
    reference_sigma_histo += reference_sigma.histc(bins=n_histo_bins, min=sigma_min, max=sigma_max) / (reference_batch_size * n_iter)

print(f"avg batch     mean: {avg_batch_mean / n_iter:{5}f}, min: {avg_batch_min / n_iter:{5}f}, max: {avg_batch_max / n_iter:{5}f}")
print(f"avg reference mean: {avg_reference_mean / n_iter:{5}f}, min: {avg_reference_min / n_iter:{5}f}, max: {avg_reference_max / n_iter:{5}f}")

multi_plot((batch_sigma_histo, "batch sigma"),
           added_plots={0: (reference_sigma_histo, "reference_sigma")},
           x_log_scale=True, y_log_scale=use_y_log_scale, x_axis_range=(sigma_min, sigma_max))