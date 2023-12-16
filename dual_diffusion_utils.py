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

import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchaudio
import torchaudio.functional as AF
import cv2

def get_activation(act_fn):
    if act_fn in ["swish", "silu"]:
        return nn.SiLU()
    elif act_fn == "mish":
        return nn.Mish()
    elif act_fn == "gelu":
        return nn.GELU()
    elif act_fn == "relu":
        return nn.ReLU()
    elif act_fn == "sin":
        return torch.sin
    elif act_fn == "sinc":
        return torch.sinc
    else:
        raise ValueError(f"Unsupported activation function: {act_fn}")
    
def compute_snr(noise_scheduler, timesteps):
    """
    Computes SNR as per
    https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr

# kaiser derived window for mclt/mdct, unused
def get_kaiser_window(window_len, beta=4*torch.pi, device="cpu"):    
    alpha = (window_len - 1) / 2
    n = torch.arange(window_len, device=device)
    return torch.special.i0(beta * torch.sqrt(1 - ((n - alpha) / alpha).square())) / torch.special.i0(torch.tensor(beta))

def get_kaiser_derived_window(window_len, beta=4*torch.pi, device="cpu"):

    kaiserw = get_kaiser_window(window_len // 2 + 1, beta, device)
    csum = torch.cumsum(kaiserw, dim=0)
    halfw = torch.sqrt(csum[:-1] / csum[-1])

    w = torch.zeros(window_len, device=device)
    w[:window_len//2] = halfw
    w[-window_len//2:] = halfw.flip(0)

    return w

# fast overlapped modified discrete cosine transform type iv - becomes mclt with complex output
def mdct(x, block_width, window_degree=1):

    padding_left = padding_right = block_width // 2
    remainder = x.shape[-1] % (block_width // 2)
    if remainder > 0:
        padding_right += block_width // 2 - remainder

    pad_tuple = (padding_left, padding_right) + (0,0,) * (x.ndim-1)
    x = F.pad(x, pad_tuple).unfold(-1, block_width, block_width//2)

    N = x.shape[-1] // 2
    n = torch.arange(2*N, device=x.device)
    k = torch.arange(0.5, N + 0.5, device=x.device)

    if window_degree == 0:
        window = 1
    else:
        window = torch.sin(torch.pi * (n + 0.5) / (2*N))
        if window_degree == 2: window = window.square()

    pre_shift = torch.exp(-1j * torch.pi / 2 / N * n)
    post_shift = torch.exp(-1j * torch.pi / 2 / N * (N + 1) * k)
    
    return torch.fft.fft(x * pre_shift * window, norm="forward")[..., :N] * post_shift * 2 * N ** 0.5

def imdct(x, window_degree=1):
    N = x.shape[-1]
    n = torch.arange(2*N, device=x.device)
    k = torch.arange(0.5, N + 0.5, device=x.device)

    if window_degree == 0:
        window = 1
    else:
        window = torch.sin(torch.pi * (n + 0.5) / (2*N))
        if window_degree == 2: window = window.square()

    pre_shift = torch.exp(-1j * torch.pi / 2 / N * n)
    post_shift = torch.exp(-1j * torch.pi / 2 / N * (N + 1) * k)
    
    x = torch.cat((x / post_shift, torch.zeros_like(x)), dim=-1)
    y = (torch.fft.ifft(x, norm="backward") / pre_shift) * window

    padded_sample_len = (y.shape[-2] + 1) * y.shape[-1] // 2
    raw_sample = torch.zeros(y.shape[:-2] + (padded_sample_len,), device=y.device, dtype=y.dtype)
    y_even = y[...,  ::2, :].reshape(*y[...,  ::2, :].shape[:-2], -1)
    y_odd  = y[..., 1::2, :].reshape(*y[..., 1::2, :].shape[:-2], -1)
    raw_sample[..., :y_even.shape[-1]] = y_even
    raw_sample[..., N:y_odd.shape[-1] + N] += y_odd

    return raw_sample[..., N:-N] * 2 * N ** 0.5

def to_ulaw(x, u=255):

    complex = False
    if torch.is_complex(x):
        complex = True
        x = torch.view_as_real(x)

    x = x / x.abs().amax(dim=tuple(range(x.ndim-2-int(complex), x.ndim)), keepdim=True)
    x = torch.sign(x) * torch.log1p(u * x.abs()) / np.log(1 + u)

    if complex:
        x = torch.view_as_complex(x)
    
    return x

def from_ulaw(x, u=255):

    complex = False
    if torch.is_complex(x):
        complex = True
        x = torch.view_as_real(x)

    x = x / x.abs().amax(dim=tuple(range(x.ndim-2-int(complex), x.ndim)), keepdim=True)
    x = torch.sign(x) * ((1 + u) ** x.abs() - 1) / u

    if complex:
        x = torch.view_as_complex(x)

    return x

def normalize_lufs(raw_samples, sample_rate, target_lufs=-20.0):

    original_shape = raw_samples.shape

    if raw_samples.ndim == 1:
        raw_samples = raw_samples.view(1, 1, -1)
    elif raw_samples.ndim == 2:
        raw_samples = raw_samples.view(raw_samples.shape[0], 1, -1)

    current_lufs = AF.loudness(raw_samples, sample_rate)
    gain = 10. ** ((target_lufs - current_lufs) / 20.0)

    gain = gain.view((*gain.shape,) + (1,) * (raw_samples.ndim - gain.ndim))
    return (raw_samples * gain).view(original_shape)

def save_flac(raw_samples, sample_rate, output_path, target_lufs=-20.0):
    
    raw_samples = raw_samples.detach().real
    if raw_samples.ndim == 1:
        raw_samples = raw_samples.view(1, -1)
    elif raw_samples.ndim == 2:
        raw_samples = raw_samples.view(raw_samples.shape[0], -1)
    elif raw_samples.ndim == 3:
        raw_samples = raw_samples.permute(1, 2, 0).view(raw_samples.shape[1], -1)

    if target_lufs is not None:
        raw_samples = normalize_lufs(raw_samples, sample_rate, target_lufs)

    directory = os.path.dirname(output_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    torchaudio.save(output_path, raw_samples.cpu(), sample_rate, bits_per_sample=16)

def save_raw(tensor, output_path):  
    directory = os.path.dirname(output_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    if tensor.dtype == torch.float16:
        tensor = tensor.float()
    elif tensor.dtype == torch.complex32:
        tensor = tensor.complex64()
    tensor.detach().resolve_conj().cpu().numpy().tofile(output_path)

def dtype_size_in_bytes(dtype):
    if dtype == torch.float16 or dtype == np.float16:
        return 2
    elif dtype == torch.float32 or dtype == np.float32:
        return 4
    elif dtype == torch.float64 or dtype == np.float64:
        return 8
    elif dtype == torch.int8 or dtype == np.int8:
        return 1
    elif dtype == torch.int16 or dtype == np.int16:
        return 2
    elif dtype == torch.int32 or dtype == np.int32:
        return 4
    elif dtype == torch.int64 or dtype == np.int64:
        return 8
    elif dtype == torch.complex32:
        return 4
    elif dtype == torch.complex64 or dtype == np.complex64:
        return 8
    elif dtype == torch.complex128 or dtype == np.complex128:
        return 16
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

def torch_dtype_to_numpy(dtype):
    if dtype == torch.float16 or dtype == np.float16:
        return np.float16
    elif dtype == torch.float32 or dtype == np.float32:
        return np.float32
    elif dtype == torch.float64 or dtype == np.float64:
        return np.float64
    elif dtype == torch.int8 or dtype == np.int8:
        return np.int8
    elif dtype == torch.int16 or dtype == np.int16:
        return np.int16
    elif dtype == torch.int32 or dtype == np.int32:
        return np.int32
    elif dtype == torch.int64 or dtype == np.int64:
        return np.int64
    elif dtype == torch.complex32:
        raise ValueError("Numpy does not support equivalent dtype: torch.complex32")
    elif dtype == torch.complex64 or dtype == np.complex64:
        return np.complex64
    elif dtype == torch.complex128 or dtype == np.complex128:
        return np.complex128
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    
def load_raw(input_path, dtype=torch.int16, start=0, count=-1):
    dtype = torch_dtype_to_numpy(dtype)
    offset = start * dtype_size_in_bytes(dtype)
    tensor = torch.from_numpy(np.fromfile(input_path, dtype=dtype, count=count, offset=offset))
    
    if dtype == np.int8:
        return tensor / 128.
    elif dtype == np.int16:
        return tensor / 32768.
    elif dtype == np.int32:
        return tensor / 2147483648.
    elif dtype == np.uint8:
        return tensor / 255.
    elif dtype == np.uint16:
        return tensor / 65535.
    elif dtype == np.uint32:
        return tensor / 4294967295.

    return tensor
    
@torch.no_grad()
def save_sample_img(sample, img_path, include_phase=False):
    
    num_chunks = sample.shape[2]
    chunk_len = sample.shape[3]
    bsz = sample.shape[0]

    sample = torch.view_as_complex(sample.clone().detach().permute(0, 2, 3, 1).contiguous().float())
    sample *= torch.arange(num_chunks//8, num_chunks+num_chunks//8).to(sample.device).view(1, num_chunks, 1)
    sample = sample.view(bsz * num_chunks, chunk_len)
    amp = sample.abs(); amp = (amp / torch.max(amp)).cpu().numpy()
    
    if include_phase:
        phase = sample.angle().cpu().numpy()
        cv2_img = np.zeros((bsz * num_chunks, chunk_len, 3), dtype=np.uint8)
        cv2_img[:, :, 0] = (np.sin(phase) + 1) * 255/2 * amp
        cv2_img[:, :, 1] = (np.sin(phase + 2*np.pi/3) + 1) * 255/2 * amp
        cv2_img[:, :, 2] = (np.sin(phase + 4*np.pi/3) + 1) * 255/2 * amp
    else:
        cv2_img = (amp * 255).astype(np.uint8)
        cv2_img = cv2.applyColorMap(cv2_img, cv2.COLORMAP_JET)

    cv2.imwrite(img_path, cv2.flip(cv2_img, -1))

class ScaleNorm(nn.Module):
    def __init__(self, num_features, init_scale=1):
        super(ScaleNorm, self).__init__()

        self.num_features = num_features
        self.scale = nn.Parameter(torch.ones(num_features) * np.log(np.exp(init_scale) - 1))

    def forward(self, x):

        view_shape = (1, -1,) + (1,) * (x.ndim - 2)
        return x * F.softplus(self.scale).view(view_shape)

def hz_to_mels(hz):
    return 1127. * torch.log(1 + hz / 700.)

def mels_to_hz(mels):
    return 700. * (torch.exp(mels / 1127.) - 1)

def get_mel_density(hz):
    return 1127. / (700. + hz)

def get_hann_window(window_len, device="cpu"):
    n = torch.arange(window_len, device=device) / (window_len - 1)
    return 0.5 - 0.5 * torch.cos(2 * torch.pi * n)
"""
a = load_raw("./dataset/samples/80.raw")
a = mdct(a, 128, window_degree=1)
a_abs = a.abs()
a_abs = a_abs / a_abs.amax(dim=-1, keepdim=True)
a_kld = -a_abs * (a_abs+1e-8).log()
a_kld /= a_kld.amax(dim=-1, keepdim=True)
a_kld = a_kld ** 0.5
signal_mask = (a_kld < a_kld.mean(dim=-1, keepdim=True)).type(torch.float32)
signal_mask /= signal_mask.amax(dim=-1, keepdim=True)
noise_mask = 1 - signal_mask

noise_mask =1
signal_mask = 1
a_imdct = imdct(a, window_degree=1).real
save_raw(a_imdct, "./debug/test_a.raw")

noise_masked_a_imdct = imdct(a * noise_mask, window_degree=1).real
save_raw(noise_masked_a_imdct, "./debug/test_noise_masked_a.raw")

signal_masked_a_imdct = imdct(a * signal_mask, window_degree=1).real
save_raw(signal_masked_a_imdct, "./debug/test_signal_masked_a.raw")

a = a / a.abs().square().sum().sqrt()
b = torch.randn_like(a) * a.abs()
b = b / b.abs().square().sum().sqrt()

reconstructed_imdct = imdct(a*noise_mask, window_degree=1).real + imdct(b*signal_mask, window_degree=2).real
save_raw(reconstructed_imdct, "./debug/test_reconstructed.raw")
"""

"""
a = torch.randn(65536*2)
b = torch.randn(65536*2)
#b = load_raw("./dataset/samples/80.raw", count=65536*2)

a = mdct(a, 128, window_degree=2)
a_abs = a.abs()
a_abs = a_abs / a_abs.amax(dim=-1, keepdim=True)
a_ln = (a_abs * 16000).log1p()

b = mdct(b, 128, window_degree=2)
b_abs = b.abs()
b_abs[:, :] = torch.randn_like(b_abs).abs()** 0.25
b_abs = b_abs / b_abs.amax(dim=-1, keepdim=True)
b_ln = (b_abs * 16000).log1p()

print(b.shape, a.shape)
rmse_error = F.mse_loss(a_abs, b_abs).sqrt()
ln_error = F.l1_loss(a_ln, b_ln)

abs_delta = a_abs - b_abs
ln_delta = a_ln - b_ln

print(f"rmse: {rmse_error}, ln: {ln_error}, total: {rmse_error + ln_error}")
print(f"abs delta: {abs_delta.mean()}, ln delta: {ln_delta.mean()}")
"""

"""
num_filters = 8192
min_freq = 0.006
max_freq = 0.85
crop_width = 65536
std_octaves = 4
device = "cuda"

filter_q = torch.rand(num_filters, device=device) * (max_freq - min_freq) + min_freq
#filter_q = torch.linspace(min_freq, max_freq, num_filters)

filter_std = 2 * torch.exp2(torch.rand(num_filters, device=device) * std_octaves)
filter_std = filter_std * filter_q / min_freq

fft_q = torch.arange(0, crop_width // 2 + 1, device=device) / (crop_width // 2)

filters = torch.zeros((num_filters, crop_width), device=device)
filters[:, :crop_width//2 + 1] = torch.exp(-filter_std.view(-1, 1) * torch.log(filter_q.view(-1, 1) / fft_q.view(1, -1)).square())
filters[:, 0] = 0


ifft_filters = torch.fft.fftshift(torch.fft.ifft(filters, norm="ortho"), dim=-1)
ifft_filters /= ifft_filters.real.amax(dim=-1, keepdim=True)
ifft_filters_abs = ifft_filters.abs()
ifft_max_filter_abs = ifft_filters_abs.amax(dim=-1, keepdim=True)
ifft_filters_abs /= ifft_max_filter_abs

save_raw(ifft_filters_abs, "./debug/ifft_filters_abs.raw")
save_raw(ifft_filters, "./debug/ifft_filters.raw")
save_raw(filters[:, :crop_width//2 + 1], "./debug/filters.raw")

filters /= filter_q.sqrt().view(-1, 1)
coverage = filters[:, :crop_width//2 + 1].sum(dim=0)
coverage /= coverage.amax()
save_raw(coverage, "./debug/filter_coverage.raw")
"""

"""
mss_params = {
    "edge_crop_width": 0,
    "max_freq": 1,
    "min_freq": 0.006,
    "min_std": 2,
    "num_filters": 1024,
    "sample_block_width": 128,
    "std_octaves": 20,
    "u": 16000,
    "version": 3
},
"""

# get list of all samples in ./dataset/samples

"""
sample_files = []
for root, dirs, files in os.walk("./dataset/samples"):
    for file in files:
        if file.endswith(".raw"):
            sample_files.append(os.path.join(root, file))

amax = 0

sample_files = sample_files[::-1]

for file in sample_files:

    a = load_raw(file)#.to("cuda")
    a = normalize_lufs(a, 8000)

    m = mdct(a, 128, window_degree=1) / 15
    #im = imdct(m, window_degree=1).real

    #save_raw(im, "./debug/test_im.raw")
    #save_raw(a, "./debug/test_a.raw")
    amax = max(amax, m.abs().amax().item())
    print(amax)

"""