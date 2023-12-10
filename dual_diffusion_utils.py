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
import torchaudio.transforms as T
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
def _kaiser(window_len, beta, device):    
    alpha = (window_len - 1) / 2
    n = torch.arange(window_len, device=device)
    return torch.special.i0(beta * torch.sqrt(1 - ((n - alpha) / alpha).square())) / torch.special.i0(torch.tensor(beta))

def kaiser_derived_window(window_len, beta=4*torch.pi, device="cpu"):

    kaiserw = _kaiser(window_len // 2 + 1, beta, device)
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
    
    return torch.fft.fft(x * pre_shift * window)[..., :N] * post_shift * (2 ** 0.5)

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
    y = (torch.fft.ifft(x) / pre_shift) * window

    padded_sample_len = (y.shape[-2] + 1) * y.shape[-1] // 2
    raw_sample = torch.zeros(y.shape[:-2] + (padded_sample_len,), device=y.device, dtype=y.dtype)
    y_even = y[...,  ::2, :].reshape(*y[...,  ::2, :].shape[:-2], -1)
    y_odd  = y[..., 1::2, :].reshape(*y[..., 1::2, :].shape[:-2], -1)
    raw_sample[..., :y_even.shape[-1]] = y_even
    raw_sample[..., N:y_odd.shape[-1] + N] += y_odd

    return raw_sample[..., N:-N] * (2 ** 0.5)

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

def normalize_lufs(raw_samples, sample_rate, target_lufs=-16.0):

    original_shape = raw_samples.shape

    if raw_samples.ndim == 1:
        raw_samples = raw_samples.view(1, 1, -1)
    elif raw_samples.ndim == 2:
        raw_samples = raw_samples.view(raw_samples.shape[0], 1, -1)

    loudness_transform = T.Loudness(sample_rate)
    current_lufs = loudness_transform(raw_samples)    
    gain = 10. ** ((target_lufs - current_lufs) / 20.0)

    gain = gain.view((*gain.shape,) + (1,) * (raw_samples.ndim - gain.ndim))
    return (raw_samples * gain).view(original_shape)

def save_flac(raw_samples, sample_rate, output_path, target_lufs=-16.0):
    
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
    tensor.detach().cpu().numpy().tofile(output_path)

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