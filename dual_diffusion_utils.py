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
from io import BytesIO
from json import dumps

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchaudio
import torchaudio.functional as AF
import cv2
from dotenv import load_dotenv

def init_cuda():
    if not torch.cuda.is_available():
        print("Error: PyTorch not compiled with CUDA support or CUDA unavailable")
        exit(1)
    else:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.cufft_plan_cache[0].max_size = 250 # stupid cufft memory leak

def dict_str(d, indent=4):
    if d is None: return "None"
    return dumps(d, indent=indent)

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

    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    snr = (alpha / sigma) ** 2
    return snr

def get_fractal_noise2d(shape, device="cpu", generator=None, degree=1):
    
    noise_shape = shape[:-2] + (shape[-2]*2, shape[-1]*2)
    noise_complex = torch.randn(noise_shape, device=device, dtype=torch.complex64)

    linspace_x = (torch.arange(0, noise_complex.shape[-2], device=device) - noise_complex.shape[-2]//2).square() / noise_complex.shape[-2]**2
    linspace_y = (torch.arange(0, noise_complex.shape[-1], device=device) - noise_complex.shape[-1]//2).square() / noise_complex.shape[-1]**2
    linspace = (linspace_x.view(-1, 1) + linspace_y.view(1, -1)).pow(-degree/2)
    linspace[noise_complex.shape[-2]//2, noise_complex.shape[-1]//2] = 0
    linspace = linspace.view((1,)*(len(shape)-2) + (noise_complex.shape[-2], noise_complex.shape[-1]))
    linspace = torch.roll(linspace, shifts=(linspace.shape[-2]//2, linspace.shape[-1]//2), dims=(-2, -1))

    pink_noise = torch.fft.ifft2(noise_complex * linspace, norm="ortho").real[..., :shape[-2], :shape[-1]]
    return (pink_noise / pink_noise.std(dim=(-2, -1), keepdim=True)).detach().requires_grad_(False)

def get_hann_window(window_len, device="cpu"):
    n = torch.arange(window_len, device=device) / window_len
    return (0.5 - 0.5 * torch.cos(2 * torch.pi * n)).requires_grad_(False)

def get_kaiser_window(window_len, beta=4*torch.pi, device="cpu"):    
    alpha = (window_len - 1) / 2
    n = torch.arange(window_len, device=device)
    return (torch.special.i0(beta * torch.sqrt(1 - ((n - alpha) / alpha).square())) / torch.special.i0(torch.tensor(beta))).requires_grad_(False)

def get_kaiser_derived_window(window_len, beta=4*torch.pi, device="cpu"):

    kaiserw = get_kaiser_window(window_len // 2 + 1, beta, device)
    csum = torch.cumsum(kaiserw, dim=0)
    halfw = torch.sqrt(csum[:-1] / csum[-1])

    w = torch.zeros(window_len, device=device)
    w[:window_len//2] = halfw
    w[-window_len//2:] = halfw.flip(0)

    return w.requires_grad_(False)

def get_hann_poisson_window(window_len, alpha=2, device="cpu"):
    x = torch.arange(window_len, device=device) / window_len
    return (torch.exp(-alpha * (1 - 2*x).abs()) * 0.5 * (1 - torch.cos(2*torch.pi*x))).requires_grad_(False)

def get_blackman_harris_window(window_len, device="cpu"):
    x = torch.arange(window_len, device=device) / window_len * 2*torch.pi
    return (0.35875 - 0.48829 * torch.cos(x) + 0.14128 * torch.cos(2*x) - 0.01168 * torch.cos(3*x)).requires_grad_(False)

def get_flat_top_window(window_len, device="cpu"):
    x = torch.arange(window_len, device=device) / window_len * 2*torch.pi
    return (0.21557895 - 0.41663158 * torch.cos(x) + 0.277263158 * torch.cos(2*x) - 0.083578947 * torch.cos(3*x) + 0.006947368 * torch.cos(4*x)).requires_grad_(False)

def mdct(x, block_width, window_fn="hann", window_degree=1):

    padding_left = padding_right = block_width // 2
    remainder = x.shape[-1] % (block_width // 2)
    if remainder > 0:
        padding_right += block_width // 2 - remainder

    pad_tuple = (padding_left, padding_right) + (0,0,) * (x.ndim-1)
    x = F.pad(x, pad_tuple).unfold(-1, block_width, block_width//2)

    N = x.shape[-1] // 2
    n = torch.arange(2*N, device=x.device)
    k = torch.arange(0.5, N + 0.5, device=x.device)

    if window_fn == "hann":
        if window_degree == 0:
            window = 1
        else:
            window = torch.sin(torch.pi * (n + 0.5) / (2*N)).requires_grad_(False)
            if window_degree == 2: window = window.square()
    elif window_fn == "hann_poisson":
        window = get_hann_poisson_window(2*N, device=x.device)
    elif window_fn == "blackman_harris":
        window = get_blackman_harris_window(2*N, device=x.device)
    elif window_fn == "flat_top":
        window = get_flat_top_window(2*N, device=x.device)
    else:
        raise ValueError(f"Unsupported window function: {window_fn}")
    
    pre_shift = torch.exp(-1j * torch.pi / 2 / N * n).requires_grad_(False)
    post_shift = torch.exp(-1j * torch.pi / 2 / N * (N + 1) * k).requires_grad_(False)
    
    return torch.fft.fft(x * pre_shift * window, norm="forward")[..., :N] * post_shift * (2 * N ** 0.5)

def imdct(x, window_fn="hann", window_degree=1):
    N = x.shape[-1]
    n = torch.arange(2*N, device=x.device)
    k = torch.arange(0.5, N + 0.5, device=x.device)

    if window_fn == "hann":
        if window_degree == 0:
            window = 1
        else:
            window = torch.sin(torch.pi * (n + 0.5) / (2*N)).requires_grad_(False)
            if window_degree == 2: window = window.square()
    elif window_fn == "hann_poisson":
        window = get_hann_poisson_window(2*N, device=x.device)
    elif window_fn == "blackman_harris":
        window = get_blackman_harris_window(2*N, device=x.device)
    elif window_fn == "flat_top":
        window = get_flat_top_window(2*N, device=x.device)
    else:
        raise ValueError(f"Unsupported window function: {window_fn}")

    pre_shift = torch.exp(-1j * torch.pi / 2 / N * n).requires_grad_(False)
    post_shift = torch.exp(-1j * torch.pi / 2 / N * (N + 1) * k).requires_grad_(False)

    y = (torch.fft.ifft(x / post_shift, norm="backward", n=2*N) / pre_shift) * window

    padded_sample_len = (y.shape[-2] + 1) * y.shape[-1] // 2
    raw_sample = torch.zeros(y.shape[:-2] + (padded_sample_len,), device=y.device, dtype=y.dtype)
    y_even = y[...,  ::2, :].reshape(*y[...,  ::2, :].shape[:-2], -1)
    y_odd  = y[..., 1::2, :].reshape(*y[..., 1::2, :].shape[:-2], -1)
    raw_sample[..., :y_even.shape[-1]] = y_even
    raw_sample[..., N:y_odd.shape[-1] + N] += y_odd

    return raw_sample[..., N:-N] * (2 * N ** 0.5)

def stft(x, block_width, window_fn="hann", step=None, add_channelwise_fft=False):

    if step is None:
        step = block_width // 2

    x = x.unfold(-1, block_width, step)

    if window_fn == "hann":
        window = get_hann_window(block_width, device=x.device) * 2
    elif window_fn == "hann^0.5":
        window = get_hann_window(block_width, device=x.device).sqrt() * (2/torch.pi)
    elif window_fn == "hann^2":
        window = get_hann_window(block_width, device=x.device).square() * (8/3)
    elif window_fn == "blackman_harris":
        window = get_blackman_harris_window(block_width, device=x.device)
    elif window_fn == "blackman_harris2":
        window = get_blackman_harris_window(block_width, device=x.device) ** 2
    elif window_fn == "none":
        window = 1
    else:
        raise ValueError(f"Unsupported window function: {window_fn}")
    
    x = torch.fft.rfft(x * window, norm="ortho")

    if add_channelwise_fft:
        return torch.fft.fft(x, norm="ortho", dim=-3)
    else:
        return x

"""
def stft(x, block_width, step=None, window_fn="hann", add_channelwise_fft=False):

    if step is None:
        step = block_width // 2

    x = x.unfold(-1, block_width, step)

    if window_fn == "hann":
        window = get_hann_window(block_width, device=x.device) * 2
    elif window_fn == "hann^0.5":
        window = get_hann_window(block_width, device=x.device).sqrt() * (2/torch.pi)
    elif window_fn == "hann^2":
        window = get_hann_window(block_width, device=x.device).square() * (8/3)
    elif window_fn == "blackman_harris":
        window = get_blackman_harris_window(block_width, device=x.device)
    elif window_fn == "none":
        window = 1
    else:
        raise ValueError(f"Unsupported window function: {window_fn}")
    
    N = x.shape[-1] // 2
    pre_shift = (torch.exp(-1j * torch.pi / 2 / N * torch.arange(2*N, device=x.device)) * window)
    x = torch.fft.fft(x * pre_shift.requires_grad_(False), norm="forward")[..., :N] * (2 * N ** 0.5)

    if add_channelwise_fft:
        return torch.fft.fft(x, norm="ortho", dim=-3)
    else:
        return x
"""

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

STR_DTYPE_TO_NUMPY_DTYPE = {
    "float16": np.float16,
    "float32": np.float32,
    "float64": np.float64,
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "int64": np.int64,
    "uint8": np.uint8,
    "uint16": np.uint16,
    "uint32": np.uint32,
    "uint64": np.uint64,
    "complex64": np.complex64,
    "complex128": np.complex128
}

STR_DTYPE_SIZE = {
    "float16": 2,
    "float32": 4,
    "float64": 8,
    "int8": 1,
    "int16": 2,
    "int32": 4,
    "int64": 8,
    "uint8": 1,
    "uint16": 2,
    "uint32": 4,
    "uint64": 8,
    "complex64": 8,
    "complex128": 16
}

STR_DTYPE_MAX_VALUE = {
    "float16": 1.,
    "float32": 1.,
    "float64": 1.,
    "int8": 128.,
    "int16": 32768.,
    "int32": 2147483648.,
    "int64": 9223372036854775808.,
    "uint8": 255.,
    "uint16": 65535.,
    "uint32": 4294967295.,
    "uint64": 18446744073709551615.,
    "complex64": 1.,
    "complex128": 1.
}

def save_raw(tensor, output_path):  
    directory = os.path.dirname(output_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    if tensor.dtype == torch.float16 or tensor.dtype == torch.bfloat16:
        tensor = tensor.float()
    elif tensor.dtype == torch.complex32:
        tensor = tensor.complex64()
    tensor.detach().resolve_conj().cpu().numpy().tofile(output_path)

def load_raw(input_path, dtype="int16", num_channels=1, start=0, count=-1, device="cpu", return_start=False):

    np_dtype = STR_DTYPE_TO_NUMPY_DTYPE.get(dtype, None)
    if np_dtype is None:
        raise ValueError(f"Unsupported dtype: {dtype} - Supported dtypes: {list(STR_DTYPE_TO_NUMPY_DTYPE.keys())}")
    dtype_size = STR_DTYPE_SIZE[dtype]

    if isinstance(input_path, str):
        sample_len = os.path.getsize(input_path) // dtype_size // num_channels
    else:
        if not isinstance(input_path, (bytes, bytearray)):
            raise ValueError(f"Unsupported input_path type: {type(input_path)}")
        sample_len = len(input_path) // dtype_size // num_channels

    if sample_len < count:
        raise ValueError(f"Requested {count} samples, but only {sample_len} available")
    if start < 0:
        if count < 0:
            raise ValueError(f"If start < 0 count cannot be < 0")
        start = np.random.randint(0, sample_len - count + 1)
    elif start > 0 and count > 0:
        if sample_len < start + count:
            raise ValueError(f"Requested {start + count} samples, but only {sample_len} available")

    offset = start * dtype_size * num_channels

    if isinstance(input_path, str):
        tensor = torch.from_numpy(np.fromfile(input_path, dtype=np_dtype, count=count * num_channels, offset=offset))
    else:
        tensor = torch.from_numpy(np.frombuffer(input_path, dtype=np_dtype, count=count * num_channels, offset=offset))
    tensor = (tensor / STR_DTYPE_MAX_VALUE[dtype]).view(-1, num_channels).permute(1, 0).to(device)

    if return_start:
        return tensor, start
    else:
        return tensor

@torch.no_grad()
def normalize_lufs(raw_samples, sample_rate, target_lufs=-16., max_clip=0.15):
    
    original_shape = raw_samples.shape
    raw_samples = torch.nan_to_num(raw_samples, nan=0, posinf=0, neginf=0)
    
    if raw_samples.ndim == 1:
        raw_samples = raw_samples.view(1, 1, -1)
    elif raw_samples.ndim == 2:
        raw_samples = raw_samples.view(1, raw_samples.shape[0], -1)

    raw_samples /= raw_samples.abs().amax(dim=tuple(range(1, len(raw_samples.shape))), keepdim=True).clip(min=1e-16)
    current_lufs = AF.loudness(raw_samples, sample_rate)
    gain = 10. ** ((target_lufs - current_lufs) / 20.0)
    gain = gain.view((*gain.shape,) + (1,) * (raw_samples.ndim - gain.ndim))

    normalized_raw_samples = raw_samples * gain
    normalized_raw_samples /= normalized_raw_samples.abs().amax(dim=tuple(range(1, len(normalized_raw_samples.shape))), keepdim=True).clip(min=1+max_clip)

    return normalized_raw_samples.view(original_shape)

def save_audio(raw_samples, sample_rate, output_path, target_lufs=-16.):
    
    raw_samples = raw_samples.detach().real.float()
    if raw_samples.ndim == 1:
        raw_samples = raw_samples.view(1, -1)

    if target_lufs is not None:
        raw_samples = normalize_lufs(raw_samples, sample_rate, target_lufs)

    directory = os.path.dirname(output_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    torchaudio.save(output_path, raw_samples.cpu(), sample_rate, bits_per_sample=16)

def load_audio(input_path, start=0, count=-1, return_sample_rate=False, device="cpu", return_start=False):

    if isinstance(input_path, bytes):
        input_path = BytesIO(input_path)
    elif not isinstance(input_path, str):
        raise ValueError(f"Unsupported input_path type: {type(input_path)}")
    
    sample_len = torchaudio.info(input_path).num_frames
    if sample_len < count:
        raise ValueError(f"Requested {count} samples, but only {sample_len} available")
    if start < 0:
        if count < 0:
            raise ValueError(f"If start < 0 count cannot be < 0")
        start = np.random.randint(0, sample_len - count + 1)
    elif start > 0 and count > 0:
        if sample_len < start + count:
            raise ValueError(f"Requested {start + count} samples, but only {sample_len} available")

    tensor, sample_rate = torchaudio.load(input_path, frame_offset=start, num_frames=count)

    if count >= 0:
        tensor = tensor[..., :count] # for whatever reason torchaudio will return more samples than requested

    return_vals = (tensor.to(device),)
    if return_sample_rate:
        return_vals += (sample_rate,)
    if return_start:
        return_vals += (start,)
    if len(return_vals) == 1:
        return return_vals[0]
    else:
        return return_vals
    
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

def hz_to_mels(hz):
    return 1127. * torch.log(1 + hz / 700.)

def mels_to_hz(mels):
    return 700. * (torch.exp(mels / 1127.) - 1)

def get_mel_density(hz):
    return 1127. / (700. + hz)

def stft2(x, block_width, overlap=2, window_fn="hann"):

    step = block_width // overlap
    padding = block_width * (overlap - 1) // overlap // 2
    x = F.pad(x, (padding, padding))
    x = x.unfold(-1, block_width, step)

    N = x.shape[-1] // 2
    n = torch.arange(2*N, device=x.device)
    k = torch.arange(0.5, N + 0.5, device=x.device)
    pre_shift = torch.exp(-1j * torch.pi / 2 / N * n)
    post_shift = torch.exp(-1j * torch.pi / 2 / N * (N + 1) * k)

    if window_fn == "hann":
        window = get_hann_window(block_width, device=x.device) * 2
    elif window_fn == "hann^0.5":
        window = get_hann_window(block_width, device=x.device).sqrt() * (2/torch.pi)
    elif window_fn == "hann^2":
        window = get_hann_window(block_width, device=x.device).square() * (8/3)
    elif window_fn == "blackman_harris":
        window = get_blackman_harris_window(block_width, device=x.device)
    elif window_fn == "none":
        window = 1
    else:
        raise ValueError(f"Unsupported window function: {window_fn}")

    return torch.fft.fft(x * pre_shift * window, norm="forward")[..., :N] * post_shift * 2 * N ** 0.5

def istft2(x, block_width, overlap=2, window_fn="none"):

    step = block_width // overlap

    signal_len = (x.shape[-2] - 1) * step + block_width
    y = torch.zeros(x.shape[:-2] + (signal_len,), device=x.device)

    N = x.shape[-1]
    n = torch.arange(2*N, device=x.device)
    k = torch.arange(0.5, N + 0.5, device=x.device)
    pre_shift = torch.exp(-1j * torch.pi / 2 / N * n)
    post_shift = torch.exp(-1j * torch.pi / 2 / N * (N + 1) * k)

    if window_fn == "hann":
        window = get_hann_window(block_width, device=x.device) * 2
    elif window_fn == "hann^0.5":
        window = get_hann_window(block_width, device=x.device).sqrt() * (2/torch.pi)
    elif window_fn == "hann^2":
        window = get_hann_window(block_width, device=x.device).square() * (8/3)
    elif window_fn == "blackman_harris":
        window = get_blackman_harris_window(block_width, device=x.device)
    elif window_fn == "none":
        window = 1
    else:
        raise ValueError(f"Unsupported window function: {window_fn}")
    
    x = (torch.fft.ifft(x / post_shift, n=block_width, norm="backward") / pre_shift).real * window * 2 * N ** 0.5

    for i in range(overlap):
        t = x[..., i::overlap, :].reshape(*x.shape[:-2], -1)
        y[..., i*step:i*step + t.shape[-1]] += t

    if overlap > 1:
        padding = block_width * (overlap - 1) // overlap // 2
        y = y[..., padding:-padding] / overlap

    return y

def quantize_tensor(x, levels):
    min_val = x.amin()
    max_val = x.amax()
    range_val = max_val - min_val
    step = range_val / (levels - 1)
    quantized = ((x - min_val) / step).round().clamp(0, levels - 1) * step + min_val
    return quantized

def save_raw_img(x, img_path, allow_inversion=False, allow_colormap=True):
    
    x = x.clone().detach().float().resolve_conj().cpu()
    x -= x.amin(dim=(x.ndim-1, x.ndim-2), keepdim=True)
    x /= x.amax(dim=(x.ndim-1, x.ndim-2), keepdim=True).clip(min=1e-16)

    if allow_inversion:
        invert_channel_mask = ((x.mean(dim=(x.ndim-1, x.ndim-2), keepdim=True) > 0.5) * torch.ones_like(x)) > 0.5
        x[invert_channel_mask] = 1 - x[invert_channel_mask]

    if (x.ndim >= 3) and (x.ndim <=4):
        if x.ndim == 4: x = x.squeeze(0)
        x = x.permute(1, 2, 0).contiguous().numpy()
        cv2_img = (x * 255).astype(np.uint8)
        if cv2_img.shape[2] == 2:
            cv2_img = np.concatenate((cv2_img, np.zeros((cv2_img.shape[0], cv2_img.shape[1], 1))), axis=2)
            cv2_img[..., 2] = cv2_img[..., 1]
            cv2_img[..., 1] = 0
    elif x.ndim == 2:
        x = x.permute(1, 0).contiguous().numpy()
        cv2_img = (x * 255).astype(np.uint8)
        if allow_colormap:
            cv2_img = cv2.applyColorMap(cv2_img, cv2.COLORMAP_JET)
    else:
        raise ValueError(f"Unsupported number of dimensions in save_raw_img: {x.ndim}")

    cv2.imwrite(img_path, cv2.flip(cv2_img, 0))

class ScaleNorm(nn.Module):
    def __init__(self, num_features, init_scale=1):
        super(ScaleNorm, self).__init__()

        self.num_features = num_features
        self.scale = nn.Parameter(torch.ones(num_features) * np.log(np.exp(init_scale) - 1))

    def forward(self, x):

        view_shape = (1, -1,) + (1,) * (x.ndim - 2)
        return x * F.softplus(self.scale).view(view_shape)

if __name__ == "__main__": # fractal noise test

    init_cuda()
    load_dotenv(override=True)

    noise_test_iter = 5
    noise_test_dim = (696, 32)
    noise_test_degree = 2/(1+5**0.5) #0.5

    debug_path = os.environ.get("DEBUG_PATH", None)
    if debug_path is not None:
        debug_path = os.path.join(debug_path, "noise_test")

        for i in range(noise_test_iter):
            noise = get_fractal_noise2d(noise_test_dim, degree=noise_test_degree)
            save_raw_img(noise, os.path.join(debug_path, "fractal_noise{i}.png"), allow_colormap=False)

        noise_fft = torch.fft.fft2(noise, norm="ortho").abs().clip(min=4e-2).log()
        save_raw_img(noise_fft, os.path.join(debug_path, "fractal_noise_ln_psd.png"))