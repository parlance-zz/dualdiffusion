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
    
    pre_shift = torch.exp(-1j * torch.pi / 2 / N * n)
    post_shift = torch.exp(-1j * torch.pi / 2 / N * (N + 1) * k)
    
    return torch.fft.fft(x * pre_shift * window, norm="forward")[..., :N] * post_shift * 2 * N ** 0.5

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

def stft(x, block_width, window_fn="hann", step=None):

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
    
    return torch.fft.rfft(x * window, norm="ortho")

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

    if tensor.dtype == torch.float16:
        tensor = tensor.float()
    elif tensor.dtype == torch.complex32:
        tensor = tensor.complex64()
    tensor.detach().resolve_conj().cpu().numpy().tofile(output_path)

def load_raw(input_path, dtype="int16", num_channels=1, start=0, count=-1):

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
    return (tensor / STR_DTYPE_MAX_VALUE[dtype]).view(-1, num_channels).permute(1, 0)

def normalize_lufs(raw_samples, sample_rate, target_lufs=-25.0):
    return raw_samples / raw_samples.abs().amax()
    original_shape = raw_samples.shape
    raw_samples = torch.nan_to_num(raw_samples, nan=0, posinf=0, neginf=0)
    
    if raw_samples.ndim == 1:
        raw_samples = raw_samples.view(1, 1, -1)
    elif raw_samples.ndim == 2:
        raw_samples = raw_samples.view(1, raw_samples.shape[0], -1)

    current_lufs = AF.loudness(raw_samples, sample_rate)
    gain = (10. ** ((target_lufs - current_lufs) / 20.0)).clamp(min=1e-5, max=1e5)
    gain = gain.view((*gain.shape,) + (1,) * (raw_samples.ndim - gain.ndim))

    normalized_raw_samples = (raw_samples * gain).view(original_shape).clamp(min=-10, max=10)
    return torch.nan_to_num(normalized_raw_samples, nan=0, posinf=0, neginf=0)

def save_audio(raw_samples, sample_rate, output_path, target_lufs=-25.0):
    
    raw_samples = raw_samples.detach().real
    if raw_samples.ndim == 1:
        raw_samples = raw_samples.view(1, -1)

    if target_lufs is not None:
        raw_samples = normalize_lufs(raw_samples, sample_rate, target_lufs)

    directory = os.path.dirname(output_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    torchaudio.save(output_path, raw_samples.cpu(), sample_rate, bits_per_sample=16)

def load_audio(input_path, start=0, count=-1, return_sample_rate=False):

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

    if return_sample_rate:
        return tensor, sample_rate
    else:
        return tensor
    
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
    n = torch.arange(window_len, device=device) / window_len
    return 0.5 - 0.5 * torch.cos(2 * torch.pi * n)

def get_comp_pair(length=65536, n_freqs=1024, freq_similarity=0, amp_similarity=0, phase_similarity=0):

    t = torch.linspace(0, length, length).view(-1, 1)

    f1 = torch.rand(n_freqs).view(1, -1)
    f2 = f1 * freq_similarity + (1-freq_similarity) * torch.rand(n_freqs).view(1, -1)

    a1 = torch.rand(n_freqs).view(1, -1)
    a2 = a1 * amp_similarity + (1-amp_similarity) * torch.rand(n_freqs).view(1, -1)

    p1 = torch.rand(n_freqs).view(1, -1)
    p2 = p1 * phase_similarity + (1-phase_similarity) * torch.rand(n_freqs).view(1, -1)

    y1 = (torch.exp(2j * torch.pi * t * f1 + 2j*torch.pi * p1) * a1).sum(dim=-1)
    y2 = (torch.exp(2j * torch.pi * t * f2 + 2j*torch.pi * p2) * a2).sum(dim=-1)

    return y1.real, y2.real

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

class MSPSD:

    def __init__(self, low_scale=5,
                       high_scale=11,
                       overlap=2,
                       noise_floor=1e-6,
                       cepstrum_crop_factor=1,
                       window_fn="hann",
                       inv_window_fn="none",
                       use_mel_weighting=False,
                       sample_rate=0):
        
        self.low_scale = low_scale
        self.high_scale = high_scale
        self.overlap = overlap
        self.noise_floor = noise_floor
        self.cepstrum_crop_factor = cepstrum_crop_factor
        self.window_fn = window_fn
        self.inv_window_fn = inv_window_fn
        self.use_mel_weighting = use_mel_weighting
        self.sample_rate = sample_rate

    def get_sample_mspsd(self, sample, separate_scale_channels=False):

        a_stfts = []

        for i in range(self.low_scale, self.high_scale):
            block_width = int(2 ** i)

            if separate_scale_channels:
                _sample = sample[..., i-self.low_scale, :]
            else:
                _sample = sample

            a_stft_abs = stft2(_sample, block_width, overlap=self.overlap, window_fn=self.window_fn).abs()
            a_stft_abs = (a_stft_abs / a_stft_abs.amax(dim=(-1,-2), keepdim=True)).clip(min=self.noise_floor).log()
            
            if self.use_mel_weighting:
                mel_density = get_mel_density(torch.linspace(0, self.sample_rate/2, block_width//2, device=a_stft_abs.device))
                a_stft_abs = a_stft_abs * mel_density.view((1,)*(a_stft_abs.ndim-1) + (-1,))

            a_stfts.append(a_stft_abs.view(a_stft_abs.shape[:-2] + (-1,)))

        return torch.stack(a_stfts, dim=a_stfts[0].ndim-1)
    
    def mspsd_to_cepstrum(self, mspsd):

        q_stfts = []

        for i in range(self.low_scale, self.high_scale):
            block_width = int(2 ** i)   
            
            view_dims = mspsd.shape[:-1] + (-1, block_width // 2)
            a_stft_abs = mspsd.view(view_dims)[..., i-self.low_scale, :, :]

            #save_raw(a_stft_abs, f"./debug/a_stft_abs_{i-self.low_scale}.raw")
            
            q_stft = torch.fft.rfft(a_stft_abs, norm="ortho")[..., :, :block_width // 4 // self.cepstrum_crop_factor]

            #density = torch.linspace(1/(2*block_width) ** 0.5, 1, q_stft.shape[-1], device=q_stft.device)
            #q_stft = q_stft * density.view((1,)*(a_stft_abs.ndim-1) + (-1,)).square()
            #q_stft[..., 0] /= (2*block_width) ** 0.5

            #q_stft[q_stft.abs() < 1e-4] = 1e-4
            #q_stft = q_stft.abs() + 1j * q_stft.angle()
            #print(q_stft.real.amin(), q_stft.real.amax(), q_stft.imag.amin(), q_stft.imag.amax())
            #q_factor_real = 1
            #q_factor_imag = 0.5
            #q_stft = (q_stft.real/q_factor_real).round()*q_factor_real + 1j * (q_stft.imag/q_factor_imag).round()*q_factor_imag
            #q_stft = q_stft.real * (q_stft.imag.cos() + 1j * q_stft.imag.sin())

            q_stft = torch.view_as_real(q_stft)
            ndim = q_stft.ndim - 3
            q_stft = q_stft.permute(list(range(ndim)) + [ndim+2, ndim+0, ndim+1])
            
            q_stfts.append(q_stft.reshape(q_stft.shape[:-2] + (-1,)))
        
        q_stfts = torch.stack(q_stfts, dim=q_stfts[0].ndim-2)
        return q_stfts.view(q_stfts.shape[:-3] + (q_stfts.shape[-3]*2, -1))

    def cepstrum_to_mspsd(self, cepstrum):

        a_stfts = []

        for i in range(self.low_scale, self.high_scale):
            block_width = int(2 ** i)
            
            view_dims = cepstrum.shape[:-1] + (-1, block_width // 4 // self.cepstrum_crop_factor)
            q_stft = cepstrum.view(view_dims)[..., (i-self.low_scale)*2:(i-self.low_scale)*2+2, :, :]

            ndim = q_stft.ndim - 3
            q_stft = q_stft.permute(list(range(ndim)) + [ndim+1, ndim+2, ndim+0])
            q_stft = torch.view_as_complex(q_stft.contiguous())

            #q_stft = q_stft.real * (q_stft.imag.cos() + 1j * q_stft.imag.sin())
            #q_stft[..., 0] *= (2*block_width)**0.5

            a_stft_abs = torch.fft.irfft(q_stft, n=block_width//2, norm="ortho")
            a_stfts.append(a_stft_abs.view(a_stft_abs.shape[:-2] + (-1,)))

        return torch.stack(a_stfts, dim=a_stfts[0].ndim-1)

    def get_mel_weighted_mspsd(self, mspsd):

        a_stfts = []

        for i in range(self.low_scale, self.high_scale):
            block_width = int(2 ** i)
            
            view_dims = mspsd.shape[:-1] + (-1, block_width // 2)
            a_stft_abs = mspsd.view(view_dims)[..., i-self.low_scale, :, :]

            mel_density = get_mel_density(torch.linspace(0, self.sample_rate/2, block_width//2, device=mspsd.device))
            a_stft_abs = a_stft_abs * mel_density.view((1,)*(a_stft_abs.ndim-1) + (-1,))
            a_stfts.append(a_stft_abs.view(a_stft_abs.shape[:-2] + (-1,)))

        return torch.stack(a_stfts, dim=a_stfts[0].ndim-1)
    
    @torch.no_grad()
    def _shape_sample(self, x, mspsd, scale, b):
        
        block_width = int(2 ** scale)
        view_dims = mspsd.shape[:-1] + (-1, block_width // 2)
        a_stft_abs = mspsd.view(view_dims)[..., scale-self.low_scale, :, :]

        x_stft = stft2(x, block_width, overlap=self.overlap, window_fn=self.window_fn)
        x_stft_abs = x_stft.abs()
        x_stft_abs[x_stft_abs == 0] = 1e-10
        x_stft_abs_ln = x_stft_abs.log()

        x_stft = x_stft / x_stft_abs * ((1 - b) * a_stft_abs + b * x_stft_abs_ln).exp()
        x = istft2(x_stft, block_width, overlap=self.overlap, window_fn=self.inv_window_fn)
        return x / x.abs().amax(dim=-1, keepdim=True)

    @torch.no_grad() 
    def get_sample(self, mspsd, num_iterations=400):
        
        sample_len = mspsd.shape[-1] * 2 // self.overlap
        x = torch.randn(mspsd.shape[:-2] + (sample_len,), device=mspsd.device)

        if self.use_mel_weighting:

            for i in range(self.low_scale, self.high_scale):
                block_width = int(2 ** i)
                
                view_dims = mspsd.shape[:-1] + (-1, block_width // 2)
                a_stft_abs = mspsd.view(view_dims)[..., i-self.low_scale, :, :]
                
                mel_density = get_mel_density(torch.linspace(0, self.sample_rate/2, block_width//2, device=a_stft_abs.device))
                a_stft_abs /= mel_density.view((1,)*(a_stft_abs.ndim-1) + (-1,))
        
        scales = range(self.high_scale-1, self.low_scale-1, -1)
        for t in range(num_iterations):
            b = t / num_iterations
            for i in scales:
                x = self._shape_sample(x, mspsd, i, b)

        return x / x.abs().amax(dim=-1, keepdim=True)

def get_facsimile(sample, target, num_iterations=400, low_scale=5, high_scale=11, overlap=2, window_fn="hann^0.5", inv_window_fn="hann^0.5"):

    a = target
    x = sample
    
    scales = list(range(low_scale, high_scale))
    a_stfts = []
    q_stfts = []

    for i in scales:
        block_width = int(2 ** i)
        
        a_stft_abs = stft2(a, block_width, overlap=overlap, window_fn=window_fn).abs()
        a_stft_abs /= a_stft_abs.amax()

        #a_stft_abs = a_stft_abs.unfold(-1, 2, 2)
        #a_stft_abs[:, :, :] = a_stft_abs.mean(dim=-1, keepdim=True)
        #a_stft_abs = a_stft_abs.view(a_stft_abs.shape[:-2] + (-1,))
        block_hz = torch.linspace(0, 1, block_width//2) * 4000
        mel_density = get_mel_density(block_hz)
        mel_density /= mel_density.mean()
        print(mel_density.amin(), mel_density.amax())
        a_stft_abs *= (torch.randn_like(a_stft_abs)*1 / mel_density.view(1, -1)).exp()
        
        #a_stft_abs += torch.randn_like(a_stft_abs) * 1e-10

        #print(i, a_stft_abs.amax())

        #a_stft_abs /= a_stft_abs.square().mean(dim=-1, keepdim=True).sqrt()#.clip(min=1e-10)
        #a_stft_abs = a_stft_abs.clip(min=1e-6).log()
        #a_stft_abs -= a_stft_abs.mean()

        #q_stft = torch.fft.rfft(a_stft_abs, norm="ortho")[:, :block_width//8]
        norm_factor = block_width #** (5/6)#(4/5)
        
        #print(a_stft_abs.square().mean())
        q_stft = torch.fft.rfft(a_stft_abs, norm="backward")[:, :block_width//8] #/ 4#* 8
        
        #print( (q_stft.abs() < 1e-4).sum().item())
        #q_stft[q_stft.abs() < 1e-4] = 1e-4

        #q_stft_real = q_stft.abs().log()
        #q_stft_imag = q_stft.angle()

        #q_stft_real = (q_stft.real*32+0.5).round()/32
        #q_stft_imag = (q_stft.imag*32+0.5).round()/32
        #q_stft = q_stft_real + 1j * q_stft_imag

        #q_stft_imag = (q_stft_imag / torch.pi * 10).round() / 10 * torch.pi
        #q_stft_real -= torch.randn_like(q_stft_real).abs()*0.5
        #q_stft += torch.randn_like(q_stft) * q_stft.abs() / 4
        #q_stft /= q_stft.abs()

        #print(q_stft_real.amin(), q_stft_real.amax()) # min = ln(1e-4) 
        #q_stft = q_stft_real + 1j * q_stft_imag
        q_stfts.append(q_stft)
        #print(q_stft.real.abs().amax(), q_stft.imag.abs().amax())


        #a_stft_abs = torch.fft.irfft(q_stft, n=block_width//2, norm="ortho")
        #a_stft_abs = torch.fft.irfft(q_stft, n=block_width//2, norm="backward")
        #print(a_stft_abs.square().mean())
        #a_stft_abs = a_stft_abs.exp()
        #a_stft_abs /= a_stft_abs.amax()

        a_stfts.append(a_stft_abs)

    torch.stack([a.flatten() for a in q_stfts], dim=0).numpy().tofile("./debug/test_q_stfts.raw")
    torch.stack([a.flatten() for a in a_stfts], dim=0).numpy().tofile("./debug/test_a_stfts.raw")
    
    for t in range(num_iterations):

        b = (t / num_iterations)# ** 2
        #b = ((np.cos(b * np.pi)+1)/2) ** 2

        for i in reversed(scales):

            block_width = int(2 ** i)

            a_stft = a_stfts[i-low_scale]

            x_stft = stft2(x, block_width, overlap=overlap, window_fn=window_fn)
            x_stft_abs = x_stft.abs()

            x_stft_abs[x_stft_abs == 0] = 1e-10
            
            x_stft_abs_ln = x_stft_abs.log()
            a_stft_abs_ln = a_stft.clip(min=1e-6).log()

            t_stft_abs_ln = a_stft_abs_ln*(1-b) + x_stft_abs_ln*b
            x_stft = x_stft / x_stft_abs * t_stft_abs_ln.exp()

            x = istft2(x_stft, block_width, overlap=overlap, window_fn=inv_window_fn)
            x /= x.abs().amax()

    return x

def get_lpc_coefficients(X: torch.Tensor, order: int ) -> torch.Tensor:
    """Forward

    Parameters
    ----------
    X: torch.Tensor
        Input signal to be sliced into frames.
        Expected input is [ Batch, Samples ]

    Returns
    -------
    X: torch.Tensor
        LPC Coefficients computed from input signal after slicing.
        Expected output is [ Batch, Frames, Order + 1 ]
    """
    p = order + 1
    B, F, S                = X.size( )

    alphas                 = torch.zeros( ( B, F, p ),
        dtype         = X.dtype,
        device        = X.device,
    )
    alphas[ :, :, 0 ]      = 1.
    alphas_prev            = torch.zeros( ( B, F, p ),
        dtype         = X.dtype,
        device        = X.device,
    )
    alphas_prev[ :, :, 0 ] = 1.

    fwd_error              = X[ :, :, 1:   ]
    bwd_error              = X[ :, :,  :-1 ]

    den                    = (
        ( fwd_error * fwd_error ).sum( axis = -1 ) + \
        ( bwd_error * bwd_error ).sum( axis = -1 )
    ).unsqueeze( -1 )

    #den = den.clip(min=1e-8) #
    #den = den + 1
    #amin = den.amin()
    #amax = den.amax()

    for i in range( order ):
        not_ill_cond        = ( den > 0 ).float( )
        den                *= not_ill_cond

        dot_bfwd            = ( bwd_error * fwd_error ).sum( axis = -1 )\
                                                        .unsqueeze( -1 )

        reflect_coeff       = -2. * dot_bfwd / den
        alphas_prev, alphas = alphas, alphas_prev

        if torch.isnan(reflect_coeff).any() or torch.isinf(reflect_coeff).any():
            break

        for j in range( 1, i + 2 ):
            alphas = alphas.clone()
            alphas[ :, :, j ] = alphas_prev[   :, :,         j ] + \
                                reflect_coeff[ :, :,         0 ] * \
                                alphas_prev[   :, :, i - j + 1 ]

        fwd_error_tmp       = fwd_error
        fwd_error           = fwd_error + reflect_coeff * bwd_error
        bwd_error           = bwd_error + reflect_coeff * fwd_error_tmp

        q                   = 1. - reflect_coeff ** 2
        den                 = q * den - \
                                bwd_error[ :, :, -1 ].unsqueeze( -1 ) ** 2 - \
                                fwd_error[ :, :,  0 ].unsqueeze( -1 ) ** 2
        
        #den = den + 1
        #amin = torch.minimum(amin, den.amin())
        #amax = torch.maximum(amax, den.amax())
        #den = den.clip(min=1e-8) #

        fwd_error           = fwd_error[ :, :, 1:   ]
        bwd_error           = bwd_error[ :, :,  :-1 ]

    #print(alphas.amin(), alphas.amax())
    #print(alphas.amin(), alphas.amax())
    #alphas[ alphas != alphas ] = 0.
    #alphas = torch.nan_to_num(alphas.clip(min=-100, max=100), nan=0.0, posinf=100, neginf=-100)
    return alphas

def save_raw_img(x, img_path):
    
    x = x.detach().resolve_conj().cpu()
    x -= x.amin(dim=(x.ndim-1, x.ndim-2), keepdim=True)
    x /= x.amax(dim=(x.ndim-1, x.ndim-2), keepdim=True)
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
        elif cv2_img.shape[2] > 3:
            raise ValueError(f"Unsupported number of channels in save_raw_img: {cv2_img.shape[2]}")
    elif x.ndim == 2:
        x = x.permute(1, 0).contiguous().numpy()
        cv2_img = (x * 255).astype(np.uint8)
        cv2_img = cv2.applyColorMap(cv2_img, cv2.COLORMAP_JET)
    else:
        raise ValueError(f"Unsupported number of dimensions in save_raw_img: {x.ndim}")

    cv2.imwrite(img_path, cv2.flip(cv2_img, 0))

if __name__ == "__main__":

    init_cuda()
    load_dotenv(override=True)

    # MSPSD test
    """
    torch.manual_seed(0)
    np.random.seed(0)
    
    mspsd = MSPSD(sample_rate=8000, use_mel_weighting=True, overlap=16, window_fn="blackman_harris", inv_window_fn="blackman_harris")

    a = load_raw("./dataset/samples/1.raw")[:65536*2]

    psd = mspsd.get_sample_mspsd(a.unsqueeze(0).unsqueeze(0))
    cepstrum = mspsd.mspsd_to_cepstrum(psd)
    #print(torch.randn_like(cepstrum).abs().mean())

    #psd = mspsd.cepstrum_to_mspsd(cepstrum)
    x = mspsd.get_sample(psd, num_iterations=100)

    save_raw(a, "./debug/test_a.raw")
    save_raw(x, "./debug/test_x.raw")
    save_raw(cepstrum, "./debug/test_c.raw")
    save_raw(psd, "./debug/test_p.raw")

    #psd_mel_weighted = mspsd.get_mel_weighted_mspsd(psd)
    #save_raw(psd_mel_weighted, "./debug/test_p_mel_weighted.raw")

    exit()
    """

    # facsimile test
    """"
    
    dataset_path = os.environ.get("DATASET_PATH", "./dataset/samples")
    test_sample = np.random.choice(os.listdir(dataset_path), 1, replace=False)[0]
    #test_sample = "29235.raw" #"1.raw"
    print(test_sample)
    a = load_raw(os.path.join(dataset_path, test_sample))[:65536*2]
    a /= a.abs().amax()
    save_raw(a, "./debug/test_a.raw")

    torch.manual_seed(0)
    np.random.seed(0)

    x = torch.randn_like(a)
    x = get_facsimile(x, a)

    x /= x.abs().amax()
    save_raw(x, "./debug/test_x.raw")
    """