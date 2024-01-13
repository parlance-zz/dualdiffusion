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

def get_ln_window(window_len, device="cpu"):
    x = torch.linspace(0, 1, window_len, device=device)
    w = torch.exp(-torch.log2(x).square() - torch.log2(1-x).square())
    w[0] = 0; w[-1] = 0
    return (w / w.amax()).requires_grad_(False)

def get_blackman_harris2_window(window_len, device="cpu"):
    return (get_blackman_harris_window(window_len, device) * get_flat_top_window(window_len, device)).requires_grad_(False)

# fast overlapped modified discrete cosine transform type iv - becomes mclt with complex output
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
    elif window_fn == "ln":
        window = get_ln_window(2*N, device=x.device)
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
    elif window_fn == "ln":
        window = get_ln_window(2*N, device=x.device)
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

def stft(x, block_width, window_fn="hann", window_degree=2, step=None):

    if step is None:
        step = block_width // 2

    #x = F.pad(x, (0, step+1)).unfold(-1, block_width, step)
    x = x.unfold(-1, block_width, step)

    if window_fn == "hann":
        if window_degree == 0:
            window = 1
        else:
            window = torch.sin(torch.pi * (n + 0.5) / x.shape[-1]).requires_grad_(False)
            if window_degree == 2: window = window.square()
    elif window_fn == "hann_poisson":
        window = get_hann_poisson_window(x.shape[-1], device=x.device)
    elif window_fn == "blackman_harris":
        window = get_blackman_harris_window(x.shape[-1], device=x.device)
    elif window_fn == "flat_top":
        window = get_flat_top_window(x.shape[-1], device=x.device)
    elif window_fn == "ln":
        window = get_ln_window(x.shape[-1], device=x.device)
    elif window_fn == "none":
        window = 1
    elif window_fn == "blackman_harris2":
        window = get_blackman_harris2_window(x.shape[-1], device=x.device)
    else:
        raise ValueError(f"Unsupported window function: {window_fn}")
    
    return torch.fft.rfft(x * window, norm="forward")

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
    raw_samples = torch.nan_to_num(raw_samples, nan=0, posinf=0, neginf=0)
    
    if raw_samples.ndim == 1:
        raw_samples = raw_samples.view(1, 1, -1)
    elif raw_samples.ndim == 2:
        raw_samples = raw_samples.view(raw_samples.shape[0], 1, -1)

    current_lufs = AF.loudness(raw_samples, sample_rate)
    gain = (10. ** ((target_lufs - current_lufs) / 20.0)).clamp(min=1e-5, max=1e5)
    gain = gain.view((*gain.shape,) + (1,) * (raw_samples.ndim - gain.ndim))

    normalized_raw_samples = (raw_samples * gain).view(original_shape).clamp(min=-10, max=10)
    return torch.nan_to_num(normalized_raw_samples, nan=0, posinf=0, neginf=0)

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

def stft2(x, block_width, overlap=2):

    step = block_width // overlap
    x = F.pad(x, (step//2, step//2))
    x = x.unfold(-1, block_width, step)
    
    window = get_hann_window(block_width, device=x.device)
    #window = get_blackman_harris_window(block_width, device=x.device)

    N = x.shape[-1] // 2
    n = torch.arange(2*N, device=x.device)
    k = torch.arange(0.5, N + 0.5, device=x.device)
    pre_shift = torch.exp(-1j * torch.pi / 2 / N * n)
    post_shift = torch.exp(-1j * torch.pi / 2 / N * (N + 1) * k)
    return torch.fft.fft(x * pre_shift * window, norm="forward")[..., :N] * post_shift #* 2 * N ** 0.5

    #return torch.fft.rfft(x * window, norm="ortho")

def istft2(x, block_width, overlap=2):

    step = block_width // overlap

    signal_len = (x.shape[-2] - 1) * step + block_width
    y = torch.zeros(x.shape[:-2] + (signal_len,), device=x.device)

    N = x.shape[-1]
    n = torch.arange(2*N, device=x.device)
    k = torch.arange(0.5, N + 0.5, device=x.device)
    pre_shift = torch.exp(-1j * torch.pi / 2 / N * n)
    post_shift = torch.exp(-1j * torch.pi / 2 / N * (N + 1) * k)
    x = (torch.fft.ifft(x / post_shift, n=block_width, norm="backward") / pre_shift).real

    #x = torch.fft.irfft(x, n=block_width, norm="ortho")

    for i in range(overlap):
        t = x[..., i::overlap, :].reshape(*x.shape[:-2], -1)
        y[..., i*step:i*step + t.shape[-1]] += t

    y = y[..., step//2:-step//2]
    return y * (2 / overlap)

def get_facsimile(sample, target, num_iterations=100, low_scale=6, high_scale=12, overlap=2):

    a = target
    x = sample
    
    a_stfts = []
    for i in range(low_scale, high_scale):
        block_width = int(2 ** i)
        a_stfts.append(stft2(a, block_width, overlap=overlap).abs())

    for v in range(len(a_stfts)):
        a_stfts[v] = torch.fft.fft((a_stfts[v] * 8000).log1p(), norm="forward")[..., :a_stfts[v].shape[-1]//2]

    temp = torch.stack([a.flatten() for a in a_stfts], dim=1)
    print(temp.shape)
    save_raw(temp, "./debug/test_a_stfts.raw")
    exit()

    for t in range(num_iterations):
        for i in range(low_scale, high_scale):
    
            i = np.random.randint(low_scale, high_scale)  # this really does have to be random... ????            
            block_width = int(2 ** i)

            a_stft = a_stfts[i-low_scale]
            x_stft = stft2(x, block_width, overlap=overlap)
            x_stft = x_stft / x_stft.abs().clip(min=1e-15) * a_stft
            if t < (num_iterations - 1): x_stft[:, -1] /= 2
            x = istft2(x_stft, block_width, overlap=overlap)

    i = 7 # seems to induce the least artifacts ????
    block_width = int(2 ** i)

    a_stft = a_stfts[i-low_scale]
    x_stft = stft2(x, block_width, overlap=overlap)
    x_stft = x_stft / x_stft.abs().clip(min=1e-15) * a_stft.abs()
    x = istft2(x_stft, block_width, overlap=overlap)
    
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

# facsimile test   
#"""
a = load_raw("./dataset/samples/29235.raw")[:65536*2]
#a = load_raw("./debug/test_y.raw")[:65536]
a /= a.abs().amax()
save_raw(a, "./debug/test_a.raw")

x = torch.randn_like(a)

#x = torch.randn_like(a)
#x = x **2 * torch.sign(x)

#x = (torch.rand_like(a) * 2 * torch.pi).cos()

x = get_facsimile(x, a)

x /= x.abs().amax()
save_raw(x, "./debug/test_x.raw")
#"""