import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
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