import numpy as np
import torch
import torch.nn.functional as F
import os
from dotenv import load_dotenv

load_dotenv()

dataset_path = os.environ.get("DATASET_PATH", "./")
test_sample = 0
block_width = 512
crop_width = 65536*2 - block_width // 2

def mdct(x, block_width, random_phase_offset=False):

    pad_tuple = (block_width//2, block_width//2) + (0,0,) * (x.ndim-1)
    x = F.pad(x, pad_tuple).unfold(-1, block_width, block_width//2)

    N = x.shape[-1] // 2
    n = torch.arange(2*N, device=x.device)
    k = torch.arange(0.5, N + 0.5, device=x.device)

    window = torch.sin(np.pi * (n + 0.5) / (2*N))
    pre_shift = torch.exp(-1j * torch.pi / 2 / N * n)
    post_shift = torch.exp(-1j * torch.pi / 2 / N * (N + 1) * k)
    
    y = torch.fft.fft(x * pre_shift * window)[..., :N] * post_shift
    if random_phase_offset:
        y *= torch.exp(2j*torch.pi*torch.rand(1, device=y.device))

    return y.real * 2

def imdct(x):
    N = x.shape[-1]
    n = torch.arange(2*N, device=x.device)
    k = torch.arange(0.5, N + 0.5, device=x.device)

    window = torch.sin(np.pi * (n + 0.5) / (2*N))
    pre_shift = torch.exp(-1j * torch.pi / 2 / N * n)
    post_shift = torch.exp(-1j * torch.pi / 2 / N * (N + 1) * k)
    
    x = torch.cat((x / post_shift, torch.zeros_like(x)), dim=-1)
    y = (torch.fft.ifft(x) / pre_shift).real * window

    padded_sample_len = (y.shape[-2] + 1) * y.shape[-1] // 2
    raw_sample = torch.zeros(y.shape[:-2] + (padded_sample_len,), device=y.device)
    raw_sample[..., :-N]  = y[...,  ::2, :].reshape(*raw_sample[..., :-N].shape)
    raw_sample[..., N: ] += y[..., 1::2, :].reshape(*raw_sample[...,  N:].shape)

    return raw_sample[..., N:-N] * 2

def to_ulaw(x, u=255):
    return torch.sign(x) * torch.log(1 + u * torch.abs(x)) / np.log(1 + u)

def from_ulaw(x, u=255):
    return torch.sign(x) * ((1 + u) ** torch.abs(x) - 1) / u

raw_sample = np.fromfile(os.path.join(dataset_path, f"{test_sample}.raw"), dtype=np.int16, count=crop_width) / 32768.
raw_sample = torch.from_numpy(raw_sample.astype(np.float32))
raw_sample.cpu().numpy().tofile("./debug/raw_sample.raw")

#raw_sample = raw_sample.unsqueeze(0)
#raw_sample = raw_sample.repeat(2, 1)
Xk = mdct(raw_sample, block_width, random_phase_offset=False)

Xk /= Xk.abs().max()
Xk = to_ulaw(Xk, u=16384)
Xk.cpu().numpy().tofile("./debug/mdct_ulaw.raw")
#Xk += torch.randn_like(Xk) * 1e-2
#Xk = (Xk * 16).type(torch.int32) / 16.
#Xk.cpu().numpy().tofile("./debug/mdct_ulaw_quantized.raw")
Xk = from_ulaw(Xk, u=16384)

print("Xk shape:", Xk.shape, "Xk mean:", (Xk / Xk.std()).mean().item(), "Xk std:", Xk.std().item())
Xk.cpu().numpy().tofile("./debug/mdct.raw")

y = imdct(Xk)
y.cpu().numpy().tofile("./debug/imdct.raw")
