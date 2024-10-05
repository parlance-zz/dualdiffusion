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

from utils import config

import os
from io import BytesIO
from typing import Optional, Union
from json import dumps as json_dumps

import numpy as np
import torch
import torchaudio
import torchaudio.functional as AF
import cv2
import safetensors.torch as ST
import matplotlib.pyplot as plt
from scipy.special import erfinv
from mutagen import File as MTAudioFile

from utils.roseus_colormap import ROSEUS_COLORMAP

class TF32_Disabled:
    def __enter__(self):
        self.original_matmul_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        self.original_cudnn_allow_tf32 = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    def __exit__(self, exc_type, exc_value, traceback):
        torch.backends.cuda.matmul.allow_tf32 = self.original_matmul_allow_tf32
        torch.backends.cudnn.allow_tf32 = self.original_cudnn_allow_tf32


def init_cuda(default_device: Optional[torch.device] = None) -> None:

    if not torch.cuda.is_available():
        raise ValueError("Error: PyTorch not compiled with CUDA support or CUDA unavailable")
    else:
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.cufft_plan_cache[0].max_size = 250 # cufft memory leak

        if default_device is not None:
            torch.cuda.set_device(default_device)

def multi_plot(*args, layout: Optional[tuple[int, int]] = None,
               figsize: Optional[tuple[int, int]] = None,
               added_plots: Optional[dict] = None,
               x_log_scale: bool = False,
               y_log_scale: bool = False,
               x_axis_range: Optional[tuple] = None) -> None:

    if config.NO_GUI: return
    
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
    plt.show()

def dict_str(d: dict, indent: int = 4) -> str:
    return json_dumps(d, indent=indent)

def sanitize_filename(filename: str) -> str:
    return ("".join(c for c in filename
        if c.isalnum() or c in (" ",".","_"))).strip()

def save_tensor_raw(tensor: torch.Tensor, output_path: str) -> None:

    directory = os.path.dirname(output_path)
    os.makedirs(directory, exist_ok=True)

    if tensor.dtype in [torch.float16, torch.bfloat16]:
        tensor = tensor.float()
    elif tensor.dtype == torch.complex32:
        tensor = tensor.complex64()
    tensor.detach().resolve_conj().cpu().numpy().tofile(output_path)

@torch.inference_mode()
def normalize_lufs(raw_samples: torch.Tensor,
                   sample_rate: int,
                   target_lufs: float = -12.,
                   max_clip:float = 0.15) -> torch.Tensor:
    
    original_shape = raw_samples.shape
    raw_samples = torch.nan_to_num(raw_samples, nan=0, posinf=0, neginf=0)
    
    if raw_samples.ndim == 1:
        raw_samples = raw_samples.view(1, 1,-1)
    elif raw_samples.ndim == 2:
        raw_samples = raw_samples.view(1, raw_samples.shape[0], -1)

    raw_samples /= raw_samples.abs().amax(
        dim=tuple(range(1, len(raw_samples.shape))), keepdim=True).clip(min=1e-16)
    
    current_lufs = AF.loudness(raw_samples, sample_rate)
    gain = 10. ** ((target_lufs - current_lufs) / 20.0)
    gain = gain.view((*gain.shape,) + (1,) * (raw_samples.ndim - gain.ndim))

    normalized_raw_samples = raw_samples * gain
    normalized_raw_samples /= normalized_raw_samples.abs().amax(
        dim=tuple(range(1, len(normalized_raw_samples.shape))), keepdim=True).clip(min=1+max_clip)

    return normalized_raw_samples.view(original_shape)

def get_no_clobber_filepath(filepath):

    directory, filename = os.path.split(filepath)
    name, ext = os.path.splitext(filename)
    
    unique_filepath = filepath
    counter = 0
    
    while os.path.exists(unique_filepath):
        new_filename = f"{name}_{counter}{ext}"
        unique_filepath = os.path.join(directory, new_filename)
        counter += 1
    
    return unique_filepath

def save_audio(raw_samples: torch.Tensor,
               sample_rate: int,
               output_path: str,
               target_lufs: float = -12.,
               metadata: Optional[dict] = None,
               no_clobber: bool = False) -> str:
    
    raw_samples = raw_samples.detach().real.float()
    if raw_samples.ndim == 1:
        raw_samples = raw_samples.view(1, -1)

    if target_lufs is not None:
        raw_samples = normalize_lufs(raw_samples, sample_rate, target_lufs)

    directory = os.path.dirname(output_path)
    os.makedirs(directory, exist_ok=True)
    
    if no_clobber == True:
        output_path = get_no_clobber_filepath(output_path)

    torchaudio.save(output_path, raw_samples.cpu(), sample_rate, bits_per_sample=16)

    if metadata is not None:
        audio_file = MTAudioFile(output_path)
        for key in metadata:
            audio_file[key] = metadata[key]
        audio_file.save()

    return output_path

def torch_dtype(dtype: Union[str, torch.dtype]) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    elif isinstance(dtype, str):
        return getattr(torch, dtype)
    else:
        raise ValueError(f"Unsupported dtype type: {dtype} ({type(dtype)})")

def get_available_torch_devices() -> list[str]:
    available_devices = ["cpu"]
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            available_devices.append(f"cuda{i}" if i > 0 else "cuda")

    if torch.backends.mps.is_available():
        available_devices.append("mps")

    return available_devices
        
def load_audio(input_path: Union[str, bytes],
               start: int = 0, count: int = -1,
               return_sample_rate: bool = False,
               device: Optional[torch.device] = None,
               return_start:bool = False):

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

def save_safetensors(tensors_dict: dict[str, torch.Tensor], output_path: str,
                     metadata: Optional[dict[str, str]] = None) -> None:
    directory = os.path.dirname(output_path)
    os.makedirs(directory, exist_ok=True)

    for key in tensors_dict:
        val = tensors_dict[key]
        if torch.is_tensor(val):
            val = val.detach().resolve_conj().contiguous().cpu()
        else:
            val = torch.tensor(val)
        tensors_dict[key] = val

    ST.save_file(tensors_dict, output_path, metadata=metadata)

def load_safetensors(input_path: str, device: Optional[torch.device] = None) -> dict[str, torch.Tensor]:
    return ST.load_file(input_path, device=device)

def get_expected_max_normal(n: int) -> float:
    return erfinv((n - np.pi/8) / (n - np.pi/4 + 1))

@torch.inference_mode()
def get_fractal_noise2d(shape: tuple, degree: int = 1, **kwargs) -> torch.Tensor:
    
    noise_shape = shape[:-2] + (shape[-2]*2, shape[-1]*2)
    noise_complex = torch.randn(noise_shape, **kwargs)
    
    linspace_x = torch.arange(0, noise_complex.shape[-2], device=noise_complex.device)
    linspace_y = torch.arange(0, noise_complex.shape[-1], device=noise_complex.device)
    linspace_x = (linspace_x - noise_complex.shape[-2]//2).square() / noise_complex.shape[-2]**2
    linspace_y = (linspace_y - noise_complex.shape[-1]//2).square() / noise_complex.shape[-1]**2

    linspace = (linspace_x.view(-1, 1) + linspace_y.view(1, -1)).pow(-degree/2)
    linspace[noise_complex.shape[-2]//2, noise_complex.shape[-1]//2] = 0
    linspace = linspace.view((1,)*(len(shape)-2) + (noise_complex.shape[-2], noise_complex.shape[-1]))
    linspace = torch.roll(linspace, shifts=(linspace.shape[-2]//2, linspace.shape[-1]//2), dims=(-2, -1))

    pink_noise = torch.fft.ifft2(noise_complex * linspace, norm="ortho").real[..., :shape[-2], :shape[-1]]
    return (pink_noise / pink_noise.std(dim=(-2, -1), keepdim=True)).detach()

def to_ulaw(x: torch.Tensor, u: float = 255.) -> torch.Tensor:

    complex = False
    if torch.is_complex(x):
        complex = True
        x = torch.view_as_real(x)

    x = x / x.abs().amax(dim=tuple(range(x.ndim-2-int(complex), x.ndim)), keepdim=True)
    x = torch.sign(x) * torch.log1p(u * x.abs()) / np.log(1 + u)

    if complex:
        x = torch.view_as_complex(x)
    
    return x

def from_ulaw(x: torch.Tensor, u: float = 255.) -> torch.Tensor:

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
def quantize_tensor(x: torch.Tensor, levels: int) -> torch.Tensor:
    reduction_dims = tuple(range(1, x.ndim)) if x.ndim > 1 else (0,)

    min_val = x.amin(dim=reduction_dims, keepdim=True)
    max_val = x.amax(dim=reduction_dims, keepdim=True)
    scale = (max_val - min_val) / (levels - 1)

    quantized = ((x - min_val) / scale).round().clamp(0, levels - 1)
    offset_and_range = torch.stack((min_val.flatten(), scale.flatten()), dim=-1)
    return quantized, offset_and_range

@torch.no_grad()
def dequantize_tensor(x: torch.Tensor, offset_and_range: torch.Tensor) -> torch.Tensor:
    view_dims = (-1,) + ((1,) * (x.ndim-1) if x.ndim > 1 else ())
    min_val, scale = offset_and_range.unbind(-1)
    return x * scale.view(view_dims) + min_val.view(view_dims)

@torch.no_grad()
def tensor_to_img(x: torch.Tensor,
                  recenter: bool = True,
                  rescale: bool = True,
                  flip_x: bool = False,
                  flip_y: bool = False,
                  colormap: bool = False,) -> np.ndarray:
    
    x = x.clone().detach().real.float().resolve_conj().cpu()
    while x.ndim < 4: x.unsqueeze_(0)
    if x.ndim == 5: x = x.view(x.shape[0], x.shape[1], x.shape[2] * x.shape[3], x.shape[4])
    x = x.permute(0, 2, 3, 1).contiguous().view(x.shape[0] * x.shape[2], x.shape[3], x.shape[1])

    if recenter: x -= x.amin(dim=(-3,-2,-1), keepdim=True)
    if rescale:  x /= x.amax(dim=(-3,-2,-1), keepdim=True).clip(min=1e-16)

    if x.shape[-1] == 4: # show alpha channel as pre-multiplied brightness
        x = x[..., :3] * x[..., 3:4]
        if recenter: x -= x.amin(dim=(-3,-2,-1), keepdim=True)
        if rescale:  x /= x.amax(dim=(-3,-2,-1), keepdim=True).clip(min=1e-16)  
    elif x.shape[-1] == 2:
        x = torch.cat((x, torch.zeros_like(x[..., 0:1])), dim=-1)
        x[..., 2], x[..., 1] = x[..., 1], 0
    elif x.shape[-1] > 4:
        raise ValueError(f"Unsupported number of channels in tensor_to_img: {x.shape[-1]}")
    
    img = (x * 255).clip(min=0, max=255).numpy().astype(np.uint8)

    if flip_x: img = cv2.flip(img, 1)
    if flip_y: img = cv2.flip(img, 0)
    if colormap: img = ROSEUS_COLORMAP[img]

    return img

def save_img(np_img: np.ndarray, img_path: str) -> None:
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    cv2.imwrite(img_path, np_img)

def open_img_window(name: str,
                    width:  Optional[int] = None,
                    height: Optional[int] = None,
                    topmost: bool = False) -> None:

    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    if width is not None and height is not None:
        cv2.resizeWindow(name, width, height)
    if topmost:
        cv2.setWindowProperty(name, cv2.WND_PROP_TOPMOST, 1)

def show_img(np_img: np.ndarray, name: str, wait: int = 1) -> None:
    if config.NO_GUI: return
    cv2.imshow(name, np_img)
    cv2.waitKey(wait)

def close_img_window(name: str) -> None:
    if config.NO_GUI: return
    cv2.destroyWindow(name)
    
def slerp(start: torch.Tensor,
          end: torch.Tensor,
          t: Union[float, torch.Tensor],
          dtype: torch.dtype = torch.float64) -> torch.Tensor:

    if torch.is_tensor(t):
        t = t.to(dtype)
        if t.ndim < start.ndim:
            t = t.view(*t.shape, *((1,) * (start.ndim - t.ndim)))
    
    start, end = start.to(dtype), end.to(dtype)
    omega = get_cos_angle(start, end, keepdim=True, dtype=dtype)
    so = torch.sin(omega)

    return (torch.sin((1 - t) * omega) / so) * start + (torch.sin(t * omega) / so) * end

def get_cos_angle(start: torch.Tensor,
                  end: torch.Tensor,
                  keepdim: bool = False,
                  dtype: torch.dtype = torch.float64) -> torch.Tensor:
    
    reduction_dims = tuple(range(1, start.ndim)) if start.ndim > 1 else (0,)

    start, end = start.to(dtype), end.to(dtype)
    start_len = torch.linalg.vector_norm(start, dim=reduction_dims, keepdim=True, dtype=dtype)
    end_len = torch.linalg.vector_norm(end, dim=reduction_dims, keepdim=True, dtype=dtype)

    return (start / start_len * end / end_len).sum(dim=reduction_dims, keepdim=keepdim).clamp(-1, 1).acos()

def normalize(x: torch.Tensor, zero_mean: bool = False, dtype: torch.dtype = torch.float64) -> torch.Tensor:

    reduction_dims = tuple(range(1, x.ndim)) if x.ndim > 1 else (0,)
    x = x.to(dtype)

    if zero_mean:
        x = x - x.mean(dim=reduction_dims, keepdim=True)

    return x / x.square().mean(dim=reduction_dims, keepdim=True).sqrt()