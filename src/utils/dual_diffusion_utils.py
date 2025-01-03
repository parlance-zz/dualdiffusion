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

from io import BytesIO
from typing import Optional, Union, Any
from json import dumps as json_dumps
from datetime import datetime
import os
import logging

import numpy as np
import torch
import torchaudio
import cv2
import safetensors.torch as safetensors
import matplotlib.pyplot as plt
import mutagen
import mutagen.flac

from utils.roseus_colormap import ROSEUS_COLORMAP

class TF32_Disabled: # disables reduced precision tensor cores inside the context
    def __enter__(self):
        self.original_matmul_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        self.original_cudnn_allow_tf32 = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    def __exit__(self, exc_type, exc_value, traceback):
        torch.backends.cuda.matmul.allow_tf32 = self.original_matmul_allow_tf32
        torch.backends.cudnn.allow_tf32 = self.original_cudnn_allow_tf32
        return False

def init_cuda(default_device: Optional[torch.device] = None) -> None:

    if not torch.cuda.is_available():
        raise ValueError("Error: PyTorch not compiled with CUDA support or CUDA unavailable")
    else:
         # leaving these enabled these seems to make no difference for performance or stability
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        #torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True) # improves perf by ~2.5%, seems stable
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.cufft_plan_cache[0].max_size = 250 # avoid cufft memory leak
        torch.backends.cudnn.benchmark = True
        
        if default_device is not None:
            torch.cuda.set_device(default_device)

def init_logging(name: Optional[str] = None, group_name: Optional[str] = None,
        format: Union[bool, str] = False, verbose: bool = False, log_to_file: bool = True) -> logging.Logger:

    if format == True: format = f"{name}: %(message)s"
    elif format == False: format = f"%(message)s"
    formatter = logging.Formatter(format)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(formatter)
    
    if config.DEBUG_PATH is not None and log_to_file == True:
        if name is None:
            name = globals().get("__file__", None)
            if name is not None: name = os.path.splitext(os.path.basename(name))[0]
            else: name = "log"
            
        logging_dir = os.path.join(config.DEBUG_PATH, group_name or "logs")
        os.makedirs(logging_dir, exist_ok=True)

        datetime_str = datetime.now().strftime(r"%Y-%m-%d_%H_%M_%S")
        log_path = os.path.join(logging_dir, f"{name}_{datetime_str}.log")

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        file_handler = None

    logger.addHandler(console_handler)
    if file_handler is not None:
        logger.addHandler(file_handler)

    if file_handler is not None:
        logger.debug(f"\nStarted {name} at {datetime_str}")
        logger.debug(f"Logging to {log_path}")
    elif log_to_file == True:
        logger.warning("Unable to log to file, DEBUG_PATH not set")

    return logger

def multi_plot(*args, layout: Optional[tuple[int, int]] = None,
               figsize: Optional[tuple[int, int]] = None,
               added_plots: Optional[dict] = None,
               x_log_scale: bool = False,
               y_log_scale: bool = False,
               x_axis_range: Optional[tuple] = None,
               y_axis_range: Optional[tuple] = None) -> None:

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
            if y_axis_range is not None:
                axis.set_ylim(ymin=y_axis_range[0], ymax=y_axis_range[1])

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

def _is_json_serializable(value: Any) -> bool:
    if isinstance(value, (str, int, float, bool, type(None))):
        return True
    elif isinstance(value, list):
        return all(_is_json_serializable(item) for item in value)
    elif isinstance(value, dict):
        return all(_is_json_serializable(v) for v in value.values())
    else:
        return False
    
def dict_str(d: dict, indent: int = 4) -> str:
    json_dict = {k: v for k, v in d.items() if _is_json_serializable(v)}
    return json_dumps(json_dict, indent=indent)

def sanitize_filename(filename: str) -> str:
    return ("".join(c for c in filename
        if c.isalnum() or c in (" ",".","_","-","+","(",")","[","]","{","}"))).strip()

def save_tensor_raw(tensor: torch.Tensor, output_path: str) -> None:

    directory = os.path.dirname(output_path)
    os.makedirs(directory, exist_ok=True)

    if tensor.dtype in [torch.float16, torch.bfloat16]:
        tensor = tensor.float()
    elif tensor.dtype == torch.complex32:
        tensor = tensor.to(torch.complex64)
    
    tensor.detach().resolve_conj().cpu().numpy().tofile(output_path)

@torch.inference_mode()
def normalize_lufs(raw_samples: torch.Tensor,
                   sample_rate: int,
                   target_lufs: float = -15.,
                   return_old_lufs: bool = False) -> torch.Tensor:
    
    original_shape = raw_samples.shape
    raw_samples = torch.nan_to_num(raw_samples, nan=0, posinf=0, neginf=0)
    
    if raw_samples.ndim == 1:
        raw_samples = raw_samples.view(1, 1,-1)
    elif raw_samples.ndim == 2:
        raw_samples = raw_samples.view(1, raw_samples.shape[0], -1)
    elif raw_samples.ndim != 3:
        raise ValueError(f"Unsupported raw_samples shape: {raw_samples.shape}")
    
    raw_samples /= raw_samples.abs().amax(
        dim=tuple(range(1, len(raw_samples.shape))), keepdim=True).clip(min=1e-16)
    
    current_lufs = torchaudio.functional.loudness(raw_samples, sample_rate)
    gain = 10. ** ((target_lufs - current_lufs) / 20.0)
    gain = gain.view((*gain.shape,) + (1,) * (raw_samples.ndim - gain.ndim))

    normalized_raw_samples = (raw_samples * gain).view(original_shape)
    if return_old_lufs:
        return normalized_raw_samples, current_lufs.item()
    else:
        return normalized_raw_samples

# get the number of clipped samples in a raw audio tensor
# clipping is defined as at least 2 consecutive samples with an absolute value > 1.0
@torch.inference_mode()
def get_num_clipped_samples(raw_samples: torch.Tensor, eps: float = 0.) -> int:
    clips = (raw_samples.abs() > (1.0 - eps)).float()
    return int((clips[..., :-1] * clips[..., 1:]).sum().item())
    
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
               target_lufs: float = -15.,
               metadata: Optional[dict] = None,
               no_clobber: bool = False,
               compression: Optional[torchaudio.io.CodecConfig] = None) -> str:
    
    raw_samples = raw_samples.detach().real.float()
    if raw_samples.ndim == 1:
        raw_samples = raw_samples.view(1, -1)

    if target_lufs is not None:
        raw_samples = normalize_lufs(raw_samples, sample_rate, target_lufs)

    directory = os.path.dirname(output_path)
    os.makedirs(directory, exist_ok=True)
    
    if no_clobber == True:
        output_path = get_no_clobber_filepath(output_path)
    
    audio_format = os.path.splitext(output_path)[1].lower()
    bits_per_sample = 16 if audio_format in [".wav", ".flac"] else None

    torchaudio.save(output_path, raw_samples.cpu(),
        sample_rate, bits_per_sample=bits_per_sample, compression=compression)

    if metadata is not None:
        update_audio_metadata(output_path, metadata)

    return output_path

def update_audio_metadata(audio_path: str, metadata: Optional[dict] = None,
                          rating: Optional[int] = None, clear_clap_fields: bool = False) -> None:
    
    metadata = metadata or {}
    audio_format = os.path.splitext(audio_path)[1].lower()

    if rating is not None and audio_format != ".mp3":
        metadata = metadata.copy()
        # documentation on rating metadata is scarce, this works for VLC but not windows explorer/media player
        metadata.update({ 
            "RATING": str(rating),
            "RATING WMP": str(rating),
            "FMPS_RATING": f"{rating/5}"
        })

    if len(metadata) > 0 or rating is not None:

        if audio_format == ".flac":
            audio_file = mutagen.flac.FLAC(audio_path)
        else:
            audio_file = mutagen.File(audio_path)
        
        if clear_clap_fields == True:
            for key in list(audio_file.keys()):
                if key.startswith("clap_"):
                    audio_file.pop(key)

        for key in metadata:
            if audio_format != ".mp3":
                audio_file[key] = metadata[key] if type(metadata[key]) not in [int, float] else str(metadata[key])
            else: audio_file[f"TXXX:{key}"] = mutagen.id3.TXXX(encoding=3, desc=key, text=metadata[key])
        
        # this works for windows explorer/media player but not VLC
        if rating is not None and audio_format == ".mp3":
            rating = int(min(max(rating, 0), 5))
            rating = [0, 1, 64, 128, 196, 255][rating] # whoever came up with this scale is an idiot
            audio_file["POPM:Windows Media Player 9 Series"] = mutagen.id3.POPM(
                email="Windows Media Player 9 Series", rating=rating)

        audio_file.save()

def get_audio_metadata(audio_path: str) -> dict:
    audio_format = os.path.splitext(audio_path)[1].lower()
    if audio_format == ".flac":
        audio_file = mutagen.flac.FLAC(audio_path)
    else:
        audio_file = mutagen.File(audio_path)
    
    return {key: audio_file[key] for key in audio_file.keys()}

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

    safetensors.save_file(tensors_dict, output_path, metadata=metadata)

def load_safetensors(input_path: str, device: Optional[torch.device] = None) -> dict[str, torch.Tensor]:
    return safetensors.load_file(input_path, device=device)

def load_safetensors_ex(input_path: str, # returns metadata
        device: Optional[torch.device] = None) -> tuple[dict[str, torch.Tensor], dict[str, str]]:  
    
    with open(input_path,'rb') as f:
        tensors_dict = safetensors.load(f.read())
    with safetensors.safe_open(input_path, framework="pt", device=device) as f:
        metadata = f.metadata()
        if metadata is not None:
            metadata = dict(metadata)

    return tensors_dict, metadata

def update_safetensors_metadata(safetensors_path: str,
                                new_metadata: dict[str, str]) -> None:
    if new_metadata is None: return
    tensors_dict, metadata = load_safetensors_ex(safetensors_path)
    metadata = metadata or {}
    metadata.update(new_metadata)

    safetensors.save_file(tensors_dict, safetensors_path, metadata=metadata)

# recursively (through dicts and class instances) move all tensors to CPU
def move_tensors_to_cpu(instance: Union[dict, torch.Tensor, Any]) -> Any:
    if torch.is_tensor(instance):
        return instance.cpu()
    
    instance_dict = getattr(instance, "__dict__", None)
    if isinstance(instance_dict, dict):
        for key, value in instance_dict.items():
            instance_dict[key] = move_tensors_to_cpu(value)

    return instance

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

@torch.inference_mode()
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