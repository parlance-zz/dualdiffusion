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

from dataclasses import dataclass
from typing import Optional, Union, Any
from json import dumps as json_dumps
from datetime import datetime
from contextlib import ContextDecorator
import os
import logging
import shutil
import subprocess

import numpy as np
import torch
import torchaudio
import cv2
import safetensors.torch as safetensors
import mutagen
import mutagen.flac
import pyloudnorm
import librosa

from utils.roseus_colormap import ROSEUS_COLORMAP


@dataclass
class AudioInfo:
    sample_rate: int
    channels: int
    frames: int
    duration: float

class TF32_Disabled(ContextDecorator): # disables reduced precision tensor cores inside the context
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

def get_available_torch_devices() -> list[str]:
    available_devices = ["cpu"]
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            available_devices.append(f"cuda{i}" if i > 0 else "cuda")

    if torch.backends.mps.is_available():
        available_devices.append("mps")

    return available_devices

def torch_dtype(dtype: Union[str, torch.dtype]) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    elif isinstance(dtype, str):
        return getattr(torch, dtype)
    else:
        raise ValueError(f"Unsupported dtype type: {dtype} ({type(dtype)})")

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

def get_no_clobber_filepath(filepath: str) -> str:

    directory, filename = os.path.split(filepath)
    name, ext = os.path.splitext(filename)
    
    unique_filepath = filepath
    counter = 0
    
    while os.path.exists(unique_filepath):
        new_filename = f"{name}_{counter}{ext}"
        unique_filepath = os.path.join(directory, new_filename)
        counter += 1
    
    return unique_filepath

def find_files(directory: str, name_pattern: str = "*") -> list[str]:
    result = subprocess.run(
        ["find", directory, "-iname", name_pattern, "-type", "f"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True
    )
    #result.check_returncode()
    return result.stdout.splitlines()

@torch.inference_mode()
def normalize_lufs(raw_samples: torch.Tensor,
                   sample_rate: int,
                   target_lufs: float = -15.,
                   return_old_lufs: bool = False) -> torch.Tensor:
    
    if raw_samples.ndim != 2:
        raise ValueError(f"Invalid shape for normalize_lufs: {raw_samples.shape}")
    
    meter = pyloudnorm.Meter(sample_rate)
    old_lufs = meter.integrated_loudness(raw_samples.mean(dim=0, keepdim=True).T.cpu().numpy())
    normalized_raw_samples = raw_samples * (10. ** ((target_lufs - old_lufs) / 20.0))

    if return_old_lufs:
        return normalized_raw_samples, old_lufs
    else:
        return normalized_raw_samples

# get the number of clipped samples in a raw audio tensor
# clipping is defined as at least 2 consecutive samples with an absolute value >= 1.0 - eps
@torch.inference_mode()
def get_num_clipped_samples(raw_samples: torch.Tensor, eps: float = 2e-2) -> int:
    clips = (raw_samples.abs() >= (1.0 - eps)).float()
    return int((clips[..., :-1] * clips[..., 1:]).sum().item())

def save_audio(raw_samples: torch.Tensor,
               sample_rate: int,
               output_path: str,
               target_lufs: float = -15.,
               metadata: Optional[dict] = None,
               no_clobber: bool = False,
               copy_on_write: bool = False) -> None:

    if raw_samples.ndim == 3:
        raw_samples = raw_samples.squeeze(0)
    
    if raw_samples.ndim != 2:
        raise ValueError(f"Invalid shape for save_audio: {raw_samples.shape}")
    
    raw_samples = raw_samples.detach().real.float()
    if target_lufs is not None:
        raw_samples = normalize_lufs(raw_samples, sample_rate, target_lufs)
    raw_samples = raw_samples.clip(min=-1, max=1)

    directory = os.path.dirname(output_path)
    os.makedirs(directory, exist_ok=True)
    if no_clobber == True:
        output_path = get_no_clobber_filepath(output_path)
    
    audio_format = os.path.splitext(output_path)[1].lower()[1:]
    bits_per_sample = 16 if audio_format in ["wav", "flac"] else None
    
    if copy_on_write == True:
        tmp_path = f"{output_path}.tmp"
        try:
            torchaudio.save(tmp_path, raw_samples.cpu(), sample_rate,
                        format=audio_format, bits_per_sample=bits_per_sample)

            if metadata is not None:
                update_audio_metadata(tmp_path, metadata)

            os.rename(tmp_path, output_path)
            if os.path.isfile(tmp_path):
                os.remove(tmp_path)

        except Exception as e:
            try:
                if os.path.isfile(tmp_path):
                    os.remove(tmp_path)
            except: pass
            raise e
    else:
        torchaudio.save(output_path, raw_samples.cpu(), sample_rate,
                    format=audio_format, bits_per_sample=bits_per_sample)
        
        if metadata is not None:
            update_audio_metadata(output_path, metadata)

def load_audio(input_path: str,
               start: int = 0, count: int = -1,
               return_sample_rate: bool = False,
               device: Optional[torch.device] = None,
               return_start: bool = False,
               force_stereo: bool = True,
               sample_rate: Optional[int] = None) -> Union[torch.Tensor, tuple[torch.Tensor, int]]:

    if start < 0:
        if count < 0:
            raise ValueError(f"If start < 0 count cannot be < 0")
        sample_len = get_audio_info(input_path).frames
        start = np.random.randint(0, sample_len - count + 1)

    #tensor, sample_rate = torchaudio.load(input_path, frame_offset=start, num_frames=count)
    if sample_rate is None:
        sample_rate = get_audio_info(input_path).sample_rate
    data, returned_sample_rate = librosa.load(input_path, sr=None, offset=start/sample_rate, duration=count/sample_rate, mono=False)
    if returned_sample_rate != sample_rate:
        raise ValueError(f"Given sample rate ({sample_rate}) does not match file sample rate ({returned_sample_rate}) in {input_path}")
    
    tensor = torch.from_numpy(data).float()
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
        
    assert tensor.shape[-1] == count or count == -1
    assert tensor.ndim == 2

    if tensor.shape[0] == 1 and force_stereo == True:
        tensor = tensor.repeat(2, 1)
        
    return_vals = (tensor.to(device=device),)
    if return_sample_rate:
        return_vals += (sample_rate,)
    if return_start:
        return_vals += (start,)
    if len(return_vals) == 1:
        return return_vals[0]
    else:
        return return_vals

def update_audio_metadata(audio_path: str, metadata: Optional[dict] = None,
        rating: Optional[int] = None, clear_clap_fields: bool = False, copy_on_write: bool = False) -> None:
    
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
                audio_file[key] = metadata[key] if type(metadata[key]) not in [bool, int, float] else str(metadata[key])
            else: audio_file[f"TXXX:{key}"] = mutagen.id3.TXXX(encoding=3, desc=key, text=metadata[key])
        
        # this works for windows explorer/media player but not VLC
        if rating is not None and audio_format == ".mp3":
            rating = int(min(max(rating, 0), 5))
            rating = [0, 1, 64, 128, 196, 255][rating] # whoever came up with this scale is an idiot
            audio_file["POPM:Windows Media Player 9 Series"] = mutagen.id3.POPM(
                email="Windows Media Player 9 Series", rating=rating)

        if copy_on_write == True:
            tmp_path = f"{audio_path}.tmp"
            try:
                shutil.copy2(audio_path, tmp_path)
                audio_file.save(tmp_path)
                os.rename(tmp_path, audio_path)
                
                if os.path.isfile(tmp_path):
                    os.remove(tmp_path)

            except Exception as e:
                try:
                    if os.path.isfile(tmp_path):
                        os.remove(tmp_path)
                except: pass
                raise e
        else:
            audio_file.save()

def get_audio_metadata(audio_path: str) -> dict[str, list[str]]:
    audio_format = os.path.splitext(audio_path)[1].lower()
    if audio_format == ".flac":
        audio_file = mutagen.flac.FLAC(audio_path)
    else:
        audio_file = mutagen.File(audio_path)
    
    return {key: audio_file[key] for key in audio_file.keys()}

def get_audio_info(input_path: str) -> AudioInfo:
    audio_info = mutagen.flac.FLAC(input_path)
    return AudioInfo(
        sample_rate=int(audio_info.info.sample_rate),
        channels=int(audio_info.info.channels),
        frames=int(audio_info.info.total_samples),
        duration=float(audio_info.info.total_samples / audio_info.info.sample_rate),
    )

def save_safetensors(tensors_dict: dict[str, torch.Tensor], output_path: str,
        metadata: Optional[dict[str, str]] = None, copy_on_write: bool = False) -> None:
    
    for key in tensors_dict:
        val = tensors_dict[key]
        if torch.is_tensor(val):
            val = val.detach().resolve_conj().contiguous().cpu()
        else:
            val = torch.tensor(val)
        tensors_dict[key] = val

    directory = os.path.dirname(output_path)
    os.makedirs(directory, exist_ok=True)

    if copy_on_write == True:
        tmp_path = f"{output_path}.tmp"
        try:
            safetensors.save_file(tensors_dict, tmp_path, metadata=metadata)

            os.rename(tmp_path, output_path)
            if os.path.isfile(tmp_path):
                os.remove(tmp_path)

        except Exception as e:
            try:
                if os.path.isfile(tmp_path):
                    os.remove(tmp_path)
            except: pass
            raise e
    else:
        safetensors.save_file(tensors_dict, output_path, metadata=metadata)

# caution: does not actually load data from disk until tensors are copied (file handle remains open until disposed)
def load_safetensors(input_path: str, device: Optional[torch.device] = None) -> dict[str, torch.Tensor]:
    return safetensors.load_file(input_path, device=device)

def load_safetensors_ex(input_path: str, # loads all data from disk, returns metadata
        device: Optional[torch.device] = None) -> tuple[dict[str, torch.Tensor], dict[str, str]]:  
    
    with open(input_path,'rb') as f:
        tensors_dict = safetensors.load(f.read())
    with safetensors.safe_open(input_path, framework="pt", device=device) as f:
        metadata = f.metadata()
        if metadata is not None:
            metadata = dict(metadata)

    return tensors_dict, metadata

def get_safetensors_metadata(input_path: str) -> dict[str, str]:
    with safetensors.safe_open(input_path, framework="pt") as f:
        metadata = f.metadata()
        if metadata is not None:
            metadata = dict(metadata)
    
    return metadata

def update_safetensors_metadata(safetensors_path: str, new_metadata: dict[str, str],
                                copy_on_write: bool = False) -> None:
    
    if new_metadata is None or len(new_metadata) == 0:
        return

    tensors_dict, metadata = load_safetensors_ex(safetensors_path)
    metadata = metadata or {}
    metadata.update(new_metadata)

    save_safetensors(tensors_dict, safetensors_path, metadata, copy_on_write=copy_on_write)

def save_tensor_raw(tensor: torch.Tensor, output_path: str, copy_on_write: bool = False) -> None:

    directory = os.path.dirname(output_path)
    os.makedirs(directory, exist_ok=True)

    if tensor.dtype in [torch.float16, torch.bfloat16]:
        tensor = tensor.float()
    elif tensor.dtype == torch.complex32:
        tensor = tensor.to(torch.complex64)
    
    np_array = tensor.detach().resolve_conj().cpu().numpy()

    if copy_on_write == True:
        tmp_path = f"{output_path}.tmp"
        try:
            np_array.tofile(tmp_path)

            os.rename(tmp_path, output_path)
            if os.path.isfile(tmp_path):
                os.remove(tmp_path)

        except Exception as e:
            try:
                if os.path.isfile(tmp_path):
                    os.remove(tmp_path)
            except: pass
            raise e
    else:
        np_array.tofile(output_path)

# calculate the size in bytes of all tensors in any number of nested dicts
def get_tensor_dict_size(instance: Union[dict, torch.Tensor]):
    if torch.is_tensor(instance):
        return instance.element_size() * instance.numel()
    
    size = 0
    instance_dict = getattr(instance, "__dict__", None)
    if isinstance(instance_dict, dict):
        for value in instance_dict.values():
            size += get_tensor_dict_size(value)

    return size

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
def quantize_tensor(x: torch.Tensor, levels: int) -> torch.Tensor:
    reduction_dims = tuple(range(1, x.ndim)) if x.ndim > 1 else (0,)

    min_val = x.amin(dim=reduction_dims, keepdim=True)
    max_val = x.amax(dim=reduction_dims, keepdim=True)
    scale = (max_val - min_val) / (levels - 1)

    quantized = ((x - min_val) / scale).round().clamp(0, levels - 1)
    offset_and_range = torch.stack((min_val.flatten(), scale.flatten()), dim=-1)
    return quantized, offset_and_range

@torch.inference_mode()
def dequantize_tensor(x: torch.Tensor, offset_and_range: torch.Tensor) -> torch.Tensor:
    view_dims = (-1,) + ((1,) * (x.ndim-1) if x.ndim > 1 else ())
    min_val, scale = offset_and_range.unbind(-1)
    return x * scale.view(view_dims) + min_val.view(view_dims)

def tensor_5d_to_4d(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.shape[0], x.shape[1]*x.shape[2], x.shape[3], x.shape[4])

def tensor_4d_to_5d(x: torch.Tensor, num_channels: int) -> torch.Tensor:
    return x.reshape(x.shape[0], num_channels, -1, x.shape[2], x.shape[3])

def tensor_info_str(x: torch.Tensor) -> str:
    info_str = f"shape: {list(x.shape)}  device: {x.device}  dtype: {x.dtype}"
    info_str += f"  mean: {x.mean().item():.4f}  std: {x.std().item():.4f}  norm: {x.square().mean().sqrt().item():.4f}"
    return info_str

@torch.inference_mode()
def tensor_to_img(x: torch.Tensor,
                  recenter: bool = True,
                  rescale: bool = True,
                  flip_x: bool = False,
                  flip_y: bool = False,
                  colormap: bool = False,
                  channel_order: Optional[tuple[int, int, int]] = None) -> np.ndarray:
    
    x = x.clone().detach().real.float().resolve_conj().cpu()
    while x.ndim < 4: x.unsqueeze_(0)
    if x.ndim == 5: x = x.view(x.shape[0], x.shape[1], x.shape[2] * x.shape[3], x.shape[4])
    x = x.permute(0, 2, 3, 1).contiguous().view(x.shape[0] * x.shape[2], x.shape[3], x.shape[1])

    if recenter: x -= x.amin(dim=(-3,-2,-1), keepdim=True)
    if rescale:  x /= x.amax(dim=(-3,-2,-1), keepdim=True).clip(min=1e-16)

    if channel_order is not None:
        _x = x.clone()
        for i in range(3):
            x[..., i] = _x[..., channel_order[i]]

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