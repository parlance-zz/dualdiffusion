import os
from typing import Union
import numpy as np
import torch
import cv2

from diffusers.schedulers import DPMSolverMultistepScheduler, DDIMScheduler, DPMSolverSDEScheduler
from diffusers.schedulers import EulerAncestralDiscreteScheduler, KDPM2AncestralDiscreteScheduler
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from unet2d_dual import UNet2DDualModel

def to_freq(x):
    x = x.permute(0, 2, 3, 1).contiguous().float()
    x = torch.fft.fft2(torch.view_as_complex(x), norm="ortho")
    x = torch.view_as_real(x)
    return x.permute(0, 3, 1, 2).contiguous()

def to_spatial(x):
    x = x.permute(0, 2, 3, 1).contiguous().float()
    x = torch.fft.ifft2(torch.view_as_complex(x), norm="ortho")
    x = torch.view_as_real(x)
    return x.permute(0, 3, 1, 2).contiguous()

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

class DualEmbeddingFormat:

    @staticmethod
    def get_sample_crop_width(model_params):
        return model_params["sample_raw_length"]
    
    @staticmethod
    def get_num_channels(model_params):
        freq_embedding_dim = model_params["freq_embedding_dim"]
        time_embedding_dim = model_params["time_embedding_dim"]
        channels = model_params["sample_raw_channels"] * 2
        return (channels + freq_embedding_dim + time_embedding_dim, channels)

    @staticmethod
    @torch.no_grad()
    def get_window(window_len):
        x = torch.arange(0, window_len, device="cuda") / (window_len - 1)
        return (1 + torch.cos(x * 2.*np.pi - np.pi)) * 0.5

    @staticmethod
    @torch.no_grad()
    def add_embeddings(samples, freq_embedding_dim, time_embedding_dim, format_hint="", pitch_augmentation=1., tempo_augmentation=1.):

        bsz = samples.shape[0]
        num_chunks = samples.shape[2]
        chunk_len = samples.shape[3]
        num_orders = freq_embedding_dim // 2

        #k = torch.pow(1.6180339887498948482, torch.arange(0, num_orders, device=freq_samples.device))
        k = torch.exp2(torch.arange(0, num_orders, device=samples.device))
        q = torch.arange(0, num_chunks, device=samples.device) + 0.5
        q = q.log2() # ???
        t = (torch.arange(0, chunk_len, device=samples.device) + 0.5) / chunk_len - 0.5
        
        embeddings = torch.exp(1j * k.view(-1, 1, 1) * q.view(1,-1, 1) * t.view(1, 1,-1))
        embeddings = torch.view_as_real(embeddings).permute(0, 3, 1, 2).contiguous()
        embeddings = embeddings.view(1, freq_embedding_dim, num_chunks, chunk_len)
        embeddings = embeddings.repeat(bsz, 1, 1, 1)

        embeddings = torch.cat((samples, embeddings), dim=1)
        return embeddings.type(samples.dtype)

    @staticmethod
    @torch.no_grad()
    def raw_to_sample(raw_samples, model_params, window=None, random_phase_offset=False):

        num_chunks = model_params["num_chunks"]
        sample_len = model_params["sample_raw_length"]
        half_sample_len = sample_len // 2
        chunk_len = half_sample_len // num_chunks
        bsz = raw_samples.shape[0]
        half_window_len = model_params["spatial_window_length"] // 2
        raw_samples = raw_samples.clone()

        if half_window_len > 0:
            if window is None:
                window = DualEmbeddingFormat.get_window(half_window_len*2).square_() # this might need to go back to just chunk_len
            raw_samples[:, :half_window_len]  *= window[:half_window_len]
            raw_samples[:, -half_window_len:] *= window[half_window_len:]

        fft = torch.fft.fft(raw_samples, norm="ortho")[:, :half_sample_len].view(bsz, num_chunks, chunk_len)
        fft[:, 0, 0] = 0 # remove DC component
        fft = torch.fft.fft(fft, norm="ortho")
        if random_phase_offset:
            fft *= torch.exp(2j*np.pi*torch.rand(1, device=fft.device))
        samples = torch.view_as_real(fft).permute(0, 3, 1, 2).contiguous()

        # unit variance
        if "sample_std" in model_params:
            samples /= model_params["sample_std"]
        else:
            samples /= samples.std(dim=(1, 2, 3), keepdim=True).clip(min=1e-8) 
        return samples, window

    @staticmethod
    @torch.no_grad()
    def sample_to_raw(samples, model_params):

        num_chunks = samples.shape[2]
        chunk_len = samples.shape[3]
        sample_len = num_chunks * chunk_len * 2
        half_sample_len = sample_len // 2
        bsz = samples.shape[0]
        samples = samples.clone()

        samples = torch.view_as_complex(samples.permute(0, 2, 3, 1).contiguous())
        samples = torch.fft.ifft(samples, norm="ortho")
        #samples[:, :, 0] = 0; samples[:, 1:, 0] -= samples[:, 1:, 0].mean(dim=1, keepdim=True) # remove annoying clicking due to lack of windowing
        samples = samples.view(bsz, half_sample_len)
        samples = torch.cat((samples, torch.zeros_like(samples)), dim=1)

        return torch.fft.ifft(samples, norm="ortho") * 2.

    @staticmethod
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

    @staticmethod
    def get_loss(sample, target, model_params, reduction="mean"):
        return torch.nn.functional.mse_loss(sample.float(), target.float(), reduction=reduction)
    
    @staticmethod
    def get_sample_shape(model_params, bsz=1, length=1):
        _, num_output_channels = DualNormalFormat.get_num_channels(model_params)
        num_chunks = model_params["num_chunks"]
        default_length = model_params["sample_raw_length"] // num_chunks // 2
        return (bsz, num_output_channels, num_chunks, default_length*length,)


class DualOverlappedFormat:

    @staticmethod
    def get_sample_crop_width(model_params):
        return model_params["sample_raw_length"]
    
    @staticmethod
    def get_num_channels(model_params):
        freq_embedding_dim = model_params["freq_embedding_dim"]
        channels = model_params["sample_raw_channels"] * 2
        return (channels + freq_embedding_dim, channels)
    
    @staticmethod
    @torch.no_grad()
    def get_window(window_len):
        x = torch.arange(0, window_len, device="cuda") / window_len #(window_len - 1)
        return (1 + torch.cos(x * 2.*np.pi - np.pi)) * 0.5

    @staticmethod
    @torch.no_grad()
    def raw_to_sample(raw_samples, model_params, window=None, random_phase_offset=False):
        num_chunks = model_params["num_chunks"]
        sample_len = model_params["sample_raw_length"]
        half_sample_len = sample_len // 2
        chunk_len = half_sample_len // num_chunks
        half_chunk_len = chunk_len // 2
        bsz = raw_samples.shape[0]

        fftshift = model_params.get("fftshift", True)

        if window is None:
            window = DualLogFormat.get_window(chunk_len)

        raw_samples[:, :half_chunk_len]  *= window[:half_chunk_len]
        raw_samples[:, -half_chunk_len:] *= window[half_chunk_len:]

        fft = torch.fft.fft(raw_samples, norm="ortho")[:, :half_sample_len + half_chunk_len]
        fft[:, half_sample_len:] = 0.

        slices_1 = fft[:, :half_sample_len].view(bsz, num_chunks, chunk_len)
        slices_2 = fft[:,  half_chunk_len:].view(bsz, num_chunks, chunk_len)

        samples = torch.cat((slices_1, slices_2), dim=2).view(bsz, num_chunks*2, chunk_len) * window
        if fftshift:
            samples = torch.fft.fft(torch.fft.fftshift(samples, dim=-1), norm="ortho")
        else:
            samples = torch.fft.fft(samples, norm="ortho")
        if random_phase_offset:
            samples *= torch.exp(2j*np.pi*torch.rand(1, device=samples.device))
        samples = torch.view_as_real(samples).permute(0, 3, 1, 2).contiguous()
        
        if "sample_std" in model_params:
            samples /= model_params["sample_std"]
        else:
            samples /= samples.std(dim=(1, 2, 3), keepdim=True).clip(min=1e-8)

        return samples, window

    @staticmethod
    @torch.no_grad()
    def sample_to_raw(samples, model_params):
        num_chunks = samples.shape[2] // 2
        chunk_len = samples.shape[3]
        half_chunk_len = chunk_len // 2
        sample_len = num_chunks * chunk_len * 2
        half_sample_len = sample_len // 2
        bsz = samples.shape[0]

        fftshift = model_params.get("fftshift", True)
        sample_std = model_params.get("sample_std", 1.)

        samples = samples.clone().permute(0, 2, 3, 1).contiguous() * sample_std

        if fftshift:
            # this mitigates clicking artifacts somehow
            #samples -= samples.mean(dim=(1,2), keepdim=True)
            samples -= samples.mean(dim=1, keepdim=True) * (-samples.std(dim=1, keepdim=True) * 20).exp()
            #samples -= samples.mean(dim=(1,3), keepdim=True) * (-samples.std(dim=(1,3), keepdim=True) * 16).exp()

            samples = torch.fft.fftshift(torch.fft.ifft(torch.view_as_complex(samples), norm="ortho"), dim=-1)
        else:
            samples = torch.fft.ifft(torch.view_as_complex(samples), norm="ortho")
            samples[:, :, 0] = 0 # mitigate clicking artifacts

        slices_1 = samples[:, 0::2, :]
        slices_2 = samples[:, 1::2, :]

        fft = torch.zeros((bsz, sample_len), dtype=torch.complex64, device="cuda")
        fft[:, :half_sample_len] = slices_1.reshape(bsz, -1)
        fft[:,  half_chunk_len:half_sample_len+half_chunk_len] += slices_2.reshape(bsz, -1)
        fft[:, half_sample_len:half_sample_len+half_chunk_len] = 0.
        
        return torch.fft.ifft(fft, norm="ortho") * 2.

    @staticmethod
    def get_loss(sample, target, model_params, reduction="mean"):
        return torch.nn.functional.mse_loss(sample.float(), target.float(), reduction=reduction)
    
    @staticmethod
    def get_sample_shape(model_params, bsz=1, length=1):
        _, num_output_channels = DualOverlappedFormat.get_num_channels(model_params)
        num_chunks = model_params["num_chunks"]
        default_length = model_params["sample_raw_length"] // num_chunks // 2
        return (bsz, num_output_channels, num_chunks*2, default_length*length,)
    
class DualLogFormat:

    @staticmethod
    def get_sample_crop_width(model_params):
        return model_params["sample_raw_length"]
    
    @staticmethod
    def get_num_channels(model_params):
        freq_embedding_dim = model_params["freq_embedding_dim"]
        channels = model_params["sample_raw_channels"] * 3
        return (channels + freq_embedding_dim, channels)
    
    @staticmethod
    @torch.no_grad()
    def get_window(window_len):
        x = torch.arange(0, window_len, device="cuda") / (window_len - 1)
        return (1 + torch.cos(x * 2.*np.pi - np.pi)) * 0.5

    """
    @staticmethod
    @torch.no_grad()
    def phase_integral(x):
        diff = torch.zeros_like(x)
        diff[:, :, 1:] = x[:, :, 1:] - x[:, :, :-1]
        diff[:, :, 0] = x[:, :, 0]
        diff[:, :, 0].cpu().numpy().tofile("./debug/debug_cumdiff.raw")
        diff[diff[:, :, :] < -np.pi] += 2.*np.pi
        diff[diff[:, :, :] >  np.pi] -= 2.*np.pi
        return torch.cumsum(diff[:, :, :], dim=-1)
    """

    @staticmethod
    @torch.no_grad()
    def raw_to_sample(raw_samples, model_params, window=None):
        num_chunks = model_params["num_chunks"]
        sample_len = model_params["sample_raw_length"]
        half_sample_len = sample_len // 2
        chunk_len = half_sample_len // num_chunks
        half_chunk_len = chunk_len // 2
        bsz = raw_samples.shape[0]

        ln_amplitude_floor = model_params["ln_amplitude_floor"]
        ln_amplitude_mean = model_params["ln_amplitude_mean"]
        ln_amplitude_std = model_params["ln_amplitude_std"]
        phase_integral_std = model_params["phase_integral_std"]

        if window is None:
            window = DualLogFormat.get_window(chunk_len)

        raw_samples[:, :half_chunk_len]  *= window[:half_chunk_len]
        raw_samples[:, -half_chunk_len:] *= window[half_chunk_len:]

        fft = torch.fft.fft(raw_samples, norm="ortho")[:, :half_sample_len + half_chunk_len]
        fft[:, half_sample_len:] = 0.

        slices_1 = fft[:, :half_sample_len].view(bsz, num_chunks, chunk_len)
        slices_2 = fft[:,  half_chunk_len:].view(bsz, num_chunks, chunk_len)

        samples = torch.cat((slices_1, slices_2), dim=2).view(bsz, num_chunks*2, chunk_len) * window
        samples_complex = torch.fft.fft(torch.fft.fftshift(samples, dim=-1), norm="ortho")
        samples_abs = samples_complex.abs().clip(min=1e-20)
        #samples_phase = torch.view_as_real(samples_complex / samples_abs)
        samples_phase = torch.view_as_real(samples_complex)
        samples_ln_amplitude = samples_abs.log().unsqueeze(-1).clip(min=ln_amplitude_floor)
        #samples_phase *= samples_ln_amplitude - ln_amplitude_floor
        samples = torch.cat((samples_ln_amplitude, samples_phase), dim=-1)

        samples[:, :, :, 0] = (samples[:, :, :, 0] - ln_amplitude_mean) / ln_amplitude_std
        samples[:, :, :, 1:] /= phase_integral_std

        samples = samples.permute(0, 3, 1, 2).contiguous()
        return samples, window

    @staticmethod
    @torch.no_grad()
    def sample_to_raw(samples, model_params):
        num_chunks = samples.shape[2] // 2
        chunk_len = samples.shape[3]
        half_chunk_len = chunk_len // 2
        sample_len = num_chunks * chunk_len * 2
        half_sample_len = sample_len // 2
        bsz = samples.shape[0]

        ln_amplitude_mean = model_params["ln_amplitude_mean"]
        ln_amplitude_std = model_params["ln_amplitude_std"]

        samples = samples.clone().permute(0, 2, 3, 1).contiguous()
        samples[:, :, :, 0] = samples[:, :, :, 0] * ln_amplitude_std + ln_amplitude_mean
        
        samples[:, :, :, 1:] /= (samples[:, :, :, 1].square() + samples[:, :, :, 2].square()).sqrt().unsqueeze(-1).clip(min=1e-10)
        samples = samples[:, :, :, 1:] * samples[:, :, :, 0].unsqueeze(-1).exp()
        
        samples = torch.fft.fftshift(torch.fft.ifft(torch.view_as_complex(samples), norm="ortho"), dim=-1)

        slices_1 = samples[:, 0::2, :]
        slices_2 = samples[:, 1::2, :]

        fft = torch.zeros((bsz, sample_len), dtype=torch.complex64, device="cuda")
        fft[:, :half_sample_len] = slices_1.reshape(bsz, -1)
        fft[:,  half_chunk_len:half_sample_len+half_chunk_len] += slices_2.reshape(bsz, -1)
        fft[:, half_sample_len:half_sample_len+half_chunk_len] = 0.
        
        return torch.fft.ifft(fft, norm="ortho") * 2.

    @staticmethod
    def get_loss(sample, target, model_params, reduction="mean"):
        
        ln_amplitude_mean = model_params["ln_amplitude_mean"]
        ln_amplitude_std = model_params["ln_amplitude_std"]

        sample = sample.clone().float()
        sample[:, 0, :, :] = sample[:, 0, :, :] * ln_amplitude_std + ln_amplitude_mean

        target = target.clone().float()
        target[:, 0, :, :] = target[:, 0, :, :] * ln_amplitude_std + ln_amplitude_mean

        sample_phase = sample[:, 1:, :, :] / (sample[:, 1, :, :].square() + sample[:, 2, :, :].square()).sqrt().unsqueeze(1).clip(min=1e-10)
        target_phase = target[:, 1:, :, :] / (target[:, 1, :, :].square() + target[:, 2, :, :].square()).sqrt().unsqueeze(1).clip(min=1e-10)
        sample = sample_phase * sample[:, 0, :, :].unsqueeze(1).exp() / 0.03526712161930658
        target = target_phase * target[:, 0, :, :].unsqueeze(1).exp() / 0.03526712161930658

        return torch.nn.functional.mse_loss(sample, target, reduction=reduction)
    
    @staticmethod
    def get_sample_shape(model_params, bsz=1, length=1):
        _, num_output_channels = DualLogFormat.get_num_channels(model_params)
        num_chunks = model_params["num_chunks"]
        default_length = model_params["sample_raw_length"] // num_chunks // 2
        return (bsz, num_output_channels, num_chunks*2, default_length*length,)
        
class DualNormalFormat:

    @staticmethod
    def get_sample_crop_width(model_params):
        return model_params["sample_raw_length"]
    
    @staticmethod
    def get_num_channels(model_params):
        freq_embedding_dim = model_params["freq_embedding_dim"]
        channels = model_params["sample_raw_channels"] * 2
        return (channels + freq_embedding_dim, channels)

    @staticmethod
    @torch.no_grad()
    def get_window(window_len):
        x = torch.arange(0, window_len, device="cuda") / (window_len - 1)
        return (1 + torch.cos(x * 2.*np.pi - np.pi)) * 0.5

    @staticmethod
    @torch.no_grad()
    def raw_to_sample(raw_samples, model_params, window=None, random_phase_offset=False):

        num_chunks = model_params["num_chunks"]
        sample_len = model_params["sample_raw_length"]
        half_sample_len = sample_len // 2
        chunk_len = half_sample_len // num_chunks
        bsz = raw_samples.shape[0]
        half_window_len = model_params["spatial_window_length"] // 2
        raw_samples = raw_samples.clone()

        if half_window_len > 0:
            if window is None:
                window = DualNormalFormat.get_window(half_window_len*2).square_() # this might need to go back to just chunk_len
            raw_samples[:, :half_window_len]  *= window[:half_window_len]
            raw_samples[:, -half_window_len:] *= window[half_window_len:]

        fft = torch.fft.fft(raw_samples, norm="ortho")[:, :half_sample_len].view(bsz, num_chunks, chunk_len)
        fft = torch.fft.fft(fft, norm="ortho")
        if random_phase_offset:
            fft *= torch.exp(2j*np.pi*torch.rand(1, device=fft.device))
        samples = torch.view_as_real(fft).permute(0, 3, 1, 2).contiguous()

        # unit variance
        if "sample_std" in model_params:
            samples /= model_params["sample_std"]
        else:
            samples /= samples.std(dim=(1, 2, 3), keepdim=True).clip(min=1e-8) 
        return samples, window

    @staticmethod
    @torch.no_grad()
    def sample_to_raw(samples, model_params):

        num_chunks = samples.shape[2]
        chunk_len = samples.shape[3]
        sample_len = num_chunks * chunk_len * 2
        half_sample_len = sample_len // 2
        bsz = samples.shape[0]
        samples = samples.clone()

        samples = torch.view_as_complex(samples.permute(0, 2, 3, 1).contiguous())
        samples = torch.fft.ifft(samples, norm="ortho")
        #samples[:, :, 0] = 0; samples[:, 1:, 0] -= samples[:, 1:, 0].mean(dim=1, keepdim=True) # remove annoying clicking due to lack of windowing
        samples = samples.view(bsz, half_sample_len)
        samples = torch.cat((samples, torch.zeros_like(samples)), dim=1)

        return torch.fft.ifft(samples, norm="ortho") * 2.

    @staticmethod
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

    @staticmethod
    def get_loss(sample, target, model_params, reduction="mean"):
        return torch.nn.functional.mse_loss(sample.float(), target.float(), reduction=reduction)
    
    @staticmethod
    def get_sample_shape(model_params, bsz=1, length=1):
        _, num_output_channels = DualNormalFormat.get_num_channels(model_params)
        num_chunks = model_params["num_chunks"]
        default_length = model_params["sample_raw_length"] // num_chunks // 2
        return (bsz, num_output_channels, num_chunks, default_length*length,)
        
class DualDiffusionPipeline(DiffusionPipeline):

    def __init__(
        self,
        unet: UNet2DDualModel,
        scheduler: DDIMScheduler,
        vae = None, # todo: insert class
        upscaler = None, # todo: insert class
        model_params: dict = None,
    ):
        super().__init__()

        modules = {"unet": unet, "scheduler": scheduler}
        if vae is not None: modules["vae"] = vae
        if upscaler is not None: modules["upscaler"] = upscaler
        self.register_modules(**modules)
        
        if model_params is not None:
            self.config["model_params"] = model_params
        else:
            model_params = self.config["model_params"]
            
        self.tiling_mode = False

        if "sample_format" not in model_params:
            model_params["sample_format"] = "normal"
        self.format = DualDiffusionPipeline.get_sample_format(model_params)

    @staticmethod
    def get_sample_format(model_params):
        sample_format = model_params["sample_format"]
        if sample_format == "ln":
            return DualLogFormat
        elif sample_format == "normal":
            return DualNormalFormat
        elif sample_format == "overlapped":
            return DualOverlappedFormat
        elif sample_format == "embedding":
            return DualEmbeddingFormat
        else:
            raise ValueError(f"Unknown sample format '{sample_format}'")
        
    @staticmethod
    def create_new(model_params, unet_params, vae_params=None, upscaler_params=None):
        
        unet = UNet2DDualModel(**unet_params)
        
        beta_schedule = model_params["beta_schedule"]
        if beta_schedule == "trained_betas":
            def alpha_bar_fn(t):
                a = 2 #4
                y = -1/(1 + a*t)**4 + 2/(1 + a*t)**2
                y -= -1/(1 + a)**4 + 2/(1 + a)**2
                return y
            
            trained_betas = []
            num_diffusion_timesteps = 1000
            for i in range(num_diffusion_timesteps):
                t1 = i / num_diffusion_timesteps
                t2 = (i + 1) / num_diffusion_timesteps
                trained_betas.append(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1))
        else:
            trained_betas = None

        scheduler = DDIMScheduler(clip_sample_range=20.,
                                  prediction_type=model_params["prediction_type"],
                                  beta_schedule=beta_schedule,
                                  beta_start=model_params["beta_start"],
                                  beta_end=model_params["beta_end"],
                                  trained_betas=trained_betas,
                                  rescale_betas_zero_snr=model_params["rescale_betas_zero_snr"],)
        
        snr = compute_snr(scheduler, scheduler.timesteps)
        debug_path = os.environ.get("DEBUG_PATH", None)
        if debug_path is not None:
            os.makedirs(debug_path, exist_ok=True)
            snr.log().cpu().numpy().tofile(os.path.join(debug_path, "debug_ln_snr.raw"))
            np.array(trained_betas).astype(np.float32).tofile(os.path.join(debug_path, "debug_betas.raw"))

        if vae_params is not None:
            #vae = VAE(**vae_params)
            raise NotImplementedError()
        else:
            vae = None

        if upscaler_params is not None:
            #upscaler = Upscaler(**upscaler_params)
            raise NotImplementedError()
        else:
            upscaler = None

        return DualDiffusionPipeline(unet, scheduler, vae=vae, upscaler=upscaler, model_params=model_params)

    @staticmethod
    @torch.no_grad()
    def add_embeddings(freq_samples, freq_embedding_dim, time_embedding_dim, format_hint="normal", pitch_augmentation=1., tempo_augmentation=1.):
        return DualEmbeddingFormat.add_embeddings(freq_samples,
                                                  freq_embedding_dim,
                                                  time_embedding_dim,
                                                  format_hint=format_hint,
                                                  pitch_augmentation=pitch_augmentation,
                                                  tempo_augmentation=tempo_augmentation)
    
    @torch.no_grad()
    def upscale(self, raw_sample):
        raise NotImplementedError()
    
    @torch.no_grad()
    def __call__(
        self,
        steps: int = 100,
        scheduler="dpms++",
        seed: Union[int, torch.Generator]=None,
        loops: int = 0,
        batch_size: int = 1,
        length: int = 1,
    ):
        if (steps <= 0) or (steps > 1000):
            raise ValueError(f"Steps must be between 1 and 1000, got {steps}")
        if loops < 0:
            raise ValueError(f"Loops must be greater than or equal to 0, got {loops}")
        if length <= 0:
            raise ValueError(f"Length must be greater than or equal to 1, got {length}")

        self.set_tiling_mode(loops > 0)

        prediction_type = self.scheduler.config["prediction_type"]
        beta_schedule = self.scheduler.config["beta_schedule"]
        if beta_schedule == "trained_betas":
            trained_betas = self.scheduler.config["trained_betas"]
        else:
            trained_betas = None

        scheduler = scheduler.lower().strip()
        if scheduler == "ddim":
            noise_scheduler = self.scheduler
        elif scheduler == "dpms++":
            noise_scheduler = DPMSolverMultistepScheduler(prediction_type=prediction_type,
                                                          solver_order=3,
                                                          beta_schedule=beta_schedule,
                                                          trained_betas=trained_betas)
        elif scheduler == "kdpm2_a":
            noise_scheduler = KDPM2AncestralDiscreteScheduler(prediction_type=prediction_type,
                                                              beta_schedule=beta_schedule,
                                                              trained_betas=trained_betas)
        elif scheduler == "euler_a":
            noise_scheduler = EulerAncestralDiscreteScheduler(prediction_type=prediction_type,
                                                              beta_schedule=beta_schedule,
                                                              trained_betas=trained_betas)
        elif scheduler == "dpms++_sde":
            if self.unet.dtype != torch.float32:
                raise ValueError("dpms++_sde scheduler requires float32 precision")
            
            noise_scheduler = DPMSolverSDEScheduler(prediction_type=prediction_type,
                                                    beta_schedule=beta_schedule,
                                                    trained_betas=trained_betas)
        else:
            raise ValueError(f"Unknown scheduler '{scheduler}'")
        noise_scheduler.set_timesteps(steps)
        timesteps = noise_scheduler.timesteps

        if isinstance(seed, int):
            if seed == 0: seed = np.random.randint(100000,999999)
            generator = torch.Generator(device=self.device).manual_seed(seed)
        elif isinstance(seed, torch.Generator):
            generator = seed

        model_params = self.config["model_params"]
        
        if getattr(self, "vae", None) is None:
            sample_shape = self.format.get_sample_shape(model_params, bsz=batch_size, length=length)
        else:
            raise NotImplementedError()
        print(f"Sample shape: {sample_shape}")

        sample = torch.randn(sample_shape, device=self.device, dtype=self.unet.dtype, generator=generator)
        sample *= noise_scheduler.init_noise_sigma

        freq_embedding_dim = model_params["freq_embedding_dim"]
        time_embedding_dim = model_params["time_embedding_dim"]

        for _, t in enumerate(self.progress_bar(timesteps)):
            
            model_input = sample
            if freq_embedding_dim > 0 or time_embedding_dim > 0:
                model_input = DualDiffusionPipeline.add_embeddings(model_input,
                                                                   freq_embedding_dim,
                                                                   time_embedding_dim,
                                                                   format_hint=model_params["sample_format"])
            model_input = noise_scheduler.scale_model_input(model_input, t)
            model_output = self.unet(model_input, t).sample

            #model_output = model_output[:, :, :, :sample.shape[3]]

            scheduler_args = {
                "model_output": model_output,
                "timestep": t,
                "sample": sample,
            }
            if scheduler != "dpms++_sde":
                scheduler_args["generator"] = generator
            sample = noise_scheduler.step(**scheduler_args)["prev_sample"]
        
        sample = sample.float()

        debug_path = os.environ.get("DEBUG_PATH", None)
        if debug_path is not None:
            print("Sample std: ", sample.std(dim=(1,2,3)).item())
            os.makedirs(debug_path, exist_ok=True)
            sample.cpu().numpy().tofile(os.path.join(debug_path, "debug_sample.raw"))
        
        if getattr(self, "vae", None) is None:
            raw_sample = self.format.sample_to_raw(sample, model_params).real
        else:
            raw_sample = self.vae.decode(sample / self.vae.config.scaling_factor).sample
            raise NotImplementedError()
        
        if getattr(self, "upscaler", None) is not None:
            raw_sample = self.upscale(raw_sample)

        raw_sample *= 0.18215 / raw_sample.std(dim=1, keepdim=True).clip(min=1e-5)
        if loops > 0: raw_sample = raw_sample.repeat(1, loops+1)
        return raw_sample
    
    def set_module_tiling(self, module):
        F, _pair = torch.nn.functional, torch.nn.modules.utils._pair

        padding_modeX = "circular"
        padding_modeY = "constant"

        rprt = module._reversed_padding_repeated_twice
        paddingX = (rprt[0], rprt[1], 0, 0)
        paddingY = (0, 0, rprt[2], rprt[3])

        def _conv_forward(self, input, weight, bias):
            padded = F.pad(input, paddingX, mode=padding_modeX)
            padded = F.pad(padded, paddingY, mode=padding_modeY)
            return F.conv2d(
                padded, weight, bias, self.stride, _pair(0), self.dilation, self.groups
            )

        module._conv_forward = _conv_forward.__get__(module)

    def remove_module_tiling(self, module):
        try:
            del module._conv_forward
        except AttributeError:
            pass

    def set_tiling_mode(self, tiling: bool):

        if self.tiling_mode == tiling:
            return
        
        modules = [self.unet]
        if getattr(self, "vae", None) is not None:
            modules.append(self.vae)
            
        modules = filter(lambda module: isinstance(module, torch.nn.Module), modules)

        for module in modules:
            for submodule in module.modules():
                if isinstance(submodule, torch.nn.Conv2d | torch.nn.ConvTranspose2d):
                    if isinstance(submodule, torch.nn.ConvTranspose2d):
                        raise NotImplementedError(
                            "Assymetric tiling doesn't support this module"
                        )

                    if tiling is False:
                        self.remove_module_tiling(submodule)
                    else:
                        self.set_module_tiling(submodule)

        self.tiling_mode = tiling