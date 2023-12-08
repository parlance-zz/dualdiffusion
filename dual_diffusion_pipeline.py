import os
from typing import Union
import numpy as np
import torch
import torch.nn.functional as F
import cv2

from diffusers.schedulers import DPMSolverMultistepScheduler, DDIMScheduler, DPMSolverSDEScheduler
from diffusers.schedulers import EulerAncestralDiscreteScheduler, KDPM2AncestralDiscreteScheduler
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from unet2d_dual import UNet2DDualModel
from autoencoder_kl_dual import AutoencoderKLDual

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

class DualMCLTFormat:

    @staticmethod
    def get_sample_crop_width(model_params):
        block_width = model_params["num_chunks"] * 2
        return model_params["sample_raw_length"] + block_width
    
    @staticmethod
    def get_num_channels(model_params):
        channels = model_params["sample_raw_channels"] * 2
        return (channels, channels + 2)

    @staticmethod
    @torch.no_grad()
    def raw_to_sample(raw_samples, model_params, window=None, random_phase_offset=False):
        
        num_chunks = model_params["num_chunks"]
        block_width = num_chunks * 2

        samples = mdct(raw_samples, block_width, window_degree=1)[..., 1:-2, :]
        samples = samples.permute(0, 2, 1)
        
        samples = torch.view_as_real(samples).permute(0, 3, 1, 2).contiguous()
        samples = samples / samples.square().sum(dim=(1,2,3), keepdim=True).mean(dim=(1,2,3), keepdim=True).sqrt().clip(min=1e-8)

        return samples, window

    @staticmethod
    def sample_to_raw(samples, model_params):
        
        samples_abs = samples[:, 0, :, :].permute(0, 2, 1).contiguous().sigmoid()
        samples_phase = samples[:, 1:3, :, :].permute(0, 3, 2, 1).contiguous()
        samples_phase = samples_phase / samples_phase.square().sum(dim=(1,2,3), keepdim=True).mean(dim=(1,2,3), keepdim=True).sqrt().clip(min=1e-8)
        samples_phase = torch.view_as_complex(samples_phase)
        samples_waveform = samples_abs * samples_phase

        samples_noise_abs = samples[:, 3, :, :].permute(0, 2, 1).contiguous().sigmoid()
        samples_noise_phase = torch.randn_like(samples_noise_abs)
        samples_noise = samples_noise_abs * samples_noise_phase * 0.5
        
        return imdct(samples_waveform, window_degree=1).real + imdct(samples_noise, window_degree=2).real

    @staticmethod
    def get_sample_shape(model_params, bsz=1, length=1):
        _, num_output_channels = DualMCLTFormat.get_num_channels(model_params)

        crop_width = DualMCLTFormat.get_sample_crop_width(model_params)
        num_chunks = model_params["num_chunks"]
        chunk_len = crop_width // num_chunks - 2

        return (bsz, num_output_channels, num_chunks, chunk_len*length,)

class DualMDCTFormat:

    @staticmethod
    def get_sample_crop_width(model_params):
        block_width = model_params["num_chunks"] * 2
        return model_params["sample_raw_length"] - block_width // 2
    
    @staticmethod
    def get_num_channels(model_params):
        channels = model_params["sample_raw_channels"] * (1 + int(model_params.get("complex", False)))
        return (channels, channels)

    @staticmethod
    @torch.no_grad()
    def raw_to_sample(raw_samples, model_params, window=None, random_phase_offset=False):
        
        num_chunks = model_params["num_chunks"]
        block_width = num_chunks * 2
        u = model_params.get("u", None)
        complex = model_params.get("complex", False)

        samples = mdct(raw_samples, block_width)

        if random_phase_offset:
            samples *= torch.exp(2j*torch.pi*torch.rand(1, device=samples.device))

        if not complex:
            samples = samples.real
        if u is not None:
            samples = to_ulaw(samples, u=u)

        if complex:
            samples = torch.view_as_real(samples).permute(0, 3, 2, 1).contiguous()
        else:
            samples = samples.permute(0, 2, 1).contiguous().unsqueeze(1)
        
        if "sample_std" in model_params:
            samples /= model_params["sample_std"]
        else:
            samples /= samples.std(dim=(1, 2, 3), keepdim=True).clip(min=1e-8)

        return samples, window

    @staticmethod
    def sample_to_raw(samples, model_params):
        
        sample_std = model_params.get("sample_std", 1)
        if sample_std != 1:
            samples = samples * sample_std

        complex = model_params.get("complex", False)
        if complex:
            samples = torch.view_as_complex(samples.permute(0, 3, 2, 1).contiguous())
        else:
            samples = samples.squeeze(1).permute(0, 2, 1).contiguous()

        u = model_params.get("u", None)
        if u is not None:
            samples = from_ulaw(samples, u=u)
        
        return imdct(samples).real

    @staticmethod
    def get_sample_shape(model_params, bsz=1, length=1):
        _, num_output_channels = DualMDCTFormat.get_num_channels(model_params)

        crop_width = DualMDCTFormat.get_sample_crop_width(model_params)
        num_chunks = model_params["num_chunks"]
        chunk_len = crop_width // num_chunks - 1

        return (bsz, num_output_channels, num_chunks, chunk_len*length,)

class DualOverlappedFormat:

    @staticmethod
    def get_sample_crop_width(model_params):
        return model_params["sample_raw_length"] - int(model_params.get("rfft", False))
    
    @staticmethod
    def get_num_channels(model_params):
        channels = model_params["sample_raw_channels"] * 2
        return (channels, channels)
    
    @staticmethod
    @torch.no_grad()
    def get_window(window_len, device):
        x = torch.arange(0, window_len, device=device) / (window_len - 1)
        return (1 + torch.cos(x * 2.*torch.pi - torch.pi)) * 0.5

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
        ifft = model_params.get("ifft", False)
        rfft = model_params.get("rfft", False)

        if window is None:
            window = DualOverlappedFormat.get_window(chunk_len, device=raw_samples.device)

        raw_samples[:, :half_chunk_len]  *= window[:half_chunk_len]
        raw_samples[:, -half_chunk_len:] *= window[half_chunk_len:]

        if rfft:
            fft = torch.fft.rfft(raw_samples, norm="ortho")
            fft = torch.cat((fft, torch.zeros((bsz, half_chunk_len), dtype=torch.complex64, device=fft.device)), dim=1)
        else:
            fft = torch.fft.fft(raw_samples, norm="ortho")[:, :half_sample_len + half_chunk_len]
            fft[:, half_sample_len:] = 0.
            fft[:, 0] /= 2 # ?

        slices_1 = fft[:, :half_sample_len].view(bsz, num_chunks, chunk_len)
        slices_2 = fft[:,  half_chunk_len:].view(bsz, num_chunks, chunk_len)

        samples = torch.cat((slices_1, slices_2), dim=2).view(bsz, num_chunks*2, chunk_len)# * window
        if fftshift:
            if ifft:
                samples = torch.fft.ifft(torch.fft.fftshift(samples, dim=-1), norm="ortho")
            else:
                samples = torch.fft.fft(torch.fft.fftshift(samples, dim=-1), norm="ortho")
        else:
            if ifft:
                samples = torch.fft.ifft(samples, norm="ortho")
                samples[:, 0, :] /= 2 # ?
            else:
                samples = torch.fft.fft(samples, norm="ortho")
                samples[:, 0, :] /= 2 # ?

        samples -= samples.mean(dim=(2,), keepdim=True)

        if random_phase_offset:
            samples *= torch.exp(2j*torch.pi*torch.rand(1, device=samples.device))
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
        ifft = model_params.get("ifft", False)
        rfft = model_params.get("rfft", False)
        sample_std = model_params.get("sample_std", 1.)

        window = DualOverlappedFormat.get_window(chunk_len, device=samples.device)

        samples = samples - samples.mean(dim=(3,), keepdim=True)
        #samples = samples - samples.mean(dim=(2,3,), keepdim=True) #?
        #samples = samples - samples.mean(dim=(2,), keepdim=True) #?

        samples = samples.permute(0, 2, 3, 1).contiguous() * sample_std

        if fftshift:
            # this mitigates clicking artifacts somehow
            #samples -= samples.mean(dim=(1,2), keepdim=True)
            samples -= samples.mean(dim=1, keepdim=True) * (-samples.std(dim=1, keepdim=True) * 20).exp()
            #samples -= samples.mean(dim=(1,3), keepdim=True) * (-samples.std(dim=(1,3), keepdim=True) * 16).exp()

            if ifft:
                samples = torch.fft.fftshift(torch.fft.fft(torch.view_as_complex(samples), norm="ortho"), dim=-1)
            else:
                samples = torch.fft.fftshift(torch.fft.ifft(torch.view_as_complex(samples), norm="ortho"), dim=-1)
        else:
            if ifft:
                samples = torch.fft.fft(torch.view_as_complex(samples), norm="ortho")
            else:
                samples = torch.fft.ifft(torch.view_as_complex(samples), norm="ortho")
            #samples[:, :, 0] = 0 # mitigate clicking artifacts

        slices_1 = samples[:, 0::2, :] * window #!
        slices_2 = samples[:, 1::2, :] * window

        fft = torch.zeros((bsz, sample_len), dtype=torch.complex64, device=samples.device)
        fft[:, :half_sample_len] = slices_1.reshape(bsz, -1)
        fft[:,  half_chunk_len:half_sample_len+half_chunk_len] += slices_2.reshape(bsz, -1)
        fft[:, half_sample_len:half_sample_len+half_chunk_len] = 0.
        
        if rfft:
            return torch.fft.irfft(fft, sample_len - 1 , norm="ortho")
        else:
            return torch.fft.ifft(fft, norm="ortho") * 2.

    @staticmethod
    def get_sample_shape(model_params, bsz=1, length=1):
        _, num_output_channels = DualOverlappedFormat.get_num_channels(model_params)
        num_chunks = model_params["num_chunks"]
        default_length = model_params["sample_raw_length"] // num_chunks // 2
        return (bsz, num_output_channels, num_chunks*2, default_length*length,)   

class DualNormalFormat:

    @staticmethod
    def get_sample_crop_width(model_params):
        rfft = model_params.get("rfft", False)
        if rfft:
            return model_params["sample_raw_length"] - 1
        else:
            return model_params["sample_raw_length"]
    
    @staticmethod
    def get_num_channels(model_params):
        channels = model_params["sample_raw_channels"] * 2
        return (channels, channels)

    @staticmethod
    @torch.no_grad()
    def get_window(window_len, device):
        x = torch.arange(0, window_len, device=device) / (window_len - 1)
        return (1 + torch.cos(x * 2.*torch.pi - torch.pi)) * 0.5

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

        ifft = model_params.get("ifft", False)
        rfft = model_params.get("rfft", False)

        if half_window_len > 0:
            if window is None:
                window = DualNormalFormat.get_window(half_window_len*2, device=raw_samples.device).square_() # this might need to go back to just chunk_len
            raw_samples[:, :half_window_len]  *= window[:half_window_len]
            raw_samples[:, -half_window_len:] *= window[half_window_len:]

        if rfft:
            fft = torch.fft.rfft(raw_samples, norm="ortho")
        else:
            fft = torch.fft.fft(raw_samples, norm="ortho")
            fft[:, 0] /= 2 
            fft = fft[:, :half_sample_len]
            #fft = sfft(raw_samples)


        fft = fft.view(bsz, num_chunks, chunk_len)

        if ifft:
            fft = torch.fft.ifft(fft, norm="ortho")
        else:
            fft = torch.fft.fft(fft, norm="ortho")

        if random_phase_offset:
            fft *= torch.exp(2j*torch.pi*torch.rand(1, device=fft.device))
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

        ifft = model_params.get("ifft", False)
        rfft = model_params.get("rfft", False)

        samples = torch.view_as_complex(samples.permute(0, 2, 3, 1).contiguous())
        if ifft:
            samples = torch.fft.fft(samples, norm="ortho")
        else:
            samples = torch.fft.ifft(samples, norm="ortho")

        #samples[:, :, 0] = 0; samples[:, 1:, 0] -= samples[:, 1:, 0].mean(dim=1, keepdim=True) # remove annoying clicking due to lack of windowing
        #samples -= samples.mean(dim=(1, 2), keepdim=True)

        samples = samples.view(bsz, half_sample_len)
        
        if rfft:
            samples = torch.cat((samples, torch.zeros_like(samples)), dim=1)
            return torch.fft.irfft(samples, samples.shape[1]-1, norm="ortho").type(samples.dtype)
        else:
            samples = torch.cat((samples, torch.zeros_like(samples)), dim=1)
            return torch.fft.ifft(samples, norm="ortho") * 2.
            #return isfft(samples)

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
    def get_sample_shape(model_params, bsz=1, length=1):
        _, num_output_channels = DualNormalFormat.get_num_channels(model_params)
        num_chunks = model_params["num_chunks"]
        default_length = model_params["sample_raw_length"] // num_chunks // 2
        return (bsz, num_output_channels, num_chunks, default_length*length,)

class DualMultiscaleSpectralLoss2:

    @torch.no_grad()
    def __init__(self, model_params):
        self.model_params = model_params

        loss_params = model_params["multiscale_spectral_loss"]
        self.block_widths = loss_params["block_widths"]
        self.u = loss_params["u"]

        if "block_offsets" in loss_params:
            self.block_offsets = loss_params["block_offsets"]
        else:
            self.block_offsets = [0, 0.25]

        #self.block_weights = torch.arange(1, len(self.block_widths) + 1, dtype=torch.float32) #.sqrt()
        #self.block_weights /= self.block_weights.mean()

    def __call__(self, sample, target):
        
        sample_block_width = self.model_params["num_chunks"] * 2
        target = target[:, sample_block_width // 2:-sample_block_width]
        assert(sample.shape == target.shape)

        #if self.block_weights.device != sample.device:
        #    self.block_weights = self.block_weights.to(sample.device)
        #if self.block_weights.dtype != sample.dtype:
        #    self.block_weights = self.block_weights.to(sample.dtype)

        loss = torch.zeros(1, device=sample.device)

        for block_num, block_width in enumerate(self.block_widths):
                
            for block_offset in self.block_offsets:

                offset = int(block_offset * block_width)

                sample_fft_abs = mdct(sample[:, offset:], block_width, window_degree=2)[:, 3:-3, :].abs()
                target_fft_abs = mdct(target[:, offset:], block_width, window_degree=2)[:, 3:-3, :].abs()

                sample_fft_abs = sample_fft_abs / sample_fft_abs.amax(dim=(1,2), keepdim=True)
                target_fft_abs = target_fft_abs / target_fft_abs.amax(dim=(1,2), keepdim=True)

                sample_fft_abs_ln = (sample_fft_abs * self.u).log1p()
                target_fft_abs_ln = (target_fft_abs * self.u).log1p()

                #loss_weight = self.block_weights[block_num]
                loss += torch.nn.functional.l1_loss(sample_fft_abs_ln, target_fft_abs_ln,  reduction="mean")# * loss_weight
                loss += torch.nn.functional.l1_loss(sample_fft_abs, target_fft_abs, reduction="mean")# * loss_weight

        return loss / (len(self.block_widths) * len(self.block_offsets) * 2)
    
class DualMultiscaleSpectralLoss:

    @torch.no_grad()
    def __init__(self, model_params, format):
        
        loss_params = model_params["multiscale_spectral_loss"]
        self.loss_params = loss_params

        num_orders = loss_params["num_orders"]
        num_filters = loss_params["num_filters"]
        num_octaves = loss_params["num_octaves"]
        filter_std = loss_params["filter_std"]
        max_q = loss_params["max_q"]
        crop_width = format.get_sample_crop_width(model_params)

        fft_q = torch.arange(0, crop_width // 2 + 1) / (crop_width // 2)

        self.filters = None
        self.filter_weights = None

        for i in range(num_orders):
            filter_q = torch.exp2(-torch.arange(0, num_filters) / num_filters * num_octaves) * max_q
            filters = torch.exp(-filter_std * torch.log(filter_q.view(-1, 1) / fft_q.view(1, -1)).square())
            filter_weights = torch.tensor(1/num_filters).repeat(num_filters)

            if self.filters is None:
                self.filters = filters
            else:
                self.filters = torch.cat((self.filters, filters), dim=0)

            if self.filter_weights is None:
                self.filter_weights = filter_weights
            else:
                self.filter_weights = torch.cat((self.filter_weights, filter_weights), dim=0)

            filter_std *= 4
            num_filters *= 2

        padding = torch.zeros((self.filters.shape[0], crop_width // 2 - 1), device=self.filters.device)
        self.filters = torch.cat((self.filters, padding), dim=1).unsqueeze(0)

        debug_path = os.environ.get("DEBUG_PATH", None)
        if debug_path is not None:
            os.makedirs(debug_path, exist_ok=True)
            self.filters[:, :, :crop_width // 2 + 1].cpu().numpy().tofile(os.path.join(debug_path, "debug_multiscale_spectral_loss_filters.raw"))
            self.filters[:, :, :crop_width // 2 + 1].mean(dim=(0, 1)).cpu().numpy().tofile(os.path.join(debug_path, "debug_multiscale_spectral_loss_filter_coverage.raw"))
            torch.fft.fftshift(torch.fft.ifft(self.filters, norm="ortho"), dim=-1).cpu().numpy().tofile(os.path.join(debug_path, "debug_multiscale_spectral_loss_filters_ifft.raw"))

    def __call__(self, sample, target):
        
        u = self.loss_params["u"]
        bsz = sample.shape[0]

        if self.filters.device != sample.device:
            self.filters = self.filters.to(sample.device)
        if self.filter_weights.device != sample.device:
            self.filter_weights = self.filter_weights.to(sample.device)

        sample_fft = torch.fft.fft(sample, norm="ortho")
        sample_filtered_abs = torch.fft.ifft(sample_fft.view(bsz, 1, -1) * self.filters, norm="ortho").abs()
        sample_filtered_abs = sample_filtered_abs / sample_filtered_abs.amax(dim=(1,2), keepdim=True)
        sample_filtered_ln = (sample_filtered_abs * u).log1p()# / np.log(u + 1)

        target_fft = torch.fft.fft(target, norm="ortho")
        target_filtered_abs = torch.fft.ifft(target_fft.view(bsz, 1, -1) * self.filters, norm="ortho").abs()
        target_filtered_abs = target_filtered_abs / target_filtered_abs.amax(dim=(1,2), keepdim=True)
        target_filtered_ln = (target_filtered_abs * u).log1p()# / np.log(u + 1)

        """
        sample_filtered_ln.cpu().numpy().tofile("./debug/debug_multiscale_spectral_loss_filtered_sample_absln.raw")
        np.histogram(sample_filtered_ln.cpu().numpy(), bins=256)[0].astype(np.int32).tofile("./debug/debug_multiscale_spectral_loss_filtered_sample_absln_histo.raw")
        sample_filtered = torch.fft.ifft(sample_fft.view(bsz, 1, -1) * self.filters, norm="ortho")
        sample_filtered.cpu().numpy().tofile("./debug/debug_multiscale_spectral_loss_filtered_sample.raw")
        sample_filtered.sum(dim=1).cpu().numpy().tofile("./debug/debug_multiscale_spectral_loss_filtered_reconstruction.raw")
        """
        loss  = torch.nn.functional.l1_loss(sample_filtered_ln,  target_filtered_ln,  reduction="none")
        loss += torch.nn.functional.l1_loss(sample_filtered_abs, target_filtered_abs, reduction="none")
        loss = (loss.mean(dim=(0, 2), keepdim=False) * self.filter_weights).mean() * 25
        return loss
    
class DualDiffusionPipeline(DiffusionPipeline):

    @torch.no_grad()
    def __init__(
        self,
        unet: UNet2DDualModel,
        scheduler: DDIMScheduler,
        vae: AutoencoderKLDual, 
        #upscaler: UNet2DDualModel = None, 
        model_params: dict = None,
    ):
        super().__init__()

        #modules = {"unet": unet, "scheduler": scheduler}
        #if vae is not None: modules["vae"] = vae
        #if upscaler is not None: modules["upscaler"] = upscaler
        modules = {
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
        }
        self.register_modules(**modules)
        
        if model_params is not None:
            self.config["model_params"] = model_params
        else:
            model_params = self.config["model_params"]
            
        self.tiling_mode = False

        if "sample_format" not in model_params:
            model_params["sample_format"] = "normal"
        self.format = DualDiffusionPipeline.get_sample_format(model_params)

        if "multiscale_spectral_loss" in model_params:
            #self.multiscale_spectral_loss = DualMultiscaleSpectralLoss(model_params, self.format)
            self.multiscale_spectral_loss = DualMultiscaleSpectralLoss2(model_params)

    @staticmethod
    @torch.no_grad()
    def get_sample_format(model_params):
        sample_format = model_params["sample_format"]

        if sample_format == "normal":
            return DualNormalFormat
        elif sample_format == "overlapped":
            return DualOverlappedFormat
        elif sample_format == "mdct":
            return DualMDCTFormat
        elif sample_format == "mclt":
            return DualMCLTFormat
        else:
            raise ValueError(f"Unknown sample format '{sample_format}'")
        
    @staticmethod
    @torch.no_grad()
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
            snr.log().cpu().numpy().tofile(os.path.join(debug_path, "debug_schedule_ln_snr.raw"))
            np.array(trained_betas).astype(np.float32).tofile(os.path.join(debug_path, "debug_schedule_betas.raw"))

        if vae_params is not None:
            vae = AutoencoderKLDual(**vae_params)
        else:
            vae = None

        if upscaler_params is not None:
            #upscaler = Upscaler(**upscaler_params)
            raise NotImplementedError()
        else:
            upscaler = None

        #return DualDiffusionPipeline(unet, scheduler, vae=vae, upscaler=upscaler, model_params=model_params)
        return DualDiffusionPipeline(unet, scheduler, vae, model_params=model_params)

    @staticmethod
    @torch.no_grad()
    def add_embeddings(freq_samples, freq_embedding_dim, time_embedding_dim, format_hint="normal", pitch_augmentation=1., tempo_augmentation=1.):
        raise NotImplementedError()
    
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

        for _, t in enumerate(self.progress_bar(timesteps)):
            
            model_input = sample
            model_input = noise_scheduler.scale_model_input(model_input, t)
            model_output = self.unet(model_input, t).sample

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
            #raw_sample = self.vae.decode(sample / self.vae.config.scaling_factor).sample
            raise NotImplementedError()
        
        if getattr(self, "upscaler", None) is not None:
            raw_sample = self.upscale(raw_sample)

        raw_sample *= 0.18215 / raw_sample.std(dim=1, keepdim=True).clip(min=1e-5)
        if loops > 0: raw_sample = raw_sample.repeat(1, loops+1)
        return raw_sample
    
    @torch.no_grad()
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

    @torch.no_grad()
    def remove_module_tiling(self, module):
        try:
            del module._conv_forward
        except AttributeError:
            pass

    @torch.no_grad()
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