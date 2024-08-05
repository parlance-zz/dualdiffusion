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

import torch
import torch.nn.functional as F
import numpy as np

from utils.dual_diffusion_utils import get_mel_density
from utils.mclt import stft

class DualMultiscaleSpectralLoss:

    @torch.no_grad()
    def __init__(self, loss_params):
    
        self.block_widths = loss_params["block_widths"]
        self.block_overlap = loss_params["block_overlap"]
        self.window_fn = loss_params["window_fn"]
        self.low_cutoff = loss_params["low_cutoff"]
        
        self.loss_scale = 1 / len(self.block_widths)

    def __call__(self, sample, target, model_params):

        target = target["raw_samples"]
        sample1 = sample["raw_samples_orig_phase"]
        sample2 = sample["raw_samples_orig_abs"]
        sample3 = sample["raw_samples"]

        #save_raw(target, "./debug/target.raw")
        #save_raw(sample1, "./debug/sample1.raw")
        #save_raw(sample2, "./debug/sample2.raw")
        #save_raw(sample3, "./debug/sample3.raw")
        #exit()

        noise_floor = model_params["noise_floor"]
        sample_rate = model_params["sample_rate"]
        use_mixed_mss = model_params["use_mixed_mss"]
        stereo_separation_weight = model_params["stereo_separation_weight"]

        loss_real = torch.zeros(1, device=target.device)
        loss_imag = torch.zeros(1, device=target.device)

        for block_width in self.block_widths:
            
            block_width = min(block_width, target.shape[-1])
            step = max(block_width // self.block_overlap, 1)
            offset = np.random.randint(0, min(target.shape[-1] - block_width + 1, step))        
            
            if np.random.rand() < stereo_separation_weight:
                add_channelwise_fft = model_params["sample_raw_channels"] > 1
            else:
                add_channelwise_fft = False
            
            with torch.no_grad():
                target_fft = stft(target[:, :, offset:],
                                  block_width,
                                  window_fn=self.window_fn,
                                  step=step,
                                  add_channelwise_fft=add_channelwise_fft)[:, :, :, self.low_cutoff:]
                target_fft_abs = target_fft.abs().clip(min=noise_floor).log().requires_grad_(False)
                target_fft_angle = target_fft.angle().requires_grad_(False)

                block_hz = torch.linspace(self.low_cutoff / block_width * sample_rate, sample_rate/2, target_fft.shape[-1], device=target_fft.device)
                mel_density = get_mel_density(block_hz).view(1, 1, 1,-1)
                target_phase_weight = ((target_fft_abs - target_fft_abs.amin(dim=3, keepdim=True)) * mel_density).requires_grad_(False)

            if use_mixed_mss:
                sample_fft1_abs = stft(sample1[:, :, offset:],
                                    block_width,
                                    window_fn=self.window_fn,
                                    step=step,
                                    add_channelwise_fft=add_channelwise_fft)[:, :, :, self.low_cutoff:].abs().clip(min=noise_floor).log()

                sample_fft2_angle = stft(sample2[:, :, offset:],
                                        block_width,
                                        window_fn=self.window_fn,
                                        step=step,
                                        add_channelwise_fft=add_channelwise_fft)[:, :, :, self.low_cutoff:].angle()

                loss_real = loss_real + (sample_fft1_abs - target_fft_abs).abs().mean()

                error_imag = (sample_fft2_angle - target_fft_angle).abs()
                error_imag_wrap_mask = (error_imag > torch.pi).detach().requires_grad_(False)
                error_imag[error_imag_wrap_mask] = 2*torch.pi - error_imag[error_imag_wrap_mask]
                loss_imag = loss_imag + (error_imag * target_phase_weight).mean()
                
            sample_fft3 = stft(sample3[:, :, offset:],
                               block_width,
                               window_fn=self.window_fn,
                               step=step,
                               add_channelwise_fft=add_channelwise_fft)[:, :, :, self.low_cutoff:]
            sample_fft3_abs = sample_fft3.abs().clip(min=noise_floor).log()
            sample_fft3_angle = sample_fft3.angle()
            
            loss_real = loss_real + (sample_fft3_abs - target_fft_abs).abs().mean()

            error_imag = (sample_fft3_angle - target_fft_angle).abs()
            error_imag_wrap_mask = (error_imag > torch.pi).detach().requires_grad_(False)
            error_imag[error_imag_wrap_mask] = 2*torch.pi - error_imag[error_imag_wrap_mask]
            loss_imag = loss_imag + (error_imag * target_phase_weight).mean()

        return loss_real * self.loss_scale, loss_imag * self.loss_scale

class DualMultiscaleSpectralLoss2D:

    __constants__ = ["loss_scale", "block_overlap", "block_widths"]

    @torch.no_grad()
    def __init__(self, loss_params):
    
        self.block_widths = loss_params["block_widths"]
        self.block_overlap = loss_params["block_overlap"]
        self.loss_scale = 4e-3

    @torch.no_grad()
    def _flat_top_window(self, x):
        return 0.21557895 - 0.41663158 * torch.cos(x) + 0.277263158 * torch.cos(2*x) - 0.083578947 * torch.cos(3*x) + 0.006947368 * torch.cos(4*x)

    @torch.no_grad()
    def get_flat_top_window_2d(self, block_width, device):
        wx = torch.linspace(0, 2*torch.pi, block_width, device=device)
        return self._flat_top_window(wx.view(1, 1,-1, 1)) * self._flat_top_window(wx.view(1, 1, 1,-1)).requires_grad_(False)
    
    def stft2d(self, x, block_width, step, window):
        
        padding = block_width // 2
        x = F.pad(x, (padding, padding, padding, padding), mode="reflect")
        x = x.unfold(2, block_width, step).unfold(3, block_width, step)

        x = torch.fft.rfft2(x * window, norm="backward")
        
        if x.shape[1] == 2:
            x = torch.stack((x[:, 0] + x[:, 1],
                             x[:, 0] - x[:, 1]), dim=1)
        elif x.shape[1] > 2:
            x = torch.fft.fft(x, dim=1, norm="backward")

        return x
    
    def __call__(self, sample, target, model_params):

        target = target["samples"]
        sample = sample["samples"]

        loss_real = torch.zeros(1, device=target.device, dtype=torch.float64)
        loss_imag = torch.zeros(1, device=target.device, dtype=torch.float64)
        loss_scale = self.loss_scale / target.numel()

        for block_width in self.block_widths:
            
            block_width = min(block_width, target.shape[-1], target.shape[-2])
            step = max(block_width // self.block_overlap, 1)            

            with torch.no_grad():
                
                window = self.get_flat_top_window_2d(block_width, target.device)

                blockfreq_y = torch.fft.fftfreq(block_width, 1/block_width, device=target.device)
                blockfreq_x = torch.arange(block_width//2 + 1, device=target.device)
                wavelength = 1 / ((blockfreq_y.square().view(-1, 1) + blockfreq_x.square().view(1, -1)).sqrt() + 1)
                real_loss_weight = (1 / wavelength * wavelength.amin()).requires_grad_(False)
                
                target_fft = self.stft2d(target, block_width, step, window)
                target_fft_abs = target_fft.abs().requires_grad_(False)
                target_fft_angle = target_fft.angle().requires_grad_(False)
                loss_imag_weight = (wavelength.view((1,)*4 + wavelength.shape) / torch.pi).requires_grad_(False) * target_fft_abs

            sample_fft = self.stft2d(sample, block_width, step, window)
            sample_fft_abs = sample_fft.abs()
            
            loss_real = loss_real + ((sample_fft_abs - target_fft_abs).abs() * real_loss_weight).type(torch.float64).sum()

            error_imag = (sample_fft.angle() - target_fft_angle).abs()
            error_imag_wrap_mask = (error_imag > torch.pi).detach().requires_grad_(False)
            error_imag = torch.where(error_imag_wrap_mask, 2*torch.pi - error_imag, error_imag)
            loss_imag = loss_imag + (error_imag * loss_imag_weight).type(torch.float64).sum()
        
        return (loss_real * loss_scale).float(), (loss_imag * loss_scale).float()