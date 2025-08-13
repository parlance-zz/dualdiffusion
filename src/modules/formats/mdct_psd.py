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

from typing import Optional, Literal
from dataclasses import dataclass

import torch

from modules.formats.format import DualDiffusionFormat, DualDiffusionFormatConfig
from modules.formats.frequency_scale import get_mel_density
from utils.dual_diffusion_utils import tensor_to_img
from utils.mdct import MDCT, IMDCT, MDCT2, IMDCT2, get_window_fn


@dataclass()
class MDCT_PSD_FormatConfig(DualDiffusionFormatConfig):

    sample_rate: int = 32000
    num_raw_channels: int = 2

    low_cut_start_hz: float = 28.862
    low_cut_end_hz: float   = 20

    default_raw_length: int = 1409024
    width_alignment: int    = 32768

    raw_to_mdct_scale: float = 278.47124 # for stereo audio @ -20 lufs
    mdct_window_len: int = 512
    mdct_window_func: Literal["sin", "kbd", "vorbis"] = "sin"

    mdct_to_p2m_scale: float = 30.50252 # for stereo audio @ -20 lufs with midside transform
    p2m_psd_scale: float = 1.461213
    p2m_psd_eps: float   = 1e-3
    p2m_use_midside_transform: bool = False
    p2m_block_width: int = 16
    p2m_window_func: Literal["sin", "kbd", "vorbis"] = "sin"
    
    @property
    def mdct_num_frequencies(self) -> int:
        return self.mdct_window_len // 2
    
    @property
    def mdct_frame_hop_length(self) -> int:
        return self.mdct_window_len // 2
    
    @property
    def p2m_num_bins(self) -> int:
        return self.p2m_block_width ** 2 // 4
    
    @property
    def p2m_num_channels(self) -> int:
        return self.p2m_num_bins * self.num_raw_channels
    
    @property
    def p2m_num_frequencies(self) -> int:
        return self.mdct_num_frequencies // self.p2m_block_hop_length + 1
    
    @property
    def p2m_block_hop_length(self) -> int:
        return self.p2m_block_width // 2

class MDCT_PSD_Format(DualDiffusionFormat):

    def __init__(self, config: MDCT_PSD_FormatConfig) -> None:
        super().__init__()
        self.config = config

        if self.config.p2m_use_midside_transform == True:
            assert self.config.num_raw_channels == 2, "P2M Mid-side transform requires 2 raw channels"

        mdct_hz = (torch.arange(config.mdct_num_frequencies) + 0.5) * (config.sample_rate/2) / config.mdct_num_frequencies
        self.register_buffer("mdct_hz", mdct_hz, persistent=False)
        self.register_buffer("mdct_mel_density", get_mel_density(mdct_hz).view(1, 1,-1, 1), persistent=False)

        p2m_hz = (torch.arange(config.p2m_num_frequencies) + 0.5) * (config.sample_rate/2) / config.p2m_num_frequencies
        self.register_buffer("p2m_hz", p2m_hz, persistent=False)
        self.register_buffer("p2m_mel_density", get_mel_density(p2m_hz).view(1, 1,-1, 1), persistent=False)

        mdct_window_fn = get_window_fn(config.mdct_window_func)
        self.mdct = MDCT(win_length=config.mdct_window_len, window_fn=mdct_window_fn, return_complex=True)
        self.imdct = IMDCT(win_length=config.mdct_window_len, window_fn=mdct_window_fn)

        p2m_window_fn = get_window_fn(config.p2m_window_func)
        self.p2m = MDCT2(win_length=config.p2m_block_width, window_fn=p2m_window_fn, return_complex=True)
        self.ip2m = IMDCT2(win_length=config.p2m_block_width, window_fn=p2m_window_fn)

    def _high_pass(self, raw_samples: torch.Tensor) -> torch.Tensor:
        
        cutoff_freq = self.config.low_cut_end_hz

        if cutoff_freq <= 0 or (self.config.low_cut_start_hz - cutoff_freq) <= 0:
            return raw_samples
        
        raw_len = raw_samples.shape[-1]

        raw_samples = torch.nn.functional.pad(raw_samples.float(), (raw_len // 2, raw_len // 2), mode="reflect")
        rfft: torch.Tensor = torch.fft.rfft(raw_samples, dim=-1, norm="ortho")
        rfft_freq = torch.fft.rfftfreq(raw_samples.shape[-1], d=1/self.config.sample_rate, device=raw_samples.device)

        filter: torch.Tensor = (rfft_freq - cutoff_freq) / (self.config.low_cut_start_hz - cutoff_freq)
        filter = filter.clip(min=0., max=1.).view(1, 1,-1)

        raw_samples = torch.fft.irfft(rfft * filter, n=raw_samples.shape[-1], dim=-1, norm="ortho")
        return raw_samples[..., raw_len//2:-raw_len//2]
    
    def get_raw_crop_width(self, raw_length: Optional[int] = None) -> int:
        raw_length = raw_length or self.config.default_raw_length
        return raw_length // self.config.width_alignment * self.config.width_alignment - self.config.mdct_num_frequencies

    def get_mdct_shape(self, bsz: int = 1, raw_length: Optional[int] = None) -> tuple[int, int, int, int]:
        raw_length = raw_length or self.config.default_raw_length
        raw_crop_width = self.get_raw_crop_width(raw_length=raw_length + self.config.mdct_num_frequencies)
        num_mdct_bins = self.config.mdct_num_frequencies
        num_mdct_frames = (raw_crop_width + self.config.mdct_num_frequencies) // num_mdct_bins
        num_channels = self.config.num_raw_channels
        return (bsz, num_channels, num_mdct_bins, num_mdct_frames,)

    @torch.no_grad()
    def raw_to_mdct(self, raw_samples: torch.Tensor, random_phase_augmentation: bool = False) -> torch.Tensor:

        raw_samples = self._high_pass(raw_samples)
        mclt = self.mdct(raw_samples.float())

        if random_phase_augmentation == True:
            phase_rotation = torch.exp(2j * torch.pi * torch.rand(mclt.shape[0], device=mclt.device)) 
            mclt *= phase_rotation.view(-1, 1, 1, 1)

        return mclt.real.contiguous(memory_format=self.memory_format) / self.mdct_mel_density * self.config.raw_to_mdct_scale
    
    @torch.no_grad()
    def raw_to_mdct_psd(self, raw_samples: torch.Tensor) -> torch.Tensor:

        raw_samples = self._high_pass(raw_samples)
        mclt = self.mdct(raw_samples.float())

        return mclt.abs().contiguous(memory_format=self.memory_format) / self.mdct_mel_density * self.config.raw_to_mdct_scale / 2**0.5
    
    @torch.no_grad()
    def mdct_to_raw(self, mdct: torch.Tensor) -> torch.Tensor:

        mdct = mdct * self.mdct_mel_density / self.config.raw_to_mdct_scale
        raw_samples = self.imdct(mdct).real.contiguous()
        
        return raw_samples

    @torch.inference_mode()
    def psd_to_img(self, psd: torch.Tensor, transpose_p2m: bool = False) -> torch.Tensor:
        psd = psd.clip(min=0) ** 0.25

        # if input is a p2m psd reshape it to a valid 2d image
        if psd.shape[1] == self.config.p2m_num_channels:
            return self._p2m_to_img(psd, transposed=transpose_p2m)
        else:
            return tensor_to_img(psd, flip_y=True)
    
    # ************* p2m methods *************

    def get_p2m_shape(self, mdct_shape: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        bsz = mdct_shape[0]
        h_blocks = mdct_shape[2] // self.config.p2m_block_hop_length + 1
        w_blocks = mdct_shape[3] // self.config.p2m_block_hop_length + 1
        return (bsz, self.config.p2m_num_channels, h_blocks, w_blocks,)
    
    @torch.no_grad()
    def mdct_to_p2m(self, mdct: torch.Tensor) -> torch.Tensor:

        if self.config.p2m_use_midside_transform == True:
            mdct = torch.stack((mdct[:, 0] + mdct[:, 1], mdct[:, 0] - mdct[:, 1]), dim=1) / 2**0.5

        p2m: torch.Tensor = self.p2m(mdct) * self.config.mdct_to_p2m_scale
        p2m = p2m.real.reshape(p2m.shape[0],
            self.config.p2m_num_channels, p2m.shape[4], p2m.shape[5])
        
        return p2m.contiguous(memory_format=self.memory_format)
    
    @torch.no_grad()
    def mdct_to_p2m_psd(self, mdct: torch.Tensor) -> torch.Tensor:

        if self.config.p2m_use_midside_transform == True:
            mdct = torch.stack((mdct[:, 0] + mdct[:, 1], mdct[:, 0] - mdct[:, 1]), dim=1) / 2**0.5

        p2m: torch.Tensor = self.p2m(mdct) * self.config.mdct_to_p2m_scale
        p2m_psd = p2m.abs().reshape(p2m.shape[0],
            self.config.p2m_num_channels, p2m.shape[4], p2m.shape[5])
        
        return p2m_psd.contiguous(memory_format=self.memory_format)
    
    @torch.no_grad()
    def p2m_to_mdct(self, p2m: torch.Tensor) -> torch.Tensor:

        p2m = p2m.reshape(p2m.shape[0],
            self.config.num_raw_channels, self.config.p2m_block_hop_length,
            self.config.p2m_block_hop_length, p2m.shape[2], p2m.shape[3])
        
        mdct: torch.Tensor = self.ip2m(p2m / self.config.mdct_to_p2m_scale)
        mdct = mdct.real.contiguous(memory_format=self.memory_format)

        if self.config.p2m_use_midside_transform == True:
            mdct = torch.stack((mdct[:, 0] + mdct[:, 1], mdct[:, 0] - mdct[:, 1]), dim=1) / 2**0.5
        
        return mdct
    
    @torch.no_grad()
    def scale_p2m_from_psd(self, p2m: torch.Tensor, p2m_psd: torch.Tensor):
        return p2m / (p2m_psd + self.config.p2m_psd_eps) * self.config.p2m_psd_scale

    @torch.no_grad()
    def unscale_p2m_from_psd(self, p2m: torch.Tensor, p2m_psd: torch.Tensor):
        return p2m * (p2m_psd + self.config.p2m_psd_eps) / self.config.p2m_psd_scale
    
    @torch.inference_mode()
    def _p2m_to_img(self, p2m: torch.Tensor, transposed: bool = False) -> torch.Tensor:

        p2m = p2m.reshape(p2m.shape[0], self.config.num_raw_channels,
            self.config.p2m_block_hop_length, self.config.p2m_block_hop_length, p2m.shape[2], p2m.shape[3])
        
        if transposed == True:
            p2m = p2m.permute(0, 1, 4, 2, 5, 3)
            p2m = p2m.reshape(p2m.shape[0], p2m.shape[1], p2m.shape[2] * p2m.shape[3], p2m.shape[4] * p2m.shape[5])
            p2m = p2m.flip(dims=(2,))
        else:    
            p2m = p2m.transpose(3, 4).flip(dims=(3,))
            p2m = p2m.reshape(p2m.shape[0], p2m.shape[1], p2m.shape[2] * p2m.shape[3], p2m.shape[4] * p2m.shape[5])

        return tensor_to_img(p2m)