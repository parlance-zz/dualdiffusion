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

from typing import Optional, Union
from dataclasses import dataclass

import torch

from .format import DualDiffusionFormat, DualDiffusionFormatConfig
from utils.mclt import mclt, imclt
from utils.dual_diffusion_utils import tensor_to_img


@dataclass
class DualMCLTFormatConfig(DualDiffusionFormatConfig):

    window_len: int = 512
    sample_to_raw_scale: float = 1 / 0.5005

    # approximately unit variance for sample and 1:1 gain for reconstructed raw, for audio normalized to -20 lufs
    abs_exponent: float = 1
    raw_to_sample_scale: float = 19.37217829
    
    # approximately unit variance for sample and 1:1 gain for reconstructed raw, for audio normalized to -20 lufs
    #abs_exponent: float = 1/4
    #raw_to_sample_scale: float = 3.85212542

class DualMCLTFormat(DualDiffusionFormat):

    def __init__(self, config: DualMCLTFormatConfig) -> None:
        super().__init__()
        self.config = config
    
    def sample_raw_crop_width(self, length: Optional[int] = None) -> int:
        block_width = self.config.window_len
        length = length or self.config.sample_raw_length
        return length // block_width // 64 * 64 * block_width + block_width
    
    def get_sample_shape(self, bsz: int = 1, length: Optional[int] = None):
        num_output_channels = self.config.sample_raw_channels

        crop_width = self.sample_raw_crop_width(length=length)
        num_mclt_bins = self.config.window_len // 2
        chunk_len = crop_width // num_mclt_bins - 2

        return (bsz, num_output_channels, num_mclt_bins, chunk_len,)

    @torch.inference_mode()
    def raw_to_sample(self, raw_samples: torch.Tensor, random_phase_augmentation: bool = False) -> Union[torch.Tensor, dict]:

        samples_mdct = mclt(raw_samples, self.config.window_len, "hann", 0.5).permute(0, 1, 3, 2).contiguous()
        if random_phase_augmentation == True:
            samples_mdct *= torch.exp(2j * torch.pi * torch.rand(1, device=samples_mdct.device))

        return samples_mdct.real.abs() ** self.config.abs_exponent * samples_mdct.real.sign() * self.config.raw_to_sample_scale

    @torch.inference_mode()
    def sample_to_raw(self, samples: torch.Tensor) -> Union[torch.Tensor, dict]:
        
        samples = (samples.abs() / self.config.raw_to_sample_scale) ** (1/self.config.abs_exponent) * samples.sign() * self.config.sample_to_raw_scale
        return imclt(samples.permute(0, 1, 3, 2), window_fn="hann", window_degree=0.5).real
    
    @torch.no_grad()
    def get_ln_freqs(self, x: torch.Tensor) -> torch.Tensor:
        
        ln_freqs = torch.linspace(0, self.config.sample_rate/2, x.shape[2] + 2, device=x.device)[1:-1].log2()
        ln_freqs = ln_freqs.view(1, 1,-1, 1).repeat(x.shape[0], 1, 1, x.shape[3])

        return ((ln_freqs - ln_freqs.mean()) / ln_freqs.std()).to(x.dtype)
    
    @torch.inference_mode()
    def sample_to_img(self, sample: torch.Tensor):
        return tensor_to_img((torch.sign(sample) * sample.abs() **(1/4)), flip_y=True)