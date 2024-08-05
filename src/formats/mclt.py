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

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from utils.mclt import mclt, imclt
from .format import DualDiffusionFormat

class DualMCLTFormat(ModelMixin, ConfigMixin, DualDiffusionFormat):

    @torch.no_grad()
    @register_to_config
    def __init__(self,
                 noise_floor: float = 2e-5,
                 sample_rate: int = 32000,
                 sample_raw_channels: int = 2,
                 sample_raw_length: int = 1057570,
                 t_scale: Optional[float] = None,
                 mclt_window_len: int = 512,
                 mclt_window_fn: str = "hann",
                 mclt_window_exponent: float = 0.5,
                 ) -> None:
        super(DualMCLTFormat, self).__init__()

    @torch.no_grad()
    def get_raw_crop_width(self, length: Optional[int] = None) -> int:
        block_width = self.config["mclt_window_len"]
        length = length or self.config["sample_raw_length"]
        return length // block_width // 64 * 64 * block_width + block_width
    
    @torch.no_grad()
    def get_num_channels(self):
        in_channels = out_channels = self.config["sample_raw_channels"] * 2
        return (in_channels, out_channels)

    @torch.no_grad()
    def get_sample_shape(self, bsz: int = 1, length: Optional[int] = None):
        _, num_output_channels = self.get_num_channels()

        crop_width = self.get_raw_crop_width(length=length)
        num_mclt_bins = self.config["mclt_window_len"] // 2
        chunk_len = crop_width // num_mclt_bins - 2

        return (bsz, num_output_channels, num_mclt_bins, chunk_len,)

    @torch.no_grad()
    def raw_to_sample(self, raw_samples: torch.Tensor,
                      return_dict: bool = False) -> Union[torch.Tensor, dict]:

        samples_mdct = mclt(raw_samples,
                            self.config["mclt_window_len"],
                            self.config["mclt_window_exponent"])[:, :, 1:-2, :]
        samples_mdct = samples_mdct.permute(0, 1, 3, 2)
        samples_mdct *= torch.exp(2j * torch.pi * torch.rand(1, device=samples_mdct.device))

        samples_mdct_abs = samples_mdct.abs()
        samples_mdct_abs_amax = samples_mdct_abs.amax(dim=(1,2,3), keepdim=True).clip(min=1e-5)
        samples_mdct_abs = (samples_mdct_abs / samples_mdct_abs_amax).clip(min=self.config["noise_floor"])
        samples_abs_ln = samples_mdct_abs.log()
        samples_qphase1 = samples_mdct.angle().abs()
        samples = torch.cat((samples_abs_ln, samples_qphase1), dim=1)

        samples_mdct /= samples_mdct_abs_amax
        raw_samples = imclt(samples_mdct.permute(0, 1, 3, 2),
                            window_degree=self.config["mclt_window_exponent"]).real

        if return_dict:
            samples_dict = {
                "samples": samples,
                "raw_samples": raw_samples,
            }
            return samples_dict
        else:
            return samples

    def sample_to_raw(self, samples: torch.Tensor,
                      return_dict: bool = False) -> Union[torch.Tensor, dict]:
        
        samples_abs, samples_phase1 = samples.chunk(2, dim=1)
        samples_abs = samples_abs.exp()
        samples_phase = samples_phase1.cos()
        raw_samples = imclt((samples_abs * samples_phase).permute(0, 1, 3, 2),
                            window_degree=self.config["mclt_window_exponent"]).real

        if return_dict:
            return {"samples": samples, "raw_samples": raw_samples}
        else:
            return raw_samples
        
    @torch.no_grad()
    def get_ln_freqs(self, x: torch.Tensor) -> torch.Tensor:

        ln_freqs = torch.linspace(0, self.config["sample_rate"]/2, x.shape[2] + 2, device=x.device)[1:-1].log2()
        ln_freqs = ln_freqs.view(1, 1,-1, 1).repeat(x.shape[0], 1, 1, x.shape[3])

        return ((ln_freqs - ln_freqs.mean()) / ln_freqs.std()).to(x.dtype)