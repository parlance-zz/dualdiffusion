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

from typing import Optional, Literal, Union
from dataclasses import dataclass

import torch

from modules.formats.format import DualDiffusionFormat, DualDiffusionFormatConfig
from modules.formats.frequency_scale import get_mel_density
from modules.formats.mel_spec import MelSpecConfig, MelSpec
from utils.dual_diffusion_utils import tensor_to_img
from utils.mdct import MDCT, IMDCT, sin_window, kaiser_bessel_derived, vorbis


def _combine_fft_crossover(xs: list[torch.Tensor], sample_rate: float, crossovers: list[float], crossfade_hz: float = 100.0) -> torch.Tensor:
    n = xs[0].shape[-1]
    f = torch.fft.rfftfreq(n, d=1 / sample_rate).to(xs[0].device)
    X = torch.stack([torch.fft.rfft(x, dim=-1) for x in xs])  # (bands, b, c, F)

    edges = [0.0, *crossovers]
    masks = []
    for i in range(len(xs)):
        if i == 0:
            m = (f < edges[1]).to(X.real.dtype)
        elif i == len(xs) - 1:
            m = (f >= edges[i]).to(X.real.dtype)
        else:
            m = ((f >= edges[i]) & (f < edges[i + 1])).to(X.real.dtype)
        masks.append(m)

    masks = torch.stack(masks)

    for k, fc in enumerate(crossovers, start=1):
        lo, hi = fc - crossfade_hz, fc + crossfade_hz
        region = (f >= lo) & (f <= hi)
        t = (f[region] - lo) / (hi - lo)
        left = 0.5 * (1 + torch.cos(torch.pi * t))
        masks[k - 1, region] = left
        masks[k, region] = 1 - left

    Y = (X * masks[:, None, None, :]).sum(0)
    return torch.fft.irfft(Y, n=n, dim=-1)

def _get_mdct_window_func(mdct_window_func: str) -> callable:
    if mdct_window_func == "sin":
        mdct_window_fn = sin_window
    elif mdct_window_func == "kaiser_bessel_derived":
        mdct_window_fn = kaiser_bessel_derived
    elif mdct_window_func == "vorbis":
        mdct_window_fn = vorbis
    else:
        raise ValueError(f"Unsupported mdct window function: {mdct_window_func}. Supported functions are 'sin', 'kaiser_bessel_derived', and 'vorbis'.")
    return mdct_window_fn

@dataclass()
class MDCT_Config:

    window_len: int = 128
    window_func: Literal["sin", "kaiser_bessel_derived", "vorbis"] = "vorbis"

    crop_lo: float = 0
    crop_hi: float = 1

    @property
    def num_frequencies(self) -> int:
        return self.window_len // 2
    @property
    def num_cropped_frequencies(self) -> int:
        return int(self.num_frequencies * (self.crop_hi - self.crop_lo))
    
    @property
    def bin_lo(self) -> int:
        return int(self.num_frequencies * self.crop_lo)
    @property
    def bin_hi(self) -> int:
        return int(self.num_frequencies * self.crop_hi)
    
    @property
    def hop_length(self) -> int:
        return self.window_len // 2
    
@dataclass()
class MS_MDCT_DualFormatConfig(DualDiffusionFormatConfig):

    # raw audio format params
    sample_rate: int = 32000
    num_raw_channels: int = 2
    default_raw_length: int = 96000
    width_alignment: int    = 4096

    use_per_freq_preconditioning: bool = True

    # mdct params

    mdcts: list[MDCT_Config] = ()
    mdct_psd_exponent: float = 0.25
    mdct_out_crossover_freqs: list[float] = (300, 600, 1200)
    mdct_out_crossover_width_hz: float = 50

    @property
    def num_mdcts(self) -> int:
        return len(self.mdcts)
        
    # ms psd params
    ms_psd_add_center_channel: bool = True
    ms_psd_img_show_center_channel: bool = True
    
    # mel-spec params
    mel_spec_config: Optional[MelSpecConfig] = None


class MS_MDCT_DualFormat(DualDiffusionFormat):
    
    has_trainable_parameters: bool = True

    @torch.no_grad()
    def __init__(self, config: MS_MDCT_DualFormatConfig) -> None:
        super().__init__()
        self.config = config

        assert int(1/self.config.mdct_psd_exponent) == 1/self.config.mdct_psd_exponent, "mdct_psd_exponent must be the reciprocal of an integer"

        # ***** mel_spec setup *****

        if config.mel_spec_config is not None:
            self.mel_spec = MelSpec(config.mel_spec_config)
            
        # ***** mdct setup *****

        self.mdcts = torch.nn.ModuleList(); self.imdcts = torch.nn.ModuleList()
        for i in range(self.config.num_mdcts):
            
            if config.mdcts[i].window_len == config.mdcts[0].window_len:
                last_frame = -1
            elif config.mdcts[i].window_len > config.mdcts[0].window_len:
                last_frame = -2
            else:
                last_frame = None

            mdct_window_func = _get_mdct_window_func(config.mdcts[i].window_func)

            mdct = MDCT(win_length=config.mdcts[i].window_len, window_fn=mdct_window_func, return_complex=True, last_frame=last_frame)
            self.mdcts.append(mdct)
            imdct = IMDCT(win_length=config.mdcts[i].window_len, window_fn=mdct_window_func)
            self.imdcts.append(imdct)

            self.register_buffer(f"mdct_phase_scale_{i}", torch.ones(config.mdcts[i].num_frequencies).view(1, 1,-1, 1), persistent=True)
            self.register_buffer(f"mdct_psd_offset_{i}", torch.zeros(config.mdcts[i].num_frequencies).view(1, 1,-1, 1), persistent=True)
            self.register_buffer(f"mdct_psd_scale_{i}",  torch.ones(config.mdcts[i].num_frequencies).view(1, 1,-1, 1),  persistent=True)
            
            mdct_hz = (torch.arange(config.mdcts[i].num_frequencies) + 0.5) * config.sample_rate / config.mdcts[i].window_len
            self.register_buffer(f"mdct_mel_density_{i}", get_mel_density(mdct_hz).view(1, 1,-1, 1), persistent=False)

    # **************** mel-scale spectrogram methods ****************

    def get_raw_crop_width(self, raw_length: Optional[int] = None) -> int:
        raw_length = raw_length or self.config.default_raw_length
        return raw_length // self.config.width_alignment * self.config.width_alignment - self.config.mdcts[0].num_frequencies

    @torch.no_grad()
    def raw_to_ms_psd(self, raw_samples: torch.Tensor, level: int = 2) -> Union[list[torch.Tensor], torch.Tensor]:

        raw_samples = raw_samples.float()
        levels = list(range(self.config.num_mdcts)) if level < 0 else [level]

        ms_psds: list[torch.Tensor] = []
        for i in levels:

            mclt: torch.Tensor = self.mdcts[i](raw_samples)
            if self.config.ms_psd_add_center_channel == True:
                ms_psd = torch.cat((mclt, (mclt[:, 0:1] + mclt[:, 1:2]) / 2), dim=1).abs()
            else:
                ms_psd = mclt.abs()

            if self.config.use_per_freq_preconditioning == False:
                mdct_mel_density: torch.Tensor = getattr(self, f"mdct_mel_density_{i}")
                ms_psd /= mdct_mel_density

            mdct_psd_offset: torch.Tensor = getattr(self, f"mdct_psd_offset_{i}")
            mdct_psd_scale: torch.Tensor = getattr(self, f"mdct_psd_scale_{i}")                      
            ms_psd = (ms_psd.pow(self.config.mdct_psd_exponent) + mdct_psd_offset) / mdct_psd_scale

            ms_psds.append(ms_psd)

        if len(ms_psds) == 1:
            return ms_psds[0]
        else:
            return ms_psds
    
    @torch.no_grad()
    def ms_psd_to_img(self, ms_psd: torch.Tensor, use_colormap: bool = False):
        if use_colormap == True:
            return tensor_to_img(ms_psd.mean(dim=(0,1)), flip_y=True, colormap=True)
        else:
            if ms_psd.shape[1] == 3:
                l, r, c = torch.chunk(ms_psd, 3, dim=1)
                if self.config.ms_psd_img_show_center_channel == True:
                    ms_psd = torch.cat((l, c, r), dim=1)
                else:
                    ms_psd = torch.cat((l, r), dim=1)

            return tensor_to_img(ms_psd, flip_y=True)

    # **************** mdct methods ****************

    def get_mdct_phase_psd_shape(self, bsz: int = 1, raw_length: Optional[int] = None):
        raw_crop_width = self.get_raw_crop_width(raw_length=raw_length)
        num_mdct_bins = self.config.mdcts[0].num_frequencies
        num_mdct_frames = raw_crop_width // num_mdct_bins + 1
        return (bsz, self.config.num_raw_channels * 2, 1, num_mdct_bins * num_mdct_frames * self.config.num_mdcts)

    def get_mdct_shape(self, bsz: int = 1, raw_length: Optional[int] = None):
        return self.get_mdct_phase_psd_shape(bsz=bsz, raw_length=raw_length)

    def raw_to_mdct_phase_psd(self, raw_samples: torch.Tensor,
            random_phase_augmentation: Union[bool, torch.Tensor] = False, level: int = 0) -> Union[torch.Tensor, list[torch.Tensor]]:
        
        if level >= 0:
            _mclt: torch.Tensor = self.mdcts[level](raw_samples.float())

            if isinstance(random_phase_augmentation, bool):
                if random_phase_augmentation == True:
                    phase_rotation = torch.exp(2j * torch.pi * torch.rand(_mclt.shape[0], device=_mclt.device)) 
                    _mclt *= phase_rotation.view(-1, 1, 1, 1)
            else:
                _mclt *= random_phase_augmentation.view(-1, 1, 1, 1)

            mdct_psd = _mclt.abs()
            mdct_phase = (_mclt.real / mdct_psd.clip(min=1e-20)).clip(min=-1, max=1)

            mdct_psd = mdct_psd.pow(self.config.mdct_psd_exponent)
            mdct_phase = mdct_phase * mdct_psd

            if self.config.use_per_freq_preconditioning == False:
                mdct_psd /= getattr(self, f"mdct_mel_density_{level}").pow(self.config.mdct_psd_exponent)
                mdct_phase /= getattr(self, f"mdct_mel_density_{level}").pow(self.config.mdct_psd_exponent)

            mdct_phase = mdct_phase / getattr(self, f"mdct_phase_scale_{level}")
            mdct_psd = (mdct_psd + getattr(self, f"mdct_psd_offset_{level}")) / getattr(self, f"mdct_psd_scale_{level}")

            mdct_phase_psd = torch.cat((mdct_phase, mdct_psd), dim=1)
            return mdct_phase_psd
        else:
            if random_phase_augmentation == True:
                random_phase_augmentation = torch.exp(2j * torch.pi * torch.rand(raw_samples.shape[0], device=raw_samples.device))

            mdct_phase_psds = []
            for i in range(self.config.num_mdcts):
                mdct_phase_psd = self.raw_to_mdct_phase_psd(raw_samples, random_phase_augmentation=random_phase_augmentation, level=i)
                mdct_phase_psds.append(mdct_phase_psd)

            return mdct_phase_psds
    
    def mdct_phase_psd_to_raw(self, mdct_phase_psd: Union[torch.Tensor, list[torch.Tensor]], level: int = 0) -> torch.Tensor:

        if level >= 0:
            mdct_phase, mdct_psd = torch.chunk(mdct_phase_psd.float(), 2, dim=1)
            mdct_phase = mdct_phase * getattr(self, f"mdct_phase_scale_{level}")
            mdct_psd: torch.Tensor = mdct_psd * getattr(self, f"mdct_psd_scale_{level}") - getattr(self, f"mdct_psd_offset_{level}")

            if self.config.use_per_freq_preconditioning == False:
                mdct_phase *= getattr(self, f"mdct_mel_density_{level}").pow(self.config.mdct_psd_exponent)
                mdct_psd *= getattr(self, f"mdct_mel_density_{level}").pow(self.config.mdct_psd_exponent)

            #mdct_psd = mdct_psd.clip(min=0).pow(1 / self.config.mdct_psd_exponent - 1)
            recon_exp = int((1 / self.config.mdct_psd_exponent - 1) / 2) * 2 + 1
            mdct_psd = mdct_psd.pow(recon_exp)
            mdct: torch.Tensor = mdct_phase * mdct_psd

            mdct[:, :, :self.config.mdcts[level].bin_lo] = 0
            mdct[:, :, self.config.mdcts[level].bin_hi:] = 0

            raw_samples = self.imdcts[level](mdct).real.contiguous()
            return raw_samples
        else:
            output_mdct_raws: list[torch.Tensor] = []
            for i in range(len(mdct_phase_psd)):
                output_mdct_raws.append(self.mdct_phase_psd_to_raw(mdct_phase_psd[i], level=i))

            crop_length = min(x.shape[-1] for x in output_mdct_raws)
            output_mdct_raws = [x[..., :crop_length] for x in output_mdct_raws]

            #return torch.stack(output_mdct_raws).sum(dim=0)
            output_mdct_raws.reverse()
            return _combine_fft_crossover(output_mdct_raws, sample_rate=self.config.sample_rate,
                crossovers=self.config.mdct_out_crossover_freqs, crossfade_hz=self.config.mdct_out_crossover_width_hz)
    
    def flatten_mdct_phase_psd(self, mdct_phase_psds: list[torch.Tensor]) -> torch.Tensor:
        mdct_phase_psds = [x.flatten(2, 3)[:, :, None, :] for x in mdct_phase_psds]
        return torch.cat(mdct_phase_psds, dim=3)

    def unflatten_mdct_phase_psd(self, mdct_phase_psd: torch.Tensor) -> list[torch.Tensor]:
        assert mdct_phase_psd.shape[3] % self.config.num_mdcts == 0
        mdct_phase_psds: list[torch.Tensor] = []
        for i, chunk in enumerate(torch.chunk(mdct_phase_psd, chunks=self.config.num_mdcts, dim=3)):
            num_mdct_phase_psd_channels = self.config.num_raw_channels * 2
            mdct_phase_psds.append(chunk.reshape(chunk.shape[0], num_mdct_phase_psd_channels, self.config.mdcts[i].num_frequencies, -1))
        return mdct_phase_psds

    def crop_unflattened(self, mdct_phase_psds: list[torch.Tensor]) -> list[torch.Tensor]:
        cropped_mdct_phase_psds: list[torch.Tensor] = []
        for i in range(self.config.num_mdcts):
            bin_hi = self.config.mdcts[i].bin_hi
            bin_lo = self.config.mdcts[i].bin_lo
            cropped_mdct_phase_psds.append(mdct_phase_psds[i][:, :, bin_lo:bin_hi, :])
        return cropped_mdct_phase_psds

    def uncrop_unflattened(self, cropped_mdct_phase_psds: list[torch.Tensor]) -> list[torch.Tensor]:
        uncropped_mdct_phase_psds: list[torch.Tensor] = []
        for i in range(self.config.num_mdcts):
            num_mdct_bins = self.config.mdcts[i].num_frequencies
            bin_hi = self.config.mdcts[i].bin_hi
            bin_lo = self.config.mdcts[i].bin_lo
            uncropped_mdct_phase_psds.append(torch.nn.functional.pad(cropped_mdct_phase_psds[i], (0, 0, bin_lo, num_mdct_bins - bin_hi)))
        return uncropped_mdct_phase_psds

    def get_mdct_phase_psd_loss(self, pred_mdct_phase_psd: list[torch.Tensor], target_mdct_phase_psd: list[torch.Tensor]) -> torch.Tensor:
        
        pred_mdct_phase_psd = self.crop_unflattened(self.unflatten_mdct_phase_psd(pred_mdct_phase_psd))
        target_mdct_phase_psd = self.crop_unflattened(self.unflatten_mdct_phase_psd(target_mdct_phase_psd))
        mel_densities = self.crop_unflattened(self.get_mdct_mel_density(level=-1))

        level_losses: list[torch.Tensor] = []
        for pred, target, weight in zip(pred_mdct_phase_psd, target_mdct_phase_psd, mel_densities):
            
            weight = weight / weight.mean()
            level_loss = (torch.nn.functional.mse_loss(pred, target.detach(), reduction="none") * weight).mean(dim=(1,2,3))
            level_losses.append(level_loss)

        return torch.stack(level_losses, dim=1)

    def get_mdct_mel_density(self, level: int = 0) -> Union[torch.Tensor:, list[torch.Tensor]]:
        if level >= 0:
            return getattr(self, f"mdct_mel_density_{level}")
        else:
            mdct_mel_densities: list[torch.Tensor] = []
            for i in range(self.config.num_mdcts):
                mdct_mel_densities.append(getattr(self, f"mdct_mel_density_{i}"))
            return mdct_mel_densities
        
    def raw_to_mdct_psd(self, raw_samples: torch.Tensor, level: int = 0) -> torch.Tensor:

        mdct_phase_psd = self.raw_to_mdct_phase_psd(raw_samples.float(), level=level)
        _, mdct_psd = torch.chunk(mdct_phase_psd, 2, dim=1)
        return mdct_psd