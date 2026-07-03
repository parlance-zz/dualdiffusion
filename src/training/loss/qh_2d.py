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

from dataclasses import dataclass

import torch

from modules.formats.frequency_scale import get_mel_density
from modules.formats.ms_mdct_dual_10 import _h0, _h1


@dataclass
class QHLoss2DConfig:

    sample_rate: float = 32000

    block_size:  int = 37
    block_step:  int = 13

    t_scale: float = 2.37
    p_real: list[tuple[float, float]] = ( (2,0), (1,1), (-1,1), (0,1) )
    p_imag: list[tuple[float, float]] = ( (0,0), (0,0), ( 0,0), (1,0) )

    psd_eps: float = 1e-4
    loss_scale: float = 1.18

    @property
    def pad_size(self) -> int:
        return int(self.block_size // 2 + 1)

class QHLoss2D:

    @torch.no_grad()
    def __init__(self, config: QHLoss2DConfig, device: torch.device) -> None:

        self.config = config
        self.device = device

        self.wnd_h0 = _h0(self.config.block_size, self.config.t_scale).to(device=self.device)
        self.wnd_h1 = _h1(self.config.block_size, self.config.t_scale).to(device=self.device)
    
    def get_qh_psd(self, stft_h0: torch.Tensor, stft_h1: torch.Tensor) -> torch.Tensor:

        qhs: list[torch.Tensor] = []
        for i in range(4):
            qhs.append((
                 self.config.p_real[i][0] * stft_h0 + self.config.p_real[i][1] * stft_h1 +
                (self.config.p_imag[i][0] * stft_h0 + self.config.p_imag[i][1] * stft_h1) * 1j
            ).abs())

        return torch.cat(qhs, dim=1)
    
    def qh_loss(self, sample: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        sample = sample.float()
        target = target.float()
        r_dims = (0, 1, 2, 3)

        sample_h = torch.nn.functional.pad(sample, (0, 0, self.config.pad_size, self.config.pad_size), mode="reflect").unfold(2, self.config.block_size, self.config.block_step)
        sample_h_h0 = torch.fft.rfft(sample_h * self.wnd_h0, norm="ortho"); sample_h_h1 = torch.fft.rfft(sample_h * self.wnd_h1, norm="ortho")
        sample_w = torch.nn.functional.pad(sample, (self.config.pad_size, self.config.pad_size, 0, 0), mode="reflect").unfold(3, self.config.block_size, self.config.block_step)
        sample_w_h0 = torch.fft.rfft(sample_w * self.wnd_h0, norm="ortho"); sample_w_h1 = torch.fft.rfft(sample_w * self.wnd_h1, norm="ortho")
        sample_h_qh_psd = self.get_qh_psd(sample_h_h0, sample_h_h1)
        sample_w_qh_psd = self.get_qh_psd(sample_w_h0, sample_w_h1)

        with torch.no_grad():
            target_h = torch.nn.functional.pad(target, (0, 0, self.config.pad_size, self.config.pad_size), mode="reflect").unfold(2, self.config.block_size, self.config.block_step)
            target_h_h0 = torch.fft.rfft(target_h * self.wnd_h0, norm="ortho"); target_h_h1 = torch.fft.rfft(target_h * self.wnd_h1, norm="ortho")
            target_w = torch.nn.functional.pad(target, (self.config.pad_size, self.config.pad_size, 0, 0), mode="reflect").unfold(3, self.config.block_size, self.config.block_step)
            target_w_h0 = torch.fft.rfft(target_w * self.wnd_h0, norm="ortho"); target_w_h1 = torch.fft.rfft(target_w * self.wnd_h1, norm="ortho")
            target_h_qh_psd = self.get_qh_psd(target_h_h0, target_h_h1)
            target_w_qh_psd = self.get_qh_psd(target_w_h0, target_w_h1)

            loss_weight_h = target_h_qh_psd.pow(2).mean(dim=r_dims, keepdim=True).clip(min=self.config.psd_eps).pow(0.5).requires_grad_(False).detach()
            loss_weight_w = target_w_qh_psd.pow(2).mean(dim=r_dims, keepdim=True).clip(min=self.config.psd_eps).pow(0.5).requires_grad_(False).detach()

            mel_dens_h = get_mel_density(torch.arange(sample_h_qh_psd.shape[2], device=self.device) * (self.config.sample_rate / 2) / sample_h_qh_psd.shape[2])
            mel_dens_w = get_mel_density(torch.arange(sample_w_qh_psd.shape[2], device=self.device) * (self.config.sample_rate / 2) / sample_w_qh_psd.shape[2])
            mel_dens_h /= mel_dens_h.mean(); mel_dens_w /= mel_dens_w.mean()

            loss_weight_h = loss_weight_h / mel_dens_h.view(1, 1,-1, 1, 1)
            loss_weight_w = loss_weight_w / mel_dens_w.view(1, 1,-1, 1, 1)

        mse_loss_h = torch.nn.functional.mse_loss(sample_h_qh_psd.float(), target_h_qh_psd.float(), reduction="none")
        mse_loss_w = torch.nn.functional.mse_loss(sample_w_qh_psd.float(), target_w_qh_psd.float(), reduction="none")
        loss = (mse_loss_h / loss_weight_h).mean(dim=(1,2,3,4)) + (mse_loss_w / loss_weight_w).mean(dim=(1,2,3,4))

        return loss * self.config.loss_scale

    def compile(self, **kwargs) -> None:
        self.qh_loss = torch.compile(self.qh_loss, fullgraph=False, dynamic=False)


if __name__ == "__main__":

    device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = QHLoss2DConfig()
    loss_fn = QHLoss2D(config, device)

    batch_size = 4
    channels = 8
    height = 256
    width = 384

    sample = torch.randn(batch_size, channels, height, width, device=device)
    target = torch.randn(batch_size, channels, height, width, device=device)

    loss = loss_fn.qh_loss(sample, target)
    print("Loss:", loss)