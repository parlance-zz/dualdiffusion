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

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProgressiveResample(nn.Module):
    """
    Variable-rate resampling along the height dimension, with fixed geometry.

    Downsample:
        src_len -> src_len // 2

    Coordinate warp:
        y = x + x**p
        where x in [0, 1] is the output-domain coordinate and
              y in [0, 2] is the source-domain coordinate.

    Forward downsampling uses 3 mip levels:
        - full resolution
        - avg_pool1d by 2
        - avg_pool1d by 3

    and blends between them based on local scale.

    Upsampling is an approximate inverse:
        - invert the coordinate warp once in __init__
        - bicubic sample from the half-resolution signal

    This module operates on the height dimension only and supports arbitrary
    leading dimensions.

    coord_mode:
        - "centers": pixel-center coordinates, align_corners=False
        - "endpoints": endpoint-inclusive coordinates, align_corners=True
    """

    def __init__(
        self,
        src_len: int,
        p: float,
        *,
        coord_mode: Literal["centers", "endpoints"] = "centers",
        newton_steps: int = 4,
        dtype: torch.dtype = torch.float32,
        device=None,
    ):
        super().__init__()

        if not (1.0 < p <= 2.0):
            raise ValueError(f"p must satisfy 1 < p <= 2, got {p}")
        if src_len < 2:
            raise ValueError("src_len must be at least 2")
        if src_len % 2 != 0:
            raise ValueError(f"src_len must be even, got {src_len}")
        if coord_mode not in ("centers", "endpoints"):
            raise ValueError(f"coord_mode must be 'centers' or 'endpoints', got {coord_mode}")

        self.src_len = int(src_len)
        self.dst_len = self.src_len // 2
        self.lvl2_len = self.src_len // 2
        self.lvl3_len = self.src_len // 3
        self.p = float(p)
        self.coord_mode = coord_mode
        self.newton_steps = int(newton_steps)
        self.align_corners = (coord_mode == "endpoints")

        # -------------------------
        # Precompute downsample geometry
        # -------------------------
        t_fwd = self._make_forward_coords(self.dst_len, dtype=dtype, device=device)  # [dst]
        y_norm = t_fwd + t_fwd.pow(self.p)                                            # [dst]
        s = 1.0 + self.p * t_fwd.pow(self.p - 1.0)                                    # [dst]

        grid1 = self._make_grid_from_y_norm(y_norm, self.src_len)
        grid2 = self._make_grid_from_y_norm(y_norm, self.lvl2_len)
        grid3 = self._make_grid_from_y_norm(y_norm, self.lvl3_len)

        mask12 = s <= 2.0
        a12 = (s - 1.0).clamp(0.0, 1.0)
        a23 = (s - 2.0).clamp(0.0, 1.0)

        # -------------------------
        # Precompute upsample geometry
        # -------------------------
        y_inv = self._make_inverse_coords(self.src_len, dtype=dtype, device=device)  # [src]

        if self.p == 2:
            t_inv = 0.5 * (torch.sqrt(1.0 + 4.0 * y_inv) - 1.0)
            t_inv = t_inv.clamp(0.0, 1.0)
        else:
            t_inv = (0.5 * y_inv).clamp(0.0, 1.0)
            for _ in range(self.newton_steps):
                tp = t_inv.clamp_min(1e-12)
                f = t_inv + tp.pow(self.p) - y_inv
                df = 1.0 + self.p * tp.pow(self.p - 1.0)
                t_inv = (t_inv - f / df).clamp(0.0, 1.0)

        inv_grid = self._make_grid_from_unit_coords(t_inv, self.dst_len)

        self.grid1: torch.Tensor; self.grid2: torch.Tensor; self.grid3: torch.Tensor
        self.register_buffer("grid1", grid1, persistent=False)
        self.register_buffer("grid2", grid2, persistent=False)
        self.register_buffer("grid3", grid3, persistent=False)

        self.mask12: torch.Tensor; self.a12: torch.Tensor; self.a23: torch.Tensor
        self.register_buffer("mask12", mask12.view(1, self.dst_len), persistent=False)
        self.register_buffer("a12", a12.view(1, self.dst_len), persistent=False)
        self.register_buffer("a23", a23.view(1, self.dst_len), persistent=False)

        self.inv_grid: torch.Tensor
        self.register_buffer("inv_grid", inv_grid, persistent=False)

    def _make_forward_coords(self, n: int, *, dtype: torch.dtype, device) -> torch.Tensor:
        if self.coord_mode == "centers":
            return (torch.arange(n, dtype=dtype, device=device) + 0.5) / n
        else:
            if n == 1:
                return torch.zeros(1, dtype=dtype, device=device)
            return torch.linspace(0.0, 1.0, n, dtype=dtype, device=device)

    def _make_inverse_coords(self, n: int, *, dtype: torch.dtype, device) -> torch.Tensor:
        if self.coord_mode == "centers":
            return 2.0 * (torch.arange(n, dtype=dtype, device=device) + 0.5) / n
        else:
            if n == 1:
                return torch.zeros(1, dtype=dtype, device=device)
            return torch.linspace(0.0, 2.0, n, dtype=dtype, device=device)

    def _make_grid_from_y_norm(self, y_norm: torch.Tensor, level_len: int) -> torch.Tensor:
        """
        y_norm is in source-domain coordinates [0, 2].
        Returns shape [1, 1, L_out, 2].
        """
        if level_len < 1:
            raise ValueError("level_len must be >= 1")

        if self.coord_mode == "centers":
            pos = y_norm * (level_len / 2.0) - 0.5
            if level_len == 1:
                gx = torch.zeros_like(pos)
            else:
                gx = 2.0 * ((pos + 0.5) / level_len) - 1.0
        else:
            pos = y_norm * ((level_len - 1) / 2.0)
            if level_len == 1:
                gx = torch.zeros_like(pos)
            else:
                gx = 2.0 * (pos / (level_len - 1)) - 1.0

        gy = torch.zeros_like(gx)
        return torch.stack((gx, gy), dim=-1).unsqueeze(0).unsqueeze(0)

    def _make_grid_from_unit_coords(self, t: torch.Tensor, level_len: int) -> torch.Tensor:
        """
        t is in unit coordinates [0, 1].
        Returns shape [1, 1, L_out, 2].
        """
        if level_len < 1:
            raise ValueError("level_len must be >= 1")

        if self.coord_mode == "centers":
            pos = t * level_len - 0.5
            if level_len == 1:
                gx = torch.zeros_like(pos)
            else:
                gx = 2.0 * ((pos + 0.5) / level_len) - 1.0
        else:
            pos = t * (level_len - 1)
            if level_len == 1:
                gx = torch.zeros_like(pos)
            else:
                gx = 2.0 * (pos / (level_len - 1)) - 1.0

        gy = torch.zeros_like(gx)
        return torch.stack((gx, gy), dim=-1).unsqueeze(0).unsqueeze(0)

    def _sample_1d(self, level: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        """
        level: [B, 1, L]
        grid:  [1, 1, L_out, 2] or [B, 1, L_out, 2]
        return: [B, L_out]
        """
        B = level.shape[0]
        if grid.shape[0] != B:
            grid = grid.expand(B, -1, -1, -1)

        level_2d = level.unsqueeze(2)  # [B, 1, 1, L]

        out = F.grid_sample(
            level_2d,
            grid,
            mode="bicubic",
            padding_mode="border",
            align_corners=self.align_corners,
        )

        return out[:, 0, 0, :]

    def downsample(self, x: torch.Tensor) -> torch.Tensor:

        x = x.transpose(-1, -2)
        if x.shape[-1] != self.src_len:
            raise ValueError(
                f"downsample expected last dim {self.src_len}, got {x.shape[-1]}"
            )

        x_flat = x.reshape(-1, 1, self.src_len)

        lvl1 = x_flat
        lvl2 = F.avg_pool1d(x_flat, kernel_size=2, stride=2)
        lvl3 = F.avg_pool1d(x_flat, kernel_size=3, stride=3)
        #lvl3 = F.interpolate(x_flat, self.lvl3_len, mode="area")

        y1 = self._sample_1d(lvl1, self.grid1)
        y2 = self._sample_1d(lvl2, self.grid2)
        y3 = self._sample_1d(lvl3, self.grid3)

        mask12 = self.mask12.expand(y1.shape[0], -1)
        a12 = self.a12.expand(y1.shape[0], -1)
        a23 = self.a23.expand(y1.shape[0], -1)

        out12 = (1.0 - a12) * y1 + a12 * y2
        out23 = (1.0 - a23) * y2 + a23 * y3
        out = torch.where(mask12, out12, out23)

        return out.reshape(*x.shape[:-1], self.dst_len).transpose(-1, -2)

    def upsample(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(-1, -2)
        if x.shape[-1] != self.dst_len:
            raise ValueError(
                f"upsample expected last dim {self.dst_len}, got {x.shape[-1]}"
            )

        x_flat = x.reshape(-1, 1, self.dst_len)
        out = self._sample_1d(x_flat, self.inv_grid)
        return out.reshape(*x.shape[:-1], self.src_len).transpose(-1, -2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.downsample(x)

    def extra_repr(self) -> str:
        return (
            f"src_len={self.src_len}, dst_len={self.dst_len}, p={self.p}, "
            f"coord_mode='{self.coord_mode}', newton_steps={self.newton_steps}"
        )
    
if __name__ == "__main__":
    
    import os

    from utils import config
    from utils.mdct.functional import mdct, imdct
    from utils.mdct.windows import vorbis
    from utils.dual_diffusion_utils import load_audio, save_audio, tensor_to_img, save_img
    from modules.formats.frequency_scale import get_mel_density

    output_path = os.path.join(config.DEBUG_PATH, "phase_recovery_test")
    os.makedirs(output_path, exist_ok=True)

    test_audio = load_audio(os.path.join(config.DEBUG_PATH, "moonsurgent.flac")).unsqueeze(0)[..., :65536*8-64]
    print("test_audio.shape", test_audio.shape)

    from modules.formats.mel_spec import MelSpec, MelSpecConfig
    mel_spec = MelSpec(MelSpecConfig(
        sample_rate=32000,
        ms_add_center_channel=False,
        ms_abs_exponent=0.25,
        ms_num_filters=64,
        ms_hop_length=512,
        ms_window_length=1024,
        ms_window_exponent=0.5
    ))

    save_img(tensor_to_img(mel_spec.raw_to_mel_spec(test_audio), flip_y=True), os.path.join(output_path, f"mel_spec.png"))

    
    p = 2
    wnd_size = 1024
    window = vorbis(wnd_size)
    psd = mdct(test_audio, window, padding=True, return_complex=True, last_frame=-2).abs().pow(0.25)

    mdct_hz = (torch.arange(wnd_size//2) + 0.5) / wnd_size * 32000
    psd /= get_mel_density(mdct_hz).pow(0.125)[None, None, :, None]

    #downsampled = variable_downsample_lastdim_half(psd.transpose(-1,-2), p=p).transpose(-1, -2)
    #upsampled = approximate_inverse_variable_downsample_lastdim_half(downsampled.transpose(-1,-2), p=p).transpose(-1, -2)

    save_img(tensor_to_img(psd, flip_y=True), os.path.join(output_path, f"psd.png"))

    x = psd
    for i in range(3):
        src_len = wnd_size // 2 // 2**i
        resample = ProgressiveResample(src_len=src_len, p=p, coord_mode="endpoints", newton_steps=4, device=psd.device)
        x = resample.downsample(x)
        save_img(tensor_to_img(x, flip_y=True), os.path.join(output_path, f"psd_down{i}.png"))

    for i in reversed(range(3)):
        src_len = wnd_size // 2 // 2**i
        resample = ProgressiveResample(src_len=src_len, p=p, coord_mode="endpoints", newton_steps=4, device=psd.device)
        x = resample.upsample(x)
        save_img(tensor_to_img(x, flip_y=True), os.path.join(output_path, f"psd_up{i}.png"))
    
    exit()
    upsampled = resample.upsample(downsampled)

    save_img(tensor_to_img(psd[:, :, :256], flip_y=True), os.path.join(output_path, f"psd.png"))
    
    save_img(tensor_to_img(upsampled[:, :, :256], flip_y=True), os.path.join(output_path, f"psd_upsampled.png"))

    exit()

    wnd_sizes = (128, 256, 512, 1024)
    psds = []; windows = []
    num_iterations = 100

    for i, wnd_size in enumerate(wnd_sizes):

        window = vorbis(wnd_size)
        windows.append(window)

        last_frame = -2 if wnd_size > wnd_sizes[0] else -1
        _mdct = mdct(test_audio, window, padding=True, return_complex=True, last_frame=last_frame)



        _imdct = imdct(_mdct.real, window, padding=True)


        psd = _mdct.abs()
        psds.append(psd)

        print(wnd_size, _mdct.shape, _imdct.shape)

        psd = psd.pow(0.25)
        psd = torch.nn.functional.interpolate(psd, (256,256), mode="area")
        
        
        psd_img = tensor_to_img(psd, flip_y=True)
        save_img(psd_img, os.path.join(output_path, f"psd_wnd{i}.png"))

    exit()

    recon = torch.randn_like(test_audio)

    for i in range(num_iterations):

        for wnd_size, psd, window in zip(wnd_sizes, psds, windows):
            _mdct = mdct(recon, window, padding=True, return_complex=True)
            _mdct *= psd / (_mdct.abs().add(1e-16))
            _imdct = imdct(_mdct.real, window, padding=True)
            recon = _imdct

    print("Reconstruction error:", torch.mean((test_audio - _imdct) ** 2).item())
    save_audio(recon.squeeze(0), 32000, os.path.join(output_path, "recon.flac"))
    save_audio(test_audio.squeeze(0), 32000, os.path.join(output_path, "test_audio.flac"))
