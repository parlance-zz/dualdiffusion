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
from typing import Optional

import torch

from modules.daes.dae_edm2_j4 import DAE_J4
from utils.resample import FilteredDownsample2D, FilteredUpsample2D


def random_crop_8px(tensor: torch.Tensor, x_offsets: Optional[torch.Tensor] = None,
        y_offsets: Optional[torch.Tensor ] = None)-> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Randomly crop each image in the batch by an offset up to 8 pixels in both height and width,
    or use provided offsets.

    Args:
        tensor (torch.Tensor): Input tensor of shape (b, c, h, w)
        x_offsets (torch.LongTensor, optional): Tensor of shape (b,) with horizontal offsets
        y_offsets (torch.LongTensor, optional): Tensor of shape (b,) with vertical offsets

    Returns:
        cropped (torch.Tensor): Cropped tensor of shape (b, c, h - 8, w - 8)
        (y_offsets, x_offsets) (Tuple[torch.LongTensor, torch.LongTensor]): The offsets used
    """
    b, c, h, w = tensor.shape
    assert h >= 8 and w >= 8, "Input height and width must be at least 8."

    if x_offsets is None:
        x_offsets = torch.randint(1, 9, (b,), device=tensor.device)
    if y_offsets is None:
        y_offsets = torch.randint(1, 9, (b,), device=tensor.device)

    # Crop each image using offsets
    cropped = torch.stack([
        tensor[i, :, y_offsets[i]:y_offsets[i] + h - 8, x_offsets[i]:x_offsets[i] + w - 8]
        for i in range(b)
    ], dim=0)

    return cropped, x_offsets, y_offsets

@dataclass
class EquivarianceLoss_Config:

    levels: int = 4
    filter_beta: float = 1.5
    filter_k_size: int = 7

class EquivarianceLoss:

    def __init__(self, config: EquivarianceLoss_Config, device: torch.device) -> None:
        super().__init__()

        self.config = config
        
        self.downsample = FilteredDownsample2D(k_size=config.filter_k_size, beta=config.filter_beta).to(device=device)
        self.upsample = FilteredUpsample2D(k_size=config.filter_k_size*2+config.filter_k_size%2, beta=config.filter_beta).to(device=device)
        
    def equivariance_loss(self, dae: DAE_J4, mel_spec: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:

        with torch.no_grad():
            mel_spec_cropped, x_offsets, y_offsets = random_crop_8px(mel_spec)
            latents_upsampled = latents
            for _ in range(self.config.levels - 1):
                latents_upsampled = self.upsample(latents_upsampled)
            
            latents_croppped, _, _ = random_crop_8px(latents_upsampled, x_offsets, y_offsets)

            latents_downsampled = latents_croppped
            for _ in range(self.config.levels - 1):
                latents_downsampled = self.downsample(latents_downsampled)
            latents_downsampled = latents_downsampled.detach()

        latents2 = dae.encode(mel_spec_cropped, None, training=True)
        latents2 = (
            latents2 / latents2.std(dim=(1,2,3), keepdim=True).detach() * latents_downsampled.std(dim=(1,2,3), keepdim=True)
            - latents2.mean(dim=(1,2,3), keepdim=True).detach() + latents_downsampled.mean(dim=(1,2,3), keepdim=True)
        )

        return torch.nn.functional.l1_loss(latents2, latents_downsampled, reduction="none").mean(dim=(1,2,3))

    def compile(self, **kwargs) -> None:
        self.equivariance_loss = torch.compile(self.equivariance_loss, **kwargs)


