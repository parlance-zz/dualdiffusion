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
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DoG_Loss2D_Config:

    channels: int = 2
    kernel_sizes: list[int] = (3,5,7,11,17,27)
    kernel_sigma: float = 0.34

class GaussianKernel(nn.Module):

    def __init__(self, kernel_size: int, channels: int, sigma: float) -> None:
        super().__init__()

        self.channels = channels
        self.padding = kernel_size // 2

        kernel = self._gaussian_kernel_2d(kernel_size, sigma).expand(channels, 1, 1, kernel_size, kernel_size)
        self.register_buffer("kernel", kernel, persistent=False)

    def _gaussian_kernel_2d(self, kernel_size: int, sigma: float) -> torch.Tensor:
        coords = torch.linspace(-1, 1, kernel_size)
        kernel = torch.exp(-(coords.view(1, -1)**2 + coords.view(-1, 1)**2) / (2 * sigma**2))
        return kernel / kernel.sum()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        padded_x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode="reflect")
        return F.conv2d(padded_x, self.kernel.squeeze(2), groups=self.channels, padding=0)
    
    def to(self, *args, memory_format=None, **kwargs):

        if memory_format is torch.channels_last_3d:
            memory_format = torch.channels_last

        super().to(*args, memory_format=memory_format, **kwargs)

class DoG_Loss2D(nn.Module):

    def __init__(self, config: DoG_Loss2D_Config) -> None:
        super().__init__()

        self.config = config
        self.kernels = nn.ModuleList()

        for kernel_size in config.kernel_sizes:
            self.kernels.append(GaussianKernel(kernel_size, config.channels, config.kernel_sigma))

        self.mse_logvar = torch.nn.Parameter(torch.zeros(len(self.config.kernel_sizes)))
                                               
    def get_kernel_images(self) -> list[torch.Tensor]:
        return [tensor_to_img(kernel.kernel.squeeze(2)) for kernel in self.kernels]

    def get_dogs(self, x: torch.Tensor) -> list[torch.Tensor]:

        filtered_outputs = []
        for kernel in self.kernels:
            filtered_outputs.append(kernel(x))

        dog_outputs = []
        for i in range(1, len(filtered_outputs)):
            dog = filtered_outputs[i - 1] - filtered_outputs[i]
            dog_outputs.append(dog)

        return dog_outputs + [filtered_outputs[-1]]
    
    def forward(self, sample: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        
        with torch.no_grad():
            target_dogs = self.get_dogs(target)

        nll_loss = torch.zeros(sample.shape[0], device=sample.device)
        dog_losses = []

        for i, target_dog in enumerate(target_dogs):
            sample_dog = sample[:, i*2:i*2+2, :, :]
            dog_loss = F.mse_loss(sample_dog, target_dog, reduction="none").mean(dim=(1,2,3))
            nll_loss = nll_loss + (dog_loss / self.mse_logvar[i].exp() + self.mse_logvar[i])
            dog_losses.append(dog_loss.detach())

        return nll_loss, dog_losses

    @torch.no_grad()
    def reconstruct(self, sample: torch.Tensor) -> torch.Tensor:
        
        recon_image = torch.zeros_like(sample[:, 0:2])
        for i in range(len(self.config.kernel_sizes)):

            dog = sample[:, i*2:i*2+2, :, :]
            sample_energy = dog.var(dim=(1,2,3), keepdim=True)
            target_energy = sample_energy + self.mse_logvar[i].exp()

            mean = dog.mean(dim=(1,2,3), keepdim=True)
            recon_image = recon_image + (dog - mean) * (target_energy / sample_energy).sqrt() + mean

        return recon_image

if __name__ == "__main__":

    from utils import config
    import os
    from utils.dual_diffusion_utils import tensor_to_img, save_img

    output_path = os.path.join(config.DEBUG_PATH, "dog_test")

    blur_dog = DoG_Loss2D(DoG_Loss2D_Config())
    image = torch.rand(1, 2, 256, 256)
    dogs = blur_dog.get_dogs(image)

    for i, dog in enumerate(dogs):
        save_img(tensor_to_img(dog), os.path.join(output_path, f"dog_{i}.png"))

    for i, img in enumerate(blur_dog.get_kernel_images()):
        save_img(img, os.path.join(output_path, f"kernel_{i}.png"))

    recon_image = blur_dog.reconstruct(image)
    save_img(tensor_to_img(image), os.path.join(output_path, "image.png"))
    save_img(tensor_to_img(recon_image), os.path.join(output_path, "image_recon.png"))
