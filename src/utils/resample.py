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

from modules.mp_tools import mp_silu


def _kaiser_windowed_sinc_1d(size, cutoff, beta):

    x = (torch.arange(size) - (size-1)/2) * torch.pi * cutoff
    sinc = torch.where(x == 0, torch.tensor(1.), torch.sin(x) / x)
    
    kernel = sinc * torch.kaiser_window(size, beta=beta, periodic=False)
    return kernel / kernel.sum()

class FilteredResample1D(torch.nn.Module):

    def __init__(self, k_size: int = 7, stride: int = 2,
            cutoff: float = 0.5, beta: float = 1.5, gain: float = 1) -> None:
        super().__init__()

        self.k_size = k_size
        self.stride = stride
        self.beta = beta

        self.register_buffer("kernel",
            _kaiser_windowed_sinc_1d(k_size, cutoff, beta) * gain, persistent=False)

        even = k_size % 2 == 0; hk_size = k_size // 2
        if stride == 1:
            self.pad_w = torch.nn.ReflectionPad2d((hk_size, hk_size - even, 0, 0))
        else:
            self.pad_w = torch.nn.ReflectionPad2d((hk_size - even, hk_size, 0, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        kernel: torch.Tensor = self.kernel
        original_dtype = x.dtype

        kw = kernel[None, None, None, :].expand(x.shape[1], 1, 1, self.k_size)
        x = torch.nn.functional.conv2d(self.pad_w(x), kw, groups=x.shape[1], stride=(1,self.stride))

        return x.to(dtype=original_dtype)

    def get_filter(self) -> torch.Tensor:
        return self.kernel
    
    def get_window(self) -> torch.Tensor:
        return torch.kaiser_window(self.k_size, beta=self.beta, periodic=False)

class FilteredDownsample1D(FilteredResample1D):

    def __init__(self, k_size: int = 7, beta: float = 1.5, factor: int = 2) -> None:
        super().__init__(k_size, factor, 1/factor, beta, gain=1)

class FilteredUpsample1D(FilteredResample1D):

    def __init__(self, k_size: int = 15, beta: float = 1.5, factor: int = 2) -> None:
        super().__init__(k_size, 1, 1/factor, beta, gain=factor)
        
        self.factor = factor
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        b,c,h,w = x.shape
        y = torch.zeros((b, c, h, w*self.factor), device=x.device, dtype=x.dtype)
        y[..., ::self.factor] = x

        return super().forward(y)

class FilteredResample2D(torch.nn.Module):

    def __init__(self, k_size: int = 7, stride: int = 2,
            cutoff: float = 0.5, beta: float = 1.5, gain: float = 1) -> None:
        super().__init__()

        self.k_size = k_size
        self.stride = stride
        self.beta = beta

        self.register_buffer("kernel",
            _kaiser_windowed_sinc_1d(k_size, cutoff, beta) * gain, persistent=False)

        even = k_size % 2 == 0; hk_size = k_size // 2 ; h_stride = stride // 2
        if stride == 1:
            self.pad_w = torch.nn.ReflectionPad2d((hk_size, hk_size - even, 0, 0))
            self.pad_h = torch.nn.ReflectionPad2d((0, 0, hk_size, hk_size - even))
        else:
            self.pad_w = torch.nn.ReflectionPad2d((hk_size - even, hk_size, 0, 0))
            self.pad_h = torch.nn.ReflectionPad2d((0, 0, hk_size - even, hk_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        kernel: torch.Tensor = self.kernel

        kw = kernel[None, None, None, :].expand(x.shape[1], 1, 1, self.k_size)
        x = torch.nn.functional.conv2d(self.pad_w(x), kw, groups=x.shape[1], stride=(1,self.stride))

        kh = kernel[None, None, :, None].expand(x.shape[1], 1, self.k_size, 1)
        x = torch.nn.functional.conv2d(self.pad_h(x), kh, groups=x.shape[1], stride=(self.stride,1))

        return x

    def get_filter(self) -> torch.Tensor:
        return torch.outer(self.kernel, self.kernel)
    
    def get_window(self) -> torch.Tensor:
        window = torch.kaiser_window(self.k_size, beta=self.beta, periodic=False)
        return torch.outer(window, window)

class FilteredDownsample2D(FilteredResample2D):

    def __init__(self, k_size: int = 7, beta: float = 1.5, factor: int = 2) -> None:
        super().__init__(k_size, factor, 1/factor, beta, gain=1)

class FilteredUpsample2D(FilteredResample2D):

    def __init__(self, k_size: int = 15, beta: float = 1.5, factor: int = 2) -> None:
        super().__init__(k_size, 1, 1/factor, beta, gain=factor)
        
        self.factor = factor
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        b,c,h,w = x.shape
        y = torch.zeros((b, c, h*self.factor, w*self.factor), device=x.device, dtype=x.dtype)
        y[..., ::self.factor, ::self.factor] = x

        return super().forward(y)

class Filtered_MP_Silu_2D(torch.nn.Module):

    def __init__(self, k_size: int = 7, beta: float = 1.5) -> None:
        super().__init__()

        self.downsample = FilteredDownsample2D(k_size=k_size, beta=beta, factor=2)
        self.upsample = FilteredUpsample2D(k_size=k_size*2+k_size%2, beta=beta, factor=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.downsample(mp_silu(self.upsample(x)))

class FilteredResample3D(torch.nn.Module):

    def __init__(self, k_size: int = 7, stride: int = 2,
            cutoff: float = 0.5, beta: float = 1.5, gain: float = 1) -> None:
        super().__init__()

        self.k_size = k_size
        self.stride = stride
        self.beta = beta

        self.register_buffer("kernel",
            _kaiser_windowed_sinc_1d(k_size, cutoff, beta) * gain, persistent=False)
        
        even = k_size % 2 == 0; hk_size = k_size // 2 ; h_stride = stride // 2
        self.pad_w = torch.nn.ReflectionPad3d((hk_size, hk_size - even, 0, 0, 0, 0))
        self.pad_h = torch.nn.ReflectionPad3d((0, 0, hk_size, hk_size - even, 0, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        kw = self.kernel[None, None, None, None, :].expand(x.shape[1], 1, 1, 1, self.k_size)
        x = torch.nn.functional.conv3d(self.pad_w(x), kw, groups=x.shape[1], stride=(1,1,self.stride))

        kh = self.kernel[None, None, None, :, None].expand(x.shape[1], 1, 1, self.k_size, 1)
        x = torch.nn.functional.conv3d(self.pad_h(x), kh, groups=x.shape[1], stride=(1,self.stride,1))

        return x

    def get_filter(self) -> torch.Tensor:
        return torch.outer(self.kernel, self.kernel)
    
    def get_window(self) -> torch.Tensor:
        window = torch.kaiser_window(self.k_size, beta=self.beta, periodic=False)
        return torch.outer(window, window)
    
class FilteredDownsample3D(FilteredResample3D):

    def __init__(self, k_size: int = 7, beta: float = 1.5, factor: int = 2) -> None:
        super().__init__(k_size, factor, 1/factor, beta, gain=factor**0.5)

class FilteredUpsample3D(FilteredResample3D):

    def __init__(self, k_size: int = 15, beta: float = 1.5, factor: int = 2) -> None:
        super().__init__(k_size, 1, 1/factor, beta, gain=factor**0.5)
        
        self.factor = factor
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        b,c,z,h,w = x.shape
        y = torch.zeros((b, c, z, h*self.factor, w*self.factor), device=x.device, dtype=x.dtype)
        y[..., ::self.factor, ::self.factor] = x

        return super().forward(y)

class Filtered_MP_Silu_3D(torch.nn.Module):

    def __init__(self, k_size: int = 7, beta: float = 1.5) -> None:
        super().__init__()

        self.downsample = FilteredDownsample3D(k_size=k_size, beta=beta, factor=2)
        self.upsample = FilteredUpsample3D(k_size=k_size*2+k_size%2, beta=beta, factor=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.downsample(mp_silu(self.upsample(x)))

class FilteredResample1D3(torch.nn.Module):

    def __init__(self, k_size: int = 7, stride: int = 2,
            cutoff: float = 0.5, beta: float = 1.5, gain: float = 1) -> None:
        super().__init__()

        self.k_size = k_size
        self.stride = stride
        self.beta = beta

        self.register_buffer("kernel",
            _kaiser_windowed_sinc_1d(k_size, cutoff, beta) * gain, persistent=False)

        even = k_size % 2 == 0; hk_size = k_size // 2
        if stride == 1:
            self.pad_w = torch.nn.ReflectionPad3d((hk_size, hk_size - even, 0, 0, 0, 0))
        else:
            self.pad_w = torch.nn.ReflectionPad3d((hk_size - even, hk_size, 0, 0, 0, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        kernel: torch.Tensor = self.kernel
        original_dtype = x.dtype

        kw = kernel[None, None, None, None, :].expand(x.shape[1], 1, 1, 1, self.k_size)
        x = torch.nn.functional.conv3d(self.pad_w(x), kw, groups=x.shape[1], stride=(1,1,self.stride))

        return x.to(dtype=original_dtype)

    def get_filter(self) -> torch.Tensor:
        return self.kernel
    
    def get_window(self) -> torch.Tensor:
        return torch.kaiser_window(self.k_size, beta=self.beta, periodic=False)

class FilteredDownsample1D3(FilteredResample1D3):

    def __init__(self, k_size: int = 7, beta: float = 1.5, factor: int = 2) -> None:
        super().__init__(k_size, factor, 1/factor, beta, gain=1)

class FilteredUpsample1D3(FilteredResample1D3):

    def __init__(self, k_size: int = 15, beta: float = 1.5, factor: int = 2) -> None:
        super().__init__(k_size, 1, 1/factor, beta, gain=factor)
        
        self.factor = factor
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        b,c,z,h,w = x.shape
        y = torch.zeros((b, c, z, h, w*self.factor), device=x.device, dtype=x.dtype)
        y[..., ::self.factor] = x

        return super().forward(y)


if __name__ == "__main__":
    
    from utils import config

    import os

    from utils.dual_diffusion_utils import (
        init_cuda, tensor_to_img, img_to_tensor, save_img, load_img
    )
    
    init_cuda()
    output_path = os.path.join(config.DEBUG_PATH, "resample_test3")

    #beta = 1.5
    #k_size = 6

    beta = 3.437
    k_size = 23
    factor = 2

    downsample = FilteredDownsample1D(k_size=k_size, beta=beta, factor=factor).cuda().to(dtype=torch.bfloat16)
    upsample = FilteredUpsample1D(k_size=k_size*factor+k_size%factor, beta=beta, factor=factor).cuda().to(dtype=torch.bfloat16)

    #test_data = torch.randn((1, 8, 2, 65536), device="cuda", dtype=torch.bfloat16)
    #print(downsample(test_data).std(), downsample(downsample(test_data)).std())
    #print(upsample(test_data).std(), upsample(upsample(test_data)).std())
    #exit()

    downsample = FilteredDownsample2D(k_size=k_size, beta=beta, factor=factor).cuda()
    upsample = FilteredUpsample2D(k_size=k_size*factor+k_size%factor, beta=beta, factor=factor).cuda()
    filtered_silu = Filtered_MP_Silu_2D(k_size=k_size, beta=beta).cuda()

    save_img(tensor_to_img(downsample.get_filter()), os.path.join(output_path, "__down_filter_kernel.png"))
    save_img(tensor_to_img(downsample.get_window()), os.path.join(output_path, "__down_filter_window.png"))
    save_img(tensor_to_img(upsample.get_filter()), os.path.join(output_path, "__up_filter_kernel.png"))
    save_img(tensor_to_img(upsample.get_window()), os.path.join(output_path, "__up_filter_window.png"))

    #test_image = img_to_tensor(load_img(os.path.join(output_path, "_test_latents.png")))
    test_image = img_to_tensor(load_img(os.path.join(output_path, "__test_img.png"))).cuda()

    #"""
    for i in range(3):
        print("pre-upsample std:", test_image.std())
        test_image = upsample(test_image)
        print("post-upsample std:", test_image.std())
    for i in range(3):
        print("pre-downsample std:", test_image.std())
        test_image = downsample(test_image)
        print("post-downsample std:", test_image.std())
    save_img(tensor_to_img(test_image, recenter=False, rescale=False), os.path.join(output_path, "__test_img_area.png"))
    exit()
    #"""

    for i in range(17):

        img = test_image
        for i in range(3):
            img = upsample(img)

        img = torch.roll(img, shifts=(i, i), dims=(-1, -2))

        for i in range(3):
            img = downsample(img)

        save_img(tensor_to_img(img, recenter=False, rescale=False), os.path.join(output_path, f"test_img_{i:02d}.png"))