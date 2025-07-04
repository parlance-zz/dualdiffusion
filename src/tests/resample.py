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

from utils import config

import os

import torch

from modules.mp_tools import lowpass_2d, FilteredDownsample2D
from utils.dual_diffusion_utils import (
    init_cuda, tensor_to_img, img_to_tensor, save_img, load_img
)


def lowpass_downsample(x: torch.Tensor, blur_width: float, circular: bool) -> torch.Tensor:
    x = lowpass_2d(x, blur_width, circular)
    return torch.nn.functional.avg_pool2d(x, 8)

def filtered_downsample(x: torch.Tensor) -> torch.Tensor:
    blur = FilteredDownsample2D(x.shape[1], stride=8)
    return blur(x)

@torch.inference_mode()
def resample_test() -> None:

    torch.manual_seed(0)

    output_path = os.path.join(config.DEBUG_PATH, "resample_test")
    os.makedirs(output_path, exist_ok=True)
    
    def test(down_func: callable, name: str) -> None:
        
        for i in range(17):
            t1 = torch.zeros((1, 1, 256, 256))
            t1[..., 128+i, 128] = 1
            t1 = down_func(t1) * 16

            save_img(tensor_to_img(t1, recenter=False, rescale=False),
                     os.path.join(output_path, name, f"h_{i:02d}.png"))
            
        for i in range(17):
            t2 = torch.zeros((1, 1, 256, 256))
            t2[..., 128, 128+i] = 1
            t2 = down_func(t2) * 16

            save_img(tensor_to_img(t2, recenter=False, rescale=False),
                     os.path.join(output_path, name, f"w_{i:02d}.png"))

        for i in range(17):
            t3 = torch.zeros((1, 1, 256, 256))
            t3[..., 128+i, 128+i] = 1
            t3 = down_func(t3) * 16

            save_img(tensor_to_img(t3, recenter=False, rescale=False),
                     os.path.join(output_path, name, f"hw_{i:02d}.png"))

        t4 = torch.zeros((1, 1, 256, 256))
        t4[..., 0, 128] = 1
        t4 = down_func(t4) * 16
        save_img(tensor_to_img(t4, recenter=False, rescale=False),
                    os.path.join(output_path, name, f"edge_t.png"))
                
        t5 = torch.zeros((1, 1, 256, 256))
        t5[..., 128, 0] = 1
        t5 = down_func(t5) * 16
        save_img(tensor_to_img(t5, recenter=False, rescale=False),
                    os.path.join(output_path, name, f"edge_l.png"))
        
        t6 = torch.zeros((1, 1, 256, 256))
        t6[..., 255, 128] = 1
        t6 = down_func(t6) * 16
        save_img(tensor_to_img(t6, recenter=False, rescale=False),
                    os.path.join(output_path, name, f"edge_b.png"))
        
        t7 = torch.zeros((1, 1, 256, 256))
        t7[..., 128, 255] = 1
        t7 = down_func(t7) * 16
        save_img(tensor_to_img(t7, recenter=False, rescale=False),
                    os.path.join(output_path, name, f"edge_r.png"))

        if os.path.isfile(os.path.join(output_path, "test_img.png")):
            test_img = load_img(os.path.join(output_path, "test_img.png"))
            t_img = img_to_tensor(test_img)
            for i in range(17):
                t8 = torch.roll(t_img, shifts=(i, i), dims=(-1, -2))
                t8 = down_func(t8)
                save_img(tensor_to_img(t8, recenter=False, rescale=False),
                        os.path.join(output_path, name, f"test_img_{i:02d}.png"))

    test(lambda x: lowpass_downsample(x, 8, True), "lowpass_8_circular")
    test(lambda x: lowpass_downsample(x, 8, False), "lowpass_8_square")
    test(lambda x: lowpass_downsample(x, 12, True), "lowpass_12_circular")
    test(lambda x: lowpass_downsample(x, 12, False), "lowpass_12_square")
    test(lambda x: lowpass_downsample(x, 16, True), "lowpass_16_circular")
    test(lambda x: lowpass_downsample(x, 16, False), "lowpass_16_square")

    test(filtered_downsample, "filtered_downsample")

    blur = FilteredDownsample2D(1, stride=8)
    save_img(tensor_to_img(blur.filter), os.path.join(output_path, f"blur_filter.png"))


if __name__ == "__main__":

    init_cuda()
    resample_test()