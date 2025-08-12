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

from utils.mdct import mdct2, imdct2, vorbis, sin_window, kaiser_bessel_derived


if __name__ == "__main__":
    
    from utils import config

    import os

    from utils.dual_diffusion_utils import (
        init_cuda, tensor_to_img, img_to_tensor, save_img, load_img,
        quantize_tensor, dequantize_tensor, load_audio, save_audio
    )
    
    init_cuda()
    output_path = os.path.join(config.DEBUG_PATH, "mdct2d_test")

    #test_image = img_to_tensor(load_img(os.path.join(output_path, "test_img.png"))).cuda()
    test_image = img_to_tensor(load_img(os.path.join(output_path, "test_img_lenna.png"))).cuda()
    test_image = test_image[:, :, :512, :512] ** 1
    #test_image[:] = test_image.mean(dim=1, keepdim=True)
    #m = (test_image[:, 0] + test_image[:, 2]) / 2**0.5
    #s = (test_image[:, 0] - test_image[:, 2]) / 2**0.5
    #test_image[:, 0] = m
    #test_image[:, 2] = s

    save_img(tensor_to_img(test_image, recenter=False, rescale=False), os.path.join(output_path, "test_img_source.png"))

    block_width = 16
    window = sin_window(block_width, device=test_image.device)
    #window = vorbis(block_width, device=test_image.device)
    #window = kaiser_bessel_derived(block_width, beta=18, device=test_image.device)
    tformed = mdct2(test_image, window, return_complex=True)
    print("tformed_shape", tformed.shape)

    tformed_abs = tformed.abs()
    tformed /= tformed_abs + 1e-4

    #tformed_abs_dc = tformed_abs[:, :, 0, 0, :, :]
    quantized_abs, offsets = quantize_tensor(tformed_abs ** 0.47, 64)
    dequantized_abs = dequantize_tensor(quantized_abs, offsets) ** (1 / 0.47)
    #dequantized_abs[:, :, 0, 0, :, :] = tformed_abs_dc
    #dequantized_abs = tformed_abs

    tformed *= dequantized_abs + 1e-4

    untformed = imdct2(tformed, window)
    save_img(tensor_to_img(untformed, recenter=False, rescale=False), os.path.join(output_path, "test_img_untformed.png"))

    #tformed_img = tformed_abs.transpose(3, 4)
    
    tformed_img = dequantized_abs.transpose(3, 4)
    #tformed_img /= tformed_img.mean(dim=(1, 3, 5), keepdim=True)#.clip(min=1e-8)
    tformed_img = tformed_img.reshape(tformed.shape[0], tformed.shape[1], tformed.shape[2] * tformed_img.shape[3], tformed_img.shape[4] * tformed_img.shape[5])
    save_img(tensor_to_img(tformed_img, recenter=False, rescale=True), os.path.join(output_path, "_test_img_tformed.png"))
    