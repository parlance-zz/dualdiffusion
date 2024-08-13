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

import utils.config as config

import os

import numpy as np
import torch

from modules.unet_edm2 import MPFourier
from utils.dual_diffusion_utils import save_raw, save_raw_img


if __name__ == "__main__": # fourier embedding inner product test

    steps = 512
    cnoise = 128#192*4
    target_snr = 32
    sigma_data = 1
    sigma_max = 200
    sigma_min = sigma_data / target_snr
    

    #emb_fourier = MPFourier(cnoise, bandwidth=100)
    emb_fourier = MPFourier(cnoise, bandwidth=512, flavor="pos")
    
    sigma = torch.linspace(np.log(sigma_min), np.log(sigma_max), steps).exp()
    a = 1
    theta1 = np.arctan(a / sigma_max); theta0 = np.arctan(a / sigma_min)
    theta = torch.linspace(1, 0, steps) * (theta0 - theta1) + theta1
    sigma = theta.cos() / theta.sin() * a
    
    #noise_label = sigma.log() / 4
    noise_label = torch.arange(steps) / steps - 0.5#torch.linspace(-0.5, 0.5, steps)
    #noise_label = (sigma_data / sigma).atan()
    #noise_label = torch.arange(steps) / steps #torch.linspace(-0.5, 0.5, steps)

    c_skip = sigma_data ** 2 / (sigma ** 2 + sigma_data ** 2)
    c_out = sigma * sigma_data / (sigma ** 2 + sigma_data ** 2).sqrt()

    c_skip *= (sigma_data **2 + sigma ** 2).sqrt()
    emb = emb_fourier(noise_label)
    print(emb.shape)
    print(emb.std())
    emb = normalize(emb, dim=1)#*1.414
    print(emb.std())
    inner_products = (emb.view(1, steps, cnoise) * emb.view(steps, 1, cnoise)).sum(dim=2) / cnoise**0.5
    
    
    inner_products -= inner_products.amax()
    #inner_products /= inner_products.std()
    
    print(inner_products.amin())
    #inner_products /= 128

    inner_products = inner_products.exp()
    inner_products /= inner_products.amax()

    d_c_skip = c_skip[1:] - c_skip[:-1]
    d_c_out = c_out[1:] - c_out[:-1]
    #multi_plot((c_skip, "c_skip"), (d_c_skip.abs() + d_c_out.abs(), "c_skip*c_out"), added_plots={0: (c_out, "c_out")}, x_axis_range=(np.log(sigma_min), np.log(sigma_max)))
    #multi_plot((c_skip, "c_skip"), (c_skip.log() + c_out.log(), "c_skip*c_out"), added_plots={0: (c_out, "c_out")}, x_axis_range=(np.log(sigma_min), np.log(sigma_max)))

    debug_path = config.DEBUG_PATH
    if debug_path is not None:    
        save_raw(inner_products / inner_products.amax(), os.path.join(debug_path, "fourier_inner_products.raw"))
        inner_products_img = save_raw_img(inner_products, os.path.join(debug_path, "fourier_inner_products.png"))

        coverage = inner_products.sum(dim=0)
        save_raw(coverage / coverage.amax(), os.path.join(debug_path, "fourier_inner_products_coverage.raw"))

        cv2.imshow("sample / output", inner_products_img)
        cv2.waitKey(0)
