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

from modules.mp_tools import MPFourier
from utils.dual_diffusion_utils import (
    tensor_to_img, save_img, save_tensor_raw, init_cuda, show_img
)

@torch.inference_mode()
def mp_fourier_test():

    torch.manual_seed(0)

    test_params = config.load_json(
        os.path.join(config.CONFIG_PATH, "tests", "mp_fourier.json"))
    
    steps = test_params["steps"]
    emb_dim = test_params["emb_dim"]
    emb_scale = test_params["emb_scale"]
    softmax = test_params["softmax"]
    sigma_data = test_params["sigma_data"]
    sigma_max = test_params["sigma_max"]
    sigma_min = test_params["sigma_min"]
    bandwidth = test_params["bandwidth"]
    sigma_scale = test_params["sigma_scale"]

    emb_fourier = MPFourier(emb_dim, bandwidth=bandwidth)
    
    if sigma_scale == "log_linear":
        sigma = torch.linspace(np.log(sigma_min), np.log(sigma_max), steps).exp()
    elif sigma_scale == "log_sech":
        theta1 = np.arctan(sigma_data / sigma_max); theta0 = np.arctan(sigma_data / sigma_min)
        theta = torch.linspace(1, 0, steps) * (theta0 - theta1) + theta1
        sigma = theta.cos() / theta.sin() * sigma_data
    
    emb = emb_fourier(sigma.log() / 4) * emb_scale / emb_dim**0.5
    inner_products = (emb.view(1, steps, emb_dim) * emb.view(steps, 1, emb_dim)).sum(dim=2)
    
    if softmax:
        inner_products -= inner_products.amax()
        inner_products = inner_products.exp()
    else:
        inner_products /= inner_products.amax()

    debug_path = config.DEBUG_PATH
    if debug_path is not None:    
        test_output_path = os.path.join(debug_path, "mp_fourier_test")
        print(f"Saving test output to {test_output_path}")

        coverage = inner_products.sum(dim=0)
        save_tensor_raw(coverage / coverage.amax(), os.path.join(test_output_path, "coverage.raw"))

        save_tensor_raw(inner_products, os.path.join(test_output_path, "inner_products.raw"))
        inner_products_img = tensor_to_img(inner_products, colormap=True).squeeze(2)
        
        save_img(inner_products_img, os.path.join(test_output_path, "inner_products.png"))
        show_img(inner_products_img, "inner_products.png", wait=0)

if __name__ == "__main__":

    init_cuda()
    mp_fourier_test()