import utils.config as config

import os

import numpy as np
import torch

from models.unet_edm2 import MPFourier
from utils.dual_diffusion_utils import save_raw, save_raw_img


if __name__ == "__main__": # fourier embedding inner product test

    steps = 200
    cnoise = 192*4
    sigma_max = 80.
    sigma_min = 0.002

    emb_fourier = MPFourier(cnoise)
    noise_label = torch.linspace(np.log(sigma_max), np.log(sigma_min), steps) / 4

    emb = emb_fourier(noise_label)
    inner_products = (emb.view(1, steps, cnoise) * emb.view(steps, 1, cnoise)).sum(dim=2)

    debug_path = config.DEBUG_PATH
    if debug_path is not None:    
        save_raw(inner_products / inner_products.amax(), os.path.join(debug_path, "fourier_inner_products.raw"))
        save_raw_img(inner_products, os.path.join(debug_path, "fourier_inner_products.png"))

        coverage = inner_products.sum(dim=0)
        save_raw(coverage / coverage.amax(), os.path.join(debug_path, "fourier_inner_products_coverage.raw"))