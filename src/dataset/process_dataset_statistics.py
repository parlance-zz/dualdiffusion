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
import json

import torch
import numpy as np
from tqdm.auto import tqdm

from utils.dual_diffusion_utils import (
    init_cuda, save_raw, load_safetensors,
    multi_plot, save_safetensors, dequantize_tensor
)


if __name__ == "__main__":

    init_cuda()

    DATASET_PATH = config.LATENTS_DATASET_PATH

    SPLIT_FILE = "train.jsonl"
    SIGMA_MAX = 200
    SIGMA_MIN = 1/32
    SIGMA_DATA = 1
    NUM_BINS = 1000

    hist_min = np.log(SIGMA_MIN)
    hist_max = np.log(SIGMA_MAX)

    hist = torch.zeros(NUM_BINS, dtype=torch.float32)
    debug_path = config.DEBUG_PATH
    split_metadata_file = os.path.join(DATASET_PATH, SPLIT_FILE)

    with open(os.path.join(DATASET_PATH, split_metadata_file), "r") as f:
        split_metadata = [json.loads(line) for line in f.readlines()]

    print(f"Processing {len(split_metadata)} samples from {split_metadata_file}...")
    progress_bar = tqdm(total=len(split_metadata))
    progress_bar.set_description(f"Split: {split_metadata_file}")

    for i, sample in enumerate(split_metadata):
        
        file_name = sample["file_name"]
        output_filename = f"{os.path.splitext(file_name)[0]}.safetensors"
        latents_dict = load_safetensors(os.path.join(DATASET_PATH, output_filename))
        latents = latents_dict["latents"].float()
        if hasattr(latents_dict, "offset_and_range"):
            latents = dequantize_tensor(latents, latents_dict["offset_and_range"]).float()
        
        rfft2_abs = (torch.fft.rfft2(latents * SIGMA_DATA, norm="ortho").abs()).log().clip(min=hist_min, max=hist_max)
        hist += torch.histc(rfft2_abs, bins=NUM_BINS, min=hist_min, max=hist_max) / len(split_metadata)

        progress_bar.update(1)
    progress_bar.close()
    
    max_bin = hist.argmax().item()
    sigma_mode = np.exp(max_bin / NUM_BINS * (hist_max - hist_min) + hist_min)
    print(f"sigma mode: {sigma_mode}")

    hist /= hist.sum()
    multi_plot((hist, "latents_abs_histogram"),
               (hist.log(), "ln_latents_abs_histogram"),
               x_axis_range=(hist_min, hist_max))
    if debug_path is not None:
        save_raw(hist, os.path.join(debug_path, "latents_abs_histogram.raw"))

    exit()

    statistics_file_path = os.path.join(DATASET_PATH, "statistics.safetensors")
    print(f"Saving statistics to {statistics_file_path}...")
    latents_statistics = {
        "ln_sigma_pdf": hist,
        "ln_sigma_range": torch.tensor([hist_min, hist_max]),
    }
    save_safetensors(latents_statistics, statistics_file_path)