"""
import torch
import torch.distributions as dist
import time

# Create a random PDF tensor of dimension 1000
pdf_tensor = torch.rand(1000).to("cpu")
pdf_tensor /= pdf_tensor.sum()  # Normalize to make it a valid PDF

# Create a Categorical distribution from the PDF tensor
categorical_dist = dist.Categorical(probs=pdf_tensor)

# Measure the time taken to sample
start_time = time.time()
for i in range(9100):
    sampled_indices = categorical_dist.sample()
end_time = time.time()

#print("Sampled index:", sampled_indices.item())
print("Time taken for sampling:", end_time - start_time, "seconds")
exit()
"""

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

import os
import json

from dotenv import load_dotenv
import torch
import numpy as np
from tqdm.auto import tqdm

from dual_diffusion_utils import init_cuda, save_raw, load_safetensors, multi_plot, save_safetensors

DATASET_PATH = os.environ.get("LATENTS_DATASET_PATH", "./dataset/latents")
SPLIT_FILE = "train.jsonl"
SIGMA_MAX = 80
SIGMA_MIN = 0.002
SIGMA_DATA = 0.5
NUM_BINS = 1000

if __name__ == "__main__":

    init_cuda()
    load_dotenv(override=True)

    hist_min = np.log(SIGMA_MIN)
    hist_max = np.log(SIGMA_MAX)

    hist = torch.zeros(NUM_BINS, dtype=torch.float32)
    debug_path = os.environ.get("DEBUG_PATH", None)
    split_metadata_file = os.path.join(DATASET_PATH, SPLIT_FILE)

    with open(os.path.join(DATASET_PATH, split_metadata_file), "r") as f:
        split_metadata = [json.loads(line) for line in f.readlines()]

    print(f"Processing {len(split_metadata)} samples from {split_metadata_file}...")
    progress_bar = tqdm(total=len(split_metadata))
    progress_bar.set_description(f"Split: {split_metadata_file}")

    for i, sample in enumerate(split_metadata):

        file_name = sample["file_name"]
        output_filename = f"{os.path.splitext(file_name)[0]}.safetensors"
        latents = load_safetensors(os.path.join(DATASET_PATH, output_filename))["latents"].float()

        rfft2_abs = (torch.fft.rfft2(latents, norm="ortho").abs() * SIGMA_DATA).log().clip(min=hist_min, max=hist_max)
        hist += torch.histc(rfft2_abs, bins=NUM_BINS, min=hist_min, max=hist_max) / len(split_metadata)

        progress_bar.update(1)
    progress_bar.close()
    
    max_bin = hist.argmax().item()
    sigma_mode = np.exp(max_bin / NUM_BINS * (hist_max - hist_min) + hist_min)
    print(f"sigma mode: {sigma_mode}")

    hist /= hist.sum()
    multi_plot((hist, "latents_ln_sigma_histogram"))
    if debug_path is not None:
        save_raw(hist, os.path.join(debug_path, "latents_ln_sigma_histogram.raw"))

    statistics_file_path = os.path.join(DATASET_PATH, "statistics.safetensors")
    print(f"Saving statistics to {statistics_file_path}...")
    latents_statistics = {
        "ln_sigma_pdf": hist,
        "ln_sigma_range": torch.tensor([hist_min, hist_max]),
    }
    save_safetensors(latents_statistics, statistics_file_path)