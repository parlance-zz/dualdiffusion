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
from datetime import datetime

import torch
from tqdm.auto import tqdm

from utils.dual_diffusion_utils import init_cuda
from training.dataset import DatasetConfig, DualDiffusionDataset

@torch.inference_mode()
def dataloader_test():

    torch.manual_seed(0)

    test_params = config.load_json(
        os.path.join(config.CONFIG_PATH, "tests", "dataloader.json"))
    
    if test_params["use_pre_encoded_latents"]:
        data_dir = config.LATENTS_DATASET_PATH
        sample_crop_width = test_params["latents_crop_width"]
    else:
        data_dir = config.DATASET_PATH
        sample_crop_width = test_params["sample_crop_width"]
    
    dataset_config = DatasetConfig(
        data_dir=data_dir,
        cache_dir=config.CACHE_PATH,
        num_proc=test_params["dataset_num_proc"],
        sample_crop_width=sample_crop_width,
        use_pre_encoded_latents=test_params["use_pre_encoded_latents"],
        t_scale=test_params["t_scale"],
        filter_invalid_samples=test_params["filter_invalid_samples"]
    )
    
    dataset_preprocessing_start = datetime.now()
    dataset = DualDiffusionDataset(dataset_config)
    print(f"Dataset preprocessing time: {datetime.now() - dataset_preprocessing_start}")

    test_dataloader = torch.utils.data.DataLoader(
        dataset[test_params["split"]],
        shuffle=True,
        batch_size=test_params["batch_size"],
        num_workers=test_params["dataloader_num_workers"],
        pin_memory=test_params["pin_memory"],
        persistent_workers=True if test_params["dataloader_num_workers"] > 0 else False,
        prefetch_factor=test_params["prefetch_factor"],
        drop_last=True,
    )

    print(f"Using dataset path {data_dir} with split '{test_params['split']}' and batch size {test_params['batch_size']}")
    print(f"  {len(dataset['train'])} train samples ({dataset.num_filtered_samples['train']} filtered)")
    print(f"  {len(dataset['validation'])} validation samples ({dataset.num_filtered_samples['validation']} filtered)")

    print(f"Using train dataloader with {test_params['dataloader_num_workers'] or 0} workers")
    print(f"  prefetch_factor = {test_params['prefetch_factor']}")
    print(f"  pin_memory = {test_params['pin_memory']}")

    loading_start = datetime.now()
    progress_bar = tqdm(total=len(test_dataloader))

    for epoch in range(test_params["num_epochs"]):
        progress_bar.set_description(f"Epoch {epoch}")
        for _ in test_dataloader: progress_bar.update(1)
        progress_bar.reset()

    print(f"Total loading time: {datetime.now() - loading_start}")
    print("Effective examples per second: {:.2f}".format(
        len(test_dataloader) * test_params["batch_size"] / (datetime.now() - loading_start).total_seconds()))
    
if __name__ == "__main__":

    init_cuda()
    dataloader_test()