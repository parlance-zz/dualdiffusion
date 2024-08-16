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
import math
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
        num_proc=test_params["dataset_num_proc"] if test_params["dataset_num_proc"] > 0 else None,
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
        pin_memory=True,
        persistent_workers=True if test_params["dataloader_num_workers"] > 0 else False,
        prefetch_factor=2 if test_params["dataloader_num_workers"] > 0 else None,
        drop_last=True,
    )

    print(f"Using dataset path {data_dir} with split '{test_params['split']}' and batch size {test_params['batch_size']}")
    print(f"{len(dataset['train'])} train samples ({dataset.num_filtered_samples['train']} filtered)")
    print(f"{len(dataset['validation'])} validation samples ({dataset.num_filtered_samples['validation']} filtered)")

    if test_params["dataloader_num_workers"] > 0:
        print(f"Using train dataloader with {test_params['dataloader_num_workers']} workers - prefetch factor = 2")

    num_processes = 1 #self.accelerator.num_processes
    gradient_accumulation_steps = 1
    num_process_steps_per_epoch = math.floor(len(test_dataloader) / num_processes)
    num_update_steps_per_epoch = math.ceil(num_process_steps_per_epoch / gradient_accumulation_steps)

    loading_start = datetime.now()
    for epoch in range(test_params["num_epochs"]):
        progress_bar = tqdm(total=num_update_steps_per_epoch)#, disable=not self.accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for _ in test_dataloader:
            progress_bar.update(1)

        progress_bar.close()

    print(f"Total loading time: {datetime.now() - loading_start}")
    print("Effective examples per second: {:.2f}".format(
        num_process_steps_per_epoch * test_params["batch_size"] / (datetime.now() - loading_start).total_seconds()))
    
if __name__ == "__main__":

    init_cuda()
    dataloader_test()