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

# todo: t_scale should be redone, it isn't calculated correctly with pre-encoded latents

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import safetensors.torch as ST
from datasets import load_dataset

from utils.dual_diffusion_utils import load_audio, dequantize_tensor

@dataclass
class DatasetTransformConfig:

    sample_rate: int
    sample_raw_channels: int
    sample_raw_crop_width: int
    use_pre_encoded_latents: bool
    latents_crop_width: int
    t_scale: Optional[float] = None

@dataclass
class DatasetConfig:
     
     data_dir: str
     cache_dir: str
     sample_rate: int
     sample_raw_channels: int
     sample_raw_crop_width: int
     use_pre_encoded_latents: bool
     latents_crop_width: int
     num_proc: Optional[int] = None
     t_scale: Optional[float] = None
     filter_invalid_samples: Optional[bool] = False

class DatasetTransform(torch.nn.Module):

    def __init__(self, dataset_transform_config: DatasetTransformConfig) -> None:
        super().__init__()
        self.config = dataset_transform_config
    
    @torch.no_grad()
    def get_t(self, t: float) -> float:
        return t / self.config.sample_raw_crop_width * self.config.t_scale - self.config.t_scale/2
    
    @torch.no_grad()
    def __call__(self, examples: dict) -> dict:

        samples = []
        paths = []
        game_ids = []
        t_ranges = []
        #author_ids = []

        num_examples = len(next(iter(examples.values())))
        examples = [{key: examples[key][i] for key in examples} for i in range(num_examples)]

        for train_sample in examples:
            
            game_id = train_sample["game_id"]
            #author_id = train_sample["author_id"]

            if self.config.use_pre_encoded_latents:
                file_path = train_sample["latents_file_name"]
                with ST.safe_open(file_path, framework="pt") as f:
                    latents_slice = f.get_slice("latents")
                    latents_shape = latents_slice.get_shape()

                    latents_idx = np.random.randint(0, latents_shape[0]) # get random variation
                    t_offset = np.random.randint(0, latents_shape[-1] - self.config.latents_crop_width + 1)
                    sample = latents_slice[latents_idx, ..., t_offset:t_offset + self.config.latents_crop_width]

                    try:
                        offset_and_range = f.get_slice("offset_and_range")
                        sample = dequantize_tensor(sample, offset_and_range[latents_idx])
                    except Exception as _:
                        pass

                assert sample.shape[2] == self.config.latents_crop_width
            else:
                file_path = train_sample["file_name"]
                sample, sample_rate, t_offset = load_audio(file_path, start=-1,
                                                           count=self.config.sample_raw_crop_width,
                                                           return_sample_rate=True,
                                                           return_start=True)
                
                assert sample_rate == self.config.sample_rate
                assert sample.shape[0] == self.config.sample_raw_channels
                assert sample.shape[1] == self.config.sample_raw_crop_width
            
            samples.append(sample)
            paths.append(file_path)
            game_ids.append(game_id)
            #author_ids.append(author_id)

            if self.config.t_scale is not None:
                t_ranges.append(torch.tensor([self.get_t(t_offset),
                    self.get_t(t_offset + self.config.sample_raw_crop_width)]))
        
        batch_data = {
            "input": samples,
            "sample_paths": paths,
            "game_ids": game_ids,
            #"author_ids": author_ids}
        }

        if self.config.t_scale is not None:
            batch_data["t_ranges"] = t_ranges

        return batch_data

class DualDiffusionDataset:

    def __init__(self, dataset_config: DatasetConfig) -> None:
        
        self.config = dataset_config
        self.dataset_dict = load_dataset(
            self.config.data_dir,
            cache_dir=self.config.cache_dir,
            num_proc=self.config.num_proc,
        )

        self.preprocess_dataset()

    def __getitem__(self, split: str) -> dict:
        return self.dataset_dict[split]

    def preprocess_dataset(self) -> None:
        
        def resolve_absolute_path(example):
            if example["file_name"] is not None:
                example["file_name"] = os.path.join(self.config.data_dir, example["file_name"])
            if example["latents_file_name"] is not None:
                example["latents_file_name"] = os.path.join(self.config.data_dir, example["latents_file_name"])
            return example
        
        self.dataset_dict = self.dataset_dict.map(resolve_absolute_path)
        
        def invalid_sample_filter(example):
            if self.config.use_pre_encoded_latents:
                if example["latents_file_name"] is None:
                    return False
                if example["latents_length"] is not None:
                    return example["latents_length"] >= self.config.latents_crop_width
                return False
            else:
                if example["file_name"] is None:
                    return False
                if example["sample_length"] is not None and example["num_channels"] is not None and example["sample_rate"] is not None:
                    if example["sample_length"] < self.config.sample_raw_crop_width:
                        return False
                    if example["num_channels"] != self.config.sample_raw_channels:
                        return False
                    if example["sample_rate"] != self.config.sample_rate:
                        return False
                    return True
                else:
                    return False

        pre_filter_n_samples = {split: len(ds) for split, ds in self.dataset_dict.items()}
        if self.config.filter_invalid_samples: self.dataset_dict = self.dataset_dict.filter(invalid_sample_filter)
        self.num_filtered_samples = {split: (len(ds) - pre_filter_n_samples[split]) for split, ds in self.dataset_dict.items()}

        dataset_transform_config = DatasetTransformConfig(
            sample_rate=self.config.sample_rate,
            sample_raw_channels=self.config.sample_raw_channels,
            sample_raw_crop_width=self.config.sample_raw_crop_width,
            use_pre_encoded_latents=self.config.use_pre_encoded_latents,
            latents_crop_width=self.config.latents_crop_width,
            t_scale=self.config.t_scale
        )
        dataset_transform = DatasetTransform(dataset_transform_config)
        self.dataset_dict.set_transform(dataset_transform)