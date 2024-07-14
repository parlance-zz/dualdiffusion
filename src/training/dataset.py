import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torchaudio
import safetensors.torch as ST
from torch.utils.data import Dataset
from datasets import load_dataset, DatasetDict

from utils.dual_diffusion_utils import load_audio, dequantize_tensor

@dataclass
class DatasetTransformConfig:

    use_pre_encoded_latents: bool
    sample_crop_width: int
    t_scale: Optional[float] = None

@dataclass
class DatasetConfig:
     
     train_data_dir: str
     cache_dir: str
     num_proc: Optional[int] = None
     sample_crop_width: int
     use_pre_encoded_latents: bool
     t_scale: Optional[float] = None

class DatasetTransform(torch.nn.Module):

    def __init__(self, dataset_transform_config: DatasetTransformConfig):
        self.config = dataset_transform_config
    
    def get_t(self, t):
        return t / self.config.sample_crop_width * self.config.t_scale - self.config.t_scale/2
    
    def __call__(self, examples):

        samples = []
        paths = []
        game_ids = []
        t_ranges = []
        #author_ids = []

        num_examples = len(next(iter(examples.values())))
        examples = [{key: examples[key][i] for key in examples} for i in range(num_examples)]

        for train_sample in examples:
            
            file_path = train_sample["file_name"]
            game_id = train_sample["game_id"]
            #author_id = train_sample["author_id"]

            if self.config.use_pre_encoded_latents:

                with ST.safe_open(file_path, framework="pt") as f:
                    latents_slice = f.get_slice("latents")
                    latents_shape = latents_slice.get_shape()

                    latents_idx = np.random.randint(0, latents_shape[0])
                    t_offset = np.random.randint(0, latents_shape[-1] - self.config.sample_crop_width + 1)
                    sample = latents_slice[latents_idx, ..., t_offset:t_offset + self.config.sample_crop_width]

                    try:
                        offset_and_range = f.get_slice("offset_and_range")
                        sample = dequantize_tensor(sample, offset_and_range[latents_idx])
                    except Exception as _:
                        pass
            else:
                sample, t_offset = load_audio(file_path,
                                            start=-1,
                                            count=self.sample_crop_width,
                                            return_start=True)
            
            samples.append(sample)
            paths.append(file_path)
            game_ids.append(game_id)
            #author_ids.append(author_id)

            if self.config.t_scale is not None:
                t_ranges.append(torch.tensor([self.get_t(t_offset), self.get_t(t_offset + self.config.sample_crop_width)]))
        
        batch_data = {
            "input": samples,
            "sample_paths": paths,
            "game_ids": game_ids,
            #"author_ids": author_ids}
        }

        if self.config.t_scale is not None:
            batch_data["t_ranges"] = t_ranges

        return batch_data

class SwitchableDataset(Dataset):

    def __init__(self, dataset_config: DatasetConfig):
        
        self.config = dataset_config
        self.split = "train"

        self.dataset_dict = load_dataset(
            self.config.train_data_dir,
            cache_dir=self.config.cache_dir,
            num_proc=self.config.num_proc,
        )

        processed_datasets = {}; num_filtered_samples = {}
        for split, dataset in self.dataset_dict.items():
            processed_datasets[split], num_filtered_samples[split] = self.preprocess_dataset(dataset)

        self.dataset_dict = DatasetDict(processed_datasets)
        self.num_filtered_samples = num_filtered_samples

    def set_split(self, split):
        self.split = split

    def get_num_filtered_samples(self, split=None):
        return self.num_filtered_samples[split or self.split]
    
    def __len__(self):
        return len(self.dataset_dict[self.split])

    def __getitem__(self, idx):
        return self.dataset_dict[self.split][idx]

    def preprocess_dataset(self, dataset: Dataset):
            
        def resolve_absolute_path(example):
            relative_path = example['file_name']
            absolute_path = os.path.join(self.config.train_data_dir, relative_path)
            example['file_name'] = absolute_path
            return example
        
        dataset = dataset.map(resolve_absolute_path)
        
        if self.config.use_pre_encoded_latents:
            def min_length_filter(example):          
                with ST.safe_open(example["file_name"], framework="pt") as f:
                    latents_slice = f.get_slice("latents")
                    latents_shape = latents_slice.get_shape()
                    return latents_shape[-1] >= self.config.sample_crop_width
        else:
            def min_length_filter(example):          
                return torchaudio.info(example["file_name"]).num_frames >= self.config.sample_crop_width
        
        pre_filter_num_samples = len(dataset)
        dataset = dataset.filter(min_length_filter)
        num_filtered_samples = pre_filter_num_samples - len(dataset)

        dataset_transform_config = DatasetTransformConfig(
            use_pre_encoded_latents=self.config.use_pre_encoded_latents,
            sample_crop_width=self.config.sample_crop_width,
            t_scale=self.config.t_scale
        )
        dataset_transform = DatasetTransform(dataset_transform_config)
        dataset = dataset.with_transform(dataset_transform)

        return dataset, num_filtered_samples
