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
from typing import Optional, Literal

import numpy as np
import torch
import safetensors.torch as ST
from datasets import load_dataset

from modules.formats.format import DualDiffusionFormatConfig
from utils.dual_diffusion_utils import load_audio, dequantize_tensor


@dataclass
class DatasetConfig:

     data_dir: str
     cache_dir: str
     latents_crop_width: int
     num_proc: Optional[int] = None
     load_datatypes: list[Literal["audio", "latents", "audio_embeddings", "text_embeddings"]] = ("audio", "audio_embeddings")
     filter_invalid_samples: Optional[bool] = True

class DualDiffusionDataset(torch.nn.Module):

    def __init__(self, dataset_config: DatasetConfig, format_config: DualDiffusionFormatConfig) -> None:
        super().__init__()
        self.config = dataset_config
        self.format_config = format_config

        self.dataset_dict = load_dataset(
            self.config.data_dir,
            cache_dir=self.config.cache_dir,
            num_proc=self.config.num_proc,
        )
        self.preprocess_dataset()
        self.dataset_dict.set_transform(self)

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
            if not example["latents_has_audio_embeddings"] and "audio_embeddings" in self.config.load_datatypes: return False
            if not example["latents_has_text_embeddings"] and "text_embeddings" in self.config.load_datatypes: return False

            if "latents" in self.config.load_datatypes:
                if example["latents_length"] < self.config.latents_crop_width: return False

            if "audio" in self.config.load_datatypes:
                if example["sample_length"] < self.format_config.sample_raw_length: return False
                if example["sample_rate"] != self.format_config.sample_rate: return False

            return True

        pre_filter_n_samples = {split: len(ds) for split, ds in self.dataset_dict.items()}
        if self.config.filter_invalid_samples: self.dataset_dict = self.dataset_dict.filter(invalid_sample_filter)
        self.num_filtered_samples = {split: (len(ds) - pre_filter_n_samples[split]) for split, ds in self.dataset_dict.items()}
    
    @torch.no_grad()
    def __call__(self, examples: dict) -> dict:

        batch_audios = []
        batch_latents = []
        batch_paths = []
        batch_audio_embeddings = []
        batch_text_embeddings = []

        num_examples = len(next(iter(examples.values())))
        examples = [{key: examples[key][i] for key in examples} for i in range(num_examples)]
        t_offset = None

        for train_sample in examples:
            
            batch_paths.append(file_path)

            if "latents" in self.config.load_datatypes:
                file_path = train_sample["latents_file_name"]
                with ST.safe_open(file_path, framework="pt") as f:
                    latents_slice = f.get_slice("latents")
                    latents_shape = latents_slice.get_shape()

                    latents_idx = np.random.randint(0, latents_shape[0]) # get random variation
                    t_offset = np.random.randint(0, latents_shape[-1] - self.config.latents_crop_width + 1)
                    latents = latents_slice[latents_idx, ..., t_offset:t_offset + self.config.latents_crop_width]

                batch_latents.append(latents)
                
            if "audio_embeddings" in self.config.load_datatypes:
                sample_audio_embeddings = f.get_slice("clap_audio_embeddings")
                if t_offset is not None:
                    batch_audio_embeddings.append(sample_audio_embeddings[:]) # todo: probably need to pad to max length in batch
                else: # todo: remove these hard-coded constants with references to the pipeline embedding module config
                    seconds_per_latent_pixel = self.format_config.sample_raw_length / self.format_config.sample_rate / self.config.latents_crop_width
                    audio_embed_start = int(t_offset * seconds_per_latent_pixel / 10 + 0.5)
                    audio_embed_end = int((t_offset + self.config.latents_crop_width) * seconds_per_latent_pixel / 10 + 0.5)
                    audio_embed_end = min(audio_embed_end, sample_audio_embeddings.get_shape()[-1])

                    sample_audio_embeddings = sample_audio_embeddings[audio_embed_start:audio_embed_end].mean(dim=0)
                    batch_audio_embeddings.append(sample_audio_embeddings)

            if "text_embeddings" in self.config.load_datatypes:
                sample_text_embeddings = f.get_slice("clap_text_embeddings")[:].mean(dim=0)
                batch_text_embeddings.append(sample_text_embeddings)

            if "audio" in self.config.load_datatypes:
                file_path = train_sample["file_name"]
                audio, sample_rate, t_offset = load_audio(file_path, start=-1,
                                                    count=self.format_config.sample_raw_length,
                                                    return_sample_rate=True, return_start=True)
                
                # duplicate to stereo or downmix to mono if needed
                if audio.shape[0] < self.format_config.sample_raw_channels:
                    audio = audio.repeat(self.format_config.sample_raw_channels // audio.shape[0], 1)
                elif audio.shape[0] < self.format_config.sample_raw_channels:
                    audio = audio.mean(dim=0, keepdim=True)

                assert sample_rate == self.format_config.sample_rate
                assert audio.shape[0] == self.format_config.sample_raw_channels
                assert audio.shape[1] == self.format_config.sample_raw_length

                batch_audios.append(audio)

        batch_data = {}

        if "latents" in self.config.load_datatypes:
            batch_data["latents"] = batch_latents
        if "audio_embeddings" in self.config.load_datatypes:
            batch_data["audio_embeddings"] = batch_audio_embeddings
        if "text_embeddings" in self.config.load_datatypes:
            batch_data["text_embeddings"] = batch_text_embeddings
        if "audio" in self.config.load_datatypes:
            batch_data["audio"] = batch_audios

        return batch_data