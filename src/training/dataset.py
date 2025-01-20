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

from dataclasses import dataclass
from typing import Optional, Literal
from glob import glob
import os

import numpy as np
import torch
import safetensors.torch as ST
from datasets import load_dataset

from modules.formats.format import DualDiffusionFormatConfig
from modules.embeddings.clap import CLAP_Config
from utils.dual_diffusion_utils import load_audio


@dataclass
class DatasetConfig:

     data_dir: str
     cache_dir: str
     sample_crop_width: int
     latents_crop_width: int
     num_proc: Optional[int] = None
     load_datatypes: list[Literal["audio", "latents", "audio_embeddings", "text_embeddings"]] = ("audio", "audio_embeddings")
     filter_invalid_samples: Optional[bool] = True

class DualDiffusionDataset(torch.nn.Module):

    def __init__(self, dataset_config: DatasetConfig, format_config: DualDiffusionFormatConfig, clap_config: CLAP_Config) -> None:
        super().__init__()
        self.config = dataset_config
        self.format_config = format_config
        self.clap_config = clap_config

        split_files = glob(f"{self.config.data_dir}/*.jsonl")
        data_files = {os.path.splitext(os.path.basename(split_file))[0]: split_file for split_file in split_files}

        self.dataset_dict = load_dataset(
            "json",
            data_files=data_files,
            num_proc=self.config.num_proc,
            keep_in_memory=True,

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
                if example["latents_length"] is None or example["latents_length"] < self.config.latents_crop_width: return False

            if "audio" in self.config.load_datatypes:
                if example["file_name"] is None: return False
                if example["sample_length"] is None or example["sample_length"] < self.config.sample_crop_width: return False
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

        for train_sample in examples:
            
            batch_paths.append(train_sample["file_name"])
            audio_t_offset = None; latents_t_offset = None

            if "audio" in self.config.load_datatypes:
                audio, sample_rate, audio_t_offset = load_audio(train_sample["file_name"], start=-1,
                                                    count=self.config.sample_crop_width,
                                                    return_sample_rate=True, return_start=True)
                
                # duplicate to stereo or downmix to mono if needed
                if audio.shape[0] < self.format_config.sample_raw_channels:
                    audio = audio.repeat(self.format_config.sample_raw_channels // audio.shape[0], 1)
                elif audio.shape[0] > self.format_config.sample_raw_channels:
                    audio = audio.mean(dim=0, keepdim=True)

                assert sample_rate == self.format_config.sample_rate
                assert audio.shape[0] == self.format_config.sample_raw_channels
                assert audio.shape[1] == self.config.sample_crop_width

                batch_audios.append(audio)

            if "latents" in self.config.load_datatypes:
                with ST.safe_open(train_sample["latents_file_name"], framework="pt") as f:
                    latents_slice = f.get_slice("latents")
                    latents_shape = latents_slice.get_shape()

                    latents_idx = np.random.randint(0, latents_shape[0]) # get random variation
                    latents_t_offset = np.random.randint(0, latents_shape[-1] - self.config.latents_crop_width + 1)
                    latents = latents_slice[latents_idx, ..., latents_t_offset:latents_t_offset + self.config.latents_crop_width]

                batch_latents.append(latents)
                
            if "audio_embeddings" in self.config.load_datatypes:
                with ST.safe_open(train_sample["latents_file_name"], framework="pt") as f:
                    sample_audio_embeddings = f.get_slice("clap_audio_embeddings")

                audio_emb_duration = self.clap_config.audio_embedding_duration

                if audio_t_offset is not None:
                    seconds_per_sample = 1 / self.format_config.sample_rate
                    audio_embed_start = int(audio_t_offset * seconds_per_sample / audio_emb_duration + 0.5)
                    audio_embed_end = int((audio_t_offset + self.config.sample_crop_width) * seconds_per_sample / audio_emb_duration + 0.5)
                elif latents_t_offset is not None:
                    seconds_per_latent_pixel = self.config.sample_crop_width / self.format_config.sample_rate / self.config.latents_crop_width
                    audio_embed_start = int(latents_t_offset * seconds_per_latent_pixel / audio_emb_duration + 0.5)
                    audio_embed_end = int((latents_t_offset + self.config.latents_crop_width) * seconds_per_latent_pixel / audio_emb_duration + 0.5)
                else:
                    audio_embed_start = 0
                    audio_embed_end = sample_audio_embeddings.get_shape()[0] + 1
                
                # handle an edge case when training VAEs with crop widths shorter than audio_emb_duration seconds
                audio_embed_start = max(audio_embed_start, 0)
                audio_embed_start = min(audio_embed_start, sample_audio_embeddings.get_shape()[0]-1)
                audio_embed_end = max(audio_embed_end, 0)
                audio_embed_end = min(audio_embed_end, sample_audio_embeddings.get_shape()[0])
                
                if audio_embed_start == audio_embed_end:
                    if audio_embed_start > 0: audio_embed_start -= 1
                    else: audio_embed_end += 1

                assert audio_embed_start >= 0 and audio_embed_start < sample_audio_embeddings.get_shape()[0]
                assert audio_embed_end > 0 and audio_embed_end <= sample_audio_embeddings.get_shape()[0]
                assert audio_embed_start != audio_embed_end

                sample_audio_embeddings = sample_audio_embeddings[audio_embed_start:audio_embed_end].mean(dim=0)
                batch_audio_embeddings.append(sample_audio_embeddings)

            if "text_embeddings" in self.config.load_datatypes:
                with ST.safe_open(train_sample["latents_file_name"], framework="pt") as f:
                    sample_text_embeddings = f.get_slice("clap_text_embeddings")[:].mean(dim=0)

                batch_text_embeddings.append(sample_text_embeddings)

        batch_data = {"sample_paths": batch_paths}

        if "latents" in self.config.load_datatypes:
            batch_data["latents"] = batch_latents
        if "audio_embeddings" in self.config.load_datatypes:
            batch_data["audio_embeddings"] = batch_audio_embeddings
        if "text_embeddings" in self.config.load_datatypes:
            batch_data["text_embeddings"] = batch_text_embeddings
        if "audio" in self.config.load_datatypes:
            batch_data["audio"] = batch_audios

        return batch_data