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
import logging
from dataclasses import dataclass
from typing import Optional

from accelerate.logging import get_logger, MultiProcessAdapter

from utils.dual_diffusion_utils import init_cuda

@dataclass
class DatasetProcessorConfig:

    formats: list[str]
    sample_rate: int
    num_channels: int
    min_sample_length: Optional[int] = None
    max_sample_length: Optional[int] = None
    pre_encoded_latents_vae: Optional[str] = None

class DatasetSplit:

    def __init__(self, path: str, logger: MultiProcessAdapter) -> None:

        self.logger = logger
        self.path = path
        self.name = os.path.splitext(os.path.basename(path))[0]

        self.init_split()

    def init_split(self) -> None:

        self.empty_sample = {
            "file_name": None,
            "system": None,
            "song": None,
            "game": None,
            "author": None,
            "latents_file_name": None,
            "system_id": None,
            "game_id": None,
            "author_id": None,
            "sample_rate": None,
            "num_channels": None,
            "sample_length": None,
            "pre_encoded_latents_vae": None,
        }
        if os.path.isfile(self.path):
            self.logger.info(f"Loading split from {self.path}")
            with open(self.path, "r") as f:
                self.samples = [self.empty_sample | json.loads(line) for line in f]
        else:
            self.logger.warning(f"Split not found at {self.path}, creating new split")
            self.samples = []

    def remove_samples(self, indices: list[int]) -> None:
        self.samples = [sample for index, sample in enumerate(self.samples) if index not in indices]
    
    def add_samples(self, samples: list[dict]) -> None:
        for sample in samples:
            self.samples.append(self.empty_sample | sample)

    def save(self) -> None:
        with open(self.path, "w") as f:
            for sample in self.samples:
                f.write(json.dumps(sample) + "\n")

class DatasetProcessor:
    
    def __init__(self) -> None:        
        self.config = DatasetProcessorConfig(
            **config.load_json(os.path.join(config.CONFIG_PATH, "dataset.json")))

        self.init_logging()
        self.init_dataset()

    def init_logging(self) -> None:

        self.logger = get_logger("dualdiffusion_dataset_processing", log_level="INFO")

        if config.DEBUG_PATH is not None:
            logging_dir = os.path.join(config.DEBUG_PATH, "dataset_processing")
            os.makedirs(logging_dir, exist_ok=True)

            log_path = os.path.join(logging_dir, "dataset_processing.log")
            logging.basicConfig(
                handlers=[
                    logging.FileHandler(log_path),
                    logging.StreamHandler()
                ],
                format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                datefmt="%m/%d/%Y %H:%M:%S",
                level=logging.INFO,
            )
            self.logger.info(f"Logging to {log_path}")
        else:
            self.logger.warning("DEBUG_PATH not defined, logging to file is disabled")

    def init_dataset(self) -> None:
        
        # init dataset info
        self.dataset_info = {
            "features": {
                "system": {"type": "string"},
                "song": {"type": "string"},
                "game": {"type": "string"},
                "author": {
                    "type": "list",
                    "value_type": {"type": "string"},
                },
                "latents_file_name": {"type": "string"},
                "system_id": {
                    "type": "int",
                    "num_classes": 0,
                },
                "game_id": {
                    "type": "int",
                    "num_classes": 0,
                },
                "author_id": {
                    "type": "list",
                    "value_type": {
                        "type": "int",
                        "num_classes": 0,
                    }
                },
                "sample_rate": {"type": "int"},
                "num_channels": {"type": "int"},
                "sample_length": {"type": "int"},
                "pre_encoded_latents_vae": {"type": "string"},
            },
            "system_id": {},
            "game_id": {},
            "author_id": {},
            "num_total_samples": 0,
            "num_train_samples": 0,
            "num_validation_samples": 0,
            "processor_config": self.config.__dict__,
        }
        self.dataset_info_path = os.path.join(
            config.DATASET_PATH, "dataset_infos", "dataset_info.json")
        if os.path.isfile(self.dataset_info_path):
            self.logger.info(f"Loading dataset info from {self.dataset_info_path}")

            self.dataset_info = self.dataset_info | config.load_json(self.dataset_info_path)
            self.dataset_info["processor_config"] = self.config.__dict__
        else:
            self.logger.warning(f"Dataset info not found at {self.dataset_info_path}, creating new dataset")

        # load / create splits
        splits = [
            DatasetSplit(os.path.join(config.DATASET_PATH, f), self.logger)
            for f in os.listdir(config.DATASET_PATH)
            if f.lower().endswith(".jsonl")
        ]
        self.splits = {split.name: split for split in splits}

        if "train" not in self.splits:
            self.splits["train"] = DatasetSplit(
                os.path.join(config.DATASET_PATH, "train.jsonl"), self.logger)
        if "validation" not in self.splits:
            self.splits["validation"] = DatasetSplit(
                os.path.join(config.DATASET_PATH, "validation.jsonl"), self.logger)

    def get_id(self, id_type: str, name: str) -> int:

        id = self.dataset_info[id_type].get(name, None)
        if id is None:

            if "value_type" in self.dataset_info["features"][id_type]:
                id = self.dataset_info["features"][id_type]["value_type"]["num_classes"]
                self.dataset_info["features"][id_type]["value_type"]["num_classes"] += 1
            else:
                id = self.dataset_info["features"][id_type]["num_classes"]
                self.dataset_info["features"][id_type]["num_classes"] += 1

            self.dataset_info[id_type][name] = name

        return id
    
    def __iter__(self):
        for _, split in self.splits.items():
            yield from ((split, index, sample) for index, sample in enumerate(split.samples))

    def validate(self) -> None:
        # todo: search for any samples with the same file_name in any splits
        # if any found, prompt to remove them
        # todo: search for any system, game, or author ids that don't match the dataset_info
        # if any found, prompt to either rebuild dataset info, or clear invalid metadata
        # todo: search for any unused system, game, or author ids
        # if any found, prompt to remove them from dataset_info and rebase ids in sample metadata
        pass

    def scan(self) -> None:
        # todo: search for any file_name in splits that no longer exists
        # if any found, prompt to remove them from splits
        # todo: search for any other loose files that are not a valid audio format
        # if any found, prompt to remove them
        # todo: search for any valid audio files not currently in any splits
        # if any found, prompt to add them to train split
        pass

    def populate_metadata(self) -> None:
        # todo: search for any samples in splits that have any null metadata fields
        # if any found, prompt to extract metadata
        pass
    
    def transcode(self) -> None:
        # todo: search for any samples with with a sample_rate / sample_length < config values
        # if any found, prompt to remove them
        # todo: search for any samples not in the selected format list in config,
        # or with a sample_rate / num_channels / sample_length > config values
        # if any found, prompt to transcode / crop (and update metadata)
        pass

    def filter(self) -> None:
        # todo: detect any duplicate / highly similar samples in splits,
        # detect any abnormal / anomalous samples in splits
        # if any found, prompt to remove them
        pass
    
    def train_validation_split(self) -> None:
        # todo: prompt to resplit the aggregated dataset into train / validation splits
        pass

    def encode_latents(self) -> None:
        # todo: search for any samples with a null pre_encoded_latents_vae, or
        # a pre_encoded_latents_vae that doesn't match the config vae model name
        # or a null latents_file_name or latents_file_name that doesn't exist
        # if any found, prompt to encode them and update metadata
        # will need to launch subprocess to use accelerate
        # accelerate config for pre_encoding is in config.DATASET_PATH/dataset_accelerate.yaml
        pass
 
    def save(self) -> None:
        # todo: prompt to save and backup existing files to config.DEBUG_PATH
        self.dataset_info["num_total_samples"] = sum(len(split.samples) for split in self.splits.values())
        self.dataset_info["num_train_samples"] = len(self.splits["train"].samples)
        self.dataset_info["num_validation_samples"] = len(self.splits["validation"].samples)
        #os.makedirs(os.path.dirname(self.dataset_info_path), exist_ok=True)
        #config.save_json(self.dataset_info_path, self.dataset_info)

        #for split in self.splits:
        #    split.save()
        pass

    def create_hf_dataset(self) -> None:
        # todo: prompt to create a copy of the full dataset using symbolic links
        # with no more than 10000 files per folder to be compatible with online huggingface dataset storage
        pass

    def process_dataset(self) -> None:
        self.validate()
        self.scan()
        self.populate_metadata()
        self.transcode()
        self.filter()
        self.train_validation_split()
        self.encode_latents()
        self.save()
        self.create_hf_dataset()

    
if __name__ == "__main__":

    init_cuda()

    processor = DatasetProcessor()
    processor.process_dataset()