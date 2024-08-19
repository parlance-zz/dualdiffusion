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
import argparse
import json
from dataclasses import dataclass
from typing import Optional

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

    def __init__(self, path: str) -> None:

        self.path = path
        self.data_path = os.path.dirname(path)
        self.name = os.path.splitext(os.path.basename(path))[0]

        sample = {
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
        if os.path.isfile(path):
            with open(path, "r") as f:
                self.samples = [sample | json.loads(line) for line in f]
        else:
            self.samples = []

    def save(self) -> None:
        with open(self.path, "w") as f:
            for sample in self.samples:
                f.write(json.dumps(sample) + "\n")

class DatasetProcessor:
    
    def __init__(self, args: argparse.Namespace) -> None:
        
        self.config = DatasetProcessorConfig(
            **config.load_json(os.path.join(config.CONFIG_PATH, "dataset.json")))

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
            "processor_config": self.config.__dict__,
        }
        self.dataset_info_path = os.path.join(
            config.DATASET_PATH, "dataset_infos", "dataset_info.json")
        if os.path.isfile(self.dataset_info_path):
            self.dataset_info = self.dataset_info | config.load_json(self.dataset_info_path)

        splits = [
            DatasetSplit(os.path.join(config.DATASET_PATH, f))
            for f in os.listdir(config.DATASET_PATH)
            if f.lower().endswith(".jsonl")
        ]
        self.splits = {split.name: split for split in splits}

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
            yield from split.samples

    def process_dataset(self, args: argparse.Namespace) -> None:
        pass

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DualDiffusion dataset processing script.")
    """
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to pretrained / new model",
    )
    parser.add_argument(
        "--train_config_path",
        type=str,
        required=True,
        help="Path to training configuration json file",
    )
    """
    return parser.parse_args()
    
if __name__ == "__main__":

    init_cuda()
    args = parse_args()

    processor = DatasetProcessor(args)
    processor.process_dataset(args)

    for sample in processor:
        sample["pre_encoded_latents_vae"] = "test"
    from utils.dual_diffusion_utils import dict_str
    for split in processor.splits.values():
        print(dict_str(split.samples[-10:]))