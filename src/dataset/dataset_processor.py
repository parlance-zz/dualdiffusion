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
from typing import Generator, Optional


@dataclass
class DatasetProcessorConfig:

    dataset_formats: list[str]
    source_formats: list[str]
    sample_rate: int
    num_channels: int
    min_sample_length: Optional[int] = None
    max_sample_length: Optional[int] = None
    pre_encoded_latents_vae: Optional[str] = None
    show_debug_info: bool = False

class DatasetSplit:

    def __init__(self, path: str) -> None:

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
            logging.getLogger().info(f"Loading split from {self.path}")
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

class SampleList:

    def __init__(self) -> None:
        self.samples: dict[DatasetSplit, list[int]] = {}

    def add_sample(self, split: DatasetSplit, index: int) -> None:
        if split not in self.samples:
            self.samples[split] = []
        self.samples[split].append(index)

    def __len__(self) -> int:
        return sum(len(indices) for indices in self.samples.values())
    
    def __iter__(self) -> Generator[DatasetSplit, int, dict]:
        for split, indices in self.samples.items():
            yield from ((split, index, split.samples[index]) for index in indices)
    
    def remove_samples_from_dataset(self) -> None:
        for split, indices in self.samples.items():
            split.remove_samples(indices)

    def show_samples(self) -> None:
        logger = logging.getLogger()
        for split, index, sample in self:
            logger.debug(f"{split.name}_{index}: {sample['file_name']}")

class DatasetProcessor:
    
    def __init__(self) -> None:        
        self.config = DatasetProcessorConfig(
            **config.load_json(os.path.join(config.CONFIG_PATH, "dataset.json")))

        self.init_logging()
        self.init_dataset()

    def init_logging(self) -> None:

        self.logger = logging.getLogger()

        if config.DEBUG_PATH is not None:
            logging_dir = os.path.join(config.DEBUG_PATH, "dataset_processing")
            os.makedirs(logging_dir, exist_ok=True)

            log_path = os.path.join(logging_dir, "dataset_processing.log")
            logging.basicConfig(
                handlers=[
                    logging.FileHandler(log_path),
                    logging.StreamHandler()
                ],
                format="",
                level=logging.DEBUG if self.config.show_debug_info else logging.INFO,
            )
            self.logger.info(f"\nLogging to {log_path}")
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
            DatasetSplit(os.path.join(config.DATASET_PATH, f))
            for f in os.listdir(config.DATASET_PATH)
            if f.lower().endswith(".jsonl")
        ]
        self.splits = {split.name: split for split in splits}

        if "train" not in self.splits:
            self.logger.warning("Train split not found, creating new split")
            self.splits["train"] = DatasetSplit(os.path.join(config.DATASET_PATH, "train.jsonl"))
        if "validation" not in self.splits:
            self.logger.warning("Validation split not found, creating new split")
            self.splits["validation"] = DatasetSplit(os.path.join(config.DATASET_PATH, "validation.jsonl"))

        # show dataset summary info
        self.logger.info(f"\nLoaded dataset with {self.num_samples()} samples")
        self.logger.info("Splits:")
        for split in self.splits.values():
            self.logger.info(f"  {split.name}: {len(split.samples)} samples")
        self.logger.info("Dataset info:")
        self.logger.info(f"  {len(self.dataset_info['system_id'])} system id(s)")
        self.logger.info(f"  {len(self.dataset_info['game_id'])} game id(s)")
        self.logger.info(f"  {len(self.dataset_info['author_id'])} author id(s)")
        self.logger.info("")

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
    
    def all_samples(self) -> Generator[DatasetSplit, int, dict]:
        for _, split in self.splits.items():
            yield from ((split, index, sample) for index, sample in enumerate(split.samples))

    def num_samples(self) -> int:
        return sum(len(split.samples) for split in self.splits.values())

    def validate(self) -> None:

        # search for any sample file_name in splits that no longer exists (or null file_name)
        # if any found, prompt to remove the samples from splits
        missing_samples = SampleList()
        for split, index, sample in self.all_samples():
            if sample["file_name"] is None or (not os.path.isfile(os.path.join(config.DATASET_PATH, sample["file_name"]))):
                missing_samples.add_sample(split, index)
        
        num_missing = len(missing_samples)
        if num_missing > 0:
            self.logger.warning(f"Found {num_missing} samples in dataset with missing files or no file_name")
            missing_samples.show_samples()
            remove_missing = input(f"Remove {num_missing} samples with missing files? (y/n): ")
            if remove_missing.lower() == "y":
                missing_samples.remove_samples_from_dataset()
                self.logger.info(f"{num_missing} samples with missing files removed")
            self.logger.info("")

        # search for any samples with a file_name that is not in the source formats list
        # if any found, prompt to remove them from splits
        invalid_format_samples = SampleList()
        valid_sample_file_formats = self.config.source_formats + self.config.dataset_formats
        for split, index, sample in self.all_samples():
            if os.path.splitext(sample["file_name"])[1].lower() not in valid_sample_file_formats:
                invalid_format_samples.add_sample(split, index)

        num_invalid_format = len(invalid_format_samples)
        if num_invalid_format > 0:
            self.logger.warning(f"Found {num_invalid_format} samples with file formats not in the source format list ({self.config.source_formats})")
            invalid_format_samples.show_samples()
            remove_invalid_format = input(f"Remove {num_invalid_format} samples with invalid file formats? (this will not delete the files) (y/n): ")
            if remove_invalid_format.lower() == "y":
                invalid_format_samples.remove_samples_from_dataset()
                self.logger.info(f"{num_invalid_format} samples with invalid file formats removed")
            self.logger.info("")

        # search for any samples with the same file_name in any splits
        # if any found, prompt to remove duplicates
        sample_files: dict[str, bool] = {}
        duplicate_samples = SampleList()
        for split, index, sample in self.all_samples():
            if sample["file_name"] in sample_files:
                duplicate_samples.add_sample(split, index)
            else:
                sample_files[sample["file_name"]] = True

        num_duplicates = len(duplicate_samples)
        if num_duplicates > 0:
            self.logger.warning(f"Found {num_duplicates} samples with duplicated file_names")
            duplicate_samples.show_samples()
            remove_duplicates = input(f"Remove {num_duplicates} samples with duplicate file_names? (y/n): ")
            if remove_duplicates.lower() == "y":
                duplicate_samples.remove_samples_from_dataset()
                self.logger.info(f"{num_duplicates} samples with duplicate file_names removed")
            self.logger.info("")

        # search for any system, game, or author ids that don't match the dataset_info
        # if any found, prompt to clear invalid metadata
        invalid_id_samples = SampleList()
        for split, index, sample in self.all_samples():

            if sample["system"] is not None:
                ds_info_system_id = self.dataset_info["system_id"].get(sample["system"], None)
                if ds_info_system_id is None:
                    invalid_id_samples.add_sample(split, index)
                    continue
                else:
                    if ds_info_system_id != sample["system_id"]:
                        invalid_id_samples.add_sample(split, index)
                        continue
            
            if sample["game"] is not None: 
                ds_info_game_id = self.dataset_info["game_id"].get(sample["game"], None)
                if ds_info_game_id is None:
                    invalid_id_samples.add_sample(split, index)
                    continue
                else:
                    if ds_info_game_id != sample["game_id"]:
                        invalid_id_samples.add_sample(split, index)
                        continue

            if sample["author"] is not None:
                for i, author in enumerate(sample["author"]):
                    ds_info_author_id = self.dataset_info["author_id"].get(author, None)
                    if ds_info_author_id is None:
                        invalid_id_samples.add_sample(split, index)
                        break
                    else:
                        if ds_info_author_id != sample["author_id"][i]:
                            invalid_id_samples.add_sample(split, index)
                            break

        num_invalid_id = len(invalid_id_samples)
        if num_invalid_id > 0:
            self.logger.warning(f"Found {num_invalid_id} samples with game, system, or author ids inconsistent with dataset_info")
            invalid_id_samples.show_samples()
            remove_invalid = input(f"Clear invalid id metadata for {num_invalid_id} samples? (y/n): ")
            if remove_invalid.lower() == "y":
                for _, _, sample in invalid_id_samples:
                    sample["system_id"] = None
                    sample["game_id"] = None
                    sample["author_id"] = None
                self.logger.info(f"Cleared invalid id metadata for {num_invalid_id} samples")
            self.logger.info("")

    def scan(self) -> None:

        # search for any files (excluding dataset metadata files) that are not in the source formats list
        # if any found, prompt to permanently delete them
        invalid_format_files = []
        valid_dataset_file_formats = self.config.source_formats + self.config.dataset_formats + [".jsonl", ".json"]
        for root, _, files in os.walk(config.DATASET_PATH):
            for file in files:
                if os.path.splitext(file)[1].lower() not in valid_dataset_file_formats:
                    invalid_format_files.append(os.path.join(root, file))

        num_invalid_format_files = len(invalid_format_files)
        if num_invalid_format_files > 0:
            self.logger.warning(f"Found {num_invalid_format_files} files with formats not in the source format list ({self.config.source_formats})")
            for file in invalid_format_files:
                self.logger.debug(file)
            delete_invalid_format_files = input(f"Delete {num_invalid_format_files} files with invalid formats? (WARNING: this is permanent and cannot be undone) (y/n): ")
            if delete_invalid_format_files.lower() == "y":
                for file in invalid_format_files:
                    os.remove(file)
                self.logger.info(f"Deleted {num_invalid_format_files} files with invalid formats")
            self.logger.info("")

        # search for any valid new source audio files not currently dataset
        # if any found, prompt to add them to train split
        sample_files: dict[str, bool] = {}
        for _, _, sample in self.all_samples():
            sample_files[sample["file_name"]] = True

        new_audio_files = []
        valid_sample_formats = self.config.source_formats + self.config.dataset_formats
        for root, _, files in os.walk(config.DATASET_PATH):
            for file in files:
                if os.path.splitext(file)[1].lower() in valid_sample_formats:
                    rel_path = os.path.relpath(os.path.join(root, file), config.DATASET_PATH)
                    if rel_path not in sample_files:
                        new_audio_files.append(rel_path)

        num_new_audio_files = len(new_audio_files)
        if num_new_audio_files > 0:
            self.logger.info(f"Found {num_new_audio_files} new audio files not currently in the dataset")
            for file in new_audio_files:
                self.logger.debug(file)
            add_new_audio_files = input(f"Add {num_new_audio_files} new audio files to train split? (y/n): ")
            if add_new_audio_files.lower() == "y":
                self.splits["train"].add_samples([{"file_name": file} for file in new_audio_files])
                self.logger.info(f"Added {num_new_audio_files} new audio files to train split")
            self.logger.info("")

    def populate_metadata(self) -> None:
        # search for any samples in splits that have any null metadata fields
        # if any found, prompt to extract metadata
        need_metadata_samples = SampleList()
        metadata_check_exclude_keys = ["latents_file_name", "pre_encoded_latents_vae"]
        for split, index, sample in self.all_samples():
            if any(value is None for key, value in sample.items() if key not in metadata_check_exclude_keys):
                need_metadata_samples.add_sample(split, index)

        num_need_metadata = len(need_metadata_samples)
        if num_need_metadata > 0:
            self.logger.warning(f"Found {num_need_metadata} samples with missing metadata")
            need_metadata_samples.show_samples()
            extract_metadata = input(f"Extract missing metadata for {num_need_metadata} samples? (y/n): ")
            if extract_metadata.lower() == "y":
                for split, index, sample in need_metadata_samples:
                    pass # todo: extract metadata
                self.logger.info(f"Extracted missing metadata for {num_need_metadata} samples")
            self.logger.info("")

        # search for any unused system, game, or author ids
        # if any found, prompt to rebuild dataset_info / ids from current metadata
        pass
    
    def transcode(self) -> None:
        # todo: search for any samples with with a sample_rate / sample_length < config values
        # if any found, prompt to remove them

        # todo: search for any samples with a format not in the dataset_formats list in config,
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
        self.dataset_info["num_total_samples"] = self.num_samples()
        self.dataset_info["num_train_samples"] = len(self.splits["train"].samples)
        self.dataset_info["num_validation_samples"] = len(self.splits["validation"].samples)
        #os.makedirs(os.path.dirname(self.dataset_info_path), exist_ok=True)
        #config.save_json(self.dataset_info_path, self.dataset_info)

        #for split in self.splits:
        #    split.save()
        pass

    def export_to_hf(self) -> None:
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
        self.export_to_hf()