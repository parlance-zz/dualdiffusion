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
import shutil
from dataclasses import dataclass
from typing import Generator, Optional
from datetime import datetime

import torch
import mutagen
import safetensors.torch as ST
from tqdm.auto import tqdm

from utils.dual_diffusion_utils import dict_str, normalize, save_safetensors, get_audio_metadata


@dataclass
class DatasetProcessorConfig:

    dataset_formats: list[str]
    source_formats: list[str]
    sample_rate: int
    num_channels: int
    min_sample_rate: Optional[int] = None
    min_kbps: Optional[int] = None
    min_sample_length: Optional[int] = None
    max_sample_length: Optional[int] = None
    min_num_class_samples: Optional[int] = None
    dataset_processor_verbose: bool = False

    pre_encoded_latents_vae: Optional[str] = None
    pre_encoded_latents_device_batch_size: int = 1
    pre_encoded_latents_num_time_offset_augmentations: int = 8
    pre_encoded_latents_pitch_offset_augmentations: list[int] = ()
    pre_encoded_latents_stereo_mirroring_augmentation: bool = False
    pre_encoded_latents_enable_quantization: bool = False

    clap_embedding_labels: Optional[dict[str, list[str]]] = None
    clap_embedding_tags: Optional[list[str]] = None
    clap_enable_fusion: bool = False
    clap_audio_encoder: str = "HTSAT-base"
    clap_text_encoder: str = "roberta"
    clap_compile_options: Optional[dict] = None
    clap_max_batch_size: int = 32

class DatasetSplit:

    def __init__(self, path: str) -> None:
        self.path = path
        self.name = os.path.splitext(os.path.basename(path))[0]

        self.init_split()

    def init_split(self) -> None:

        self.empty_sample = {
            "file_name": None,
            "file_size": None,
            "system": None,
            "game": None,
            "song": None,
            "author": None,
            "system_id": None,
            "game_id": None,
            "author_id": None,
            "sample_rate": None,
            "num_channels": None,
            "sample_length": None,
            "bit_rate": None,
            "prompt": None,
            "latents_file_name": None,
            "latents_file_size": None,
            "latents_length": None,
            "latents_num_variations": None,
            "latents_quantized": None,
            "latents_vae_model": None,
            "latents_has_audio_embeddings": None,
            "latents_has_text_embeddings": None,
        }
        if os.path.isfile(self.path):
            logging.getLogger().info(f"Loading split from {self.path}")
            with open(self.path, "r") as f:
                self.samples = [self.empty_sample | json.loads(line) for line in f]
        else:
            logging.getLogger().warning(f"Split not found at {self.path}, creating new split")
            self.samples: list[dict] = []

    def remove_samples(self, indices: list[int]) -> None:
        self.samples = [sample for index, sample in enumerate(self.samples) if index not in indices]
    
    def add_samples(self, samples: list[dict]) -> None:
        for sample in samples:
            self.samples.append(self.empty_sample | sample)

    def save(self, path: Optional[str] = None) -> None:
        with open(path or self.path, "w") as f:
            for sample in self.samples:
                f.write(json.dumps(sample) + "\n")

class SampleList:

    def __init__(self) -> None:
        self.samples: dict[DatasetSplit, list[int]] = {}
        self.annotations : dict[tuple[DatasetSplit, int], str] = {}

    def add_sample(self, split: DatasetSplit, index: int, annotation: Optional[str] = None) -> None:
        if split not in self.samples:
            self.samples[split] = []
        self.samples[split].append(index)

        if annotation is not None:
            self.annotations[(split, index)] = annotation

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
            sample_str = f'{split.name}_{index}: "{sample["file_name"]}"'
            annotation = self.annotations.get((split, index), None)
            if annotation is not None: sample_str += f" ({annotation})"
            logger.debug(sample_str)

class DatasetProcessor:
    
    def __init__(self) -> None:

        if config.CONFIG_PATH is None:
            raise ValueError("ERROR: CONFIG_PATH not defined")
        if not os.path.isdir(config.CONFIG_PATH):
            raise ValueError(f"ERROR: CONFIG_PATH '{config.CONFIG_PATH}' not found")
        if config.DATASET_PATH is None:
            raise ValueError("ERROR: DATASET_PATH not defined")
        if not os.path.isdir(config.DATASET_PATH):
            raise ValueError(f"ERROR: DATASET_PATH '{config.DATASET_PATH}' not found")
        
        self.config = DatasetProcessorConfig(
            **config.load_json(os.path.join(config.CONFIG_PATH, "dataset", "dataset.json")))
        
        self.datetime_str = datetime.now().strftime(r"%Y-%m-%d_%H_%M_%S")
        if config.DEBUG_PATH is not None:
            self.backup_path = os.path.join(config.DEBUG_PATH, "dataset_processing", f"backup_{self.datetime_str}")
        else:
            self.backup_path = None
        
        self.init_logging()
        self.init_dataset()

    def init_logging(self) -> None:

        self.logger = logging.getLogger()

        if config.DEBUG_PATH is not None:
            logging_dir = os.path.join(config.DEBUG_PATH, "dataset_processing")
            os.makedirs(logging_dir, exist_ok=True)

            log_path = os.path.join(logging_dir, f"dataset_processing_{self.datetime_str}.log")
            logging.basicConfig(
                handlers=[
                    logging.FileHandler(log_path),
                    logging.StreamHandler()
                ],
                format="",
                level=logging.DEBUG if self.config.dataset_processor_verbose else logging.INFO,
            )
            self.logger.info(f"\nStarted DatasetProcessor at {self.datetime_str}")
            self.logger.info(f"Logging to {log_path}")
        else:
            self.logger.warning("WARNING: DEBUG_PATH not defined, logging to file and metadata backup is disabled")

    def init_dataset(self) -> None:
        
        # init dataset info
        self.dataset_info = {
            "features": {
                "system": {"type": "string"},
                "game": {"type": "string"},
                "song": {"type": "string"},
                "author": {
                    "type": "list",
                    "value_type": {"type": "string"},
                },
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
                "prompt": {"type": "string"},
                "sample_rate": {"type": "int"},
                "num_channels": {"type": "int"},
                "sample_length": {"type": "int"},
                "file_size": {"type": "int"},
                "bit_rate": {"type": "int"},
                "latents_file_name": {"type": "string"},
                "latent_file_size": {"type": "int"},
                "latents_length": {"type": "int"},
                "latents_num_variations": {"type": "int"},
                "latents_quantized": {"type": "bool"},
                "latents_vae_model": {"type": "string"},
                "latents_has_audio_embeddings": {"type": "bool"},
                "latents_has_text_embeddings": {"type": "bool"},
            },
            "system_id": {},
            "game_id": {},
            "author_id": {},
            "num_total_samples": 0,
            "num_train_samples": 0,
            "num_validation_samples": 0,
            "system_train_sample_counts": {},
            "game_train_sample_counts": {},
            "author_train_sample_counts": {},
            "processor_config": None,
        }

        self.dataset_info_path = os.path.join(config.DATASET_PATH, "dataset_infos", "dataset_info.json")
        if os.path.isfile(self.dataset_info_path):
            self.logger.info(f"Loading dataset info from {self.dataset_info_path}")
            dataset_info = self.dataset_info | config.load_json(self.dataset_info_path)
            dataset_info["features"] = self.dataset_info["features"] | dataset_info["features"]
        else:
            self.logger.warning(f"Dataset info not found at {self.dataset_info_path}, creating new dataset")
            dataset_info = self.dataset_info

        dataset_info["processor_config"] = self.config.__dict__
        self.dataset_info = dataset_info

        # load / create splits
        splits = [
            DatasetSplit(os.path.join(config.DATASET_PATH, f))
            for f in os.listdir(config.DATASET_PATH)
            if f.lower().endswith(".jsonl")
        ]
        self.splits = {split.name: split for split in splits}

        if "train" not in self.splits:
            self.splits["train"] = DatasetSplit(os.path.join(config.DATASET_PATH, "train.jsonl"))
        if "validation" not in self.splits:
            self.splits["validation"] = DatasetSplit(os.path.join(config.DATASET_PATH, "validation.jsonl"))

        self.show_dataset_summary()

    def show_dataset_summary(self) -> None:

        self.logger.info(f"\nLoaded dataset with {self.num_samples()} samples")
        self.logger.info("Splits:")
        total_samples = 0
        for split in self.splits.values():
            self.logger.info(f"  {split.name}: {len(split.samples)} samples")
            total_samples += len(split.samples)

        self.logger.info("Dataset info:")
        self.logger.info(f"  {len(self.dataset_info['system_id'])} system id(s)")
        self.logger.info(f"  {len(self.dataset_info['game_id'])} game id(s)")
        self.logger.info(f"  {len(self.dataset_info['author_id'])} author id(s)")

        if total_samples > 0:
            min_sample_length = min(sample["sample_length"] for _, _, sample in self.all_samples() if sample["sample_length"] is not None)
            max_sample_length = max(sample["sample_length"] for _, _, sample in self.all_samples() if sample["sample_length"] is not None)
            min_sample_length_seconds = min(sample["sample_length"] / sample["sample_rate"]
                for _, _, sample in self.all_samples() if sample["sample_length"] is not None and sample["sample_rate"] is not None)
            max_sample_length_seconds = max(sample["sample_length"] / sample["sample_rate"]
                for _, _, sample in self.all_samples() if sample["sample_length"] is not None and sample["sample_rate"] is not None)
            
            try:
                min_latents_length = min(sample["latents_length"] for _, _, sample in self.all_samples() if sample["latents_length"] is not None)
                max_latents_length = max(sample["latents_length"] for _, _, sample in self.all_samples() if sample["latents_length"] is not None)
            except Exception as e:
                min_latents_length = 0
                max_latents_length = 0
        else:
            min_sample_length = 0
            max_sample_length = 0
            min_sample_length_seconds = 0
            max_sample_length_seconds = 0
            min_latents_length = 0
            max_latents_length = 0

        self.logger.info(f"  min sample_length: {min_sample_length} ({min_sample_length_seconds:.2f}s)")
        self.logger.info(f"  max sample_length: {max_sample_length} ({max_sample_length_seconds:.2f}s)")
        self.logger.info(f"  min latents_length: {min_latents_length}")
        self.logger.info(f"  max latents_length: {max_latents_length}")

    def get_id(self, id_type: str, name: str) -> int:

        id = self.dataset_info[id_type].get(name, None)
        if id is None:

            if "value_type" in self.dataset_info["features"][id_type]:
                id = self.dataset_info["features"][id_type]["value_type"]["num_classes"]
                self.dataset_info["features"][id_type]["value_type"]["num_classes"] += 1
            else:
                id = self.dataset_info["features"][id_type]["num_classes"]
                self.dataset_info["features"][id_type]["num_classes"] += 1

            self.dataset_info[id_type][name] = id

        return id
    
    def all_samples(self) -> Generator[DatasetSplit, int, dict]:
        for _, split in self.splits.items():
            yield from ((split, index, sample) for index, sample in enumerate(split.samples))

    def num_samples(self) -> int:
        return sum(len(split.samples) for split in self.splits.values())

    def get_unused_ids(self) -> tuple[dict[str, int]]:

        used_system_ids, used_game_ids, used_author_ids = set(), set(), set()
        for _, _, sample in self.all_samples():
            if sample["system_id"] is not None: used_system_ids.add(sample["system_id"])
            if sample["game_id"] is not None: used_game_ids.add(sample["game_id"])
            if sample["author_id"] is not None: used_author_ids.update(id for id in sample["author_id"])

        unused_system_ids: dict[str, int] = {}
        unused_game_ids: dict[str, int] = {}
        unused_author_ids: dict[str, int] = {}
        for system, system_id in self.dataset_info["system_id"].items():
            if system_id not in used_system_ids:
                unused_system_ids[system] = system_id
        for game, game_id in self.dataset_info["game_id"].items():
            if game_id not in used_game_ids:
                unused_game_ids[game] = game_id
        for author, author_id in self.dataset_info["author_id"].items():
            if author_id not in used_author_ids:
                unused_author_ids[author] = author_id

        return unused_system_ids, unused_game_ids, unused_author_ids

    def validate_files(self) -> None:

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
            if input(f"Remove {num_missing} samples with missing files? (y/n): ").lower() == "y":
                missing_samples.remove_samples_from_dataset()
                self.logger.info(f"{num_missing} samples with missing files removed")
            self.logger.info("")

        # search for any sample latents_file_name in splits that no longer exists
        # if any found, prompt to clear latents metadata for those samples
        missing_latents_samples = SampleList()
        for split, index, sample in self.all_samples():
            if sample["latents_file_name"] is not None:
                if not os.path.isfile(os.path.join(config.DATASET_PATH, sample["latents_file_name"])):
                    missing_latents_samples.add_sample(split, index)

        num_missing_latents = len(missing_latents_samples)
        if num_missing_latents > 0:
            self.logger.warning(f"Found {num_missing_latents} samples with nonexistent latents_file_name")
            missing_latents_samples.show_samples()
            if input(f"Clear latents metadata for {num_missing_latents} samples? (y/n): ").lower() == "y":
                for _, _, sample in missing_latents_samples:
                    sample["latents_file_name"] = None
                    sample["latents_file_size"] = None
                    sample["latents_length"] = None
                    sample["latents_num_variations"] = None
                    sample["latents_quantized"] = None
                    sample["latents_vae_model"] = None
                    sample["latents_has_audio_embeddings"] = None
                    sample["latents_has_text_embeddings"] = None
                self.logger.info(f"Cleared latents metadata for {num_missing_latents} samples")
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
            if input(f"Remove {num_invalid_format} samples with invalid file formats? (this will not delete the files) (y/n): ").lower() == "y":
                invalid_format_samples.remove_samples_from_dataset()
                self.logger.info(f"{num_invalid_format} samples with invalid file formats removed")
            self.logger.info("")

        # search for any samples with the same file_name in any splits
        # if any found, prompt to remove duplicates
        sample_files = set()
        duplicate_samples = SampleList()
        for split, index, sample in self.all_samples():
            norm_path = os.path.normpath(sample["file_name"])
            if norm_path in sample_files:
                duplicate_samples.add_sample(split, index)
            else:
                sample_files.add(norm_path)

        num_duplicates = len(duplicate_samples)
        if num_duplicates > 0:
            self.logger.warning(f"Found {num_duplicates} samples with duplicated file_names")
            duplicate_samples.show_samples()
            if input(f"Remove {num_duplicates} samples with duplicate file_names? (y/n): ").lower() == "y":
                duplicate_samples.remove_samples_from_dataset()
                self.logger.info(f"{num_duplicates} samples with duplicate file_names removed")
            self.logger.info("")

    def scan(self) -> None:

        # search for any files (excluding dataset metadata and latents files) that are not in the source formats list
        # if any found, prompt to permanently delete them
        invalid_format_files = []
        valid_dataset_file_formats = self.config.dataset_formats + self.config.source_formats + [".jsonl", ".json", ".safetensors", ".md"]
        for root, _, files in os.walk(config.DATASET_PATH):
            for file in files:
                if os.path.splitext(file)[1].lower() not in valid_dataset_file_formats:
                    invalid_format_files.append(os.path.join(root, file))

        num_invalid_format_files = len(invalid_format_files)
        if num_invalid_format_files > 0:
            self.logger.warning(f"Found {num_invalid_format_files} files with formats not in the source format list ({self.config.source_formats})")
            for file in invalid_format_files:
                self.logger.debug(f'"{file}"')
            if input(f"Delete {num_invalid_format_files} files with invalid formats? (WARNING: this is permanent and cannot be undone) (type 'delete' to confirm): ").lower() == "delete":
                for file in invalid_format_files:
                    os.remove(file)
                self.logger.info(f"Deleted {num_invalid_format_files} files with invalid formats")
            self.logger.info("")
    
        # search for any valid new source audio files not currently dataset
        # if any found, prompt to add them to train split
        sample_files = set()
        for _, _, sample in self.all_samples():
            sample_files.add(os.path.normpath(sample["file_name"]))

        new_audio_files = []
        valid_sample_formats = self.config.source_formats + self.config.dataset_formats
        for root, _, files in os.walk(config.DATASET_PATH):
            for file in files:
                if os.path.splitext(file)[1].lower() in valid_sample_formats:
                    rel_path = os.path.normpath(os.path.relpath(os.path.join(root, file), config.DATASET_PATH))
                    if rel_path not in sample_files:
                        new_audio_files.append(rel_path)

        num_new_audio_files = len(new_audio_files)
        if num_new_audio_files > 0:
            self.logger.info(f"Found {num_new_audio_files} new audio files not currently in the dataset")
            for file in new_audio_files:
                self.logger.debug(f'"{file}"')
            if input(f"Add {num_new_audio_files} new audio files to train split? (y/n): ").lower() == "y":
                new_samples = []
                for file in new_audio_files:
                    new_sample = {"file_name": file}
                    new_sample_latents_file = os.path.join(config.DATASET_PATH, os.path.splitext(file)[0] + ".safetensors")
                    if os.path.isfile(new_sample_latents_file):
                        new_sample["latents_file_name"] = os.path.relpath(new_sample_latents_file, config.DATASET_PATH)
                    new_samples.append(new_sample)
                self.splits["train"].add_samples(new_samples)
                self.logger.info(f"Added {num_new_audio_files} new audio files to train split")
            self.logger.info("")

        # search for any .safetensors file that has a filename corresponding an existing sample in
        # the dataset, but that sample does not have a latents_file_name set
        # if any found, prompt to set latents_file_name to the existing .safetensors file
        samples_with_safetensors = SampleList()
        for split, index, sample in self.all_samples():
            if sample["file_name"] is not None and sample["latents_file_name"] is None:
                latents_file_name = f"{os.path.splitext(sample['file_name'])[0]}.safetensors"
                if os.path.isfile(os.path.join(config.DATASET_PATH, latents_file_name)):
                    samples_with_safetensors.add_sample(split, index)
        
        num_samples_with_safetensors = len(samples_with_safetensors)
        if num_samples_with_safetensors > 0:
            self.logger.warning(f"Found {num_samples_with_safetensors} samples with no latents_file_name but matching latents file exists")
            samples_with_safetensors.show_samples()
            if input(f"Use existing pre-encoded latents files for {num_samples_with_safetensors} samples? (y/n): ").lower() == "y":
                for _, _, sample in samples_with_safetensors:
                    sample["latents_file_name"] = f"{os.path.splitext(sample['file_name'])[0]}.safetensors"
                self.logger.info(f"Set latents_file_name for {num_samples_with_safetensors} samples")

            self.logger.info("")

        # search for any .safetensors file that isn't referenced as a latents_file_name in the dataset
        # if any found, prompt to delete them
        referenced_latents_files = set()
        for _, _, sample in self.all_samples():
            if sample["latents_file_name"] is not None:
                referenced_latents_files.add(os.path.normpath(sample["latents_file_name"]))

        unreferenced_latents_files = []
        for root, _, files in os.walk(config.DATASET_PATH):
            for file in files:
                if os.path.splitext(file)[1].lower() == ".safetensors":
                    rel_path = os.path.normpath(os.path.relpath(os.path.join(root, file), config.DATASET_PATH))
                    if rel_path not in referenced_latents_files and os.path.dirname(rel_path).lower() != "dataset_infos":
                        unreferenced_latents_files.append(rel_path)

        num_unreferenced_latents_files = len(unreferenced_latents_files)
        if num_unreferenced_latents_files > 0:
            self.logger.warning(f"Found {num_unreferenced_latents_files} unreferenced .safetensors (latents) files")
            for file in unreferenced_latents_files:
                self.logger.debug(f'"{file}"')
            if input(f"Delete {num_unreferenced_latents_files} unreferenced .safetensors (latents) files? (WARNING: this is permanent and cannot be undone) (type 'delete' to confirm): ").lower() == "delete":
                for file in unreferenced_latents_files:
                    os.remove(os.path.join(config.DATASET_PATH, file))
                self.logger.info(f"Deleted {num_unreferenced_latents_files} unreferenced .safetensors (latents) files")
            self.logger.info("")

        # search for any empty folders, if any found prompt to delete
        empty_folders = []
        for root, dirs, files in os.walk(config.DATASET_PATH):
            if len(dirs) == 0 and len(files) == 0:
                empty_folders.append(root)
        
        if len(empty_folders) > 0:
            self.logger.warning(f"Found {len(empty_folders)} empty folders in dataset ({config.DATASET_PATH})")
            for folder in empty_folders:
                self.logger.debug(f'"{folder}"')
            if input(f"Delete {len(empty_folders)} empty folders? (WARNING: this is permanent and cannot be undone) (type 'delete' to confirm): ").lower() == "delete":
                for folder in empty_folders:
                    os.rmdir(folder)
                self.logger.info(f"Deleted {len(empty_folders)} empty folders")
            self.logger.info("")

    def transcode(self) -> None:

        # search for any samples in splits that have any null audio metadata fields
        # if any found, prompt to extract audio metadata
        need_metadata_samples = SampleList()
        metadata_check_keys = ["file_size", "sample_rate", "num_channels", "sample_length", "bit_rate"]
        for split, index, sample in self.all_samples():
            if any(sample[key] is None for key in metadata_check_keys):
                need_metadata_samples.add_sample(split, index)

        def get_audio_metadata(sample: dict) -> None:
            if sample["file_name"] is None:
                sample["file_size"] = None
                sample["sample_rate"] = None
                sample["num_channels"] = None
                sample["sample_length"] = None
                sample["bit_rate"] = None
                return
            audio_info = mutagen.File(os.path.join(config.DATASET_PATH, sample["file_name"])).info
            sample["file_size"] = os.path.getsize(os.path.join(config.DATASET_PATH, sample["file_name"]))
            sample["sample_rate"] = audio_info.sample_rate
            sample["num_channels"] = audio_info.channels
            sample["sample_length"] = int(audio_info.length * sample["sample_rate"])
            sample["bit_rate"] = int(sample["file_size"] / 128 / audio_info.length)

        num_need_metadata = len(need_metadata_samples)
        if num_need_metadata > 0:
            self.logger.warning(f"Found {num_need_metadata} samples with missing audio metadata")
            need_metadata_samples.show_samples()
            if input(f"Extract missing audio metadata for {num_need_metadata} samples? (y/n): ").lower() == "y":
                failed_metadata_extraction_samples = SampleList()
                for split, index, sample in tqdm(need_metadata_samples, total=num_need_metadata, mininterval=1):
                    try:
                        get_audio_metadata(sample)
                    except Exception as e:
                        failed_metadata_extraction_samples.add_sample(split, index, str(e))
                        continue

                num_failed_metadata = len(failed_metadata_extraction_samples)
                if num_failed_metadata > 0:
                    self.logger.warning(f"Failed to extract audio metadata for {len(failed_metadata_extraction_samples)} samples")
                    failed_metadata_extraction_samples.show_samples()
                self.logger.info(f"Extracted missing audio metadata for {num_need_metadata - num_failed_metadata} samples")

            self.logger.info("")

        # search for any samples with with sample_rate / sample_length / kbps < config values
        # if any found, prompt to delete them. if not, prompt to remove them
        invalid_audio_samples = SampleList()
        min_sample_length = self.config.min_sample_length / self.config.sample_rate if self.config.min_sample_length is not None else None
        for split, index, sample in self.all_samples():
            if sample["sample_rate"] is not None and self.config.min_sample_rate is not None:
                if sample["sample_rate"] < self.config.min_sample_rate:
                    invalid_audio_samples.add_sample(split, index, f"sample_rate {sample['sample_rate']} < {self.config.min_sample_rate}")
                    continue
            if sample["sample_length"] is not None and sample["sample_rate"] is not None and min_sample_length is not None:
                sample_length = sample["sample_length"] / sample["sample_rate"]
                if sample_length < min_sample_length:
                    invalid_audio_samples.add_sample(split, index, f"sample_length {sample_length:.2f}s < {min_sample_length:.2f}s")
                    continue
            if sample["bit_rate"] is not None and self.config.min_kbps is not None:
                if sample["bit_rate"] < self.config.min_kbps:
                    invalid_audio_samples.add_sample(split, index, f"kbps {sample['bit_rate']} < {self.config.min_kbps}")

        num_invalid_audio = len(invalid_audio_samples)
        if num_invalid_audio > 0:
            self.logger.warning(f"Found {num_invalid_audio} samples with sample/bit rate or length below config minimum values")
            invalid_audio_samples.show_samples()
            if input(f"Delete {num_invalid_audio} samples with insufficient sample/bit rate or length? (WARNING: this is permanent and cannot be undone) (type 'delete' to confirm): ").lower() == "delete":
                for _, _, sample in invalid_audio_samples:
                    os.remove(os.path.join(config.DATASET_PATH, sample["file_name"]))
                invalid_audio_samples.remove_samples_from_dataset()
                self.logger.info(f"{num_invalid_audio} samples with insufficient sample/bit rate or length deleted")
            elif input(f"Remove {num_invalid_audio} samples with insufficient sample/bit rate or length? (y/n): ").lower() == "y":
                invalid_audio_samples.remove_samples_from_dataset()
                self.logger.info(f"{num_invalid_audio} samples with insufficient sample/bit rate or length removed from dataset")
            self.logger.info("")

        # transcoding in dataset_processor removed, it is done better in an external tool (foobar2000)
        
    def filter(self) -> None:

        # find any game ids with a low number of samples
        # if any found, prompt to merge them into a "misc" game id
        # else prompt to remove them
        if self.config.min_num_class_samples is not None:
            game_id_to_name = {v: k for k, v in self.dataset_info["game_id"].items()}
            game_id_counts = {game_id: 0 for game_id in self.dataset_info["game_id"].values()}
            for _, _, sample in self.all_samples():
                if sample["game_id"] is not None:
                    game_id_counts[sample["game_id"]] += 1

            low_sample_game_ids = {}
            total_low_samples = 0
            for game_id, count in game_id_counts.items():
                if count < self.config.min_num_class_samples and count > 0:
                    low_sample_game_ids[game_id] = count
                    total_low_samples += count

            if len(low_sample_game_ids) > 0:
                self.logger.warning(f"Found {len(low_sample_game_ids)} game ids with fewer than {self.config.min_num_class_samples} samples")
                sorted_game_ids = dict(sorted(low_sample_game_ids.items(), key=lambda x: x[1]))
                for game_id, count in sorted_game_ids.items():
                    self.logger.debug(f"Game id '{game_id_to_name[game_id]}' has only {count} samples")
                self.logger.debug(f"Total sample count in low_sample_game_ids: {total_low_samples}")

                #input("")
                #if input(f"Remove {len(low_sample_game_ids)} game ids with fewer than {self.config.min_class_samples} samples? (y/n): ").lower() == "y":

        # todo: detect any duplicate / highly similar samples in splits,
        # detect any abnormal / anomalous samples in splits
        # if any found, prompt to remove them
        pass
    
    def validate_metadata(self) -> None:
        
        # search for any samples in splits that have a latents_file_name and any null latents metadata fields
        # if any found, prompt to extract latents metadata
        need_metadata_samples = SampleList()
        metadata_check_keys = [key for key in self.dataset_info["features"].keys() if key.startswith("latents_") and key != "latents_file_name"]
        for split, index, sample in self.all_samples():
            if sample["latents_file_name"] is not None:
                if any(sample[key] is None for key in metadata_check_keys) or (sample["prompt"] is not None and sample["latents_has_text_embeddings"] == False):
                    need_metadata_samples.add_sample(split, index)

        num_need_metadata = len(need_metadata_samples)
        if num_need_metadata > 0:
            self.logger.warning(f"Found {num_need_metadata} samples with missing latents metadata")
            need_metadata_samples.show_samples()
            if input(f"Extract missing latents metadata for {num_need_metadata} samples? (y/n): ").lower() == "y":
                failed_metadata_extraction_samples = SampleList()
                for split, index, sample in tqdm(need_metadata_samples, total=num_need_metadata, mininterval=1):
                    try:
                        latents_file_path = os.path.join(config.DATASET_PATH, sample["latents_file_name"])
                        sample["latents_file_size"] = os.path.getsize(latents_file_path)

                        with ST.safe_open(latents_file_path, framework="pt") as f:
                            latents_shape = f.get_slice("latents").get_shape()
                            sample["latents_length"] = latents_shape[-1]
                            sample["latents_num_variations"] = latents_shape[0]
                            
                            try:
                                _ = f.get_slice("offset_and_range")
                                sample["latents_quantized"] = True
                            except Exception as _:
                                sample["latents_quantized"] = False

                            st_metadata = f.metadata()
                            if st_metadata is not None:
                                sample["latents_vae_model"] = st_metadata.get("latents_vae_model", None)

                            try:
                                _ = f.get_slice("clap_audio_embeddings")
                                sample["latents_has_audio_embeddings"] = True
                            except Exception as _:
                                sample["latents_has_audio_embeddings"] = False
                            
                            try:
                                _ = f.get_slice("clap_text_embeddings")
                                sample["latents_has_text_embeddings"] = True
                            except Exception as _:
                                sample["latents_has_text_embeddings"] = False

                    except Exception as e:
                        failed_metadata_extraction_samples.add_sample(split, index, str(e))
                        continue

                num_failed_metadata = len(failed_metadata_extraction_samples)
                if num_failed_metadata > 0:
                    self.logger.warning(f"Failed to extract latents metadata for {len(failed_metadata_extraction_samples)} samples")
                    failed_metadata_extraction_samples.show_samples()
                self.logger.info(f"Extracted missing latents metadata for {num_need_metadata - num_failed_metadata} samples")

            self.logger.info("")

        # search for any system, game, or author ids that don't match the dataset_info
        # if any found, prompt to clear invalid metadata
        invalid_id_samples = SampleList()
        for split, index, sample in self.all_samples():
            if sample["system"] is not None:
                ds_info_system_id = self.dataset_info["system_id"].get(sample["system"], None)
                if ds_info_system_id != sample["system_id"]:
                    invalid_id_samples.add_sample(split, index, f"system_id '{sample['system_id']}' != {ds_info_system_id}")
                    continue
            if sample["game"] is not None: 
                ds_info_game_id = self.dataset_info["game_id"].get(sample["game"], None)
                if ds_info_game_id != sample["game_id"]:
                    invalid_id_samples.add_sample(split, index, f"game_id '{sample['game_id']}' != {ds_info_game_id}")
                    continue
            if sample["author"] is not None:
                for i, author in enumerate(sample["author"]):
                    ds_info_author_id = self.dataset_info["author_id"].get(author, None)
                    if ds_info_author_id != sample["author_id"][i]:
                        invalid_id_samples.add_sample(split, index, f"author_id '{sample['author_id'][i]}' != {ds_info_author_id}")
                        break

        num_invalid_id = len(invalid_id_samples)
        if num_invalid_id > 0:
            self.logger.warning(f"Found {num_invalid_id} samples with game, system, or author ids inconsistent with dataset_info")
            invalid_id_samples.show_samples()
            if input(f"Clear invalid id metadata for {num_invalid_id} samples? (y/n): ").lower() == "y":
                for _, _, sample in invalid_id_samples:
                    sample["system_id"] = None
                    sample["game_id"] = None
                    sample["author_id"] = None
                self.logger.info(f"Cleared invalid id metadata for {num_invalid_id} samples")
            self.logger.info("")

        # search for any samples in splits that have any null id metadata fields
        # if any found, prompt to extract metadata
        need_metadata_samples = SampleList()
        metadata_check_keys = ["system", "game", "song", "author", "system_id", "game_id", "author_id"]
        for split, index, sample in self.all_samples():
            if any(sample[key] is None for key in metadata_check_keys):
                need_metadata_samples.add_sample(split, index)

        num_need_metadata = len(need_metadata_samples)
        if num_need_metadata > 0:
            self.logger.warning(f"Found {num_need_metadata} samples with missing id metadata")
            need_metadata_samples.show_samples()
            if input(f"Extract missing id metadata for {num_need_metadata} samples? (y/n): ").lower() == "y":

                num_pre_update_system_ids = len(self.dataset_info["system_id"])
                num_pre_update_game_ids = len(self.dataset_info["game_id"])
                num_pre_update_author_ids = len(self.dataset_info["author_id"])

                metadata_check_keys = ["system", "game", "song", "author"]
                failed_metadata_extraction_samples = SampleList()
                for split, index, sample in tqdm(need_metadata_samples, total=num_need_metadata, mininterval=1):
                    try:
                        if any(sample[key] is None for key in metadata_check_keys):
                            file_path_list = os.path.normpath(sample["file_name"]).split(os.sep)
                            if len(file_path_list) == 3:
                                if sample["system"] is None: sample["system"] = file_path_list[0]
                                if sample["game"] is None: sample["game"] = f"{file_path_list[0]}/{file_path_list[1]}"
                                if sample["song"] is None: sample["song"] = os.path.splitext(os.path.basename(sample["file_name"]))[0]
                                if sample["author"] is None:
                                    sample["author"] = []
                                    sample_metadata = get_audio_metadata(os.path.join(config.DATASET_PATH, sample["file_name"]))
                                    if "author" in sample_metadata:
                                        for author in sample_metadata["author"]:
                                            sample["author"].extend([author.strip() for author in author.split(",")])
                            else:
                                failed_metadata_extraction_samples.add_sample(split, index, "path does not match system/game/song format")
                                continue

                        if sample["system_id"] is None: sample["system_id"] = self.get_id("system_id", sample["system"])
                        if sample["game_id"] is None: sample["game_id"] = self.get_id("game_id", sample["game"])
                        if sample["author_id"] is None: sample["author_id"] = [self.get_id("author_id", author) for author in sample["author"]]

                    except Exception as e:
                        failed_metadata_extraction_samples.add_sample(split, index, str(e))
                        continue
                
                num_failed_metadata = len(failed_metadata_extraction_samples)
                if num_failed_metadata > 0:
                    self.logger.warning(f"Failed to extract id metadata for {len(failed_metadata_extraction_samples)} samples")
                    failed_metadata_extraction_samples.show_samples()

                self.logger.info(f"Extracted missing id metadata for {num_need_metadata - num_failed_metadata} samples")
                if num_pre_update_system_ids != len(self.dataset_info["system_id"]):
                    self.logger.info(f"Added {len(self.dataset_info['system_id']) - num_pre_update_system_ids} new system id(s)")
                if num_pre_update_game_ids != len(self.dataset_info["game_id"]):
                    self.logger.info(f"Added {len(self.dataset_info['game_id']) - num_pre_update_game_ids} new game id(s)")
                if num_pre_update_author_ids != len(self.dataset_info["author_id"]):
                    self.logger.info(f"Added {len(self.dataset_info['author_id']) - num_pre_update_author_ids} new author id(s)")

            self.logger.info("")

        # search for any unused system, game, or author ids
        # if any found, prompt to rebuild dataset_info / ids from current metadata
        unused_system_ids, unused_game_ids, unused_author_ids = self.get_unused_ids()
        if len(unused_system_ids) > 0 or len(unused_game_ids) > 0 or len(unused_author_ids) > 0:
            self.logger.warning("Found unused system, game, or author ids in dataset info")
            if len(unused_system_ids) > 0:
                self.logger.warning(f"Unused system ids: {len(unused_system_ids)}")
                self.logger.debug(f"{dict_str(unused_system_ids)}")
            if len(unused_game_ids) > 0:
                self.logger.warning(f"Unused game ids: {len(unused_game_ids)}")
                self.logger.debug(f"{dict_str(unused_game_ids)}")
            if len(unused_author_ids) > 0:
                self.logger.warning(f"Unused author ids: {len(unused_author_ids)}")
                self.logger.debug(f"{dict_str(unused_author_ids)}")

            if input("Rebuild all dataset ids from current metadata? (WARNING: any models trained with current ids will have incorrect class labels) (type 'rebuild' to confirm): ").lower() == "rebuild":
                self.dataset_info["system_id"] = {}
                self.dataset_info["features"]["system_id"]["num_classes"] = 0
                self.dataset_info["game_id"] = {}
                self.dataset_info["features"]["game_id"]["num_classes"] = 0
                self.dataset_info["author_id"] = {}
                self.dataset_info["features"]["author_id"]["value_type"]["num_classes"] = 0

                for _, _, sample in self.all_samples():
                    sample["system_id"] = self.get_id("system_id", sample["system"])
                    sample["game_id"] = self.get_id("game_id", sample["game"])
                    sample["author_id"] = [self.get_id("author_id", author) for author in sample["author"]]

                self.logger.info("Rebuilt all dataset ids from current metadata")
            self.logger.info("")

        # search for any samples in splits that have any null prompt field
        # if any found, prompt to to create a default prompt for each sample
        need_prompt_samples = SampleList()
        for split, index, sample in self.all_samples():
            if sample["prompt"] is None:
                if (sample["song"] is not None and sample["song"] != "") or (sample["game"] is not None and sample["game"] != ""):
                    need_prompt_samples.add_sample(split, index)

        num_need_prompt = len(need_prompt_samples)
        if num_need_prompt > 0:
            self.logger.warning(f"Found {num_need_prompt} samples with song metadata and missing prompt")
            need_prompt_samples.show_samples()
            if input(f"Create default prompt for {num_need_prompt} samples? (y/n): ").lower() == "y":
                for _, _, sample in need_prompt_samples:
                    song = sample["song"] or ""
                    game = sample["game"] or ""
                    if game != "": game = game.split("/")[1]
                    if "miscellaneous" in game.lower(): game = ""
                    if game.lower().split(" ")[0] in song.lower().split(" ")[0]: game = ""

                    prompt = f"{game} - {song}" if game != "" else song
                    if sample["author"] is not None and len(sample["author"]) > 0:
                        prompt += f" by {', '.join(sample['author'])}"
                    sample["prompt"] = prompt
                    self.logger.debug(prompt)

                self.logger.info(f"Created default prompt for {num_need_prompt} samples")
            self.logger.info("")

    def train_validation_split(self) -> None:
        # todo: prompt to resplit the aggregated dataset into train / validation splits
        pass

    def encode_latents(self) -> None:

        # todo: pre-encoding latents and embeddings is done in separate scripts, ideally launching them from here would be nice
        """
        # if self.config.pre_encoded_latents_vae is null skip encode_latents step
        if self.config.pre_encoded_latents_vae is None:
            self.logger.warning("Skipping encode_latents because config.pre_encoded_latents_vae is not defined")

        # search for any samples with a null latents_vae_model, or
        # a latents_vae_model that doesn't match the config vae model name
        # or a null latents_file_name
        # if any found, prompt to encode them with configured vae and update metadata
        # will need to launch subprocess to use accelerate
        # accelerate config for pre_encoding is in config.CONFIG_PATH/dataset/dataset_accelerate.yaml
        """
        

        samples_with_audio_embeddings = SampleList()
        samples_with_text_embeddings = SampleList()
        samples_with_embeddings = SampleList()
        for split, index, sample in self.all_samples():
            if sample["latents_has_audio_embeddings"] == True:
                samples_with_audio_embeddings.add_sample(split, index)
            if sample["latents_has_text_embeddings"] == True:
                samples_with_text_embeddings.add_sample(split, index)
            if sample["latents_has_audio_embeddings"] == True or sample["latents_has_text_embeddings"] == True:
                samples_with_embeddings.add_sample(split, index)

        num_samples_with_embeddings = len(samples_with_embeddings)
        if num_samples_with_embeddings > 0:
            num_samples_with_audio_embeddings = len(samples_with_audio_embeddings)
            num_samples_with_text_embeddings = len(samples_with_text_embeddings)
            self.logger.info(f"Found {num_samples_with_audio_embeddings} samples with audio embeddings and {num_samples_with_text_embeddings} samples with text embeddings")
            samples_with_embeddings.show_samples()

            if input(f"Aggregate embeddings? (you only need to do this when the dataset embeddings have changed) (y/n): ").lower() == "y":
                self.logger.info("Aggregating dataset audio and text embeddings...")
                dataset_embeddings_dict = {
                    "_unconditional_audio": torch.zeros(512, dtype=torch.float64),
                    "_unconditional_text": torch.zeros(512, dtype=torch.float64),
                }

                for split, index, sample in tqdm(samples_with_embeddings, total=num_samples_with_embeddings, mininterval=1):
                    latents_path = os.path.join(config.DATASET_PATH, sample["latents_file_name"])
                    with ST.safe_open(latents_path, framework="pt") as f:
                        
                        if sample["latents_has_audio_embeddings"] == True:
                            dataset_embeddings_dict["_unconditional_audio"].add_(
                                f.get_slice("clap_audio_embeddings")[:].to(torch.float64).mean(dim=0), alpha=1./num_samples_with_audio_embeddings)
                            if sample["game"] is not None:
                                game_audio_embeddings = dataset_embeddings_dict.get(f"{sample['game']}_audio",
                                    torch.zeros_like(dataset_embeddings_dict["_unconditional_audio"]))
                                game_audio_embeddings.add_(f.get_slice("clap_audio_embeddings")[:].to(torch.float64).mean(dim=0))
                                dataset_embeddings_dict[f"{sample['game']}_audio"] = game_audio_embeddings

                        if sample["latents_has_text_embeddings"] == True:
                            dataset_embeddings_dict["_unconditional_text"].add_(
                                f.get_slice("clap_text_embeddings")[:].to(torch.float64).mean(dim=0), alpha=1./num_samples_with_text_embeddings)
                            if sample["game"] is not None:
                                game_text_embeddings = dataset_embeddings_dict.get(f"{sample['game']}_text",
                                    torch.zeros_like(dataset_embeddings_dict["_unconditional_text"]))
                                game_text_embeddings.add_(f.get_slice("clap_text_embeddings")[:].to(torch.float64).mean(dim=0))
                                dataset_embeddings_dict[f"{sample['game']}_text"] = game_text_embeddings
                
                dataset_embeddings_dict = {k: normalize(v).float() for k, v in dataset_embeddings_dict.items()}
                output_path = os.path.join(config.DATASET_PATH, "dataset_infos", "dataset_embeddings.safetensors")
                self.logger.info(f"Saving aggregated dataset embeddings to '{output_path}'...")
                save_safetensors(dataset_embeddings_dict, output_path)
                self.logger.info("")

 
    def save(self, dataset_path: Optional[str] = None) -> None:
    
        dataset_path = dataset_path or config.DATASET_PATH

        # add total number of samples in train / validation splits to dataset_info
        self.dataset_info["num_total_samples"] = self.num_samples()
        self.dataset_info["num_train_samples"] = len(self.splits["train"].samples)
        self.dataset_info["num_validation_samples"] = len(self.splits["validation"].samples)
        
        # add number of samples in training set for each system / game / author id to dataset_info
        game_train_sample_counts = {game: 0 for game in self.dataset_info["game_id"].keys()}
        system_train_sample_counts = {system: 0 for system in self.dataset_info["system_id"].keys()}
        author_train_sample_counts = {author: 0 for author in self.dataset_info["author_id"].keys()}
        for sample in self.splits["train"].samples:
            if sample["system"] is not None: system_train_sample_counts[sample["system"]] += 1
            if sample["game"] is not None: game_train_sample_counts[sample["game"]] += 1
            if sample["author"] is not None:
                for author in sample["author"]:
                    author_train_sample_counts[author] += 1
        
        self.dataset_info["game_train_sample_counts"] = game_train_sample_counts
        self.dataset_info["system_train_sample_counts"] = system_train_sample_counts
        self.dataset_info["author_train_sample_counts"] = author_train_sample_counts

        # prompt to save and backup existing metadata files to config.DEBUG_PATH

        if os.path.isfile(self.dataset_info_path):
            if self.backup_path is None:
                backup_warning = " (WARNING: Dataset metadata backup is NOT enabled)"
            else:
                backup_warning = f" (Backing up to '{self.backup_path}')"
        else:
            backup_warning = " No existing dataset metadata to backup"
            self.backup_path = None

        if input(f"Save changes to dataset metadata? (path: '{dataset_path}') (y/n){backup_warning}: ").lower() == "y":
            if self.backup_path is not None:
                self.logger.info(f"Backing up dataset metadata to '{self.backup_path}'")
                backup_dataset_info_path = os.path.join(self.backup_path, "dataset_infos", "dataset_info.json")
                os.makedirs(os.path.dirname(backup_dataset_info_path), exist_ok=True)
                
                shutil.copy(self.dataset_info_path, backup_dataset_info_path)
                for split in self.splits.values():
                    shutil.copy(split.path, os.path.join(self.backup_path, f"{split.name}.jsonl"))

            self.logger.info(f"Saving dataset metadata to '{dataset_path}'")
            config.save_json(self.dataset_info, os.path.join(dataset_path, "dataset_infos", "dataset_info.json"))
            for split in self.splits.values():
                split.save(os.path.join(dataset_path, f"{split.name}.jsonl"))
            self.logger.info(f"Saved dataset metadata to '{dataset_path}' successfully")
        else:
            self.logger.info(f"Finished without saving changes")

    def process_dataset(self) -> None:
        
        self.validate_files()
        self.scan()
        self.transcode()
        self.filter()
        self.validate_metadata()
        self.train_validation_split()
        self.encode_latents()
        self.save()


if __name__ == "__main__":

    from utils.dual_diffusion_utils import init_cuda

    init_cuda()

    processor = DatasetProcessor()
    processor.process_dataset()