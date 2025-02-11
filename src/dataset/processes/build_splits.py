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

from utils import config

from typing import Optional, Union, Any, Literal
from dataclasses import dataclass
import os
import logging

import torch
import safetensors.torch as safetensors

from dataset.dataset_processor import DatasetProcessor, DatasetProcessStage
from utils.dual_diffusion_utils import get_audio_metadata, get_audio_info


@dataclass
class BuildSplitsProcessConfig:
    output_dataset_path: str = None  # if set, overrides the default output path of $DATASET_PATH
    input_dataset_path: str  = None  # if set, overrides the default input path of $DATASET_PATH

class BuildSplits(DatasetProcessStage):

    def __init__(self, process_config: BuildSplitsProcessConfig) -> None:
        self.process_config = process_config
        
    def get_stage_type(self) -> Literal["io", "cpu", "cuda"]:
        return "cpu"
    
    def info_banner(self, logger: logging.Logger):
        logger.info(f"Input path: {self.process_config.input_dataset_path or config.DATASET_PATH}")
        logger.info(f"Output path: {self.process_config.output_dataset_path or config.DATASET_PATH}")

    def summary_banner(self, logger: logging.Logger, completed: bool) -> None:
        
        # if the process was aborted don't write any split / dataset info files
        if completed != True: return
        
        # aggregate samples from each worker process
        
        dataset_samples: dict[str, dict] = {}
        while self.output_queue.queue.qsize() > 0:
            process_dataset_samples_dict: dict[str, dict] = self.output_queue.get()
            for sample_name, sample_data in process_dataset_samples_dict.items():

                if sample_name not in dataset_samples:
                    dataset_samples[sample_name] = sample_data
                    continue

                dataset_samples[sample_name].update(
                    {k: v for k, v in sample_data.items() if v is not None})

        # build splits - todo: add option for auto-validation split
        dataset_splits: dict[str, list[dict]] = {}
        dataset_samples = [*dataset_samples.values()]
        for sample in dataset_samples: 
            if sample["split"] is None: sample["split"] = ["train"]

            splits = []
            for split in sample["split"]:
                rating = sample["rating"]
                
                if rating is not None:
                    if rating <= 1:
                        splits.append(f"{split}_negative")
                    elif rating == 2:
                        splits.append(split)
                    elif rating >= 3:
                        splits.append(split)
                        splits.append(f"{split}_positive")
                else:
                    splits.append(split)
            
            for split in splits:
                if split not in dataset_splits:
                    dataset_splits[split] = []

                dataset_splits[split].append(sample)

        # just makes the summary neater / easier to read
        dataset_splits = dict(sorted(dataset_splits.items()))

        # write dataset info
        dataset_info = {
            "features": {
                "file_name": {"type": "string"},
                "sample_rate": {"type": "int"},
                "num_channels": {"type": "int"},
                "sample_length": {"type": "int"},
                "system": {"type": "string"},
                "game": {"type": "string"},
                "song": {"type": "string"},
                "author": {"type": "list", "value_type": {"type": "string"}},
                "split": {"type": "list", "value_type": {"type": "string"}},
                "prompt": {"type": "string"},
                "rating": {"type": "int"},
                "latents_file_name": {"type": "string"},
                "latents_length": {"type": "int"},
                "latents_num_variations": {"type": "int"},
                "latents_has_audio_embeddings": {"type": "bool"},
                "latents_has_text_embeddings": {"type": "bool"},
            },
            "num_split_samples": {split: len(split_samples) for split, split_samples in dataset_splits.items()},
            "total_num_samples": len(dataset_samples),
            "processor_config": self.processor_config.__dict__,
        }

        dataset_path = self.process_config.output_dataset_path or config.DATASET_PATH
        dataset_info_path = os.path.join(dataset_path, "dataset_infos", "dataset_info.json")
        config.save_json(dataset_info, dataset_info_path, copy_on_write=True)

        # write splits and summarize results
        summary_str = f"\nTotal samples: {len(dataset_samples)}\nSplits:"
        for split, split_samples in dataset_splits.items():
            summary_str += f"\n  {split}: {len(split_samples)} samples"

            split_path = os.path.join(dataset_path, f"{split}.jsonl")
            config.save_json(split_samples, split_path, copy_on_write=True)

        logger.info(summary_str)

        # fix stage processed/skipped stats
        self.input_queue.total_count.value = self.skip_counter.value
        self.input_queue.processed_count.value = len(dataset_samples)
        self.skip_counter.value -= len(dataset_samples)
        self.input_queue.total_count.value += self.skip_counter.value
        self.input_queue.processed_count.value += self.skip_counter.value

    def get_dataset_sample(self, file_name: str) -> dict[str, Any]:

        if file_name in self.dataset_samples:
            return self.dataset_samples[file_name]
        
        new_sample = {
            "file_name": file_name,
            "sample_rate": None,
            "num_channels": None,
            "sample_length": None,
            "post_norm_lufs": None,
            "system": None,
            "game": None,
            "song": None,
            "author": None,
            "prompt": None,
            "rating": None,
            "split": None,
            "latents_file_name": None,
            "latents_length": None,
            "latents_num_variations": None,
            "latents_has_audio_embeddings": None,
            "latents_has_text_embeddings": None,
        }

        self.dataset_samples[file_name] = new_sample
        return new_sample

    @torch.inference_mode()
    def start_process(self) -> None:
        self.dataset_samples = {}

    @torch.inference_mode()
    def finish_process(self) -> None:
        self.output_queue.put(self.dataset_samples)

    @torch.inference_mode()
    def process(self, input_dict: dict) -> Optional[Union[dict, list[dict]]]:
        
        file_path = input_dict["file_path"]
        file_ext:str = os.path.splitext(file_path)[1]

        if file_ext == ".flac": # load audio / general metadata
            
            file_name = os.path.normpath(os.path.relpath(file_path, input_dict["scan_path"]))
            sample_dict = self.get_dataset_sample(file_name)

            try:
                # if sample is too short remove it from the dataset and skip it
                audio_info = get_audio_info(file_path)
                if (self.processor_config.min_audio_length is not None and
                    audio_info.duration < self.processor_config.min_audio_length): 
                    del self.dataset_samples[file_name]
                    return None
                
                sample_dict["sample_rate"] = audio_info.sample_rate
                sample_dict["num_channels"] = audio_info.channels
                sample_dict["sample_length"] = audio_info.frames
            except Exception as e:
                self.logger.error(f"Error reading audio info for \"{file_path}\": {e}")

            try:
                audio_metadata = get_audio_metadata(file_path)
                for field in ["system", "game", "song", "prompt", "rating", "post_norm_lufs"]:
                    sample_dict[field] = audio_metadata[field][0] if field in audio_metadata else None

                # parse rating metadata as integer
                if sample_dict["rating"] is not None:
                    try:
                        sample_dict["rating"] = int(sample_dict["rating"])
                    except Exception as e:
                        self.logger.warning(f"invalid rating \"{sample_dict['rating']}\" in \"{file_path}\"")
                        sample_dict["rating"] = None

                # parse post_norm_lufs metadata as float
                if sample_dict["post_norm_lufs"] is not None:
                    try:
                        sample_dict["post_norm_lufs"] = float(sample_dict["post_norm_lufs"])
                    except Exception as e:
                        self.logger.warning(f"invalid post_norm_lufs \"{sample_dict['post_norm_lufs']}\" in \"{file_path}\"")
                        sample_dict["post_norm_lufs"] = None

                if "split" in audio_metadata:
                    sample_dict["split"] = audio_metadata["split"]

                if "author" in audio_metadata:
                    sample_dict["author"] = []
                    for author in audio_metadata["author"]:
                        sample_dict["author"].extend([author.strip() for author in author.split(",")])

            except Exception as e:
                self.logger.error(f"Error reading audio metadata for \"{file_path}\": {e}")

        elif file_ext == ".safetensors": # load latents metadata
            
            file_name = os.path.normpath(os.path.relpath(f"{os.path.splitext(file_path)[0]}.flac", input_dict["scan_path"]))
            sample_dict = self.get_dataset_sample(file_name)

            with safetensors.safe_open(file_path, framework="pt") as f:
                sample_dict["latents_file_name"] = os.path.normpath(os.path.relpath(file_path, input_dict["scan_path"]))

                try:
                    latents_shape = f.get_slice("latents").get_shape()
                    sample_dict["latents_length"] = latents_shape[-1]
                    sample_dict["latents_num_variations"] = latents_shape[0]
                except:
                    sample_dict["latents_length"] = 0
                    sample_dict["latents_num_variations"] = 0

                try:
                    f.get_slice("clap_audio_embeddings")
                    sample_dict["latents_has_audio_embeddings"] = True
                except:
                    sample_dict["latents_has_audio_embeddings"] = False

                try:
                    f.get_slice("clap_text_embeddings")
                    sample_dict["latents_has_text_embeddings"] = True
                except:
                    sample_dict["latents_has_text_embeddings"] = False
        
        return None


if __name__ == "__main__":

    process_config: BuildSplitsProcessConfig = config.load_config(BuildSplitsProcessConfig,
                        os.path.join(config.CONFIG_PATH, "dataset", "build_splits.json"))
    
    dataset_processor = DatasetProcessor()
    dataset_processor.process(
        "Build",
        [BuildSplits(process_config)],
        input=process_config.input_dataset_path or config.DATASET_PATH
    )

    exit(0)