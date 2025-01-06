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

from typing import Optional, Union, Any
import os
import logging

import torch
import mutagen
import safetensors.torch as safetensors

from dataset.dataset_processor import DatasetProcessor, DatasetProcessStage
from utils.dual_diffusion_utils import get_audio_metadata


class Build(DatasetProcessStage):

    def summary_banner(self, logger: logging.Logger) -> None:
        
        # aggregate samples from each worker process
        dataset_samples: list[dict] = []
        while self.output_queue.queue.qsize() > 0:
            process_dataset_samples_dict: dict[str, Any] = self.output_queue.get()
            dataset_samples.append(process_dataset_samples_dict.values())

        # build splits - todo: add option for auto-validation split
        dataset_splits: dict[str, list[dict]] = {}

        for sample in dataset_samples: 
            sample["split"] = sample["split"] or ["train"]

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
                    dataset_splits[split] = {}

                dataset_splits[split].append(sample)

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
        dataset_info_path = os.path.join(config.DATASET_PATH, "dataset_infos", "dataset_info.json")
        config.save_json(dataset_info, dataset_info_path, copy_on_write=self.processor_config.copy_on_write)

        # write splits and summarize results
        logger.info(f"\nTotal samples: {len(dataset_samples)}")
        logger.info("Splits:")
        for split, split_samples in dataset_splits.items():
            self.logger.info(f"  {split}: {len(split_samples)} samples")

            split_path = os.path.join(config.DATASET_PATH, f"{split}.jsonl")
            config.save_json(split_samples, split_path, copy_on_write=self.processor_config.copy_on_write)
    
    def get_dataset_sample(self, file_name: str) -> dict[str, Any]:

        if file_name in self.dataset_samples:
            return self.dataset_samples[file_name]
        
        new_sample = {
            "file_name": file_name,
            "sample_rate": None,
            "num_channels": None,
            "sample_length": None,
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
                audio_info = mutagen.File(file_path).info
                sample_dict["sample_rate"] = int(audio_info.sample_rate)
                sample_dict["num_channels"] = int(audio_info.channels)
                sample_dict["sample_length"] = int(audio_info.length * audio_info.sample_rate)
            except Exception as e:
                self.logger.error(f"Error reading audio info for \"{file_path}\": {e}")

            try:
                audio_metadata = get_audio_metadata(file_path)
                for field in ["system", "game", "song", "prompt", "rating"]:
                    sample_dict[field] = audio_metadata[field][0] if field in audio_metadata else None

                if sample_dict["rating"] is not None:
                    try:
                        sample_dict["rating"] = int(sample_dict["rating"])
                    except Exception as e:
                        self.logger.warning(f"invalid rating \"{sample_dict['rating']}\" in \"{file_path}\"")
                        sample_dict["rating"] = None

                if "split" in audio_metadata:
                    sample_dict[field] = audio_metadata["split"]

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

    dataset_processor = DatasetProcessor()
    dataset_processor.process(
        "Build",
        [Build()],
        input=config.DATASET_PATH
    )

    os._exit(0)