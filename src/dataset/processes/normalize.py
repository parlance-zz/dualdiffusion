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

from typing import Optional, Union
import logging
import os

import torch

from dataset.dataset_processor import DatasetProcessor, DatasetProcessStage
from utils.dual_diffusion_utils import (
    get_audio_metadata, load_audio, save_audio, normalize_lufs, get_num_clipped_samples
)

class NormalizeLoad(DatasetProcessStage):

    def info_banner(self, logger: logging.Logger) -> None:
        logger.info(f"Normalizing to target_lufs: {self.processor_config.target_lufs}")

    @torch.inference_mode()
    def process(self, input_dict: dict) -> Optional[Union[dict, list[dict]]]:

        if os.path.splitext(input_dict["file_path"])[1].lower() == ".flac":
            
            target_lufs = self.processor_config.target_lufs

            audio_path = input_dict["file_path"]
            audio_metadata = get_audio_metadata(audio_path)
            current_lufs = audio_metadata.get("post_norm_lufs", None)
            if current_lufs is not None: current_lufs = float(current_lufs[0])

            if current_lufs != target_lufs or self.processor_config.force_overwrite == True:
                audio, sample_rate = load_audio(audio_path, return_sample_rate=True)

                return {
                    "audio_path": audio_path,
                    "target_lufs": target_lufs,
                    "audio": audio,
                    "audio_metadata": audio_metadata,
                    "sample_rate": sample_rate,
                }
        
        return None

    def get_max_output_queue_size(self):
        return 5
    
    def get_stage_type(self):
        return "cpu"

class NormalizeProcess(DatasetProcessStage):

    @torch.inference_mode()
    def process(self, input_dict: dict) -> Optional[Union[dict, list[dict]]]:
        
        audio: torch.Tensor = input_dict["audio"]
        audio_path = input_dict["audio_path"]
        target_lufs = input_dict["target_lufs"]
        sample_rate = input_dict["sample_rate"]
        audio_metadata = input_dict["audio_metadata"]

        pre_norm_peaks = get_num_clipped_samples(audio)
        normalized_audio, old_lufs = normalize_lufs(raw_samples=audio,
            sample_rate=sample_rate, target_lufs=target_lufs, return_old_lufs=True)
        post_norm_peaks = get_num_clipped_samples(normalized_audio)
        
        audio_metadata["pre_norm_peaks"] = pre_norm_peaks
        audio_metadata["post_norm_peaks"] = post_norm_peaks
        audio_metadata["post_norm_lufs"] = target_lufs

        self.logger.debug(f"\"{audio_path}\" lufs: {old_lufs} -> {target_lufs}")

        return {
            "audio_path": audio_path,
            "audio": normalized_audio,
            "audio_metadata": audio_metadata,
            "sample_rate": sample_rate,
        }

    def get_max_output_queue_size(self):
        return 5
    
    def get_stage_type(self):
        return "cpu"

class NormalizeSave(DatasetProcessStage):

    @torch.inference_mode()
    def process(self, input_dict: dict) -> Optional[Union[dict, list[dict]]]:

        if self.processor_config.test_mode == False:
            with self.critical_lock:
                save_audio(
                    raw_samples=input_dict["audio"],
                    sample_rate=input_dict["sample_rate"],
                    output_path=input_dict["audio_path"],
                    target_lufs=None,
                    metadata=input_dict["audio_metadata"],
                )

        return None
    
    def get_stage_type(self):
        return "cpu"
    
if __name__ == "__main__":

    dataset_processor = DatasetProcessor()
    dataset_processor.process(
        "Normalize",
        [
            NormalizeLoad(),
            NormalizeProcess(),
            NormalizeSave()
        ],
        input=config.DATASET_PATH,
    )

    os._exit(0) # for whatever reason exiting is unreliable on ctrl+c without this :(