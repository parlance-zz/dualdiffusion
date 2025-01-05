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

    def get_stage_type(self):
        return "cpu"
    
    def info_banner(self, logger: logging.Logger) -> None:
        logger.info(f"Normalizing to target_lufs: {self.processor_config.normalize_target_lufs}")
        if self.processor_config.normalize_trim_max_length:
            logger.info(f"Trim max length: {self.processor_config.normalize_trim_max_length}s")
        if self.processor_config.normalize_trim_silence == True:
            logger.info("Trim leading / trailing silence enabled")

    def limit_output_queue_size(self):
        return True
    
    @torch.inference_mode()
    def process(self, input_dict: dict) -> Optional[Union[dict, list[dict]]]:

        if os.path.splitext(input_dict["file_path"])[1].lower() == ".flac":
            
            target_lufs = self.processor_config.normalize_target_lufs

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

class NormalizeProcess(DatasetProcessStage):

    def get_stage_type(self):
        return "cpu"
    
    def limit_output_queue_size(self):
        return True
    
    @torch.inference_mode()
    def process(self, input_dict: dict) -> Optional[Union[dict, list[dict]]]:
        
        audio: torch.Tensor = input_dict["audio"]
        audio_path = input_dict["audio_path"]
        target_lufs = input_dict["target_lufs"]
        sample_rate = input_dict["sample_rate"]
        audio_metadata = input_dict["audio_metadata"]

        # first trim the max length, if set
        if self.processor_config.normalize_trim_max_length is not None:
            max_samples = self.processor_config.normalize_trim_max_length * sample_rate
            if max_samples > 0 and audio.shape[-1] > max_samples:
                audio = audio[..., :max_samples]

        # then trim leading / trailing silence
        if self.processor_config.normalize_trim_silence == True:
            assert audio.ndim == 2
            mask = audio.abs().mean(dim=0) > 6e-5
            if len(mask) == 0:
                indices = (0, 0)
            else:
                indices = torch.nonzero(mask, as_tuple=True)
            audio = audio[:, indices[0]:indices[-1]]

        # count peaks and normalize loudness
        if audio.shape[-1] >= 12800: # this is required for torchaudio.functional.loudness for some reason
            pre_norm_peaks = get_num_clipped_samples(audio)
            normalized_audio, old_lufs = normalize_lufs(raw_samples=audio,
                sample_rate=sample_rate, target_lufs=target_lufs, return_old_lufs=True)
            post_norm_peaks = get_num_clipped_samples(normalized_audio)
            
            if "pre_norm_peaks" not in audio_metadata: # we want the preserve this from the original transcoded file
                audio_metadata["pre_norm_peaks"] = pre_norm_peaks
            audio_metadata["post_norm_peaks"] = post_norm_peaks
            audio_metadata["post_norm_lufs"] = target_lufs

            # add metadata for the "effective sample rate" (frequencies below which contain 99% of the signal energy)
            rfft = torch.cumsum(torch.fft.rfft(normalized_audio, dim=-1, norm="ortho").abs().mean(dim=0)) + 1e-20
            indices = torch.nonzero((rfft / rfft.amax()) > 0.99, as_tuple=True)
            effective_sample_rate = (indices[0] / rfft.shape[-1]).item() * sample_rate
            audio_metadata["effective_sample_rate"] = effective_sample_rate

        return {
            "audio_path": audio_path,
            "audio": normalized_audio,
            "audio_metadata": audio_metadata,
            "sample_rate": sample_rate,
            "old_lufs": old_lufs,
            "target_lufs": target_lufs,
        }

class NormalizeSave(DatasetProcessStage):

    def get_stage_type(self):
        return "cpu"
    
    def summary_banner(self, logger: logging.Logger) -> None:
        verb = "Normalized"
        if self.processor_config.test_mode == True:
            logger.info(f"(Would have) {verb} {self.output_queue.queue.qsize()} files.")
        else:
            logger.info(f"{verb} {self.output_queue.queue.qsize()} files.")

    @torch.inference_mode()
    def process(self, input_dict: dict) -> Optional[Union[dict, list[dict]]]:

        with self.critical_lock:

            audio_path = input_dict["audio_path"]
            old_lufs = input_dict["old_lufs"]
            target_lufs = input_dict["target_lufs"]
            self.logger.debug(f"\"{audio_path}\" lufs: {old_lufs} -> {target_lufs}")
            
            if self.processor_config.test_mode == False:
                save_audio(
                    raw_samples=input_dict["audio"],
                    sample_rate=input_dict["sample_rate"],
                    output_path=input_dict["audio_path"],
                    target_lufs=None,
                    metadata=input_dict["audio_metadata"],
                    copy_on_write=self.processor_config.copy_on_write
                )
                return {}

        return None


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

    os._exit(0)