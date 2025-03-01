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

from typing import Optional, Union, Literal
from dataclasses import dataclass
import logging
import os

import torch
import torchaudio
import pyloudnorm

from dataset.dataset_processor import DatasetProcessor, DatasetProcessStage
from utils.dual_diffusion_utils import (
    load_audio, save_audio, get_num_clipped_samples,
    get_audio_info, update_audio_metadata, get_audio_metadata
)


@dataclass
class NormalizeProcessConfig:
    target_lufs: float               = -20.  # desired loudness level for dataset audio in the normalization process
    trim_silence: bool               = True  # removes any silence at the beginning or end of the audio file
    trim_max_length: Optional[float] = 180   # if set, truncates the length of the audio to this max length (in seconds)
    sample_rate: Optional[int]       = None  # if set, resamples audio to this sample rate during normalization (if needed)
    remove_dc_offset: bool           = True  # zeros the mean / "zero frequency" of each audio channel if enabled
    clipping_eps: float              = 2e-2  # controls sensitivity for clipping detection
    silence_eps: float               = 2e-4  # controls sensitivity for leading / trailing silence trimming
    frequency_eps: float             = 3e-5  # controls sensitivity for max frequency detection
    max_peaks_per_second: float      = 10    # if normalizing to target lufs would cause clipping, back off until this level of clipping is reached

class NormalizeLoad(DatasetProcessStage):

    def __init__(self, process_config: NormalizeProcessConfig) -> None:
        self.process_config = process_config

    def get_stage_type(self) -> Literal["io", "cpu", "cuda"]:
        return "cpu"
    
    def get_proc_weight(self) -> float:
        return 0.5
    
    def info_banner(self, logger: logging.Logger) -> None:
        logger.info(f"Normalizing to target_lufs: {self.process_config.target_lufs}")
        logger.info(f"Max peaks per second after normalizing gain: {self.process_config.max_peaks_per_second}")

        if self.process_config.sample_rate is not None:
            logger.info(f"Target sample rate: {self.process_config.sample_rate}hz")
        if self.process_config.trim_max_length:
            logger.info(f"Trim max length: {self.process_config.trim_max_length}s")
        if self.process_config.remove_dc_offset == True:
            logger.info(f"Remove per-channel DC offset enabled")
        if self.process_config.trim_silence == True:
            logger.info("Trim leading / trailing silence enabled")
            logger.info(f"Silence detection eps: {self.process_config.silence_eps}")

        logger.info(f"Clipping detection eps: {self.process_config.clipping_eps}")
        logger.info(f"Max frequency detection eps: {self.process_config.frequency_eps}")

    def limit_output_queue_size(self) -> bool:
        return True
    
    @torch.inference_mode()
    def process(self, input_dict: dict) -> Optional[Union[dict, list[dict]]]:

        if os.path.splitext(input_dict["file_path"])[1].lower() == ".flac":
            
            target_lufs = self.process_config.target_lufs
            max_length = self.process_config.trim_max_length
            target_sample_rate = self.process_config.sample_rate
            do_normalize = False

            audio_path = input_dict["file_path"]
            audio_metadata = get_audio_metadata(audio_path)

            # check if normalization _would_ put this sample below minimum length
            if "below_min_length" in audio_metadata: # if so, skip it
                post_norm_length = float(audio_metadata["below_min_length"][0]) 
                if self.processor_config.min_audio_length is not None and post_norm_length < self.processor_config.min_audio_length:
                    return None
            
            current_lufs = audio_metadata.get("post_norm_lufs", None)
            if current_lufs is not None: current_lufs = float(current_lufs[0])
            if current_lufs is None:# or (current_lufs - target_lufs)**2 >= 1:
                do_normalize = True

            audio_info = get_audio_info(audio_path)
            current_length = audio_info.duration
            if (self.processor_config.min_audio_length is not None and
                current_length < self.processor_config.min_audio_length):
                return None # too short, skip
            if max_length is not None and current_length > max_length:
                do_normalize = True

            current_sample_rate = audio_info.sample_rate
            if target_sample_rate is not None and current_sample_rate != target_sample_rate:
                do_normalize = True
                
            if do_normalize == True or self.processor_config.force_overwrite == True:
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

    def __init__(self, process_config: NormalizeProcessConfig) -> None:
        self.process_config = process_config

    def get_stage_type(self) -> Literal["io", "cpu", "cuda"]:
        return "cpu"
    
    def get_proc_weight(self) -> float:
        return 2
    
    def limit_output_queue_size(self) -> bool:
        return True
    
    @torch.inference_mode()
    def start_process(self):
        self.loudness_meters: dict[int, pyloudnorm.Meter] = {}

    @torch.inference_mode()
    def process(self, input_dict: dict) -> Optional[Union[dict, list[dict]]]:
        
        audio: torch.Tensor = input_dict["audio"]
        old_length = audio.shape[-1]
        audio_path = input_dict["audio_path"]
        target_lufs = input_dict["target_lufs"]
        sample_rate = input_dict["sample_rate"]
        audio_metadata = input_dict["audio_metadata"]
        
        # first resample if needed
        old_sample_rate = sample_rate
        target_sample_rate = self.process_config.sample_rate
        if target_sample_rate is not None and sample_rate != target_sample_rate:
            audio = torchaudio.functional.resample(audio, sample_rate, target_sample_rate)
            sample_rate = target_sample_rate

        # trim the max length, if set
        if self.process_config.trim_max_length is not None:
            max_samples = self.process_config.trim_max_length * sample_rate
            if max_samples > 0 and audio.shape[-1] > max_samples:
                audio = audio[..., :max_samples]

        # trim leading / trailing silence before dc offset removal
        if self.process_config.trim_silence == True:
            mask = audio.abs().mean(dim=0) > self.process_config.silence_eps
            indices = torch.nonzero(mask)
            if indices.numel() == 0: indices = (0,)
            audio = audio[:, indices[0]:indices[-1]]

        # remove dc offset
        dc_offset = audio.mean().item()
        if "dc_offset" not in audio_metadata: # keep result from first normalize
            audio_metadata["dc_offset"] = f"{dc_offset:.2f}"

        if self.process_config.remove_dc_offset == True:
            # ensure we aren't going to make clipping even worse when removing dc offset
            if dc_offset > 0: dc_offset = min(dc_offset, audio.amin().item() + 1)
            if dc_offset < 0: dc_offset = max(dc_offset, audio.amax().item() - 1)
            audio -= dc_offset

        # then trim leading / trailing silence again _after_ dc offset removal
        if self.process_config.trim_silence == True:
            mask = audio.abs().mean(dim=0) > self.process_config.silence_eps
            indices = torch.nonzero(mask)
            if indices.numel() == 0: indices = (0,)
            audio = audio[:, indices[0]:indices[-1]]

        # at this point if the sample has fallen below the minimum length, mark it as such then skip it
        current_length = audio.shape[-1] / sample_rate
        if self.processor_config.min_audio_length is not None and current_length < self.processor_config.min_audio_length:
            update_audio_metadata(audio_path, {"below_min_length": current_length}, copy_on_write=True)
            return None
        else:
            if "below_min_length" in audio_metadata:   # if our min length has changed and this sample is now valid
                del audio_metadata["below_min_length"] # remove the below min length metadata

        # count peaks and normalize loudness
        if audio.shape[-1] >= 12800: # this is required for loudness calculation
            pre_norm_peaks = get_num_clipped_samples(audio / audio.abs().amax().clip(min=1e-8),
                                                eps=self.process_config.clipping_eps)
            if sample_rate not in self.loudness_meters:
                self.loudness_meters[sample_rate] = pyloudnorm.Meter(sample_rate)
            
            old_lufs = self.loudness_meters[sample_rate].integrated_loudness(audio.T.numpy())
            normalized_audio = (audio * (10. ** ((target_lufs - old_lufs) / 20.0))).clip(min=-1, max=1)

            while True: # limit peaking to acceptable levels, but unfortunately this means we can't reach our target gain
                post_norm_peaks = get_num_clipped_samples(normalized_audio)
                if post_norm_peaks <= self.process_config.max_peaks_per_second: break
                target_lufs -= 0.5
                normalized_audio = (audio * (10. ** ((target_lufs - old_lufs) / 20.0))).clip(min=-1, max=1)

            if "pre_norm_peaks" not in audio_metadata: # keep result from first normalize
                audio_metadata["pre_norm_peaks"] = f"{(pre_norm_peaks / audio.shape[-1] * sample_rate):.2f}" # avg peaks per second
            audio_metadata["post_norm_peaks"] = f"{(post_norm_peaks / audio.shape[-1] * sample_rate):.2f}"
            audio_metadata["post_norm_lufs"] = target_lufs

            # add metadata for the "effective sample rate" (frequencies below which contain 99% of the signal energy)
            rfft = torch.fft.rfft(normalized_audio, dim=-1, norm="ortho").abs().mean(dim=0)
            indices = torch.nonzero((rfft / rfft.amax().clip(min=1e-10)) > self.process_config.frequency_eps)
            if indices.numel() == 0: indices = (0,)
            effective_sample_rate = int((indices[-1] / rfft.shape[-1]).item() * sample_rate)
            effective_sample_rate = (effective_sample_rate + 50) // 100 * 100 # round to nearest 100hz
            audio_metadata["effective_sample_rate"] = effective_sample_rate
        else:
            normalized_audio = audio
            old_lufs = None

        return {
            "audio_path": audio_path,
            "audio": normalized_audio,
            "audio_metadata": audio_metadata,
            "old_sample_rate": old_sample_rate,
            "new_sample_rate": sample_rate,
            "old_length": old_length,
            "new_length": audio.shape[-1],
            "old_lufs": old_lufs,
            "target_lufs": target_lufs,
        }

class NormalizeSave(DatasetProcessStage):

    def __init__(self, process_config: NormalizeProcessConfig) -> None:
        self.process_config = process_config

    def get_stage_type(self) -> Literal["io", "cpu", "cuda"]:
        return "cpu"
    
    def get_proc_weight(self) -> float:
        return 0.5
    
    def summary_banner(self, logger: logging.Logger, completed: bool) -> None:
        verb = "Normalized"
        if self.processor_config.test_mode == True:
            logger.info(f"(Would have) {verb} {self.output_queue.queue.qsize()} files.")
        else:
            logger.info(f"{verb} {self.output_queue.queue.qsize()} files.")

    @torch.inference_mode()
    def process(self, input_dict: dict) -> Optional[Union[dict, list[dict]]]:

        audio = input_dict["audio"]
        audio_path = input_dict["audio_path"]
        old_sample_rate = input_dict["old_sample_rate"]
        new_sample_rate = input_dict["new_sample_rate"]
        old_length = input_dict["old_length"]
        new_length = input_dict["new_length"]
        old_lufs = input_dict["old_lufs"]
        new_lufs = input_dict["target_lufs"]
        
        info_str = ""
        if old_lufs is not None:
            info_str += f" {old_lufs}lufs -> {new_lufs}lufs"
        if old_sample_rate != new_sample_rate:
            info_str += f" {old_sample_rate}hz -> {new_sample_rate}hz"
        if old_length != new_length:
            info_str += f" {int(old_length/old_sample_rate)}s -> {int(new_length/new_sample_rate)}s"

        self.logger.debug(f"\"{audio_path}\" ({info_str})")
        
        if self.processor_config.test_mode == False:
            save_audio(
                raw_samples=audio,
                sample_rate=new_sample_rate,
                output_path=audio_path,
                target_lufs=None,
                metadata=input_dict["audio_metadata"],
                copy_on_write=True
            )

        return {}


if __name__ == "__main__":

    process_config: NormalizeProcessConfig = config.load_config(NormalizeProcessConfig,
                        os.path.join(config.CONFIG_PATH, "dataset", "normalize.json"))
    
    dataset_processor = DatasetProcessor()
    dataset_processor.process(
        "Normalize",
        [
            NormalizeLoad(process_config),
            NormalizeProcess(process_config),
            NormalizeSave(process_config)
        ],
        input=config.DATASET_PATH,
    )

    exit(0)