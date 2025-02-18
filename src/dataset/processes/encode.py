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
from copy import deepcopy
import logging
import os

import torch
import torchaudio
import safetensors.torch as safetensors

from pipelines.dual_diffusion_pipeline import DualDiffusionPipeline
from modules.formats.spectrogram import SpectrogramFormat, SpectrogramFormatConfig
from modules.embeddings.embedding import DualDiffusionEmbedding
from modules.daes.dae import DualDiffusionDAE
from dataset.dataset_processor import DatasetProcessor, DatasetProcessStage
from utils.dual_diffusion_utils import (
    get_audio_metadata, load_audio, get_audio_info, dict_str,
    save_safetensors, load_safetensors_ex, normalize
)

@dataclass
class EncodeProcessConfig:
    model: Optional[str]                          = None  # use the format, dae, and embeddings from this model (under $MODELS_PATH)
    dae_ema: Union[str, bool]                     = True  # use the specified ema if str, the first ema if true, and no ema if false for latents encoding
    compile_models: bool                          = True  # compile the dae before encoding
    latents_batch_size: int                       = 1     # batch size for encoding latents. choose a value that works with your vram capacity
    latents_num_time_offset_augmentations: int    = 8     # add augmentations for sub-pixel (latent pixel) offsets
    latents_pitch_offset_augmentations: list[int] = ()    # add augmentations for list of pitch offsets (in semitones)
    latents_stereo_mirroring_augmentation: bool   = True  # add augmentation with swapped stereo channels
    latents_force_overwrite: bool                 = False # (re)encode and overwrite latents
    audio_embeddings_force_overwrite: bool        = False # (re)encode and overwrite existing audio embeddings
    text_embeddings_force_overwrite: bool         = False # (re)encode and overwrite existing text embeddings
    embeddings_only: bool                         = False # only encodes audio/text embeddings and skips latents

class EncodeLoad(DatasetProcessStage):

    def __init__(self, process_config: EncodeProcessConfig) -> None:
        self.process_config = process_config

    def get_stage_type(self) -> Literal["io", "cpu", "cuda"]:
        return "io"
    
    def info_banner(self, logger: logging.Logger) -> None:

        if self.process_config.model is None:
            logger.error("encode_model undefined in DatasetProcessorConfig")
            raise ValueError()
        
        encode_model_path = os.path.join(config.MODELS_PATH, self.process_config.model)
        if os.path.isdir(encode_model_path) == False:
            logger.error(f"encode_model \"{encode_model_path}\" not found")
            raise ValueError()
        
        if self.process_config.latents_force_overwrite == True and self.process_config.embeddings_only == False:
            logger.warning("WARNING: Force latents overwrite is enabled - existing data will be overwritten")
        if self.process_config.audio_embeddings_force_overwrite == True:
            logger.warning("WARNING: Force audio embeddings overwrite is enabled - existing data will be overwritten")
        if self.process_config.text_embeddings_force_overwrite == True:
            logger.warning("WARNING: Force text embeddings overwrite is enabled - existing data will be overwritten")
        
        logger.info(f"Encode model: {self.process_config.model}  compile: {self.process_config.compile_models}")
        if self.process_config.embeddings_only == True:
            logger.info(f"Skipping latents, encoding audio / text embeddings only")

        if self.process_config.embeddings_only == False:
            logger.info(f"Latents number of time offset augmentations: {self.process_config.latents_num_time_offset_augmentations}")
            logger.info(f"Latents number of pitch offset augmentations: {len(self.process_config.latents_pitch_offset_augmentations)}")
            logger.info(f"Latents stereo mirroring augmentation: {self.process_config.latents_stereo_mirroring_augmentation}")

    def limit_output_queue_size(self) -> bool:
        return True
    
    @torch.inference_mode()
    def process(self, input_dict: dict) -> Optional[Union[dict, list[dict]]]:
        
        file_path = input_dict["file_path"]
        file_ext = os.path.splitext(file_path)[1].lower()

        if file_ext == ".flac":
            
            # if sample is too short skip it
            audio_info = get_audio_info(input_dict["file_path"])
            if (self.processor_config.min_audio_length is not None and
                audio_info.duration < self.processor_config.min_audio_length): 
                return None
            
            # check if we've already encoded anything
            safetensors_file_path = f"{os.path.splitext(file_path)[0]}.safetensors"
            if os.path.isfile(safetensors_file_path):
                with safetensors.safe_open(safetensors_file_path, framework="pt") as f:
                    try:
                        f.get_slice("latents").get_shape()
                        has_latents = True
                    except:
                        has_latents = False

                    try:
                        f.get_slice("clap_audio_embeddings")
                        has_audio_embeddings = True
                    except:
                        has_audio_embeddings = False

                    try:
                        f.get_slice("clap_text_embeddings")
                        has_text_embeddings = True
                    except:
                        has_text_embeddings = False
            else:
                has_latents = False
                has_audio_embeddings = False
                has_text_embeddings = False
            
            if self.process_config.latents_force_overwrite == True: has_latents = False
            if self.process_config.audio_embeddings_force_overwrite == True: has_audio_embeddings = False
            if self.process_config.text_embeddings_force_overwrite == True: has_text_embeddings = False

            # if we need to encode latents or audio embeddings, load the audio file content
            if has_latents == False or has_audio_embeddings == False:
                audio, sample_rate = load_audio(file_path, return_sample_rate=True)
            else:
                audio = None
                sample_rate = None

            # if no existing text embeddings, try to get prompt metadata
            if has_text_embeddings == False:
                audio_metadata = get_audio_metadata(file_path)
                if "prompt" in audio_metadata:
                    prompt = audio_metadata["prompt"][0]
                else: # no prompt, cannot encode text embedding
                    prompt = None
            else:
                prompt = None

            # if nothing needs to / can be encoded, skip this file
            if audio is None and prompt is None:
                return None
            
            # if anything is already encoded we need to read the whole safetensors file to rewrite it
            if has_latents == True or has_text_embeddings == True or has_audio_embeddings == True:
                latents, latents_metadata = load_safetensors_ex(safetensors_file_path)

                # if the prompt has changed (re)encode the new text embeddings
                if "prompt" in latents_metadata and prompt is not None:
                    if prompt != latents_metadata["prompt"]:
                        has_text_embeddings = False
            else:
                latents = None
                latents_metadata = None

            return {
                "file_path": file_path,
                "safetensors_file_path": safetensors_file_path,
                "audio": audio,
                "sample_rate": sample_rate,
                "prompt": prompt,
                "latents": latents,
                "latents_metadata": latents_metadata,
                "has_latents": has_latents,
                "has_audio_embeddings": has_audio_embeddings,
                "has_text_embeddings": has_text_embeddings
            }
        
        return None

class EncodeProcess(DatasetProcessStage):

    def __init__(self, process_config: EncodeProcessConfig) -> None:
        self.process_config = process_config

    def get_stage_type(self) -> Literal["io", "cpu", "cuda"]:
        return "cuda"
    
    def limit_output_queue_size(self) -> bool:
        return True
    
    def get_pitch_augmentation_format(self, shift_semitones: float) -> SpectrogramFormat:
        shift_rate = 2 ** (shift_semitones / 12)
        augmented_config = deepcopy(self.format_config)
        augmented_config.min_frequency *= shift_rate
        augmented_config.max_frequency *= shift_rate
        return SpectrogramFormat(augmented_config)

    @torch.inference_mode()
    def start_process(self):

        # load pipeline and compile dae / embedding models
        model_path = os.path.join(config.MODELS_PATH, self.process_config.model)
        self.pipeline = DualDiffusionPipeline.from_pretrained(
            model_path, load_checkpoints=True,
            device={"dae": self.device, "format": self.device, "embedding": self.device},
            )#load_emas={"dae": self.process_config.dae_ema})
        
        self.logger.info(f"Model metadata:\n{dict_str(self.pipeline.model_metadata)}")
        self.format: SpectrogramFormat = self.pipeline.format
        self.embedding: DualDiffusionEmbedding = self.pipeline.embedding
        self.dae: DualDiffusionDAE = self.pipeline.dae

        if self.dae.config.last_global_step == 0 and self.process_config.embeddings_only == False:
            self.logger.error(f"Error: DAE has not been trained, unable to encode latents. Aborting...")
            exit(1)

        self.dae = self.dae.to(dtype=torch.bfloat16)
        if self.process_config.compile_models == True:
            self.dae.compile(fullgraph=True, dynamic=True)
            self.embedding.compile(fullgraph=False, dynamic=True)
        
        # encode latents setup
        
        self.format_config: SpectrogramFormatConfig = self.format.config
        
        num_encode_offsets = self.process_config.latents_num_time_offset_augmentations
        self.dae_encode_offset_padding = self.format_config.hop_length * num_encode_offsets
        self.dae_encode_offsets = [i * self.format_config.hop_length for i in range(num_encode_offsets)]
        self.dae_batch_size = self.process_config.latents_batch_size
        self.dae_num_batches_per_sample = (num_encode_offsets + self.dae_batch_size - 1) // self.dae_batch_size
        
        pitch_shifts = self.process_config.latents_pitch_offset_augmentations
        pitch_augmentation_formats = [
            self.get_pitch_augmentation_format(shift).to(self.device) for shift in pitch_shifts]
        self.dae_encode_formats: list[SpectrogramFormat] = [self.format] + pitch_augmentation_formats
    
    @torch.inference_mode()
    def process(self, input_dict: dict) -> Optional[Union[dict, list[dict]]]:
        
        safetensors_file_path: str = input_dict["safetensors_file_path"]
        audio: Optional[torch.Tensor] = input_dict["audio"]
        sample_rate: Optional[int] = input_dict["sample_rate"]
        prompt: Optional[str] = input_dict["prompt"]
        latents: dict[str, torch.Tensor] = input_dict["latents"] or {}
        latents_metadata: dict[str, str] = input_dict["latents_metadata"] or {}

        # if audio is shorter than 1 audio embedding chunk we can't use it
        if audio.shape[-1] < self.embedding.config.sample_crop_width:
            self.logger.debug(f"Skipping \"{input_dict['file_path']}\" due to insufficient length ({audio.shape[-1]})")
            return None
        
        # move audio and latents to device
        if audio is not None: audio = audio.to(self.device)
        latents = {name: tensor.to(self.device) for name, tensor in latents.items()}
            
        # encode audio embeddings if needed
        if audio is not None and input_dict["has_audio_embeddings"] == False:
            self.logger.debug(f"encoding audio embeddings: \"{safetensors_file_path}\"")
            latents["clap_audio_embeddings"] = self.embedding.encode_audio(audio, sample_rate=sample_rate).to(dtype=torch.bfloat16)
        elif input_dict["has_audio_embeddings"] == True:
            self.logger.debug(f"existing audio embeddings: \"{safetensors_file_path}\"")

        # encode text embeddings if a prompt is available and they are not yet encoded
        if prompt is not None and input_dict["has_text_embeddings"] == False:
            self.logger.debug(f"encoding text embeddings: \"{safetensors_file_path}\"")
            latents["clap_text_embeddings"] = self.embedding.encode_text([prompt]).to(dtype=torch.bfloat16)
            latents_metadata["prompt"] = prompt
        elif input_dict["has_text_embeddings"] == True:
            self.logger.debug(f"existing text embeddings: \"{safetensors_file_path}\"")

        # encode latents
        if audio is not None and input_dict["has_latents"] == False and self.process_config.embeddings_only == False:
            
            # resample audio to model format sample_rate
            crop_width = self.format.sample_raw_crop_width(audio.shape[-1] - self.dae_encode_offset_padding)
            if sample_rate != self.format_config.sample_rate:
                audio = torchaudio.functional.resample(
                    audio, sample_rate, self.format_config.sample_rate)
            
            # create raw audio augmentations
            input_audio = []
            for offset in self.dae_encode_offsets:
                input_raw_offset_sample = audio[:, offset:offset + crop_width].unsqueeze(0)
                input_audio.append(input_raw_offset_sample)
                if self.process_config.latents_stereo_mirroring_augmentation == True:
                    input_audio.append(torch.flip(input_raw_offset_sample, dims=(1,)))
            audio = torch.cat(input_audio, dim=0)

            # encode spectrograms of all audios
            input_samples = []; bsz = self.process_config.latents_batch_size
            for format in self.dae_encode_formats:
                for b in range(self.dae_num_batches_per_sample):
                    batch_input_raw_sample = audio[b*bsz:(b+1)*bsz]
                    input_sample = format.raw_to_sample(batch_input_raw_sample).type(torch.bfloat16)
                    input_samples.append(input_sample)
            input_sample = torch.cat(input_samples, dim=0)
            
            # finally, encode the latents
            self.logger.debug(f"encoding latents: \"{safetensors_file_path}\" ({input_sample.shape[0]} variations)")
            audio_embeddings = latents["clap_audio_embeddings"].mean(dim=0, keepdim=True)
            audio_embeddings = normalize(audio_embeddings).to(device=self.dae.device, dtype=self.dae.dtype)
            dae_embeddings = self.dae.get_embeddings(audio_embeddings)
            encoded_latents: list[torch.Tensor] = []
            for b in range(input_sample.shape[0] // bsz):
                batch_input_sample = input_sample[b*bsz:(b+1)*bsz]
                batch_latents = self.dae.encode(batch_input_sample, dae_embeddings)
                encoded_latents.append(batch_latents)
            latents["latents"] = torch.cat(encoded_latents, dim=0).to(dtype=torch.bfloat16)
            assert latents["latents"].ndim == 4
            
        elif input_dict["has_latents"] == True:
            self.logger.debug(f"existing latents: \"{safetensors_file_path}\" ({latents['latents'].shape[0]} variations)")

        return {
            "safetensors_file_path": safetensors_file_path,
            "latents": {tensor_name: tensor.cpu() for tensor_name, tensor in latents.items()},
            "latents_metadata": latents_metadata,
        }

class EncodeSave(DatasetProcessStage):

    def __init__(self, process_config: EncodeProcessConfig) -> None:
        self.process_config = process_config

    def get_stage_type(self) -> Literal["io", "cpu", "cuda"]:
        return "io"
    
    def summary_banner(self, logger: logging.Logger, completed: bool) -> None:
        verb = "Encoded"
        if self.processor_config.test_mode == True:
            logger.info(f"(Would have) {verb} {self.output_queue.queue.qsize()} files.")
        else:
            logger.info(f"{verb} {self.output_queue.queue.qsize()} files.")

    @torch.inference_mode()
    def process(self, input_dict: dict) -> Optional[Union[dict, list[dict]]]:
        
        safetensors_file_path = input_dict["safetensors_file_path"]
        latents = input_dict["latents"]
        latents_metadata = input_dict["latents_metadata"]

        self.logger.debug(f"saving \"{safetensors_file_path}\"")
        
        if self.processor_config.test_mode == False:
            save_safetensors(
                tensors_dict=latents,
                output_path=safetensors_file_path,
                metadata=latents_metadata,
                copy_on_write=True
            )

        return {}


if __name__ == "__main__":

    process_config: EncodeProcessConfig = config.load_config(EncodeProcessConfig,
                        os.path.join(config.CONFIG_PATH, "dataset", "encode.json"))
    
    dataset_processor = DatasetProcessor()
    dataset_processor.process(
        "Encode",
        [
            EncodeLoad(process_config),
            EncodeProcess(process_config),
            EncodeSave(process_config)
        ],
        input=config.DATASET_PATH,
    )

    exit(0)