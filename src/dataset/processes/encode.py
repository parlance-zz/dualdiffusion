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
from copy import deepcopy
import logging
import os

import torch
import torchaudio
import safetensors.torch as safetensors

from pipelines.dual_diffusion_pipeline import DualDiffusionPipeline
from modules.formats.spectrogram import SpectrogramFormat, SpectrogramFormatConfig
from modules.embeddings.embedding import DualDiffusionEmbedding
from modules.vaes.vae import DualDiffusionVAE
from dataset.dataset_processor import DatasetProcessor, DatasetProcessStage
from utils.dual_diffusion_utils import (
    get_audio_metadata, load_audio, move_tensors_to_cpu,
    save_safetensors, load_safetensors_ex, init_cuda, normalize
)


class EncodeLoad(DatasetProcessStage):

    def get_stage_type(self) -> Literal["io", "cpu", "cuda"]:
        return "cpu"
    
    def info_banner(self, logger: logging.Logger) -> None:

        if self.processor_config.encode_model is None:
            logger.error("encode_model undefined in DatasetProcessorConfig")
            raise ValueError()
        
        encode_model_path = os.path.join(config.MODELS_PATH, self.processor_config.encode_model)
        if os.path.isdir(encode_model_path) == False:
            logger.error(f"encode_model \"{encode_model_path}\" not found")
            raise ValueError()
        
        if self.processor_config.encode_latents_force_overwrite == True:
            logger.warning("WARNING: Force latents overwrite is enabled - existing data will be overwritten")
        if self.processor_config.encode_audio_embeddings_force_overwrite == True:
            logger.warning("WARNING: Force audio embeddings overwrite is enabled - existing data will be overwritten")
        if self.processor_config.encode_text_embeddings_force_overwrite == True:
            logger.warning("WARNING: Force text embeddings overwrite is enabled - existing data will be overwritten")
        
        logger.info(f"Encode model: {self.processor_config.encode_model}  compile: {self.processor_config.encode_compile_models}")
        logger.info(f"Latents number of time offset augmentations: {self.processor_config.encode_latents_num_time_offset_augmentations}")
        logger.info(f"Latents number of pitch offset augmentations: {len(self.processor_config.encode_latents_pitch_offset_augmentations)}")
        logger.info(f"Latents stereo mirroring augmentation: {self.processor_config.encode_latents_stereo_mirroring_augmentation}")

    def limit_output_queue_size(self) -> bool:
        return True
    
    @torch.inference_mode()
    def process(self, input_dict: dict) -> Optional[Union[dict, list[dict]]]:
        
        file_path = input_dict["file_path"]
        file_ext = os.path.splitext(file_path)[1].lower()

        if file_ext == ".flac":
            
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
            
            if self.processor_config.encode_latents_force_overwrite == True: has_latents = False
            if self.processor_config.encode_audio_embeddings_force_overwrite == True: has_audio_embeddings = False
            if self.processor_config.encode_text_embeddings_force_overwrite == True: has_text_embeddings = False

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

        init_cuda()

        # load pipeline and compile vae / embedding models
        model_path = os.path.join(config.MODELS_PATH, self.processor_config.encode_model)
        self.pipeline = DualDiffusionPipeline.from_pretrained(model_path,
            device={"vae": self.device, "format": self.device, "embedding": self.device})
        
        self.vae: DualDiffusionVAE = self.pipeline.vae
        self.embedding: DualDiffusionEmbedding = self.pipeline.embedding

        self.vae = self.vae.to(dtype=torch.bfloat16)
        if self.processor_config.encode_compile_models == True:
            self.vae.compile(fullgraph=True, dynamic=True)
            self.embedding.compile(fullgraph=True, dynamic=True)
        
        # encode latents setup
        self.format: SpectrogramFormat = self.pipeline.format
        self.format_config: SpectrogramFormatConfig = self.format.config
        
        num_encode_offsets = self.processor_config.encode_latents_num_time_offset_augmentations
        self.vae_encode_offset_padding = self.format_config.hop_length * num_encode_offsets
        self.vae_encode_offsets = [i * self.format_config.hop_length for i in range(num_encode_offsets)]
        self.vae_batch_size = self.processor_config.encode_latents_batch_size
        self.vae_num_batches_per_sample = (num_encode_offsets + self.vae_batch_size - 1) // self.vae_batch_size
        
        pitch_shifts = self.processor_config.encode_latents_pitch_offset_augmentations
        pitch_augmentation_formats = [
            self.get_pitch_augmentation_format(shift).to(self.device) for shift in pitch_shifts]
        self.vae_encode_formats: list[SpectrogramFormat] = [self.format] + pitch_augmentation_formats
    
    @torch.inference_mode()
    def process(self, input_dict: dict) -> Optional[Union[dict, list[dict]]]:
        
        safetensors_file_path: str = input_dict["safetensors_file_path"]
        audio: Optional[torch.Tensor] = input_dict["audio"]
        sample_rate: Optional[int] = input_dict["sample_rate"]
        prompt: Optional[str] = input_dict["prompt"]
        latents: dict[str, torch.Tensor] = input_dict["latents"] or {}
        latents_metadata: dict[str, str] = input_dict["latents_metadata"] or {}

        # move audio and latents to device
        if audio is not None: audio = audio.to(self.device)
        latents = {name: tensor.to(self.device) for name, tensor in latents.items()}
            
        # encode audio embeddings if needed
        if audio is not None and input_dict["has_audio_embeddings"] == False:
            latents["clap_audio_embeddings"] = self.embedding.encode_audio(audio, sample_rate=sample_rate).to(dtype=torch.bfloat16)

        # encode text embeddings if a prompt is available and they are not yet encoded
        if prompt is not None and input_dict["has_text_embeddings"] == False:
            latents["clap_text_embeddings"] = self.embedding.encode_text([prompt]).to(dtype=torch.bfloat16)
            latents_metadata["prompt"] = prompt

        # encode latents
        if audio is not None and input_dict["has_latents"] == False:
            
            # resample audio to model format sample_rate
            crop_width = self.format.sample_raw_crop_width(audio.shape[-1] - self.vae_encode_offset_padding)
            if sample_rate != self.format_config.sample_rate:
                audio = torchaudio.functional.resample(
                    audio, sample_rate, self.format_config.sample_rate)
            
            # create raw audio augmentations
            input_audio = []
            for offset in self.vae_encode_offsets:
                input_raw_offset_sample = audio[:, offset:offset + crop_width].unsqueeze(0)
                input_audio.append(input_raw_offset_sample)
                if self.processor_config.encode_latents_stereo_mirroring_augmentation == True:
                    input_audio.append(torch.flip(input_raw_offset_sample, dims=(1,)))
            audio = torch.cat(input_audio, dim=0)

            # encode spectrograms of all audios
            input_samples = []; bsz = self.processor_config.encode_latents_batch_size
            for format in self.vae_encode_formats:
                for b in range(self.vae_num_batches_per_sample):
                    batch_input_raw_sample = audio[b*bsz:(b+1)*bsz]
                    input_sample = format.raw_to_sample(batch_input_raw_sample).type(torch.bfloat16)
                    input_samples.append(input_sample)
            input_sample = torch.cat(input_samples, dim=0)
            
            # finally, encode the latents
            vae_class_embeddings = latents["clap_audio_embeddings"][:, :self.vae.emb_dim].mean(dim=0, keepdim=True)
            vae_class_embeddings = normalize(vae_class_embeddings).to(device=self.vae.device, dtype=self.vae.dtype)
            encoded_latents: list[torch.Tensor] = []
            for b in range(input_sample.shape[0] // bsz):
                batch_input_sample = input_sample[b*bsz:(b+1)*bsz]
                batch_latents = self.vae.encode(batch_input_sample, vae_class_embeddings, self.format).mode()
                encoded_latents.append(batch_latents)
            latents["latents"] = torch.cat(encoded_latents, dim=0).to(dtype=torch.bfloat16)

        return {
            "safetensors_file_path": safetensors_file_path,
            "latents": move_tensors_to_cpu(latents),
            "latents_metadata": latents_metadata,
        }

class EncodeSave(DatasetProcessStage):

    def get_stage_type(self) -> Literal["io", "cpu", "cuda"]:
        return "io"
    
    def summary_banner(self, logger: logging.Logger) -> None:
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

        with self.critical_lock:
            self.logger.debug(f"saving \"{safetensors_file_path}\"")
            
            if self.processor_config.test_mode == False:
                save_safetensors(
                    tensors_dict=latents,
                    output_path=safetensors_file_path,
                    metadata=latents_metadata,
                    copy_on_write=self.processor_config.copy_on_write
                )

        return {}


if __name__ == "__main__":

    dataset_processor = DatasetProcessor()
    dataset_processor.process(
        "Encode",
        [
            EncodeLoad(),
            EncodeProcess(),
            EncodeSave()
        ],
        input=config.DATASET_PATH,
    )

    os._exit(0)