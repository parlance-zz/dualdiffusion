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
from dataset.dataset_processor import DatasetProcessor, DatasetProcessStage
from utils.dual_diffusion_utils import (
    get_audio_metadata, load_audio, move_tensors_to_cpu,
    save_safetensors, load_safetensors_ex
)


def get_pitch_augmentation_format(original_config: SpectrogramFormatConfig,
                                  shift_semitones: float) -> SpectrogramFormat:
    shift_rate = 2 ** (shift_semitones / 12)
    augmented_config = deepcopy(original_config)
    augmented_config.min_frequency *= shift_rate
    augmented_config.max_frequency *= shift_rate
    return SpectrogramFormat(augmented_config)

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
                if "text_embedding_prompt" in latents_metadata:
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
        return "cpu"
    
    def limit_output_queue_size(self) -> bool:
        return True
    
    @torch.inference_mode()
    def start_process(self):

        # load pipeline and compile vae / embedding models
        model_path = os.path.join(config.MODELS_PATH, self.processor_config.encode_model)
        self.pipeline = DualDiffusionPipeline.from_pretrained(model_path,
            device={"vae": self.device, "format": self.device, "embedding": self.device})
        
        self.pipeline.vae = self.pipeline.vae.to(torch.bfloat16)
        if self.processor_config.encode_compile_models == True:
            self.pipeline.vae.compile(fullgraph=True, dynamic=True)
            self.pipeline.embedding.compile(fullgraph=True, dynamic=True)
        
        # encode latents setup
        format_config: SpectrogramFormatConfig = self.pipeline.format.config
        spectrogram_hop_length = format_config.hop_length
        num_encode_offsets = self.processor_config.encode_latents_num_time_offset_augmentations
        self.encode_offset_padding = spectrogram_hop_length * num_encode_offsets
        encode_offsets = [i * spectrogram_hop_length for i in range(num_encode_offsets)]
        batch_size = self.processor_config.encode_latents_batch_size
        num_batches_per_sample = (num_encode_offsets + batch_size - 1) // batch_size
        
        pitch_shifts = self.processor_config.encode_latents_pitch_offset_augmentations
        pitch_augmentation_formats = [
            get_pitch_augmentation_format(format_config, shift).to(self.device) for shift in pitch_shifts]
        formats: list[SpectrogramFormat] = [self.pipeline.format] + pitch_augmentation_formats
    
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
        latents = {tensor.to(self.device) for tensor in latents}
        
        # encode audio embeddings if needed
        if audio is not None and input_dict["has_audio_embeddings"] == False:
            
            # resample to embedding sample rate / channels
            if sample_rate != 48000:
                emb_audio = torchaudio.functional.resample(audio, sample_rate, 48000).mean(dim=0)
            else:
                emb_audio = audio.mean(dim=0)

            # chunkify embedding audio
            chunk_size = 48000 * 10
            emb_audio = emb_audio[:emb_audio.shape[0] // chunk_size * chunk_size].reshape(-1, chunk_size)

            # get embeddings in chunks using max_batch_size
            audio_embeddings = torch.cat([
                normalize(clap_model.get_audio_embedding_from_data(chunk, use_tensor=True)).float() for chunk in audio.split(max_batch_size)], dim=0)

        # encode text embeddings if a prompt is available and they are not yet encoded
        if prompt is not None and input_dict["has_text_embeddings"] == False:
            text_embeddings = normalize(clap_model.get_text_embedding([sample_prompt], use_tensor=True)).float()

        # encode latents
        if audio is not None and input_dict["has_latents"] == False:

            crop_width = self.pipeline.format.sample_raw_crop_width(audio.shape[-1] - self.encode_offset_padding)

            if sample_rate != self.processor_config.sample_rate:
                audio = torchaudio.functional.resample(
                    audio, sample_rate, self.processor_config.sample_rate)
                
            """
            input_raw_samples = []
            for offset in encode_offsets:
                input_raw_offset_sample = input_raw_sample[:, offset:offset+crop_width].unsqueeze(0).to(device)
                input_raw_samples.append(input_raw_offset_sample)
                if stereo_mirroring:
                    input_raw_samples.append(torch.flip(input_raw_sample, dims=1))
            input_raw_sample = torch.cat(input_raw_samples, dim=0)

            input_samples = []
            for format in formats:
                for b in range(num_batches_per_sample):
                    batch_input_raw_sample = input_raw_sample[b*batch_size:(b+1)*batch_size]
                    input_sample = format.raw_to_sample(batch_input_raw_sample).type(torch.bfloat16)
                    input_samples.append(input_sample)
            input_sample = torch.cat(input_samples, dim=0)
            
            vae_class_embeddings = pipeline.vae.get_class_embeddings(pipeline.get_class_labels(game_id, module_name="vae"))            
            latents = []
            for b in range(input_sample.shape[0] // batch_size):
                batch_input_sample = input_sample[b*batch_size:(b+1)*batch_size]
                batch_latents = pipeline.vae.encode(batch_input_sample, vae_class_embeddings, pipeline.format).mode()
                latents.append(batch_latents)
            latents = torch.cat(latents, dim=0).type(torch.bfloat16)

            if quantize_latents:
                latents_quantized, offset_and_range = quantize_tensor(latents, 256)
                latents_dict = {"latents": latents_quantized.type(torch.uint8), "offset_and_range": offset_and_range}
            else:
                latents_dict = {"latents": latents}
            existing_latents.update(latents_dict)
            save_safetensors(existing_latents, output_path)
            del existing_latents
            """

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