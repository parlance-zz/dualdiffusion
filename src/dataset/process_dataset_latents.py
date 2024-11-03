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
from copy import deepcopy

import torch
from tqdm.auto import tqdm
from accelerate import PartialState

from pipelines.dual_diffusion_pipeline import DualDiffusionPipeline
from modules.formats.spectrogram import SpectrogramFormat, SpectrogramFormatConfig
from dataset.dataset_processor import DatasetProcessorConfig
from utils.dual_diffusion_utils import (
    init_cuda, load_audio, save_safetensors, quantize_tensor
)

def get_pitch_augmentation_format(original_config: SpectrogramFormatConfig, shift_semitones: float) -> SpectrogramFormat:
    shift_rate = 2 ** (shift_semitones / 12)
    augmented_config = deepcopy(original_config)
    augmented_config.min_frequency *= shift_rate
    augmented_config.max_frequency *= shift_rate
    return SpectrogramFormat(augmented_config)

@torch.inference_mode()
def pre_encode_latents():

    dataset_processor_config = DatasetProcessorConfig(
        **config.load_json(os.path.join(config.CONFIG_PATH, "dataset", "dataset.json")))

    model_name = dataset_processor_config.pre_encoded_latents_vae
    num_encode_offsets = dataset_processor_config.pre_encoded_latents_num_time_offset_augmentations
    batch_size = dataset_processor_config.pre_encoded_latents_device_batch_size
    quantize_latents = dataset_processor_config.pre_encoded_latents_enable_quantization
    pitch_shifts = dataset_processor_config.pre_encoded_latents_pitch_offset_augmentations
    stereo_mirroring = dataset_processor_config.pre_encoded_latents_stereo_mirroring_augmentation

    distributed_state = PartialState()
    device = distributed_state.device
    
    model_path = os.path.join(config.MODELS_PATH, model_name)
    print(f"Loading DualDiffusion model from '{model_path}'...")
    pipeline = DualDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16,
        load_latest_checkpoints=True, device={"vae": device, "format": device})
    
    format_config: SpectrogramFormatConfig = pipeline.format.config
    spectrogram_hop_length = format_config.hop_length
    encode_offset_padding = spectrogram_hop_length * num_encode_offsets
    encode_offsets = [i * spectrogram_hop_length for i in range(num_encode_offsets)]
    num_batches_per_sample = num_encode_offsets // batch_size
    assert num_encode_offsets % batch_size == 0, "pre_encoded_latents_num_time_offset_augmentations must be divisible by pre_encoded_latents_device_batch_size"

    pitch_augmentation_formats = [get_pitch_augmentation_format(format_config, shift).to(device) for shift in pitch_shifts]
    formats: list[SpectrogramFormat] = [pipeline.format] + pitch_augmentation_formats
    
    # get split metadata
    split_metadata_files = []
    for filename in os.listdir(config.DATASET_PATH):
        if os.path.isfile(os.path.join(config.DATASET_PATH, filename)) and filename.lower().endswith(".jsonl"):
            split_metadata_files.append(filename)

    # process split samples
    for split_metadata_file in split_metadata_files:
        with open(os.path.join(config.DATASET_PATH, split_metadata_file), "r") as f:
            split_metadata = [json.loads(line) for line in f.readlines()]

        # filter split samples that already encoded
        encode_samples = []
        for sample in split_metadata:
            if sample["latents_file_name"] is not None: continue
            sample["latents_file_name"] = f"{os.path.splitext(sample['file_name'])[0]}.safetensors"

            if not os.path.exists(os.path.join(config.DATASET_PATH, sample["latents_file_name"])):
                encode_samples.append(sample)
        
        if distributed_state.is_main_process:
            print(f"Processing {len(split_metadata)} samples from {split_metadata_file} ({len(encode_samples)} samples left to process)...")
            
        with distributed_state.split_between_processes(encode_samples) as samples:

            if distributed_state.is_main_process:
                progress_bar = tqdm(total=len(samples))
                progress_bar.set_description(f"Split: {split_metadata_file}")
            
            for sample in samples:
                game_id = sample["game_id"]
                file_name = sample["file_name"]
                output_filename = sample["latents_file_name"]
                output_path = os.path.join(config.DATASET_PATH, output_filename)

                if os.path.exists(output_path):
                    if distributed_state.is_main_process:
                        progress_bar.update(1)
                    continue

                input_raw_sample = load_audio(os.path.join(config.DATASET_PATH, file_name))
                crop_width = pipeline.format.sample_raw_crop_width(input_raw_sample.shape[-1] - encode_offset_padding)

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
                save_safetensors(latents_dict, output_path)

                if distributed_state.is_main_process:
                    progress_bar.update(1)
                    progress_bar.set_postfix({"file_name": file_name})

        print("Main process finished, waiting for distributed processes...")
        distributed_state.wait_for_everyone()

        if distributed_state.is_main_process:
            progress_bar.close()
        
    print("Pre-encoding latents complete")

if __name__ == "__main__":

    init_cuda()

