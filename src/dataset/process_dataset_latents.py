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
import shutil
from copy import deepcopy

import torch
from tqdm.auto import tqdm
from accelerate import PartialState

from pipelines.dual_diffusion_pipeline import DualDiffusionPipeline
from formats.spectrogram import DualSpectrogramFormat
from utils.dual_diffusion_utils import (
    init_cuda, load_audio, save_safetensors, load_safetensors,
    quantize_tensor, dequantize_tensor, save_raw_img, save_audio
)

@torch.no_grad()
def get_pitch_augmentation_format(original_format, shift_semitones):
    shift_rate = 2 ** (shift_semitones / 12)
    aug_model_params = deepcopy(original_format.model_params)
    aug_model_params["spectrogram_params"]["min_frequency"] *= shift_rate
    aug_model_params["spectrogram_params"]["max_frequency"] *= shift_rate
    return DualSpectrogramFormat(aug_model_params)

if __name__ == "__main__":

    init_cuda()

    model_name = "edm2_vae7_5"
    num_encode_offsets = 8 # should be equal to latent downsample factor
    pitch_shifts = [-1, 1]
    batch_size = 8 # num_encode_offsets should be divisible by batch_size
    sample_latents = False
    quantize_latents = False
    seed = 2000
    write_debug_files = False
    fp16 = True

    distributed_state = PartialState()
    device = distributed_state.device
    torch.manual_seed(seed)
    
    model_path = config.MODEL_PATH
    model_dtype = torch.bfloat16 if fp16 else torch.float32
    print(f"Loading DualDiffusion model from '{model_path}' (dtype={model_dtype})...")
    pipeline = DualDiffusionPipeline.from_pretrained(model_path,
                                                     torch_dtype=model_dtype,
                                                     load_latest_checkpoints=True,
                                                     device=device)
    pipeline.unet = pipeline.unet.to("cpu")
    last_global_step = pipeline.vae.config["last_global_step"]
    spectrogram_hop_length = pipeline.format.spectrogram_params.hop_length
    encode_offset_padding = spectrogram_hop_length * num_encode_offsets
    encode_offsets = [i * spectrogram_hop_length for i in range(num_encode_offsets)]
    num_batches_per_sample = num_encode_offsets // batch_size
    assert num_encode_offsets % batch_size == 0, "num_encode_offsets must be divisible by batch_size"

    pipeline.format.spectrogram_params.num_griffin_lim_iters = 200
    model_params = pipeline.format.model_params
    pitch_augmentation_formats = [get_pitch_augmentation_format(pipeline.format, shift).to(device) for shift in pitch_shifts]
    formats = [pipeline.format] + pitch_augmentation_formats
    
    dataset_cfg = config.load_json(os.path.join(config.CONFIG_PATH, "dataset.json"))
    dataset_path = config.DATASET_PATH
    debug_path = config.DEBUG_PATH
    dataset_format = dataset_cfg["dataset_format"]

    latents_dataset_path = config.LATENTS_DATASET_PATH
    os.makedirs(latents_dataset_path, exist_ok=True)
    
    # get split metadata
    split_metadata_files = []
    for filename in os.listdir(dataset_path):
        if os.path.isfile(os.path.join(dataset_path, filename)) and filename.lower().endswith(".jsonl"):
            split_metadata_files.append(filename)

    # process split samples
    for split_metadata_file in split_metadata_files:
        with open(os.path.join(dataset_path, split_metadata_file), "r") as f:
            split_metadata = [json.loads(line) for line in f.readlines()]

        # filter split samples that already encoded
        encode_samples = []
        for sample in split_metadata:
            file_name = sample["file_name"]
            output_filename = f"{os.path.splitext(file_name)[0]}.safetensors"
            output_path = os.path.join(latents_dataset_path, output_filename)

            if os.path.exists(output_path):
                continue
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
                output_filename = f"{os.path.splitext(file_name)[0]}.safetensors"
                output_path = os.path.join(latents_dataset_path, output_filename)

                if os.path.exists(output_path):
                    if distributed_state.is_main_process:
                        progress_bar.update(1)
                    continue

                file_ext = os.path.splitext(file_name)[1]
                input_raw_sample = load_audio(os.path.join(dataset_path, file_name))
                crop_width = pipeline.format.get_sample_crop_width(length=input_raw_sample.shape[-1] - encode_offset_padding)

                input_raw_samples = []
                for offset in encode_offsets:
                    input_raw_offset_sample = input_raw_sample[:, offset:offset+crop_width].unsqueeze(0).to(device)
                    input_raw_samples.append(input_raw_offset_sample)
                input_raw_sample = torch.cat(input_raw_samples, dim=0)

                input_samples = []
                for format in formats:
                    for j in range(num_batches_per_sample):
                        batch_input_raw_sample = input_raw_sample[j*batch_size:(j+1)*batch_size]
                        input_sample = format.raw_to_sample(batch_input_raw_sample, return_dict=True)["samples"].type(model_dtype)
                        input_samples.append(input_sample)
                input_sample = torch.cat(input_samples, dim=0)
                
                class_labels = pipeline.get_class_labels(game_id)
                vae_class_embeddings = pipeline.vae.get_class_embeddings(class_labels)            
                latents = []
                for j in range(input_sample.shape[0] // batch_size):
                    batch_input_sample = input_sample[j*batch_size:(j+1)*batch_size]
                    posterior = pipeline.vae.encode(batch_input_sample, vae_class_embeddings, pipeline.format)
                    batch_latents = posterior.sample(torch.randn) if sample_latents else posterior.mode()
                    latents.append(batch_latents.float())
                latents = torch.cat(latents, dim=0)

                if quantize_latents:
                    latents_quantized, offset_and_range = quantize_tensor(latents, 256)
                    latents_dict = {"latents": latents_quantized.type(torch.uint8), "offset_and_range": offset_and_range}
                else:
                    latents_dict = {"latents": latents }
                save_safetensors(latents_dict, output_path)

                # debug
                if write_debug_files:
                    if distributed_state.is_main_process:
                        
                        latents_dict = load_safetensors(output_path)
                        if quantize_latents:
                            latents = dequantize_tensor(latents_dict["latents"], latents_dict["offset_and_range"]).to(device)
                        else:
                            latents = latents_dict["latents"].to(device)
                            
                        for j, latent in enumerate(latents.unbind(0)):
                            save_raw_img(latent, os.path.join(debug_path, "latents", f"latents_{j:02}.png"))

                            decoded = pipeline.vae.decode(latent.unsqueeze(0).to(pipeline.vae.dtype), vae_class_embeddings, pipeline.format)
                            raw_sample = pipeline.format.sample_to_raw(decoded.float()).squeeze(0)
                            save_audio(raw_sample, model_params["sample_rate"], os.path.join(debug_path, "latents", f"audio_{j:02}.flac"))
                    exit()

                if distributed_state.is_main_process:
                    progress_bar.update(1)
                    progress_bar.set_postfix({"file_name": file_name})

        distributed_state.wait_for_everyone()

        if distributed_state.is_main_process:

            progress_bar.close()

            # update filename in split metadatato new safetensors latents
            for i, sample in enumerate(split_metadata):
                file_name = sample["file_name"]
                output_filename = f"{os.path.splitext(file_name)[0]}.safetensors"
                split_metadata[i]["file_name"] = output_filename

            # save split metadata for latents dataset
            split_metadata_output_path = os.path.join(latents_dataset_path, split_metadata_file)
            print(f"Saving split metadata to {split_metadata_output_path}...")
            with open(split_metadata_output_path, "w") as f:
                for sample in split_metadata:
                    f.write(json.dumps(sample) + "\n")

    if distributed_state.is_main_process:
        # save model info so we know which model / checkpoint was used to generate the latents
        vae_model_info_path = os.path.join(latents_dataset_path, "vae_model_info.md")
        print(f"Saving model info to {vae_model_info_path}...")
        with open(vae_model_info_path, "w") as f:
            f.write(f"model_name: {model_name}\n")
            f.write(f"last_global_step: {last_global_step}\n")
            f.write(f"dtype: {model_dtype}\n")

        #lastly, copy the dataset_info folder to the latents dataset
        dataset_infos_path = os.path.join(dataset_path, "dataset_infos")
        latents_dataset_infos_path = os.path.join(latents_dataset_path, "dataset_infos")
        print(f"Copying dataset_infos from {dataset_infos_path} to {latents_dataset_infos_path}...")
        shutil.copytree(dataset_infos_path, latents_dataset_infos_path, dirs_exist_ok=True)