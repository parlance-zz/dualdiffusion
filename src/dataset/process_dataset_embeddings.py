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

from copy import deepcopy
import os
import json

from tqdm.auto import tqdm
from accelerate import PartialState
import torch
import librosa
import laion_clap

from dataset.dataset_processor import DatasetProcessorConfig
from utils.dual_diffusion_utils import init_cuda, load_safetensors, save_safetensors, normalize, update_audio_metadata


@torch.inference_mode()
def pre_encode_embeddings():

    dataset_processor_config = DatasetProcessorConfig(
        **config.load_json(os.path.join(config.CONFIG_PATH, "dataset", "dataset.json")))
    
    labels = dataset_processor_config.clap_embedding_labels
    tags = dataset_processor_config.clap_embedding_tags
    enable_fusion = dataset_processor_config.clap_enable_fusion
    audio_encoder = dataset_processor_config.clap_audio_encoder
    text_encoder = dataset_processor_config.clap_text_encoder
    compile_options = dataset_processor_config.clap_compile_options
    max_batch_size = dataset_processor_config.clap_max_batch_size
    resume_progress = True

    distributed_state = PartialState()
    device = distributed_state.device
    
    if config.CLAP_MODEL_PATH is None:
        raise ValueError("CLAP_MODEL_PATH is not set")

    with distributed_state.main_process_first():
        if distributed_state.is_main_process:
            print("Warning: This script will modify all audio file metadata and latents in the dataset path.")
            if input("Are you sure you want to continue? (y/n): ").lower() not in ["y", "yes"]:
                return

        clap_model = laion_clap.CLAP_Module(device=device, enable_fusion=enable_fusion, amodel=audio_encoder, tmodel=text_encoder)
        clap_model.load_ckpt(config.CLAP_MODEL_PATH, verbose=False)
        clap_model = clap_model.to(device)

    distributed_state.wait_for_everyone()

    if compile_options is not None:
        clap_model.get_audio_embedding_from_data = torch.compile(clap_model.get_audio_embedding_from_data, **compile_options)
        clap_model.get_text_embedding = torch.compile(clap_model.get_text_embedding, **compile_options)

    if tags is not None: 
        tag_labels = {tag: [tag] for tag in tags}
        if labels is not None:
            labels.update(tag_labels)
        else:
            labels = tag_labels

    if labels is not None:

        print(f"Calculating text embeddings for {len(labels)} labels...")
        label_embeddings = []
        for label, tag_list in labels.items():
            if isinstance(tag_list, str):
                tag_list = [tag_list]
            elif not isinstance(tag_list, list):
                raise ValueError(f"Invalid tag list for label '{label}': {tag_list}")
            
            tag_embeddings = clap_model.get_text_embedding(tag_list, use_tensor=True)
            label_embeddings.append(tag_embeddings.mean(dim=0))
        label_embeddings = torch.stack(label_embeddings, dim=0).to(device=device, dtype=torch.float32)
        label_embeddings = normalize(label_embeddings).float()

    # get split metadata
    split_metadata_files = []
    for filename in os.listdir(config.DATASET_PATH):
        if os.path.isfile(os.path.join(config.DATASET_PATH, filename)) and filename.lower().endswith(".jsonl"):
            split_metadata_files.append(filename)

    # process split samples
    for split_metadata_file in split_metadata_files:
        with open(os.path.join(config.DATASET_PATH, split_metadata_file), "r") as f:
            split_metadata = [json.loads(line) for line in f.readlines()]

        if distributed_state.is_main_process:
            print(f"Processing {len(split_metadata)} samples from {split_metadata_file}...")
            
        with distributed_state.split_between_processes(split_metadata) as samples:

            if distributed_state.is_main_process:
                progress_bar = tqdm(total=len(samples))
                progress_bar.set_description(f"Split: {split_metadata_file}")
            
            for sample in samples:

                if sample["latents_file_name"] is None:
                    sample["latents_file_name"] = f"{os.path.splitext(sample['file_name'])[0]}.safetensors"
                    
                file_name = sample["file_name"]
                input_path = os.path.join(config.DATASET_PATH, file_name)
                output_filename = sample["latents_file_name"]
                output_path = os.path.join(config.DATASET_PATH, output_filename)
                sample_prompt = sample.get("prompt", None)
                save_latents = False

                existing_latents = None
                audio_embeddings = None
                text_embeddings = None
            
                if os.path.isfile(output_path): # deep copying required for safetensors to close the file handle
                    existing_latents = deepcopy(load_safetensors(output_path))
                    if resume_progress == True:
                        if "clap_audio_embeddings" in existing_latents:
                            audio_embeddings = existing_latents["clap_audio_embeddings"].to(device=device, dtype=torch.float32)
                        if "clap_text_embeddings" in existing_latents:
                            text_embeddings = existing_latents["clap_text_embeddings"].to(device=device, dtype=torch.float32)

                # get audio embeddings if they are not yet encoded
                if audio_embeddings is None:
                    save_latents = True
                
                    audio, sample_rate = librosa.load(input_path, sr=48000, mono=True)
                    chunk_size = sample_rate * 10

                    audio = audio[:audio.shape[0] // chunk_size * chunk_size] # crop out last chunk if it's too small
                    audio = torch.tensor(audio.reshape(-1, chunk_size), dtype=torch.float32, device=device)

                    # get embeddings in chunks using max_batch_size
                    audio_embeddings = torch.cat([
                        normalize(clap_model.get_audio_embedding_from_data(chunk, use_tensor=True)).float() for chunk in audio.split(max_batch_size)], dim=0)

                # get text embeddings if a prompt is available and they are not yet encoded
                if sample_prompt is not None and text_embeddings is None:
                    save_latents = True
                    text_embeddings = normalize(clap_model.get_text_embedding([sample_prompt], use_tensor=True)).float()

                if labels is not None:
                    # gets similarity for each label and chunk individually
                    #cos_similarity = torch.mm(label_embeddings / label_embeddings.shape[1]**0.5,
                    #                          audio_embeddings.T / audio_embeddings.shape[1]**0.5).clip(-1, 1)

                    # update audio file metadata with label similarity scores
                    label_scores = (torch.einsum("ld,d->l", label_embeddings, audio_embeddings.mean(dim=0)) / label_embeddings.shape[1])
                    label_scores = ((label_scores + 1) / 2).clip(0, 1).tolist() # keep score positive to preserve textual sorting order
                    labels_metadata = {f"clap_{label}": f"{score:01.4f}" for label, score in zip(labels.keys(), label_scores)}
                    labels_metadata["clap_all_labels"] = f"{sum(label_scores) / len(label_scores):01.4f}"
                    
                    try:
                        update_audio_metadata(input_path, labels_metadata, clear_clap_fields=True)
                    except Exception as e:
                        print(f"Failed to update metadata for {file_name}: {e}")

                if save_latents == True:
                    latents_dict = existing_latents or {}
                    latents_dict["clap_audio_embeddings"] = audio_embeddings.to(torch.bfloat16)
                    if text_embeddings is not None:
                        latents_dict["clap_text_embeddings"] = text_embeddings.to(torch.bfloat16)

                    save_safetensors(latents_dict, output_path)
                    del latents_dict

                if distributed_state.is_main_process:
                    progress_bar.update(1)
                    # doesn't respect min_interval, disabled for performance
                    #progress_bar.set_postfix({"file_name": file_name}) 

        print("Main process finished, waiting for distributed processes...")
        distributed_state.wait_for_everyone()

        if distributed_state.is_main_process:
            progress_bar.close()
        
    print("Pre-encoding embeddings complete")

if __name__ == "__main__":

    init_cuda()
    pre_encode_embeddings()