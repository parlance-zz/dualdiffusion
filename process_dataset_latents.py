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

import os
import json
import shutil

from dotenv import load_dotenv
import torch
from tqdm.auto import tqdm

from dual_diffusion_pipeline import DualDiffusionPipeline
from dual_diffusion_utils import init_cuda, load_raw, load_audio, save_safetensors
        
if __name__ == "__main__":

    init_cuda()
    load_dotenv(override=True)

    model_name = "edm2_vae_test7_2"
    device = "cuda"
    fp16 = True

    model_path = os.path.join(os.environ.get("MODEL_PATH", "./"), model_name)
    model_dtype = torch.bfloat16 if fp16 else torch.float32
    print(f"Loading DualDiffusion model from '{model_path}' (dtype={model_dtype})...")
    pipeline = DualDiffusionPipeline.from_pretrained(model_path,
                                                     torch_dtype=model_dtype,
                                                     load_latest_checkpoints=True,
                                                     device=device)
    pipeline.unet = pipeline.unet.to("cpu")
    last_global_step = pipeline.vae.config["last_global_step"]

    latents_dataset_path = os.environ.get("LATENTS_DATASET_PATH", "./dataset/latents")
    os.makedirs(latents_dataset_path, exist_ok=True)

    dataset_path = os.environ.get("DATASET_PATH", "./dataset/samples")
    dataset_format = os.environ.get("DATASET_FORMAT", ".flac")
    dataset_raw_format = os.environ.get("DATASET_RAW_FORMAT", "int16")
    
    # get split metadata
    split_metadata_files = []
    for filename in os.listdir(dataset_path):
        if os.path.isfile(os.path.join(dataset_path, filename)) and filename.lower().endswith(".jsonl"):
            split_metadata_files.append(filename)

    # process split samples
    for split_metadata_file in split_metadata_files:
        with open(os.path.join(dataset_path, split_metadata_file), "r") as f:
            split_metadata = [json.loads(line) for line in f.readlines()]

        print(f"Processing {len(split_metadata)} samples from {split_metadata_file}...")
        progress_bar = tqdm(total=len(split_metadata))
        progress_bar.set_description(f"Split: {split_metadata_file}")

        for i, sample in enumerate(split_metadata):

            game_id = sample["game_id"]
            file_name = sample["file_name"]
            output_filename = f"{os.path.splitext(file_name)[0]}.safetensors"
            output_path = os.path.join(latents_dataset_path, output_filename)
            split_metadata[i]["file_name"] = output_filename

            if os.path.exists(output_path):
                progress_bar.update(1)
                continue

            file_ext = os.path.splitext(file_name)[1]
            if dataset_format == ".raw":
                input_raw_sample = load_raw(os.path.join(dataset_path, file_name), dtype=dataset_raw_format)
            else:
                input_raw_sample = load_audio(os.path.join(dataset_path, file_name))
            crop_width = pipeline.format.get_sample_crop_width(length=input_raw_sample.shape[-1])
            input_raw_sample = input_raw_sample[:, :crop_width].unsqueeze(0).to(device)

            class_labels = pipeline.get_class_labels(game_id)
            vae_class_embeddings = pipeline.vae.get_class_embeddings(class_labels)
            input_sample = pipeline.format.raw_to_sample(input_raw_sample, return_dict=True)["samples"]
            posterior = pipeline.vae.encode(input_sample.type(model_dtype), vae_class_embeddings, pipeline.format)
            latents = posterior.mode()

            save_safetensors({"latents": latents}, output_path)
            progress_bar.update(1)
            progress_bar.set_postfix({"file_name": file_name})

        progress_bar.close()

        # save split metadata for latents dataset
        split_metadata_output_path = os.path.join(latents_dataset_path, split_metadata_file)
        print(f"Saving split metadata to {split_metadata_output_path}...")
        with open(split_metadata_output_path, "w") as f:
            for sample in split_metadata:
                f.write(json.dumps(sample) + "\n")

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
