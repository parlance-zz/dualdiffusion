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

from huggingface_hub import HfApi, list_repo_files


if __name__ == "__main__":

    dataset_cfg = config.load_json(os.path.join(config.CONFIG_PATH, "dataset.json"))

    datasets = [
        (config.LATENTS_DATASET_PATH, dataset_cfg["latents_hf_repository"]),
        (config.DATASET_PATH, dataset_cfg["dataset_hf_repository"]),
    ]

    removal_list = [
        "Super Mario World 2 - Yoshi's Island - 114b Overworld (Fuzzy)",
        "Super Mario World 2 - Yoshi's Island - 113d Athletic (Fuzzy 3)",
        "Super Mario World 2 - Yoshi's Island - 107b Flower Garden (Fuzzy 1)",
        "Super Mario World - 37b Cast List (Looped)",
        "Super Mario World - 30d Bowser's Last Attack (Faster)",
        "Super Mario World - 21b Boss Battle (Hurry)",
        "Super Mario World - 19b Bonus Game (Hurry)",
        "Super Mario World - 16b Fortress (Hurry)",
        "Super Mario World - 15b Haunted House (Hurry)",
        "Super Mario World - 14d Swimming (Hurry Yoshi)",
        "Super Mario World - 14c Swimming (Hurry)",
        "Super Mario World - 13d Underground (Yoshi Hurry)",
        "Super Mario World - 13c Underground (Hurry)",
        "Super Mario World - 12d Athletic (Hurry Yoshi)",
        "Super Mario World - 12c Athletic (Hurry)",
        "Super Mario World - 11d Overworld (Hurry Yoshi)",
        "Super Mario World - 11c Overworld (Hurry)",
        "Super Mario Kart - 22 Rainbow Road (Final Lap)",
        "Super Mario Kart - 18 Koopa Beach (Final Lap)",
        "Super Mario Kart - 39 Battle Mode (Faster)",
        "Super Mario Kart - 22 Rainbow Road (Final Lap)",
        "Super Mario Kart - 18 Koopa Beach (Final Lap)",
        "Super Mario Kart - 16 Choco Island (Final Lap)",
        "Super Mario Kart - 14 Bowser's Castle (Final Lap)",
        "Super Mario Kart - 10 Donut Plains (Final Lap)",
        "Super Mario Kart - 08 Mario Circuit (Final Lap)",
        "Super Mario All-Stars - 340 Athletic (Hurry Up)",
        "Super Mario All-Stars - 122 Swimming (Hurry Up)",
        "Super Mario All-Stars - 119 Overworld (Hurry Up)",
        "Super Mario World 2 - Yoshi's Island - 118b Big Boss (No Intro)",
    ]

    hf_api = HfApi()

    for dataset_path, hf_dataset_name in datasets:
        
        print(f"Processing dataset {dataset_path}...")
        if not os.path.exists(dataset_path):
            print(f"Dataset path {dataset_path} does not exist, skipping...")
            continue

        # get split metadata
        split_metadata_files = []
        for filename in os.listdir(dataset_path):
            if os.path.isfile(os.path.join(dataset_path, filename)) and filename.lower().endswith(".jsonl"):
                split_metadata_files.append(filename)

        # process split samples
        for split_metadata_file in split_metadata_files:
            with open(os.path.join(dataset_path, split_metadata_file), "r") as f:
                split_metadata = [json.loads(line) for line in f.readlines()]

            original_split_metadata = deepcopy(split_metadata)
            assert len(split_metadata) == len(original_split_metadata)

            # remove samples from split metadata
            for sample in original_split_metadata:
                file_name = sample["file_name"]

                if os.path.splitext(os.path.basename(file_name))[0] in removal_list:
                    
                    sample_path = os.path.join(dataset_path, file_name)
                    if os.path.exists(sample_path):
                        print(f"Removing {sample_path}...")
                        os.remove(sample_path)
                    else:
                        print(f"File {file_name} not found in {dataset_path}, skipping...")

                    split_metadata.remove(sample)

            # save filtered split metadata for dataset
            shutil.copyfile(os.path.join(dataset_path, split_metadata_file), os.path.join(dataset_path, f"{split_metadata_file}.old"))
            split_metadata_output_path = os.path.join(dataset_path, split_metadata_file)
            print(f"Saving updated split metadata to {split_metadata_output_path}...")
            with open(split_metadata_output_path, "w") as f:
                for sample in split_metadata:
                    f.write(json.dumps(sample) + "\n")

            # update split metadata on hf hub
            if hf_dataset_name is not None:
                try:
                    print(f"Updating split metadata {split_metadata_file} in {hf_dataset_name}...")
                    hf_api.upload_file(
                        path_or_fileobj=split_metadata_output_path,
                        path_in_repo=split_metadata_file,
                        repo_id=hf_dataset_name,
                        repo_type="dataset",
                    )
                except Exception as e:
                    print(f"Error uploading file to huggingface hub: {e}")
                    continue
        
        # remove files from hf hub dataset
        if hf_dataset_name is not None:

            print(f"Getting list of files in {hf_dataset_name}...")
            try:
                hf_dataset_file_names = list_repo_files(hf_dataset_name, repo_type="dataset")
            except Exception as e:
                print(f"Error listing files in huggingface hub dataset: {e}")
                continue

            for file_name in hf_dataset_file_names:
                if os.path.splitext(os.path.basename(file_name))[0] in removal_list:
                    print(f"Removing {file_name} from {hf_dataset_name}...")
                    try:
                        hf_api.delete_file(path_in_repo=file_name,
                                        repo_id=hf_dataset_name, repo_type="dataset")
                    except Exception as e:
                        print(f"Error deleting {file_name} from huggingface hub: {e}")