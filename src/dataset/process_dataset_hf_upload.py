import utils.config as config

import os
from huggingface_hub import HfApi
import time



if __name__ == "__main__":

    dataset_cfg = config.load_json(os.path.join(config.CONFIG_PATH, "dataset.json"))

    TARGET_HF_REPOSITORY = dataset_cfg["dataset_hf_repository"]
    #OUTPUT_SAMPLE_DIR = os.environ.get("DATASET_PATH")
    OUTPUT_SAMPLE_DIR = os.environ.get("LATENTS_DATASET_PATH")

    if not os.path.exists(OUTPUT_SAMPLE_DIR):        
        print(f"Dataset path {OUTPUT_SAMPLE_DIR} does not exist")
        exit(1)
    
    # push to huggingface hub
    while True:
        try:
            api = HfApi()
            api.upload_folder(folder_path=OUTPUT_SAMPLE_DIR, repo_id=TARGET_HF_REPOSITORY, repo_type="dataset", 
                            multi_commits=True, multi_commits_verbose=True)
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(60 * 65)
            continue

        break

    print(f"Uploaded dataset to {TARGET_HF_REPOSITORY} successfully")