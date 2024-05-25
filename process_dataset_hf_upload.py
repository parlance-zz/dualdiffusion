import os
from dotenv import load_dotenv
from huggingface_hub import HfApi
import time

TARGET_HF_REPOSITORY = "parlance/spc_audio_latents"
#OUTPUT_SAMPLE_DIR = os.environ.get("DATASET_PATH")
OUTPUT_SAMPLE_DIR = os.environ.get("LATENTS_DATASET_PATH")

if __name__ == "__main__":

    load_dotenv(override=True)

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