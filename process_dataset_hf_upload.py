import os
from dotenv import load_dotenv
from huggingface_hub import HfApi

TARGET_HF_REPOSITORY = "parlance/spc_audio"

if __name__ == "__main__":

    load_dotenv(override=True)

    OUTPUT_SAMPLE_DIR = os.environ.get("DATASET_PATH")

    if not os.path.exists(OUTPUT_SAMPLE_DIR):        
        print(f"Dataset path {OUTPUT_SAMPLE_DIR} does not exist")
        exit(1)
    
    # push to huggingface hub
    api = HfApi()
    api.upload_folder(folder_path=OUTPUT_SAMPLE_DIR, repo_id=TARGET_HF_REPOSITORY, repo_type="dataset", 
                      multi_commits=True, multi_commits_verbose=True)