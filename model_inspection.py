import os
from dotenv import load_dotenv

import numpy as np
import torch

from dual_diffusion_pipeline import DualDiffusionPipeline

if __name__ == "__main__":

    load_dotenv()

    model_name = "dualdiffusion2d_90"

    model_path = os.path.join(os.environ.get("MODEL_PATH", "./"), model_name)
    print(f"Loading DualDiffusion model from '{model_path}'...")
    pipeline = DualDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float32)
    sample_rate = pipeline.config["model_params"]["sample_rate"]

    debug_weights_path = os.path.join(os.environ.get("DEBUG_PATH", "./"), "debug_model_weights.raw")
    debug_weights_file = open(debug_weights_path, "wb")

    debug_weight_labels_path = os.path.join(os.environ.get("DEBUG_PATH", "./"), "debug_model_weight_labels.txt")
    debug_weight_labels_file = open(debug_weight_labels_path, "w")

    debug_weight_info_path = os.path.join(os.environ.get("DEBUG_PATH", "./"), "debug_model_weight_info.txt")
    debug_weight_info_file = open(debug_weight_info_path, "w")

    with torch.no_grad():
        num_weights = 0
        for name, param in pipeline.unet.named_parameters():
            param_mean = param.mean().item()
            param_std = param.std().item()
            param_numel = param.numel()
            info_str = f"name: {name}: shape: {param.shape} mean: {param_mean} std: {param_std}"

            print(info_str)
            debug_weight_info_file.write(info_str)

            param.cpu().numpy().tofile(debug_weights_file)
            debug_weight_labels_file.write(f"{num_weights/sample_rate}\t{(num_weights+param_numel)/sample_rate}\t{name}\n")

            num_weights += param_numel