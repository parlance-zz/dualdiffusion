import os
from dotenv import load_dotenv

import numpy as np
import torch

from dual_diffusion_pipeline import DualDiffusionPipeline

if __name__ == "__main__":

    load_dotenv()

    model_name = "dualdiffusion2d_147"
    compare_model_name = None
    #compare_model_name = "dualdiffusion2d_132"

    model_path = os.path.join(os.environ.get("MODEL_PATH", "./"), model_name)
    print(f"Loading DualDiffusion model from '{model_path}'...")
    pipeline = DualDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float32)
    sample_rate = pipeline.config["model_params"]["sample_rate"]

    if compare_model_name is not None:
        model_path = os.path.join(os.environ.get("MODEL_PATH", "./"), compare_model_name)
        print(f"Loading DualDiffusion model from '{model_path}'...")
        compare_pipeline = DualDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float32)

        debug_weight_info_path = os.path.join(os.environ.get("DEBUG_PATH", "./"), "debug_model_compare_info.txt")
        debug_weight_info_file = open(debug_weight_info_path, "w")

        with torch.no_grad():
            
            avg_diff = 0
            total_numel = 0

            for (name1, param1), (name2, param2) in zip(pipeline.unet.named_parameters(), compare_pipeline.unet.named_parameters()):
                
                param_std1 = param1.std().item()
                param_std2 = param2.std().item()
                
                info_str = f"name: {name1}: std1: {param_std1} std2: {param_std2} diff: {param_std1-param_std2}"
                print(info_str); debug_weight_info_file.write(info_str+"\n")
                
                param_numel = param1.numel()
                avg_diff += (param_std1-param_std2)*param_numel
                total_numel += param_numel

            avg_diff /= total_numel
            info_str = f"Average std difference: {avg_diff}"
            print(info_str); debug_weight_info_file.write(info_str+"\n")

    else:
        debug_weights_path = os.path.join(os.environ.get("DEBUG_PATH", "./"), "debug_model_weights.raw")
        debug_weights_file = open(debug_weights_path, "wb")
        debug_weight_labels_path = os.path.join(os.environ.get("DEBUG_PATH", "./"), "debug_model_weight_labels.txt")
        debug_weight_labels_file = open(debug_weight_labels_path, "w")
        debug_weight_info_path = os.path.join(os.environ.get("DEBUG_PATH", "./"), "debug_model_weight_info.txt")
        debug_weight_info_file = open(debug_weight_info_path, "w")

        with torch.no_grad():

            num_weights = 0
            avg_std = 0

            for name, param in pipeline.unet.named_parameters():
                
                param_mean = param.mean().item()
                param_std = param.std().item()
                param_numel = param.numel()

                info_str = f"name: {name}: shape: {param.shape} mean: {param_mean} std: {param_std}"
                print(info_str); debug_weight_info_file.write(info_str+"\n")

                param.cpu().numpy().tofile(debug_weights_file)
                debug_weight_labels_file.write(f"{num_weights/sample_rate}\t{(num_weights+param_numel)/sample_rate}\t{name}\n")
                
                avg_std += param_std * param_numel
                num_weights += param_numel

            avg_std /= num_weights
            info_str = f"Average std: {avg_std}"
            print(info_str); debug_weight_info_file.write(info_str+"\n")