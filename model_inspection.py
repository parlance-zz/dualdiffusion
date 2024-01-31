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
from dotenv import load_dotenv

import fnmatch
import torch

from dual_diffusion_pipeline import DualDiffusionPipeline

if __name__ == "__main__":

    load_dotenv(override=True)

    model_name = "dualdiffusion2d_330_v8_256embed_3_noskip"
    #module_name_filter = ["*attentions*group_norm_*.weight"]
    #module_name_filter = ["*attentions*group_norm_embedding.weight"]
    #module_name_filter = ["*resnets*conv*.weight"]
    module_name_filter = ["*"]

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
        avg_std = 0

        for name, param in pipeline.unet.named_parameters():
            
            if not any([fnmatch.fnmatch(name, filter) for filter in module_name_filter]):
                continue

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