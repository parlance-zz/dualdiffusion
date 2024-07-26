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

from utils import config

import os
import shutil
import importlib

import torch

from pipelines.dual_diffusion_pipeline import DualDiffusionPipeline

def print_module_info(module):

    module_params = {}
    num_emb_params = num_conv_params = num_attn_params = num_other_params = num_total_params = 0
    for name, param in module.named_parameters():

        module_name = name.replace(".weight", "").split(".")[-1]
        module_params[module_name] = module_params.get(module_name, 0) + param.numel()

        if "emb" in name: num_emb_params += param.numel()
        elif "conv" in name: num_conv_params += param.numel()
        elif "attn" in name: num_attn_params += param.numel()
        else: num_other_params += param.numel()
        num_total_params += param.numel()

    module_params = sorted(module_params.items(), key=lambda x:x[1])
    print("\nNumber of module params: ")
    for name, count in module_params:
        print(f"  {name.ljust(11)}: {count/1000000:{4}f}m")
    print("")
    print(f"Number of emb params:   {num_emb_params/1000000:{4}f}m")
    print(f"Number of conv params:  {num_conv_params/1000000:{4}f}m")
    print(f"Number of attn params:  {num_attn_params/1000000:{4}f}m")
    print(f"Number of other params: {num_other_params/1000000:{4}f}m")
    print(f"Estimated size (MB): {num_total_params*4/1000000:{4}f}m")
    print("")


if __name__ == "__main__":

    model_name = input(f"Enter model name:")
    model_seed = input(f"Enter random seed (or none for random seed):")

    if model_seed != "":
        torch.manual_seed(int(model_seed))
    
    model_config_source_path = os.path.join(config.CONFIG_PATH, model_name)
    if not os.path.isdir(model_config_source_path):
        raise FileNotFoundError(f"Model config path '{model_config_source_path}' not found")
    
    model_modules = {}
    model_index = config.load_json(os.path.join(model_config_source_path, "model_index.json"))
    for module_name in model_index:
        if module_name.startswith("_") or not isinstance(model_index[module_name], list): continue

        module_package_name, module_class_name = model_index[module_name][0], model_index[module_name][1]
        module_package = importlib.import_module(module_package_name)
        module_class = getattr(module_package, module_class_name)

        module_config_path = os.path.join(model_config_source_path, f"{module_name}.json")
        if not os.path.isfile(module_config_path):
            raise FileNotFoundError(f"Module config '{module_config_path}' not found")
        module_config = config.load_json(module_config_path)

        model_modules[module_name] = module_class(**module_config)
        print_module_info(model_modules[module_name])

    pipeline = DualDiffusionPipeline.create_new(**model_modules)

    new_model_path = os.path.join(config.MODEL_PATH, model_name)
    if os.path.exists(new_model_path):
        print(f"Warning: Output folder already exists '{new_model_path}'")
        prompt = "Overwrite existing model? (y/n): "
    else:
        prompt = f"Create new model with config at '{new_model_path}'? (y/n): "
    if input(prompt).lower() not in ["y","yes"]: exit()
    
    pipeline.save_pretrained(new_model_path, safe_serialization=True)
    print(f"Created new DualDiffusion model with config at '{new_model_path}'")

    model_train_config_path = os.path.join(new_model_path, "training")
    os.makedirs(model_train_config_path, exist_ok=True)

    for module_name in model_modules:
        
        module_train_config_source_path = os.path.join(model_config_source_path, f"{module_name}_train.json")
        if not os.path.isfile(module_train_config_source_path): continue

        module_accelerate_config_source_path = os.path.join(model_config_source_path, f"{module_name}_accelerate.yaml")
        if not os.path.isfile(module_accelerate_config_source_path):
            module_accelerate_config_source_path = os.path.join(model_config_source_path, f"accelerate.yaml")
        if not os.path.isfile(module_accelerate_config_source_path):
            raise FileNotFoundError(f"Training accelerate config '{module_accelerate_config_source_path}' not found")
        
        shutil.copy(module_train_config_source_path, model_train_config_path)
        shutil.copy(module_accelerate_config_source_path, model_train_config_path)
        module_train_config_dest_path = os.path.join(model_train_config_path, f"{module_name}_train.json")
        module_accelerate_config_dest_path = os.path.join(model_train_config_path, os.path.basename(module_accelerate_config_source_path))

        module_train_sh_path = os.path.join(model_train_config_path, f"{module_name}_train.sh")
        with open(module_train_sh_path, "w") as f:
            f.write(f"""#!/bin/bash
                    accelerate launch \
                    --config_file "{module_accelerate_config_dest_path}" \
                    "{module_train_sh_path}" \
                    --model_path="{new_model_path}" \
                    --train_config_path="{module_train_config_dest_path}"
            """)