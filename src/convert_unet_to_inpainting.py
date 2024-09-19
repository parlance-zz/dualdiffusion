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
import importlib

from modules.unets.unet import DualDiffusionUNet

if __name__ == "__main__":

    model_name = input(f"Enter model name: ")
    model_path = os.path.join(config.MODELS_PATH, model_name)
    
    model_index = config.load_json(os.path.join(model_path, f"model_index.json"))
    module_import_dict = model_index["modules"]["unet"]
    module_package = importlib.import_module(module_import_dict["package"])
    module_class = getattr(module_package, module_import_dict["class"])
    
    unet: DualDiffusionUNet = module_class.from_pretrained(model_path, subfolder="unet")
    unet.convert_to_inpainting()
    
    inpainting_unet_path = os.path.join(model_path, "unet_inpainting")
    if os.path.exists(inpainting_unet_path):
        print(f"Warning: Output folder already exists '{inpainting_unet_path}'")
        prompt = "Overwrite existing model? (y/n): "
    else:
        prompt = f"Save inpainting model with config at '{inpainting_unet_path}'? (y/n): "
    if input(prompt).lower() not in ["y","yes"]: exit()

    unet.save_pretrained(inpainting_unet_path)