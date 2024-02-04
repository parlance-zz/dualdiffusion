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
import datetime

from dotenv import load_dotenv
import numpy as np
import torch

from dual_diffusion_pipeline import DualDiffusionPipeline
from dual_diffusion_utils import init_cuda, save_audio, save_raw, load_raw, load_audio

if __name__ == "__main__":

    init_cuda()
    load_dotenv(override=True)

    model_name = "dualdiffusion2d_1000_6"
    num_samples = 1
    #device = "cuda"
    device = "cpu"
    fp16 = False
    #fp16 = True

    model_path = os.path.join(os.environ.get("MODEL_PATH", "./"), model_name)
    model_dtype = torch.float16 if fp16 else torch.float32
    print(f"Loading DualDiffusion model from '{model_path}' (dtype={model_dtype})...")
    pipeline = DualDiffusionPipeline.from_pretrained(model_path,
                                                     torch_dtype=model_dtype,
                                                     load_latest_checkpoints=True)
    model_params = pipeline.config["model_params"]
    crop_width = pipeline.format.get_sample_crop_width(model_params)
    vae = pipeline.vae.to(device)
    last_global_step = vae.config["last_global_step"]

    dataset_path = os.environ.get("DATASET_PATH", "./dataset/samples")
    dataset_format = os.environ.get("DATASET_FORMAT", ".flac")
    dataset_raw_format = os.environ.get("DATASET_RAW_FORMAT", "int16")
    test_samples = np.random.choice(os.listdir(dataset_path), num_samples, replace=False)
    print("Sample shape: ", pipeline.format.get_sample_shape(model_params))

    output_path = os.path.join(model_path, "output")
    os.makedirs(output_path, exist_ok=True)

    start_time = datetime.datetime.now()

    for filename in test_samples:
        
        file_ext = os.path.splitext(filename)[1]
        if dataset_format == ".raw":
            input_raw_sample = load_raw(os.path.join(dataset_path, filename),
                                        dtype=dataset_raw_format, start=-1, count=crop_width)
        else:
            input_raw_sample = load_audio(os.path.join(dataset_path, filename), start=-1, count=crop_width)
        input_raw_sample = input_raw_sample.unsqueeze(0).to(device)

        input_sample_dict = pipeline.format.raw_to_sample(input_raw_sample, model_params, return_dict=True)
        input_sample = input_sample_dict["samples"]
        posterior = vae.encode(input_sample.type(model_dtype), return_dict=False)[0]
        latents = posterior.sample()
        model_output = vae.decode(latents, return_dict=False)[0]
        output_sample_dict = pipeline.format.sample_to_raw(model_output.type(torch.float32), model_params, return_dict=True)
        #output_sample_dict = pipeline.format.sample_to_raw(model_output.type(torch.float32), model_params, return_dict=True, abs_replacement=input_sample[:, 0:model_params["sample_raw_channels"]])
        output_sample = output_sample_dict["samples"]
        output_raw_sample = output_sample_dict["raw_samples"]

        save_raw(latents, os.path.join(output_path,f"step_{last_global_step}_{filename.replace(file_ext, '_latents.raw')}"))
        save_raw(input_sample, os.path.join(output_path, f"step_{last_global_step}_{filename.replace(file_ext, '_input_sample.raw')}"))
        save_raw(output_sample, os.path.join(output_path, f"step_{last_global_step}_{filename.replace(file_ext, '_sample.raw')}"))
        save_raw(posterior.parameters, os.path.join(output_path, f"step_{last_global_step}_{filename.replace(file_ext, '_posterior.raw')}"))

        output_flac_file_path = os.path.join(output_path, f"step_{last_global_step}_{filename.replace(file_ext, '_original.flac')}")
        save_audio(input_sample_dict["raw_samples"].squeeze(0), model_params["sample_rate"], output_flac_file_path)
        print(f"Saved flac output to {output_flac_file_path}")

        output_flac_file_path = os.path.join(output_path, f"step_{last_global_step}_{filename.replace(file_ext, '_decoded.flac')}")
        save_audio(output_raw_sample.squeeze(0), model_params["sample_rate"], output_flac_file_path)
        print(f"Saved flac output to {output_flac_file_path}")

    print(f"Finished in: {datetime.datetime.now() - start_time}")