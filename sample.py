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
import time
from dotenv import load_dotenv

import numpy as np
import torch
import datetime

from dual_diffusion_pipeline import DualDiffusionPipeline
from dual_diffusion_utils import init_cuda, save_audio

if __name__ == "__main__":

    init_cuda()
    load_dotenv(override=True)

    model_name = "edm2_100_2"

    num_samples = 1
    batch_size = 1
    length = 0
    #length = 30 * 32000
    steps = 50
    cfg_scale = 5
    game_ids = [787]
    loops = 0 #1
    fp16 = False #True
    device = "cuda" #"cpu"
    use_midpoint_integration = True#True
    use_perturbation = True

    seed = np.random.randint(10000, 99999-num_samples)
    #seed = 2000
    #seed = 43820
    seed = 21308

    model_dtype = torch.bfloat16 if fp16 else torch.float32
    model_path = os.path.join(os.environ.get("MODEL_PATH", "./"), model_name)
    print(f"Loading DualDiffusion model from '{model_path}' (dtype={model_dtype})...")
    pipeline = DualDiffusionPipeline.from_pretrained(model_path,
                                                     torch_dtype=model_dtype,
                                                     #device=device, # this doesn't work for some fucking reason????
                                                     load_latest_checkpoints=True).to(device)
    pipeline.format = pipeline.format.to(device); pipeline.vae = pipeline.vae.to(device)
    last_global_step = pipeline.unet.config["last_global_step"]

    output_path = os.path.join(model_path, "output")
    os.makedirs(output_path, exist_ok=True)

    start_time = datetime.datetime.now()

    """
    #pipeline.unet.emb_fourier.thing()
    t = torch.tensor([1/50], device="cuda")
    #mid = pipeline.unet.emb_noise(pipeline.unet.emb_fourier(t))
    mid = pipeline.unet.emb_fourier(t)
    results = torch.zeros(25, device="cuda")
    for i in range(25):
        t = torch.tensor([i / (25 - 1)], device="cuda")
        #e = pipeline.unet.emb_noise(pipeline.unet.emb_fourier(t))
        e = pipeline.unet.emb_fourier(t)
        results[i] = (e*mid).sum()
    from dual_diffusion_utils import save_raw
    results /= results.amax()
    save_raw(results, "./debug/results.raw")
    exit()
    """

    for i in range(num_samples):
        print(f"Generating sample {i+1}/{num_samples}...")

        start = time.time()
        output = pipeline(steps=steps,
                          seed=seed,
                          loops=loops,
                          batch_size=batch_size,
                          length=length,
                          cfg_scale=cfg_scale,
                          game_ids=game_ids,
                          use_midpoint_integration=use_midpoint_integration,
                          use_perturbation=use_perturbation)
        print(f"Time taken: {time.time()-start}")

        output_flac_file_path = os.path.join(output_path, f"step_{last_global_step}_{steps}_cfg{cfg_scale}_s{seed}.flac")
        save_audio(output.squeeze(0), pipeline.config["model_params"]["sample_rate"], output_flac_file_path)
        print(f"Saved flac output to {output_flac_file_path}")

        seed += 1
    
    print(f"Finished in: {datetime.datetime.now() - start_time}")