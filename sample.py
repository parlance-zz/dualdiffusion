import os
import time
import glob

import torch
import numpy as np
import ffmpeg

from dual_diffusion_pipeline import DualDiffusionPipeline
from diffusers import StableDiffusionImg2ImgPipeline

def embedding_test():
    num_positions = 256
    embedding_dim = 32

    positions = ((torch.arange(0, num_positions, 1, dtype=torch.float32) + 0.5) / num_positions).log()
    positions = positions / positions[0] * num_positions / 4

    pe = DualDiffusionPipeline.get_positional_embedding(positions, embedding_dim)
    output = torch.zeros((num_positions, num_positions), dtype=torch.float32)

    for x in range(num_positions):
        a = pe[:, x]
        for y in range(num_positions):
            output[x, y] = ( pe[:, y] * a).sum()

    output.cpu().numpy().tofile("./output/debug_embeddings.raw")
    exit()

def reconstruction_test(model_params):
    
    crop_width = DualDiffusionPipeline.get_sample_crop_width(model_params)
    raw_sample = np.fromfile("./dataset/samples/400.raw", dtype=np.int16, count=crop_width) / 32768.
    raw_sample = torch.from_numpy(raw_sample).unsqueeze(0).to("cuda")
    freq_sample = DualDiffusionPipeline.raw_to_freq(raw_sample, model_params) #.type(torch.float16)
    #phases = DualDiffusionPipeline.raw_to_freq(raw_sample, model_params, format_override="complex")
    #raw_sample = DualDiffusionPipeline.freq_to_raw(freq_sample, model_params, phases)
    raw_sample = DualDiffusionPipeline.freq_to_raw(freq_sample, model_params)
    raw_sample.type(torch.complex64).cpu().numpy().tofile("./output/debug_reconstruction.raw")
    
    exit()

if __name__ == "__main__":

    if not torch.cuda.is_available():
        print("Error: PyTorch not compiled with CUDA support or CUDA unavailable")
        exit(1)

    model_path = "./models/new_lgdiffusion2"
    print(f"Loading LGDiffusion model from '{model_path}'...")
    pipeline = DualDiffusionPipeline.from_pretrained(model_path).to("cuda")
    sample_rate = pipeline.config["model_params"]["sample_rate"]

    #reconstruction_test(pipeline.config["model_params"])

    num_samples = 10
    batch_size = 1
    length = 1
    scheduler = "dpms++"
    #scheduler = "ddim"
    steps = 100
    loops = 1
    renormalize = False
    rebalance = False

    seed = np.random.randint(10000, 99999-num_samples)

    for i in range(num_samples):
        print(f"Generating sample {i+1}/{num_samples}...")

        start = time.time()
        output = pipeline(steps=steps,
                          scheduler=scheduler,
                          seed=seed,
                          loops=loops,
                          batch_size=batch_size,
                          length=length,
                          renormalize_sample=renormalize,
                          rebalance_mean=rebalance)
        print(f"Time taken: {time.time()-start}")

        model_name = os.path.basename(model_path)
        output_path = f"./models/{model_name}/output"
        os.makedirs(output_path, exist_ok=True)

        last_global_step = pipeline.config["model_params"]["last_global_step"]
        output_path = os.path.join(output_path, f"step_{last_global_step}_{scheduler}{steps}_s{seed}.raw")
        
        output.cpu().numpy().tofile(output_path)
        output_flac_file = os.path.splitext(output_path)[0] + '.flac'
        ffmpeg.input(output_path, f="f32le", ac=2, ar=sample_rate).output(output_flac_file).run(quiet=True)
        print(f"Saved flac output to {output_flac_file}")
        os.remove(output_path)

        seed += 1