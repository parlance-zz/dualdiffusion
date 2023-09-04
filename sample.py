import os
import time
import glob

import torch
import numpy as np
import ffmpeg

from lg_diffusion_pipeline import LGDiffusionPipeline
from diffusers import StableDiffusionImg2ImgPipeline

def embedding_test():
    num_positions = 256
    embedding_dim = 32

    positions = ((torch.arange(0, num_positions, 1, dtype=torch.float32) + 0.5) / num_positions).log()
    positions = positions / positions[0] * num_positions / 4

    pe = LGDiffusionPipeline.get_positional_embedding(positions, embedding_dim)
    output = torch.zeros((num_positions, num_positions), dtype=torch.float32)

    for x in range(num_positions):
        a = pe[:, x]
        for y in range(num_positions):
            output[x, y] = ( pe[:, y] * a).sum()

    output.cpu().numpy().tofile("./output/debug_embeddings.raw")
    exit()

def reconstruction_test(model_params):
    crop_width = LGDiffusionPipeline.get_sample_crop_width(model_params)
    raw_sample = np.fromfile("./dataset/samples/700.raw", dtype=np.int16, count=crop_width) / 32768.
    raw_sample = torch.from_numpy(raw_sample).unsqueeze(0).to("cuda")
    freq_sample = LGDiffusionPipeline.raw_to_freq(raw_sample, model_params) #.type(torch.float16)
    phases = LGDiffusionPipeline.raw_to_freq(raw_sample, model_params, format_override="complex")
    raw_sample = LGDiffusionPipeline.freq_to_raw(freq_sample, model_params, phases)
    raw_sample.type(torch.complex64).cpu().numpy().tofile("./output/debug_reconstruction.raw")
    exit()

if __name__ == "__main__":

    if not torch.cuda.is_available():
        print("Error: PyTorch not compiled with CUDA support or CUDA unavailable")
        exit(1)

    model_path = "./models/new_lgdiffusion2-no-pe"
    print(f"Loading LGDiffusion model from '{model_path}'...")
    #my_pipeline = LGDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16).to("cuda")
    my_pipeline = LGDiffusionPipeline.from_pretrained(model_path).to("cuda")
    sample_rate = my_pipeline.config["model_params"]["sample_rate"]

    #reconstruction_test(my_pipeline.config["model_params"])

    num_samples = 10
    batch_size = 1
    length = 1
    scheduler = "dpms++"
    #scheduler = "ddim"
    steps = 100
    loops = 1
    renormalize = False
    rebalance = False

    seed = np.random.randint(10000000,
                            99999999-num_samples)

    for i in range(num_samples):
        print(f"Generating sample {i+1}/{num_samples}...")

        start = time.time()
        output = my_pipeline(steps=steps,
                             scheduler=scheduler,
                             seed=seed,
                             loops=loops,
                             batch_size=batch_size,
                             length=length,
                             renormalize_sample=renormalize,
                             rebalance_mean=rebalance)
        print(f"Time taken: {time.time()-start}")

        existing_output_count = len(glob.glob("./output/output*.raw"))
        test_output_path = f"./output/output_{existing_output_count}_{scheduler}_{steps}_{seed}.raw"
        
        output.cpu().numpy().tofile(test_output_path)
        print(f"Saved raw output to {test_output_path}")

        output_flac_file = os.path.splitext(test_output_path)[0] + '.flac'
        ffmpeg.input(test_output_path, f="f32le", ac=2, ar=sample_rate).output(output_flac_file).run(quiet=True)
        print(f"Saved flac output to {output_flac_file}")
        
        seed += 1