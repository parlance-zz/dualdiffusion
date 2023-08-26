import os
import time
import glob

import torch
import numpy as np
import ffmpeg

from lg_diffusion_pipeline import LGDiffusionPipeline

if __name__ == "__main__":

    if not torch.cuda.is_available():
        print("Error: PyTorch not compiled with CUDA support or CUDA unavailable")
        exit(1)

    model_path = "./models/lgdiffusion_freq9"
    print(f"Loading LGDiffusion model from '{model_path}'...")
    my_pipeline = LGDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16).to("cuda")
    sample_rate = my_pipeline.config["model_params"]["sample_rate"]

    num_samples = 1
    batch_size = 1
    scheduler = "dpms++"
    #scheduler = "ddim"
    steps = 100
    loops = 1

    seed = np.random.randint(10000000,
                            99999999-num_samples)

    for i in range(num_samples):
        print(f"Generating sample {i+1}/{num_samples}...")

        start = time.time()
        output = my_pipeline(steps=steps, scheduler=scheduler,
                            seed=seed,
                            loops=loops,
                            batch_size=batch_size)
        print(f"Time taken: {time.time()-start}")

        existing_output_count = len(glob.glob("./output/output*.raw"))
        test_output_path = f"./output/output_{existing_output_count}_{scheduler}_{steps}_{seed}.raw"
        
        output.cpu().numpy().tofile(test_output_path)
        print(f"Saved raw output to {test_output_path}")

        output_flac_file = os.path.splitext(test_output_path)[0] + '.flac'
        ffmpeg.input(test_output_path, f="f32le", ac=2, ar=sample_rate).output(output_flac_file).run(quiet=True)
        print(f"Saved flac output to {output_flac_file}")
        
        seed += 1