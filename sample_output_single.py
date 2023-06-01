import time
import glob
import random

import torch
import numpy as np

from single_diffusion_pipeline import SingleDiffusionPipeline

if __name__ == "__main__":
    
    if not torch.cuda.is_available():
        print("Error: PyTorch not compiled with CUDA support or CUDA unavailable")
        exit(1)

    torch.seed()

    model_path = "./models/singlediffusion"
    print(f"Loading SingleDiffusion model from '{model_path}'...")
    my_pipeline = SingleDiffusionPipeline.from_pretrained(model_path).to("cuda")

    example_files = glob.glob("./dataset/single/*.raw")
    example_file = random.choice(example_files)
    print(f"Loading example from '{example_file}'...")
    example = np.fromfile(example_file, dtype=np.complex64)
    example = torch.from_numpy(example).to("cuda")

    start = time.time()
    output = my_pipeline(steps=100, length=4, example=example)
    print(f"Time taken: {time.time()-start}")

    existing_output_count = len(glob.glob("./output/output*.raw"))
    test_output_path = f"./output/output_{existing_output_count}.raw"
    output = SingleDiffusionPipeline.resample_audio(output, 8000, 44100)
    output = output.cpu().numpy()
    (output / np.max(np.absolute(np.real(output)))).tofile(test_output_path)
    #output.tofile(test_output_path)
    print(f"Saved output to {test_output_path}")