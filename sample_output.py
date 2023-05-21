import time
import glob

import torch
import numpy as np

from dual_diffusion_pipeline import DualDiffusionPipeline

if __name__ == "__main__":
    
    if not torch.cuda.is_available():
        print("Error: PyTorch not compiled with CUDA support or CUDA unavailable")
        exit(1)

    model_path = "./models/dualdiffusion"
    print(f"Loading DualDiffusion model from '{model_path}'...")
    my_pipeline = DualDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16).to("cuda")
    
    start = time.time(); output = my_pipeline(batch_size=32, steps=50, eta=0.)
    print(f"Time taken: {time.time()-start}")

    existing_output_count = len(glob.glob("./output/*.raw"))
    test_output_path = f"./output/output_{existing_output_count}.raw"
    output = output.cpu().numpy()
    (output / np.max(np.absolute(output))).tofile(test_output_path)
    print(f"Saved output to {test_output_path}")