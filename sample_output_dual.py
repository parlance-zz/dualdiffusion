import time
import glob

import torch
import numpy as np

from dual_diffusion_pipeline import DualDiffusionPipeline

if __name__ == "__main__":
    
    if not torch.cuda.is_available():
        print("Error: PyTorch not compiled with CUDA support or CUDA unavailable")
        exit(1)

    torch.seed()

    model_path = "./models/dualdiffusion_s"
    print(f"Loading DualDiffusion model from '{model_path}'...")
    my_pipeline = DualDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16).to("cuda")

    #noise = np.fromfile("./dataset/dual/01 - front line base.raw", dtype=np.complex64, count=2048*1024)
    #noise = np.fft.fft(noise, norm="ortho")
    #noise = np.exp(np.random.uniform(-np.pi, np.pi, size=noise.shape) * 1j) * np.absolute(noise)
    #noise[len(noise)//2:] = 0.
    #noise = np.fft.ifft(noise, norm="ortho").astype(np.complex64)
    #noise = torch.from_numpy(noise).to("cuda")

    noise_profile = np.fromfile("./dataset/avg_freq_response.raw", dtype=np.float32)
    noise_profile = torch.from_numpy(noise_profile).to("cuda")
    noise = torch.exp(torch.rand(len(noise_profile), device=noise_profile.device) * 1j * 2 * np.pi) * noise_profile
    noise = torch.cat((noise, torch.zeros(len(noise_profile), device=noise_profile.device)))
    #noise[:512] = 0.
    noise = torch.fft.ifft(noise, norm="ortho")

    start = time.time()
    output = my_pipeline(batch_size=64, steps=1000, length=2048 * 1024, noise=noise)
    print(f"Time taken: {time.time()-start}")

    existing_output_count = len(glob.glob("./output/output*.raw"))
    test_output_path = f"./output/output_{existing_output_count}.raw"
    output = output.cpu().numpy()
    (output / np.max(np.absolute(output))).tofile(test_output_path)
    print(f"Saved output to {test_output_path}")