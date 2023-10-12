import os
import time
from dotenv import load_dotenv

import numpy as np
import torch
import torchaudio

from dual_diffusion_pipeline import DualDiffusionPipeline, DualLogFormat, DualNormalFormat, DualOverlappedFormat

def get_dataset_stats(format):
    model_params = {
        "sample_raw_length": 65536,
        "num_chunks": 128,
        "ln_amplitude_mean": 0.,
        "ln_amplitude_std": 1.,
        "phase_integral_mean": 0.,
        "phase_integral_std": 1,
        "sample_std": 1,
        "spatial_window_length": 256,
    }
    crop_width = model_params["sample_raw_length"]
    format = DualOverlappedFormat
    #format = DualLogFormat
    #format = DualNormalFormat

    if format == DualLogFormat:
        ln_amplitude_mean = 0.
        ln_amplitude_std = 0.
        phase_integral_mean = 0.
        phase_integral_std = 0.
    else:
        sample_std = 0.
    num_samples = 0
    window = None
    
    sample_list = os.listdir("./dataset/samples")
    for filename in sample_list:
        if filename.endswith(".raw"):
            raw_sample = np.fromfile(os.path.join("./dataset/samples", filename), dtype=np.int16, count=crop_width) / 32768.
            raw_sample = torch.from_numpy(raw_sample.astype(np.float32)).unsqueeze(0).to("cuda")
            
            sample, window = format.raw_to_sample(raw_sample, model_params, window)

            if format == DualLogFormat:
                ln_amplitude_mean += sample[:, 0, :, :].mean(dim=(0,1,2)).item()
                ln_amplitude_std += sample[:, 0, :, :].std(dim=(0,1,2)).item()
                phase_integral_mean += sample[:, 1:, :, :].mean(dim=(0,1,2,3)).item()
                phase_integral_std += sample[:, 1:, :, :].std(dim=(0,1,2,3)).item()
            else:
                sample_std += sample.std().item()

            num_samples += 1
            if num_samples % 100 == 0:
                print(f"Processed {num_samples}/{len(sample_list)} samples")

    if format == DualLogFormat:
        ln_amplitude_mean /= num_samples
        ln_amplitude_std /= num_samples
        phase_integral_mean /= num_samples
        phase_integral_std /= num_samples
        print(f"ln_amplitude_mean: {ln_amplitude_mean}")
        print(f"ln_amplitude_std: {ln_amplitude_std}")
        print(f"phase_integral_mean: {phase_integral_mean}")
        print(f"phase_integral_std: {phase_integral_std}")
        print(f"total samples processed: {num_samples}")
    else:
        sample_std /= num_samples
        print(f"sample_std: {sample_std}")
        print(f"total samples processed: {num_samples}")
        
    exit()

def reconstruction_test(format, sample_num=1):

    model_params = {
        "sample_raw_length": 65536,
        "num_chunks": 128,
        "fftshift": True,
        #"ln_amplitude_floor": -12,
        #"ln_amplitude_mean": -6.1341057,
        #"ln_amplitude_std": 1.66477387,
        #"phase_integral_mean": 0,
        #"phase_integral_std": 0.0212208259651,
        #"spatial_window_length": 256,
        #"sample_std": 0.021220825965105643,
    }
    crop_width = model_params["sample_raw_length"]

    raw_sample = np.fromfile(f"./dataset/samples/{sample_num}.raw", dtype=np.int16, count=crop_width) / 32768.
    raw_sample = torch.from_numpy(raw_sample.astype(np.float32)).unsqueeze(0).to("cuda")
    raw_sample.cpu().numpy().tofile("./debug/debug_raw_original.raw")

    freq_sample, _ = format.raw_to_sample(raw_sample, model_params)
    print("Sample shape:", freq_sample.shape)
    print("Sample mean:", freq_sample.mean(dim=(2,3)), freq_sample.mean().item())
    print("Sample std:", freq_sample.std().item())
    freq_sample.cpu().numpy().tofile("./debug/debug_sample.raw")

    raw_sample = format.sample_to_raw(freq_sample, model_params).real
    raw_sample /= raw_sample.abs().max()
    raw_sample.cpu().numpy().tofile("./debug/debug_reconstruction.raw")
    
    exit()

if __name__ == "__main__":

    if not torch.cuda.is_available():
        print("Error: PyTorch not compiled with CUDA support or CUDA unavailable")
        exit(1)
    else:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cuda.cufft_plan_cache[0].max_size = 32 # stupid cufft memory leak

    load_dotenv()

    #reconstruction_test(DualOverlappedFormat, sample_num=100)
    #get_dataset_stats(DualOverlappedFormat)

    model_name = "dualdiffusion2d_112"
    num_samples = 7
    batch_size = 1
    length = 1
    scheduler = "dpms++"
    #scheduler = "ddim"
    #scheduler = "kdpm2_a"
    #scheduler = "euler_a"
    steps = 125
    loops = 1
    #fp16 = False
    fp16 = True

    #seed = np.random.randint(10000, 99999-num_samples)
    seed = 100

    model_dtype = torch.float16 if fp16 else torch.float32
    model_path = os.path.join(os.environ.get("MODEL_PATH", "./"), model_name)
    print(f"Loading DualDiffusion model from '{model_path}' (dtype={model_dtype})...")
    pipeline = DualDiffusionPipeline.from_pretrained(model_path, torch_dtype=model_dtype).to("cuda")
    sample_rate = pipeline.config["model_params"]["sample_rate"]

    for i in range(num_samples):
        print(f"Generating sample {i+1}/{num_samples}...")

        start = time.time()
        output = pipeline(steps=steps,
                          scheduler=scheduler,
                          seed=seed,
                          loops=loops,
                          batch_size=batch_size,
                          length=length).real.cpu()
        print(f"Time taken: {time.time()-start}")

        output_path = os.path.join(model_path, "output")
        os.makedirs(output_path, exist_ok=True)

        last_global_step = pipeline.config["model_params"]["last_global_step"]
        output_flac_file_path = os.path.join(output_path, f"step_{last_global_step}_{scheduler}{steps}_s{seed}.flac")
        torchaudio.save(output_flac_file_path, output, sample_rate, bits_per_sample=16)
        print(f"Saved flac output to {output_flac_file_path}")

        seed += 1


"""
def img_test():
    model_params = {
        "sample_raw_length": 65536*2,
        "num_chunks": 128,
    }
    sample = np.fromfile("./dataset/samples/66.raw", dtype=np.int16, count=model_params["sample_raw_length"]) / 32768.
    sample = torch.from_numpy(sample).unsqueeze(0).to("cuda")
    sample, window = DualDiffusionPipeline.raw_to_sample(sample, model_params)
    DualDiffusionPipeline.save_sample_img(sample, "test.png")
    exit()

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

def attention_shaping_test():
    from unet2d_dual_blocks import shape_for_attention, unshape_for_attention

    hidden_states_original = torch.randn((4, 32, 256, 256), dtype=torch.float32).to("cuda")

    for attn_dim in range(2, 3+1):
        hidden_states = hidden_states_original.clone()
        original_shape = hidden_states.shape

        hidden_states = shape_for_attention(hidden_states, attn_dim)
        hidden_states = unshape_for_attention(hidden_states, attn_dim, original_shape)

        assert(torch.equal(hidden_states_original, hidden_states))
    
    exit()
"""