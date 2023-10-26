import os
import time
from dotenv import load_dotenv

import numpy as np
import torch
import torchaudio

from dual_diffusion_pipeline import DualDiffusionPipeline, DualLogFormat, DualNormalFormat, DualOverlappedFormat

def get_dataset_stats():
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
    format = DualDiffusionPipeline.get_sample_format(model_params)

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

def reconstruction_test(sample_num=1):

    model_params = {
        "sample_raw_length": 65536*2,
        "num_chunks": 256,
        #"sample_format": "overlapped",
        "sample_format": "normal",
        "freq_embedding_dim": 24,
        #"fftshift": False,
        #"ln_amplitude_floor": -12,
        #"ln_amplitude_mean": -6.1341057,
        #"ln_amplitude_std": 1.66477387,
        #"phase_integral_mean": 0,
        #"phase_integral_std": 0.0212208259651,
        "spatial_window_length": 1024,
        #"sample_std": 0.021220825965105643,
    }
    crop_width = model_params["sample_raw_length"]
    format = DualDiffusionPipeline.get_sample_format(model_params)
    
    raw_sample = np.fromfile(f"./dataset/samples/{sample_num}.raw", dtype=np.int16, count=crop_width) / 32768.
    raw_sample = torch.from_numpy(raw_sample.astype(np.float32)).unsqueeze(0).to("cuda")
    raw_sample.cpu().numpy().tofile("./debug/debug_raw_original.raw")

    freq_sample, _ = format.raw_to_sample(raw_sample, model_params)
    print("Sample shape:", freq_sample.shape)
    print("Sample mean:", freq_sample.mean(dim=(2,3)), freq_sample.mean().item())
    print("Sample std:", freq_sample.std().item())
    freq_sample.cpu().numpy().tofile("./debug/debug_sample.raw")
    
    # you _can_ change the tempo without changing frequency with this sample format, however,
    # for good quality you need to resample nicely (ideally sinc)
    #a = torch.zeros((1, 2, 256, 1024,), device=freq_sample.device, dtype=freq_sample.dtype)
    #a[:, :, :, ::2] = freq_sample
    #a[:, :, :, 1::2] = a[:, :, :, ::2]
    #a[:, :, :, 1:-1:2] += a[:, :, :, 2::2]
    #a[:, :, :, 1::2] /= 2
    #freq_sample = a

    raw_sample = format.sample_to_raw(freq_sample, model_params).real
    raw_sample /= raw_sample.abs().max()
    raw_sample.cpu().numpy().tofile("./debug/debug_reconstruction.raw")
    
    freq_sample = DualDiffusionPipeline.add_freq_embedding(freq_sample,
                                                           model_params["freq_embedding_dim"],
                                                           model_params["sample_format"])
    freq_sample.cpu().numpy().tofile("./debug/debug_sample_with_freq_embedding.raw")
    
    exit()

def embedding_test():
    freq_embedding_dim = 60
    ref_pitch_pos = 64
    ref_time_pos = 64
    exp_scale = 10

    sample = torch.zeros((1, 2, 256, 256), dtype=torch.float32).to("cuda")
    sample = DualDiffusionPipeline.add_freq_embedding(sample, freq_embedding_dim, "normal")

    ref_pitch_embed = sample[:, 2:freq_embedding_dim//2, ref_pitch_pos:ref_pitch_pos+1, :]
    ref_time_embed = sample[:, 2+freq_embedding_dim//2:, :, ref_time_pos:ref_time_pos+1]
    
    sample_pitch_embed = sample[:, 2:freq_embedding_dim//2, :, :]
    sample_time_embed = sample[:, 2+freq_embedding_dim//2:, :, :]

    pitch_response = (ref_pitch_embed * sample_pitch_embed).mean(dim=(0,1,3))
    time_response = (ref_time_embed * sample_time_embed).mean(dim=(0,1,2))
    
    pitch_response -= pitch_response.max()
    time_response -= time_response.max()
    pitch_response = (pitch_response*exp_scale).exp()
    time_response = (time_response*exp_scale).exp()
    pitch_response /= pitch_response.max()
    time_response /= time_response.max()

    print("Pitch response accuracy: ", pitch_response[ref_pitch_pos].item() / pitch_response.sum().item())
    print("Time response accuracy: ", time_response[ref_time_pos].item() / time_response.sum().item())

    pitch_response.cpu().numpy().tofile("./debug/debug_embed_pitch_response.raw")
    time_response.cpu().numpy().tofile("./debug/debug_embed_time_response.raw")

    exit()


if __name__ == "__main__":

    if not torch.cuda.is_available():
        print("Error: PyTorch not compiled with CUDA support or CUDA unavailable")
        exit(1)
    else:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cuda.cufft_plan_cache[0].max_size = 32 # stupid cufft memory leak

    load_dotenv()

    #reconstruction_test(sample_num=100)
    #get_dataset_stats(DualOverlappedFormat)
    #embedding_test()

    model_name = "dualdiffusion2d_135"
    #model_name = "dualdiffusion2d_118"
    num_samples = 3
    batch_size = 1
    length = 1
    scheduler = "dpms++"
    #scheduler = "ddim"
    #scheduler = "kdpm2_a"
    #scheduler = "euler_a"
    #scheduler = "dpms++_sde"
    steps = 250#337 #250
    loops = 1
    fp16 = False
    #fp16 = True
    
    seed = np.random.randint(10000, 99999-num_samples)
    #seed = 2000

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
                          length=length).cpu()
        print(f"Time taken: {time.time()-start}")

        output_path = os.path.join(model_path, "output")
        os.makedirs(output_path, exist_ok=True)

        last_global_step = pipeline.config["model_params"].get("unet_last_global_step", None)
        if last_global_step is None: last_global_step = pipeline.config["model_params"].get("last_global_step", 0)

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