import os
import time
from dotenv import load_dotenv

import numpy as np
import torch
import torchaudio

from dual_diffusion_pipeline import DualDiffusionPipeline, DualLogFormat, DualNormalFormat, DualOverlappedFormat
from attention_processor_dual import add_embeddings

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
        "sample_format": "embedding",
        "freq_embedding_dim": 128,
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
    #raw_sample = torch.sin(torch.arange(0, crop_width) / 400).unsqueeze(0).to("cuda")
    raw_sample.cpu().numpy().tofile("./debug/debug_raw_original.raw")

    freq_sample, _ = format.raw_to_sample(raw_sample, model_params)
    print("Sample shape:", freq_sample.shape)
    print("Sample mean:", freq_sample.mean(dim=(2,3)), freq_sample.mean())
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
    
    if model_params["freq_embedding_dim"] > 0 or model_params["time_embedding_dim"] > 0:
        freq_sample = DualDiffusionPipeline.add_embeddings(freq_sample,
                                                        model_params["freq_embedding_dim"],
                                                        model_params["time_embedding_dim"],
                                                        model_params["sample_format"])
        print("Sample shape (with freq embedding):", freq_sample.shape)
        print("Sample mean (with freq embedding):", freq_sample.mean().item())
        print("Sample std (with freq embedding):", freq_sample.std().item())
        freq_sample.cpu().numpy().tofile("./debug/debug_sample_with_freq_embedding.raw")
    
    exit()

def get_embedding_response(query_embed, key_embed, exp_scale):
    response = (query_embed * key_embed).sum(dim=(0))
    response -= response.max()
    response = (response*exp_scale).exp()
    return response / response.max()

def get_query(query_embed, weight):
    return (weight.view(1, -1) * query_embed).sum(dim=1).view(-1, 1)

def embedding_test():
    base_n_channels = 128
    freq_embedding_dim = 64
    time_embedding_dim = 64
    sample_resolution_freq = 256
    sample_resolution_time = 256
    freq_exp_scale = (base_n_channels + freq_embedding_dim)**-0.5 # / 2
    time_exp_scale = (base_n_channels + time_embedding_dim)**-0.5 # / 2

    sample = torch.zeros((1, base_n_channels, sample_resolution_freq, sample_resolution_time), dtype=torch.float32)
    
    #sample = add_embeddings(sample, freq_embedding_dim, time_embedding_dim)
    sample = DualDiffusionPipeline.add_embeddings(sample, freq_embedding_dim, time_embedding_dim)

    freq_embed = sample[0, base_n_channels:base_n_channels+freq_embedding_dim, :, 0]
    time_embed = sample[0, base_n_channels+freq_embedding_dim:, 0, :]

    def g(dim, x, std):
        x = torch.linspace(-1, 1, dim) - x
        w = torch.exp(-0.5*(x/std)**2)
        return w / w.max()
    
    def lg(dim, x, std):
        x = torch.linspace(0, 1, dim) / x
        w = torch.exp(-0.5*(torch.log2(x)/std)**2)
        return w / w.max()
    
    #freq_test_weight = lg(sample_resolution_freq, 0.4, 0.05)
    #freq_test_weight += lg(sample_resolution_freq, 0.2, 0.05)
    #freq_test_weight += lg(sample_resolution_freq, 0.1, 0.05)

    freq_test_weight = lg(sample_resolution_freq, 0.1, 0.02)# /0.1
    freq_test_weight += lg(sample_resolution_freq, 0.24, 0.02) #/ 0.24
    freq_test_weight += lg(sample_resolution_freq, 0.63, 0.02)# / 0.63
    freq_test_weight /= torch.arange(1, len(freq_test_weight)+1e-3)

    freq_test_weight /= freq_test_weight.max()
    freq_test_weight.cpu().numpy().tofile("./debug/debug_embed_freq_weight.raw")

    freq_query = get_query(freq_embed, freq_test_weight)
    freq_response = get_embedding_response(freq_query, freq_embed, freq_exp_scale)
    freq_response.cpu().numpy().tofile("./debug/debug_embed_freq_response.raw")
    
    time_test_weight = g(sample_resolution_time, -0.5, 0.015)
    time_test_weight += g(sample_resolution_time, -0.3, 0.015)
    time_test_weight += g(sample_resolution_time, -0.1, 0.015)
    time_test_weight += g(sample_resolution_time, 0., 0.015)
    time_test_weight += g(sample_resolution_time, 0.3, 0.015)
    time_test_weight /= time_test_weight.max()
    time_test_weight.cpu().numpy().tofile("./debug/debug_embed_time_weight.raw")

    time_query = get_query(time_embed, time_test_weight)
    time_response = get_embedding_response(time_query, time_embed, time_exp_scale)
    time_response.cpu().numpy().tofile("./debug/debug_embed_time_response.raw")

    print("freq response error:", (freq_response - freq_test_weight).square().mean().item())
    print("time response error:", (time_response - time_test_weight).square().mean().item())
    #print("freq response error:", (freq_response.log() - freq_test_weight.log()).square().mean().item())
    #print("time response error:", (time_response.log() - time_test_weight.log()).square().mean().item())
    exit()


if __name__ == "__main__":

    if not torch.cuda.is_available():
        print("Error: PyTorch not compiled with CUDA support or CUDA unavailable")
        exit(1)
    else:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cuda.cufft_plan_cache[0].max_size = 32 # stupid cufft memory leak

    load_dotenv()

    #reconstruction_test(sample_num=200)
    #get_dataset_stats(DualOverlappedFormat)
    embedding_test()

    model_name = "dualdiffusion2d_302"
    #model_name = "dualdiffusion2d_118"
    num_samples = 1
    batch_size = 1
    length = 1
    scheduler = "dpms++"
    #scheduler = "ddim"
    #scheduler = "kdpm2_a"
    #scheduler = "euler_a"
    #scheduler = "dpms++_sde"
    steps = 999#337 #250
    loops = 1
    fp16 = False
    #fp16 = True
    
    seed = np.random.randint(10000, 99999-num_samples)
    #seed = 1000

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