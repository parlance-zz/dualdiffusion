import os
import time
from dotenv import load_dotenv

import numpy as np
import torch
import torchaudio
import json

from dual_diffusion_pipeline import DualDiffusionPipeline, DualLogFormat, DualNormalFormat, DualOverlappedFormat, DualMDCTFormat, DualMCLTFormat, DualMultiscaleSpectralLoss
from attention_processor_dual import SeparableAttnProcessor2_0
from autoencoder_kl_dual import AutoencoderKLDual

def multiscale_spectral_loss_test():
    torch.manual_seed(200)

    model_params = {
        "sample_raw_length": 65536*2,
        "num_chunks": 256,
        "sample_format": "mdct",
        "complex": True,
        "freq_embedding_dim": 0,
        "time_embedding_dim": 0,
        "multiscale_spectral_loss": {
            #"num_filters": 120,
            #"filter_std": 500,
            #"num_filters": 60,
            #"filter_std": 125,
            #"num_filters": 30,
            #"filter_std": 31.25,
            #"num_filters": 15,
            #"filter_std": 7.8125,
            "num_filters": 8,
            "filter_std": 2,
            "num_orders": 5,
            "num_octaves": 10,
            "max_q": 1,
            "u": 20000,
        }
    }

    sample_num = 23

    format = DualDiffusionPipeline.get_sample_format(model_params)
    crop_width = format.get_sample_crop_width(model_params)

    dataset_path = os.environ.get("DATASET_PATH", "./dataset/samples")
    raw_sample = np.fromfile(os.path.join(dataset_path, f"{sample_num}.raw"), dtype=np.int16, count=crop_width) / 32768.
    raw_sample = torch.from_numpy(raw_sample.astype(np.float32)).unsqueeze(0)

    raw_sample.cpu().numpy().tofile("./debug/debug_raw_original.raw")

    sample, _ = format.raw_to_sample(raw_sample, model_params, random_phase_offset=False)
    recon = format.sample_to_raw(sample, model_params).real

    recon.cpu().numpy().tofile("./debug/debug_reconstruction.raw")

    loss_fn = DualMultiscaleSpectralLoss(model_params, format)
    loss = loss_fn(recon, raw_sample)
    
    print(loss)
    exit()

def get_dataset_stats():
    model_params = {
        "sample_raw_length": 65536*2,
        "num_chunks": 256,
        "sample_format": "mcltbce",
        #"complex": False,
        #"u": 255,
        #"sample_std": 1,
    }
    
    format = DualDiffusionPipeline.get_sample_format(model_params)
    crop_width = format.get_sample_crop_width(model_params)
    
    if format == DualLogFormat:
        ln_amplitude_mean = 0.
        ln_amplitude_std = 0.
        phase_integral_mean = 0.
        phase_integral_std = 0.
    elif format == DualMCLTFormat:
        pos_examples = 0.
        neg_examples = 0.
    else:
        sample_std = 0.
    num_samples = 0
    window = None
    
    dataset_path = os.environ.get("DATASET_PATH", "./dataset/samples")
    sample_list = os.listdir(dataset_path)
    for filename in sample_list:
        if filename.endswith(".raw"):
            raw_sample = np.fromfile(os.path.join(dataset_path, filename), dtype=np.int16, count=crop_width) / 32768.
            raw_sample = torch.from_numpy(raw_sample.astype(np.float32)).unsqueeze(0).to("cuda")
            
            sample, window = format.raw_to_sample(raw_sample, model_params, window)

            if format == DualLogFormat:
                ln_amplitude_mean += sample[:, 0, :, :].mean(dim=(0,1,2)).item()
                ln_amplitude_std += sample[:, 0, :, :].std(dim=(0,1,2)).item()
                phase_integral_mean += sample[:, 1:, :, :].mean(dim=(0,1,2,3)).item()
                phase_integral_std += sample[:, 1:, :, :].std(dim=(0,1,2,3)).item()
            elif format == DualMCLTFormat:
                sample_abs = sample[:, 0, :, :]
                sample_abs /= sample_abs.amax(dim=(1,2), keepdim=True)
                pos_examples += sample_abs.sum().item()
                neg_examples += (1-sample_abs).sum().item()
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
    elif format == DualMCLTFormat:
        print(f"pos_examples: {pos_examples}")
        print(f"neg_examples: {neg_examples}")
        print(f"total samples processed: {num_samples}")
    else:
        sample_std /= num_samples
        print(f"sample_std: {sample_std}")
        print(f"total samples processed: {num_samples}")
        
    exit()

def reconstruction_test(sample_num=1):

    torch.manual_seed(200)

    """
    model_params = {
        "sample_raw_length": 65536*2,
        "num_chunks": 256,
        "sample_format": "time_overlapped",
        "freq_embedding_dim": 0,
        "time_embedding_dim": 0,
        "spatial_window_length": 0,
        "rfft": True,
    }
    """
    """
    model_params = {
        #"sample_raw_length": 65536*4,
        "sample_raw_length": 65536*2,
        "num_chunks": 256,
        "sample_format": "normal",
        "freq_embedding_dim": 0,
        "time_embedding_dim": 0,
        "spatial_window_length": 512,
        "ifft": True,
        #"rfft": True,
    }
    """
    """
    model_params = {
        #"sample_raw_length": 65536*4,
        "sample_raw_length": 65536*2,
        "num_chunks": 128,
        "sample_format": "overlapped",
        "freq_embedding_dim": 0,
        "time_embedding_dim": 0,
        "spatial_window_length": 512,
        "ifft": True,
        "rfft": True,
        "fftshift": False,
    }
    """
    model_params = {
        "sample_raw_length": 65536*2,
        "num_chunks": 256,
        "sample_format": "mcltbce",
        #"complex": True,
        #"u": 255,
        "freq_embedding_dim": 0,
        "time_embedding_dim": 0,
    }

    format = DualDiffusionPipeline.get_sample_format(model_params)
    crop_width = format.get_sample_crop_width(model_params)

    raw_sample = np.fromfile(f"./dataset/samples/{sample_num}.raw", dtype=np.int16, count=crop_width) / 32768.
    raw_sample = torch.from_numpy(raw_sample.astype(np.float32)).unsqueeze(0).to("cuda")
    
    """
    raw_sample = torch.randn((1, crop_width), dtype=torch.float32).to("cuda")
    raw_sample[:, (crop_width)//5:(crop_width)//5*3] = 0
    raw_sample[:, (crop_width)//5*3:] /= 2
    """

    raw_sample.cpu().numpy().tofile("./debug/debug_raw_original.raw")

    freq_sample, _ = format.raw_to_sample(raw_sample, model_params, random_phase_offset=False)
    print("Sample shape:", freq_sample.shape)
    print("Sample mean:", freq_sample.mean(dim=(2,3)), freq_sample.mean())
    print("Sample std:", freq_sample.std().item())
    freq_sample.cpu().numpy().tofile("./debug/debug_sample.raw")
    
    sample_abs = freq_sample[:, 0, :, :]
    sample_phase = freq_sample[:, 1:, :, :]
    np.histogram(sample_abs.cpu().numpy(), bins=256)[0].astype(np.int32).tofile("./debug/debug_histogram_abs.raw")
    np.histogram(sample_phase.cpu().numpy(), bins=256)[0].astype(np.int32).tofile("./debug/debug_histogram_phase.raw")

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
    ln_response = response.clone()
    response = (response*exp_scale).exp()
    return response / response.max(), ln_response

def get_query(query_embed, weight):
    return (weight.view(1, -1) * query_embed).sum(dim=1).view(-1, 1)

def embedding_test():
    base_n_channels = 128
    freq_embedding_dim = 512
    time_embedding_dim = 512
    sample_resolution_freq = 256
    sample_resolution_time = 256
    freq_exp_scale = (base_n_channels + freq_embedding_dim)**-0.5 #* 0.5
    time_exp_scale = (base_n_channels + time_embedding_dim)**-0.5 #* 0.5
    sample_shape = (1, base_n_channels, sample_resolution_freq, sample_resolution_time)

    embeddings = SeparableAttnProcessor2_0.get_embeddings(sample_shape, freq_embedding_dim, time_embedding_dim, dtype=torch.float32, device="cpu")
    freq_embed = embeddings[0, :freq_embedding_dim,  :, 0]
    time_embed = embeddings[0,  freq_embedding_dim:, 0, :]
    
    #sample = torch.zeros(sample_shape)
    #sample = DualDiffusionPipeline.add_embeddings(sample, freq_embedding_dim, time_embedding_dim)
    #freq_embed = sample[0, base_n_channels:base_n_channels+freq_embedding_dim,  :, 0]
    #time_embed = sample[0, base_n_channels+freq_embedding_dim:, 0, :]

    print("freq_embed_std: ", freq_embed.std().item(), "freq_embed_mean: ", freq_embed.mean().item())
    print("time_embed_std: ", time_embed.std().item(), "time_embed_mean: ", time_embed.mean().item())
    #print("combined_std: ", embeddings.std().item(), "combined_mean: ", embeddings.mean().item())
    print("")

    def g(dim, x, std):
        x = torch.linspace(-1, 1, dim) - x
        w = torch.exp(-0.5*(x/std)**2)
        return w / w.square().sum() ** 0.5
        #return w/w.max()
    
    def lg(dim, x, std):
        x = torch.linspace(0, 1, dim) / x
        w = torch.exp(-0.5*(torch.log2(x)/std)**2)
        return w / w.square().sum() ** 0.5
        #return w/w.max()
    
    #freq_test_weight = lg(sample_resolution_freq, 0.4, 0.05)
    #freq_test_weight += lg(sample_resolution_freq, 0.2, 0.05)
    #freq_test_weight += lg(sample_resolution_freq, 0.1, 0.05)

    freq_test_weight  = lg(sample_resolution_freq, 0.1, 0.03)# /0.1
    freq_test_weight += lg(sample_resolution_freq, 0.24, 0.03) #/ 0.24
    freq_test_weight += lg(sample_resolution_freq, 0.63, 0.03)# / 0.63
    #freq_test_weight /= torch.arange(0, len(freq_test_weight)) +1e-5#e-5 # + 1

    freq_test_weight /= freq_test_weight.max()
    freq_test_weight.cpu().numpy().tofile("./debug/debug_embed_freq_weight.raw")

    freq_query = get_query(freq_embed, freq_test_weight)
    freq_response, freq_ln_response = get_embedding_response(freq_query, freq_embed, freq_exp_scale)
    freq_response.cpu().numpy().tofile("./debug/debug_embed_freq_response.raw")
    
    time_test_weight_std = 0.01#0.003
    time_test_weight = g(sample_resolution_time, -0.5, time_test_weight_std)
    time_test_weight += g(sample_resolution_time, -0.3, time_test_weight_std)
    time_test_weight += g(sample_resolution_time, -0.1, time_test_weight_std)
    time_test_weight += g(sample_resolution_time, 0., time_test_weight_std)
    time_test_weight += g(sample_resolution_time, 0.3, time_test_weight_std)
    time_test_weight /= time_test_weight.max()
    time_test_weight.cpu().numpy().tofile("./debug/debug_embed_time_weight.raw")

    time_query = get_query(time_embed, time_test_weight)
    time_query.cpu().numpy().tofile("./debug/debug_embed_time_query.raw")
    time_response, time_ln_response = get_embedding_response(time_query, time_embed, time_exp_scale)
    time_response.cpu().numpy().tofile("./debug/debug_embed_time_response.raw")

    freq_ln_test_weight = freq_test_weight.log()
    freq_nan_mask = torch.isnan(freq_ln_test_weight).logical_or(torch.isinf(freq_ln_test_weight)).logical_or(torch.isneginf(freq_ln_test_weight))
    freq_ln_response[freq_nan_mask] = 0
    freq_ln_test_weight[freq_nan_mask] = 0

    time_ln_test_weight = time_test_weight.log()
    time_nan_mask = torch.isnan(time_ln_test_weight).logical_or(torch.isinf(time_ln_test_weight)).logical_or(torch.isneginf(time_ln_test_weight))
    time_ln_response[time_nan_mask] = 0
    time_ln_test_weight[time_nan_mask] = 0

    print("freq response ln error:", (freq_ln_response - freq_ln_test_weight).mean().item())
    print("time response ln error:", (time_ln_response - time_ln_test_weight).mean().item())
    print("freq response mse error:", (freq_response - freq_test_weight).square().mean().item())
    print("time response mse error:", (time_response - time_test_weight).square().mean().item())
    exit()

def vae_test():

    #dualdiffusion2d_330_mdct_v8_256embed_4vae
    model_name = "dualdiffusion2d_330_mclt_v8_256embed_2vae_mssloss"
    num_samples = 1
    #device = "cuda"
    device = "cpu"
    fp16 = False
    #fp16 = True

    model_path = os.path.join(os.environ.get("MODEL_PATH", "./"), model_name)
    with open(os.path.join(model_path, "model_index.json"), "r") as f:
        model_index = json.load(f)
    model_params = model_index["model_params"]
    sample_rate = model_params["sample_rate"]

    output_path = os.path.join(model_path, "output")
    os.makedirs(output_path, exist_ok=True)

    format = DualDiffusionPipeline.get_sample_format(model_params)
    crop_width = format.get_sample_crop_width(model_params)
    print("Sample shape: ", format.get_sample_shape(model_params))

    dataset_path = os.environ.get("DATASET_PATH", "./dataset/samples")
    test_samples = sorted(os.listdir(dataset_path), key=lambda x: int(x.split(".")[0]))[:num_samples]
    #test_samples = np.random.choice(os.listdir(dataset_path), num_samples, replace=False)
    
    vae_path = os.path.join(model_path, "vae")
    #vae_path = "D:/dualdiffusion/models/dualdiffusion2d_330_overlapped_v8_256embed_16vae/old checkpoints/vae_checkpoint-49400/vae"
    model_dtype = torch.float16 if fp16 else torch.float32
    vae = AutoencoderKLDual.from_pretrained(vae_path, torch_dtype=model_dtype).to(device)
    last_global_step = vae.config["last_global_step"]

    for filename in test_samples:
        raw_sample = np.fromfile(os.path.join(dataset_path, filename), dtype=np.int16, count=crop_width) / 32768.
        raw_sample = torch.from_numpy(raw_sample.astype(np.float32)).unsqueeze(0).to(device)
        sample, window = format.raw_to_sample(raw_sample, model_params)

        latents = vae.encode(sample.type(model_dtype), return_dict=False)[0].sample()
        output = vae.decode(latents, return_dict=False)[0]

        output_sample_file_path = os.path.join(output_path, f"step_{last_global_step}_{filename.replace('.raw', '_sample.raw')}")
        output.detach().type(torch.float32).cpu().numpy().tofile(output_sample_file_path)

        output = format.sample_to_raw(output.type(torch.float32), model_params).real.detach()
        
        output_latents_file_path = os.path.join(output_path, f"step_{last_global_step}_{filename.replace('.raw', '_latents.raw')}")
        latents.detach().type(torch.float32).cpu().numpy().tofile(output_latents_file_path)

        raw_sample /= raw_sample.abs().max()
        output_flac_file_path = os.path.join(output_path, f"step_{last_global_step}_{filename.replace('.raw', '_original.flac')}")
        torchaudio.save(output_flac_file_path, raw_sample.cpu(), sample_rate, bits_per_sample=16)
        print(f"Saved flac output to {output_flac_file_path}")

        output /= output.abs().max()
        output_flac_file_path = os.path.join(output_path, f"step_{last_global_step}_{filename.replace('.raw', '_decoded.flac')}")
        torchaudio.save(output_flac_file_path, output.cpu(), sample_rate, bits_per_sample=16)
        print(f"Saved flac output to {output_flac_file_path}")

    exit()

if __name__ == "__main__":

    if not torch.cuda.is_available():
        print("Error: PyTorch not compiled with CUDA support or CUDA unavailable")
        exit(1)
    else:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cuda.cufft_plan_cache[0].max_size = 32 # stupid cufft memory leak

    load_dotenv()

    #multiscale_spectral_loss_test()
    #reconstruction_test(sample_num=1)
    #get_dataset_stats()
    #embedding_test()
    vae_test()

    model_name = "dualdiffusion2d_330_v9_256embed_vae"
    #model_name = "dualdiffusion2d_118"
    num_samples = 5
    batch_size = 1
    length = 1
    scheduler = "dpms++"
    #scheduler = "ddim"
    #scheduler = "kdpm2_a"
    #scheduler = "euler_a"
    #scheduler = "dpms++_sde"
    steps = 999#337 #250
    loops = 0
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