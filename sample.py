# MIT License
#
# Copyright (c) 2023 Christopher Friesen
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import time
from dotenv import load_dotenv

import numpy as np
import torch
import torchaudio
import json

from dual_diffusion_pipeline import DualDiffusionPipeline, DualMCLTFormat
from attention_processor_dual import SeparableAttnProcessor2_0
from autoencoder_kl_dual import AutoencoderKLDual
from dual_diffusion_utils import save_flac, save_raw, load_raw

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
    
    if format == DualMCLTFormat:
        pos_examples = 0.
        neg_examples = 0.
    else:
        sample_std = 0.
    num_samples = 0
    
    dataset_path = os.environ.get("DATASET_PATH", "./dataset/samples")
    sample_list = os.listdir(dataset_path)
    for filename in sample_list:
        if filename.endswith(".raw"):
            raw_sample = np.fromfile(os.path.join(dataset_path, filename), dtype=np.int16, count=crop_width) / 32768.
            raw_sample = torch.from_numpy(raw_sample.astype(np.float32)).unsqueeze(0).to("cuda")
            
            sample = format.raw_to_sample(raw_sample, model_params)

            if format == DualMCLTFormat:
                sample_abs = sample[:, 0, :, :]
                sample_abs /= sample_abs.amax(dim=(1,2), keepdim=True)
                pos_examples += sample_abs.sum().item()
                neg_examples += (1-sample_abs).sum().item()
            else:
                sample_std += sample.std().item()

            num_samples += 1
            if num_samples % 100 == 0:
                print(f"Processed {num_samples}/{len(sample_list)} samples")

    if format == DualMCLTFormat:
        print(f"pos_examples: {pos_examples}")
        print(f"neg_examples: {neg_examples}")
        print(f"total samples processed: {num_samples}")
    else:
        sample_std /= num_samples
        print(f"sample_std: {sample_std}")
        print(f"total samples processed: {num_samples}")
        
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
    
    print("freq_embed_std: ", freq_embed.std().item(), "freq_embed_mean: ", freq_embed.mean().item())
    print("time_embed_std: ", time_embed.std().item(), "time_embed_mean: ", time_embed.mean().item())
    print("")

    def g(dim, x, std):
        x = torch.linspace(-1, 1, dim) - x
        w = torch.exp(-0.5*(x/std)**2)
        return w / w.square().sum() ** 0.5
    
    def lg(dim, x, std):
        x = torch.linspace(0, 1, dim) / x
        w = torch.exp(-0.5*(torch.log2(x)/std)**2)
        return w / w.square().sum() ** 0.5
    
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

    model_name = "dualdiffusion2d_600_mclt_4vae_15"
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
    test_samples = np.random.choice(os.listdir(dataset_path), num_samples, replace=False)
    
    #test_samples = ["27705.raw"] # extremely heavy noise
    #test_samples = ["26431.raw"] 
    #test_samples = ["34000.raw"] 
    test_samples = ["29235.raw"] 

    # try to use most recent checkpoint if one exists
    vae_checkpoints = [f for f in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, f)) and f.startswith("vae_checkpoint")]
    if len(vae_checkpoints) > 0:
        vae_checkpoints = sorted(vae_checkpoints, key=lambda x: int(x.split("-")[1]))
        model_path = os.path.join(model_path, vae_checkpoints[-1])

    vae_path = os.path.join(model_path, "vae")
    model_dtype = torch.float16 if fp16 else torch.float32
    vae = AutoencoderKLDual.from_pretrained(vae_path, torch_dtype=model_dtype).to(device)
    last_global_step = vae.config["last_global_step"]

    for filename in test_samples:
        input_raw_sample = load_raw(os.path.join(dataset_path, filename), dtype=np.int16, count=crop_width)
        input_raw_sample = input_raw_sample.unsqueeze(0).to(device)
        input_sample = format.raw_to_sample(input_raw_sample, model_params)

        posterior = vae.encode(input_sample.type(model_dtype), return_dict=False)[0]
        latents = posterior.sample()
        #latents -= latents.mean(dim=(1,2,3), keepdim=True)
        #latents /= latents.std(dim=(1,2,3), keepdim=True)
        output_sample = vae.decode(latents, return_dict=False)[0]
        output_raw_sample = format.sample_to_raw(output_sample.type(torch.float32), model_params)

        output_latents_file_path = os.path.join(output_path, f"step_{last_global_step}_{filename.replace('.raw', '_latents.raw')}")
        save_raw(latents, output_latents_file_path)
        output_sample_file_path = os.path.join(output_path, f"step_{last_global_step}_{filename.replace('.raw', '_sample.raw')}")
        save_raw(output_sample, output_sample_file_path)
        output_posterior_file_path = os.path.join(output_path, f"step_{last_global_step}_{filename.replace('.raw', '_posterior.raw')}")
        save_raw(posterior.parameters, output_posterior_file_path)

        original_raw_sample = format.raw_to_sample(input_raw_sample, model_params, return_dict=True)["raw_samples"]
        output_flac_file_path = os.path.join(output_path, f"step_{last_global_step}_{filename.replace('.raw', '_original.flac')}")
        save_flac(original_raw_sample, sample_rate, output_flac_file_path)
        print(f"Saved flac output to {output_flac_file_path}")

        output_flac_file_path = os.path.join(output_path, f"step_{last_global_step}_{filename.replace('.raw', '_decoded.flac')}")
        save_flac(output_raw_sample, sample_rate, output_flac_file_path)
        print(f"Saved flac output to {output_flac_file_path}")

    exit()

if __name__ == "__main__":

    if not torch.cuda.is_available():
        print("Error: PyTorch not compiled with CUDA support or CUDA unavailable")
        exit(1)
    else:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cuda.cufft_plan_cache[0].max_size = 250 # stupid cufft memory leak

    load_dotenv()

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