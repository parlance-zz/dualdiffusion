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

from utils import config

import os
import datetime
import random

import torch

from modules.embeddings.clap import CLAP_Embedding
from pipelines.dual_diffusion_pipeline import DualDiffusionPipeline, SampleParams, SampleOutput
from modules.unets.unet_edm2_ddec import DDec_UNet
from modules.daes.dae_edm2_a1 import DualDiffusionDAE_EDM2_A1
from modules.formats.spectrogram import SpectrogramFormat
from utils.dual_diffusion_utils import (
    init_cuda, normalize, save_audio, load_audio, load_safetensors,
    tensor_to_img, save_img, get_audio_info, dict_str, tensor_4d_to_5d
)


@torch.inference_mode()
def dae_test() -> None:

    torch.manual_seed(0)

    test_params = config.load_json(
        os.path.join(config.CONFIG_PATH, "tests", "dae_test.json"))
    
    model_name = test_params["model_name"]
    model_load_options = test_params["model_load_options"]
    length = test_params["length"]
    no_crop = test_params["no_crop"]
    num_fgla_iters = test_params["num_fgla_iters"]

    model_path = os.path.join(config.MODELS_PATH, model_name)
    print(f"Loading DualDiffusion model from '{model_path}'...")
    pipeline = DualDiffusionPipeline.from_pretrained(model_path, **model_load_options)
    dae: DualDiffusionDAE_EDM2_A1 = pipeline.dae#pipeline.vae
    format: SpectrogramFormat = pipeline.format

    format.config.num_fgla_iters = num_fgla_iters
    sample_rate = format.config.sample_rate
    if no_crop == True:
        crop_width = -1
    else:
        crop_width = format.sample_raw_crop_width(length=length)

    if test_params["test_ddec"] == True:
        ddec: DDec_UNet = pipeline.ddec
        last_global_step = ddec.config.last_global_step
    else:
        last_global_step = dae.config.last_global_step
    model_metadata = {"model_metadata": dict_str(pipeline.model_metadata)}
    print(f"{model_metadata['model_metadata']}\n")

    dataset_path = config.DATASET_PATH
    test_samples: list[str] = test_params["test_samples"] or []
    sample_shape = pipeline.get_sample_shape(length=length)
    latent_shape = pipeline.get_latent_shape(sample_shape)
    print(f"Sample shape: {sample_shape}  Latent shape: {latent_shape}")
    
    output_path = os.path.join(model_path, "output", "dae", f"step_{last_global_step}")
    os.makedirs(output_path, exist_ok=True)
    start_time = datetime.datetime.now()
    avg_point_similarity = avg_latents_mean = avg_latents_std = 0

    add_random_test_samples = test_params["add_random_test_samples"]
    if add_random_test_samples > 0:
        train_samples = config.load_json(os.path.join(config.DATASET_PATH, "train.jsonl"))
        test_samples += [sample["file_name"] for sample in random.sample(train_samples, add_random_test_samples)]
    copy_sample_source_files: bool = test_params["copy_sample_source_files"]

    for filename in test_samples:   
        
        print(f"\nfile: {filename}")
        file_ext = os.path.splitext(filename)[1]

        safetensors_file_name = os.path.join(f"{os.path.splitext(filename)[0]}.safetensors")
        latents_dict = load_safetensors(os.path.join(dataset_path, safetensors_file_name))
        audio_embedding = normalize(latents_dict["clap_audio_embeddings"].mean(dim=0, keepdim=True)).float()

        audio_len = get_audio_info(os.path.join(dataset_path, filename)).frames
        source_raw_sample = load_audio(os.path.join(dataset_path, filename), count=min(crop_width, audio_len))
        input_raw_sample = source_raw_sample.unsqueeze(0).to(format.device)
        input_sample = format.raw_to_sample(input_raw_sample)
        
        dae_embedding = dae.get_embeddings(audio_embedding)
        latents = dae.encode(input_sample.to(dtype=dae.dtype), dae_embedding)
        output_sample = dae.decode(latents, dae_embedding)

        if test_params["test_ddec"] == True:
            ddec_params = SampleParams(
                num_steps=30, length=audio_len, cfg_scale=1.5, input_perturbation=0, use_heun=True
            )
            output_sample = pipeline.diffusion_decode(
                ddec_params, audio_embedding=audio_embedding,
                x_ref=output_sample.to(dtype=ddec.dtype), module_name="ddec")

        point_similarity = (output_sample - input_sample).abs().mean().item()

        print(f"input   mean/std: {input_sample.mean().item():.4} {input_sample.std().item():.4}")
        print(f"output  mean/std: {output_sample.mean().item():.4} {output_sample.std().item():.4}")
        print(f"latents mean/std: {latents.mean().item():.4} {latents.std().item():.4}")
        print(f"decoded point similarity: {point_similarity}")
        
        output_raw_sample = format.sample_to_raw(output_sample.type(torch.float32))
        
        latents_mean = latents.mean().item()
        latents_std = latents.std().item()
        avg_point_similarity += point_similarity
        avg_latents_mean += latents_mean
        avg_latents_std += latents_std
        filename = os.path.basename(filename)

        save_img(dae.latents_to_img(latents), os.path.join(output_path, f"step_{last_global_step}_{filename.replace(file_ext, '_latents.png')}"))
        save_img(format.sample_to_img(output_sample), os.path.join(output_path, f"step_{last_global_step}_{filename.replace(file_ext, '_output_sample.png')}"))
        save_img(format.sample_to_img(input_sample), os.path.join(output_path, f"step_{last_global_step}_{filename.replace(file_ext, '_input_sample.png')}"))

        input_raw_sample = format.sample_to_raw(input_sample)
        output_flac_file_path = os.path.join(output_path, f"step_{last_global_step}_{filename.replace(file_ext, '_original.flac')}")
        save_audio(input_raw_sample, sample_rate, output_flac_file_path, target_lufs=None)
        print(f"Saved flac output to {output_flac_file_path}")

        output_flac_file_path = os.path.join(output_path, f"step_{last_global_step}_{filename.replace(file_ext, '_decoded.flac')}")
        save_audio(output_raw_sample, sample_rate, output_flac_file_path, metadata=model_metadata, target_lufs=None)
        print(f"Saved flac output to {output_flac_file_path}")

        if copy_sample_source_files == True:
            output_flac_file_path = os.path.join(output_path, f"step_{last_global_step}_{filename.replace(file_ext, '_source.flac')}")
            save_audio(source_raw_sample, sample_rate, output_flac_file_path, target_lufs=None)
            print(f"Saved flac output to {output_flac_file_path}")
        
        latents_fft = tensor_4d_to_5d(torch.fft.rfft2(latents.float(), norm="ortho").abs().clip(min=1e-6).log(), num_channels=4)
        save_img(tensor_to_img(latents_fft, flip_y=True), os.path.join(output_path, f"step_{last_global_step}_{filename.replace(file_ext, '_latents_fft_ln_psd.png')}"))
        latents_fft.cpu().numpy().tofile(os.path.join(output_path, f"step_{last_global_step}_{filename.replace(file_ext, '_latents_fft_ln_psd.raw')}"))

    print(f"\nFinished in: {datetime.datetime.now() - start_time}")
    print(f"Avg Point similarity: {avg_point_similarity / len(test_samples)}")
    print(f"Latents avg mean: {avg_latents_mean / len(test_samples)}")
    print(f"Latents avg std: {avg_latents_std / len(test_samples)}")

if __name__ == "__main__":

    init_cuda()
    dae_test()