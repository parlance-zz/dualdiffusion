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
import datetime

from dotenv import load_dotenv
import numpy as np
import torch

from pipelines.dual_diffusion_pipeline import DualDiffusionPipeline
from utils.dual_diffusion_utils import (
    init_cuda, save_audio, save_raw, load_raw,
    load_audio, save_raw_img, quantize_tensor, dequantize_tensor
)
from models.unet_edm2 import normalize


if __name__ == "__main__":

    init_cuda()
    load_dotenv(override=True)
    #np.random.seed(0)

    #os.environ["MODEL_PATH"] = "Z:/dualdiffusion/models"; model_name = "edm2_vae_test6"
    model_name = "edm2_vae_test7_4"
    
    num_samples = 1
    device = "cuda" #"cpu"
    fp16 = True
    start = 0
    length = 32000 * 45
    fgla_iterations = 300 #400
    save_output = True
    sample_latents = True
    normalize_latents = False #True
    random_latents = False #True
    quantize_latents = 0#256
    add_latent_noise = 0#1/32

    model_path = os.path.join(os.environ.get("MODEL_PATH", "./"), model_name)
    model_dtype = torch.bfloat16 if fp16 else torch.float32
    print(f"Loading DualDiffusion model from '{model_path}' (dtype={model_dtype})...")
    pipeline = DualDiffusionPipeline.from_pretrained(model_path,
                                                     torch_dtype=model_dtype,
                                                     load_latest_checkpoints=True,
                                                     device=device)
    pipeline.format.spectrogram_params.num_griffin_lim_iters = fgla_iterations
    model_params = pipeline.config["model_params"]
    crop_width = pipeline.format.get_sample_crop_width(length=length)
    noise_floor = model_params["noise_floor"]
    last_global_step = pipeline.vae.config["last_global_step"]

    dataset_path = os.environ.get("DATASET_PATH", "./dataset/samples")
    dataset_format = os.environ.get("DATASET_FORMAT", ".flac")
    dataset_raw_format = os.environ.get("DATASET_RAW_FORMAT", "int16")
    test_samples = np.random.choice(os.listdir(dataset_path), num_samples, replace=False)

    test_samples = []
    test_samples += [(666, "1/Kirby Super Star  [Kirby's Fun Pak] - 36 Mine Cart Riding.flac")] # success case
    test_samples += [(666, "1/Kirby Super Star  [Kirby's Fun Pak] - 53 Heart of Nova.flac")]
    test_samples += [(666, "1/Kirby Super Star  [Kirby's Fun Pak] - 41 Halberd ~ Nightmare Warship.flac")]
    test_samples += [(1302, "2/Super Mario RPG - The Legend of the Seven Stars - 217 Weapons Factory.flac")]
    test_samples += [(1302, "2/Super Mario RPG - The Legend of the Seven Stars - 135 Welcome to Booster Tower.flac")]
    test_samples += [(1302, "2/Super Mario RPG - The Legend of the Seven Stars - 128 Beware the Forest's Mushrooms.flac")]
    test_samples += [(788, "2/Mega Man X3 - 09 Blast Hornet.flac")]
    test_samples += [(788, "2/Mega Man X3 - 11 Toxic Seahorse.flac")]
    test_samples += [(788, "2/Mega Man X3 - 14 Crush Crawfish.flac")]
    test_samples += [(230, "1/Contra III - The Alien Wars - 05 Neo Kobe Steel Factory.flac")] 
    test_samples += [(230, "1/Contra III - The Alien Wars - 02 Ground Zero.flac")]
    test_samples += [(230, "1/Contra III - The Alien Wars - 06 Road Warriors.flac")]
    test_samples += [(471, "1/Gradius III - 03 Invitation.flac")]
    test_samples += [(471, "1/Gradius III - 04 Departure for Space.flac")]
    test_samples += [(471, "1/Gradius III - 05 Sand Storm.flac")]
    test_samples += [(387, "1/Final Fantasy VI - 104 Locke.flac")]
    test_samples += [(387, "1/Final Fantasy VI - 105 Battle Theme.flac")]
    test_samples += [(387, "1/Final Fantasy VI - 113 Cyan.flac")]
    test_samples += [(387, "1/Final Fantasy VI - 104 Locke.flac")]
    test_samples += [(387, "1/Final Fantasy VI - 215 Blackjack.flac")]
    test_samples += [(413, "1/Front Mission - 23 Arena.flac")]
    test_samples += [(413, "1/Front Mission - 24 Shop.flac")]
    test_samples += [(413, "1/Front Mission - 37 Terrible Density.flac")]
    test_samples += [(1303, "2/Super Mario World - 11a Overworld.flac")]
    test_samples += [(1303, "2/Super Mario World - 12a Athletic.flac")]
    test_samples += [(1303, "2/Super Mario World - 14a Swimming.flac")]
    test_samples += [(788, "2/Mega Man X2 - 09 Panzer des Drachens.flac")]
    test_samples += [(788, "2/Mega Man X2 - 13 Red Alert.flac")]
    test_samples += [(788, "2/Mega Man X2 - 04 The Mavericks' Last Stand.flac")]
    test_samples += [(788, "2/Mega Man X2 - 11 Volcano's Fury.flac")]
    
    sample_shape = pipeline.format.get_sample_shape(length=length)
    print(f"Sample shape: {sample_shape}  Latent shape: {pipeline.vae.get_latent_shape(sample_shape)}")
    
    output_path = os.path.join(model_path, "output", "vae")
    os.makedirs(output_path, exist_ok=True)
    start_time = datetime.datetime.now()
    point_similarity = latents_mean = latents_std = 0

    for sample in test_samples:
        
        sample_game_id = torch.tensor(sample[0], device=pipeline.vae.device, dtype=torch.long)
        filename = sample[1]

        file_ext = os.path.splitext(filename)[1]
        input_raw_sample = load_audio(os.path.join(dataset_path, filename), start=start, count=crop_width)
        input_raw_sample = input_raw_sample.unsqueeze(0).to(device)

        class_labels = pipeline.get_class_labels(sample_game_id)
        vae_class_embeddings = pipeline.vae.get_class_embeddings(class_labels)

        input_sample_dict = pipeline.format.raw_to_sample(input_raw_sample, return_dict=True)
        input_sample = input_sample_dict["samples"]

        posterior = pipeline.vae.encode(input_sample.type(model_dtype), vae_class_embeddings, pipeline.format)
        if sample_latents:
            latents = posterior.sample(torch.randn)
        else:
            latents = posterior.mode()
        if quantize_latents > 0:
            latents, offset_and_range = quantize_tensor(latents, quantize_latents)
            latents = dequantize_tensor(latents, offset_and_range)
        if add_latent_noise > 0:
            latents += torch.rand_like(latents) * add_latent_noise  
        if normalize_latents:
            latents = normalize(latents)
        if random_latents:
            latents = pipeline.noise_fn(latents.shape, dtype=latents.dtype, device=latents.device)
        model_output = pipeline.vae.decode(latents, vae_class_embeddings, pipeline.format)

        output_sample_dict = pipeline.format.sample_to_raw(model_output.type(torch.float32), return_dict=True)
        output_raw_sample = output_sample_dict["raw_samples"]
        output_sample = output_sample_dict["samples"]

        point_similarity += (output_sample - input_sample_dict["samples"]).abs().mean().item()
        latents_mean += latents.mean().item()
        latents_std += latents.std().item()

        if save_output:
            save_raw(latents, os.path.join(output_path,f"step_{last_global_step}_{filename.replace(file_ext, '_latents.raw')}"))
            save_raw_img(latents, os.path.join(output_path, f"step_{last_global_step}_{filename.replace(file_ext, '_latents.png')}"))
            save_raw(input_sample, os.path.join(output_path, f"step_{last_global_step}_{filename.replace(file_ext, '_input_sample.raw')}"))
            save_raw_img(input_sample_dict["samples"], os.path.join(output_path, f"step_{last_global_step}_{filename.replace(file_ext, '_input_sample.png')}"))
            save_raw(output_sample, os.path.join(output_path, f"step_{last_global_step}_{filename.replace(file_ext, '_sample.raw')}"))
            save_raw_img(output_sample, os.path.join(output_path, f"step_{last_global_step}_{filename.replace(file_ext, '_sample.png')}"))
            save_raw(posterior.parameters, os.path.join(output_path, f"step_{last_global_step}_{filename.replace(file_ext, '_posterior.raw')}"))
            save_raw_img(output_sample-input_sample_dict["samples"], os.path.join(output_path, f"step_{last_global_step}_{filename.replace(file_ext, '_sample_error.png')}"))

            original_sample_dict = pipeline.format.sample_to_raw(input_sample, return_dict=True)
            original_raw_sample = original_sample_dict["raw_samples"]
            output_flac_file_path = os.path.join(output_path, f"step_{last_global_step}_{filename.replace(file_ext, '_original.flac')}")
            save_audio(original_raw_sample.squeeze(0), model_params["sample_rate"], output_flac_file_path)
            print(f"Saved flac output to {output_flac_file_path}")

            output_flac_file_path = os.path.join(output_path, f"step_{last_global_step}_{filename.replace(file_ext, '_decoded.flac')}")
            save_audio(output_raw_sample.squeeze(0), model_params["sample_rate"], output_flac_file_path)
            print(f"Saved flac output to {output_flac_file_path}")

            latents_fft = torch.fft.rfft2(latents.float(), norm="ortho").abs().clip(min=noise_floor).log()
            save_raw_img(latents_fft, os.path.join(output_path, f"step_{last_global_step}_{filename.replace(file_ext, '_latents_fft_ln_psd.png')}"))

    print(f"Finished in: {datetime.datetime.now() - start_time}")
    print(f"Point similarity: {point_similarity / len(test_samples)}")
    print(f"Latents mean: {latents_mean / len(test_samples)}")
    print(f"Latents std: {latents_std / len(test_samples)}")