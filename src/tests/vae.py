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

import torch

from pipelines.dual_diffusion_pipeline import DualDiffusionPipeline
from utils.dual_diffusion_utils import (
    init_cuda, normalize, save_audio, load_audio,
    tensor_to_img, save_img, quantize_tensor, dequantize_tensor
)

@torch.inference_mode()
def vae_test() -> None:

    torch.manual_seed(0)

    test_params = config.load_json(
        os.path.join(config.CONFIG_PATH, "tests", "vae_test.json"))
    
    model_name = test_params["model_name"]
    model_load_options = test_params["model_load_options"]
    length = test_params["length"]
    num_fgla_iters = test_params["num_fgla_iters"]
    sample_latents = test_params["sample_latents"]
    normalize_latents = test_params["normalize_latents"]
    random_latents = test_params["random_latents"]
    quantize_latents = test_params["quantize_latents"]
    add_latent_noise = test_params["add_latent_noise"]

    model_path = os.path.join(config.MODELS_PATH, model_name)
    print(f"Loading DualDiffusion model from '{model_path}'...")
    pipeline = DualDiffusionPipeline.from_pretrained(model_path, **model_load_options)
    
    pipeline.format.config.num_fgla_iters = num_fgla_iters
    noise_floor = pipeline.format.config.noise_floor
    sample_rate = pipeline.format.config.sample_rate
    crop_width = pipeline.format.sample_raw_crop_width(length=length)
    last_global_step = pipeline.vae.config.last_global_step

    dataset_path = config.DATASET_PATH
    test_samples = test_params["test_samples"]
    sample_shape = pipeline.get_sample_shape(length=length)
    latent_shape = pipeline.get_latent_shape(sample_shape)
    print(f"Sample shape: {sample_shape}  Latent shape: {latent_shape}")
    
    output_path = os.path.join(model_path, "output", "vae", f"step_{last_global_step}")
    os.makedirs(output_path, exist_ok=True)
    start_time = datetime.datetime.now()
    point_similarity = latents_mean = latents_std = 0

    for sample_game_id, filename in test_samples:

        file_ext = os.path.splitext(filename)[1]
        input_raw_sample = load_audio(os.path.join(dataset_path, filename), count=crop_width)
        input_raw_sample = input_raw_sample.unsqueeze(0).to(pipeline.format.device)

        class_labels = pipeline.get_class_labels(sample_game_id)
        vae_class_embeddings = pipeline.vae.get_class_embeddings(class_labels)
        input_sample = pipeline.format.raw_to_sample(input_raw_sample)

        posterior = pipeline.vae.encode(input_sample.type(pipeline.vae.dtype),
                                        vae_class_embeddings, pipeline.format)
        latents = posterior.sample() if sample_latents else posterior.mode()

        if quantize_latents > 0:
            latents, offset_and_range = quantize_tensor(latents, quantize_latents)
            latents = dequantize_tensor(latents, offset_and_range)
        if add_latent_noise > 0:
            latents += torch.rand_like(latents) * add_latent_noise  
        if normalize_latents:
            latents = normalize(latents).float()
        if random_latents:
            latents = torch.randn_like(latents)

        output_sample = pipeline.vae.decode(latents, vae_class_embeddings, pipeline.format)
        output_raw_sample = pipeline.format.sample_to_raw(output_sample.type(torch.float32))

        point_similarity += (output_sample - input_sample).abs().mean().item()
        latents_mean += latents.mean().item()
        latents_std += latents.std().item()

        save_img(tensor_to_img(latents), os.path.join(output_path, f"step_{last_global_step}_{filename.replace(file_ext, '_latents.png')}"))
        save_img(tensor_to_img(input_sample), os.path.join(output_path, f"step_{last_global_step}_{filename.replace(file_ext, '_input_sample.png')}"))
        save_img(tensor_to_img(output_sample), os.path.join(output_path, f"step_{last_global_step}_{filename.replace(file_ext, '_output_sample.png')}"))

        input_raw_sample = pipeline.format.sample_to_raw(input_sample)
        output_flac_file_path = os.path.join(output_path, f"step_{last_global_step}_{filename.replace(file_ext, '_original.flac')}")
        save_audio(input_raw_sample.squeeze(0), sample_rate, output_flac_file_path)
        print(f"Saved flac output to {output_flac_file_path}")

        output_flac_file_path = os.path.join(output_path, f"step_{last_global_step}_{filename.replace(file_ext, '_decoded.flac')}")
        save_audio(output_raw_sample.squeeze(0), sample_rate, output_flac_file_path)
        print(f"Saved flac output to {output_flac_file_path}")

        latents_fft = torch.fft.rfft2(latents.float(), norm="ortho").abs().clip(min=noise_floor).log()
        save_img(tensor_to_img(latents_fft), os.path.join(output_path, f"step_{last_global_step}_{filename.replace(file_ext, '_latents_fft_ln_psd.png')}"))

    print(f"Finished in: {datetime.datetime.now() - start_time}")
    print(f"Point similarity: {point_similarity / len(test_samples)}")
    print(f"Latents mean: {latents_mean / len(test_samples)}")
    print(f"Latents std: {latents_std / len(test_samples)}")

if __name__ == "__main__":

    init_cuda()
    vae_test()