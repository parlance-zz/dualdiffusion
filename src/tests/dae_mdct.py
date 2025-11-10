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
import numpy as np

from modules.embeddings.clap import CLAP_Embedding
from pipelines.dual_diffusion_pipeline import DualDiffusionPipeline, SampleParams
from modules.unets.unet_edm2_p1_ddec import UNet
from modules.embeddings.clap import CLAP_Embedding
from modules.daes.dae_edm2_p1 import DAE
from modules.formats.mdct import MDCT_Format
from utils.dual_diffusion_utils import (
    init_cuda, normalize, save_audio, load_audio, load_safetensors,
    save_img, get_audio_info, dict_str
)


@torch.inference_mode()
def dae_test() -> None:
    
    test_params = config.load_json(
        os.path.join(config.CONFIG_PATH, "tests", "dae_mdct_test.json"))
    
    torch.manual_seed(0)
    if test_params["random_test_samples_seed"] is not None:
        random.seed(test_params["random_test_samples_seed"])

    model_name = test_params["model_name"]
    model_load_options = test_params["model_load_options"]
    length = test_params["length"]

    model_path = os.path.join(config.MODELS_PATH, model_name)
    print(f"Loading DualDiffusion model from '{model_path}'...")
    pipeline = DualDiffusionPipeline.from_pretrained(model_path, **model_load_options)
    dae: DAE = getattr(pipeline, "dae", None)
    ddec_mdct: UNet = getattr(pipeline, "ddec", None)
    format: MDCT_Format = pipeline.format
    embedding: CLAP_Embedding = pipeline.embedding

    sample_rate = format.config.sample_rate
    
    if test_params.get("ddec_output", False) != True:
        ddec_mdct = None
    
    if ddec_mdct is None:
        last_global_step = dae.config.last_global_step
        output_path = os.path.join(model_path, "output", "dae", f"step_{last_global_step}")
    else:
        last_global_step = ddec_mdct.config.last_global_step
        output_path = os.path.join(model_path, "output", "ddec", f"step_{last_global_step}")

    os.makedirs(output_path, exist_ok=True)

    model_metadata = {"model_metadata": dict_str(pipeline.model_metadata)}
    print(f"{model_metadata['model_metadata']}\n")

    dataset_path = config.DATASET_PATH
    test_samples: list[str] = test_params["test_samples"] or []
    length = length or format.config.default_raw_length
    sample_shape = pipeline.get_mel_spec_shape(raw_length=length)
    latent_shape = pipeline.get_latent_shape(sample_shape)
    print(f"Sample shape: {sample_shape}  Latent shape: {latent_shape}")

    if test_params.get("latents_img_use_pca", None) is not None:
        dae.config.latents_img_use_pca = test_params["latents_img_use_pca"]
    
    start_time = datetime.datetime.now()
    avg_latents_mean = avg_latents_std = 0
    collage_img = None

    add_random_test_samples = test_params["add_random_test_samples"]
    if add_random_test_samples > 0:
        train_samples = config.load_json(os.path.join(config.DATASET_PATH, "train.jsonl"))
        test_samples += [sample["file_name"] for sample in random.sample(train_samples, add_random_test_samples)]
    copy_sample_source_files: bool = test_params["copy_sample_source_files"]

    for i, filename in enumerate(test_samples):
        
        print(f"\nfile {i+1}/{len(test_samples)}: {filename}")
        file_ext = os.path.splitext(filename)[1]

        file_path = os.path.join(dataset_path, filename)
        if os.path.isfile(file_path) == False:
            file_path = os.path.join(config.DEBUG_PATH, filename)

        audio_len = get_audio_info(file_path).frames
        if audio_len < length:
            print(f"WARNING: audio length {audio_len} is shorter than the test length {length}, skipping...")
            continue
        count = format.get_raw_crop_width(raw_length=min(length, audio_len))
        source_raw_sample = load_audio(file_path, count=count)
        input_raw_sample = source_raw_sample.unsqueeze(0).to(format.device)
        input_mdct = format.raw_to_mdct(input_raw_sample, random_phase_augmentation=True)

        safetensors_file_name = os.path.join(f"{os.path.splitext(filename)[0]}.safetensors")
        safetensors_file_path = os.path.join(dataset_path, safetensors_file_name)
        if os.path.isfile(safetensors_file_path) == False:
            safetensors_file_path = os.path.join(config.DEBUG_PATH, safetensors_file_name)
        if os.path.isfile(safetensors_file_path):
            latents_dict = load_safetensors(safetensors_file_path)
            audio_embedding = normalize(latents_dict["clap_audio_embeddings"].mean(dim=0, keepdim=True)).float()
        else:
            audio_embedding = normalize(embedding.encode_audio(
                input_raw_sample, sample_rate=format.config.sample_rate).mean(dim=0, keepdim=True)).float()

        filename = os.path.basename(filename)

        # ***************** dae stage *****************

        dae_embedding = dae.get_embeddings(audio_embedding.to(dtype=dae.dtype))

        if test_params["latents_tiled_encode"] == True:
            latents = dae.tiled_encode(input_mdct.to(dtype=dae.dtype), dae_embedding,
                max_chunk=test_params["latents_tiled_max_chunk_size"], overlap=test_params["latents_tiled_overlap"])
        else:
            latents = dae.encode(input_mdct.to(dtype=dae.dtype), dae_embedding).float()
        
        ddec_cond = dae.decode(latents.to(dtype=dae.dtype), dae_embedding).float()

        # ***************** ddec mdct stage ***************** 
        x_ref_mdct = ddec_cond

        if ddec_mdct is not None:
            ddec_mdct_params = SampleParams(
                seed=5000,
                num_steps=100, length=audio_len, cfg_scale=5, input_perturbation=1, input_perturbation_offset=0.3,
                use_heun=False, schedule="linear", rho=7, sigma_max=11, sigma_min=0.0002, stereo_fix=0
            )

            output_ddec_mdct = pipeline.diffusion_decode(
                ddec_mdct_params, audio_embedding=audio_embedding,
                sample_shape=format.get_mdct_shape(raw_length=count),
                x_ref=x_ref_mdct.to(dtype=ddec_mdct.dtype), module=ddec_mdct)
            
            output_raw = format.mdct_to_raw(output_ddec_mdct.float())
        else:
            output_raw = None

        print(f"input_mdct   mean/std: {input_mdct.mean().item():.4} {input_mdct.std().item():.4}")
        print(f"ddec_cond    mean/std: {ddec_cond.mean().item():.4} {ddec_cond.std().item():.4}")
        if latents is not None:
            latents_mean = latents.mean().item()
            latents_std = latents.std().item()
            avg_latents_mean += latents_mean
            avg_latents_std += latents_std
            print(f"latents mean/std: {latents_mean:.4} {latents_std:.4}")
        
        metadata = {**model_metadata}
        metadata["ddec_mdct_metadata"] = dict_str(ddec_mdct_params.__dict__) if ddec_mdct is not None else "null"

        if latents is not None:
            latents_img = dae.latents_to_img(latents)
            save_img(latents_img, os.path.join(output_path, "1", f"step_{last_global_step}_{filename.replace(file_ext, '_latents.png')}"))

            if test_params.get("latents_img_save_collage", False) == True:
                if collage_img is None:
                    collage_img = latents_img
                else:
                    collage_img = np.concatenate([collage_img, latents_img], axis=0)
        
        if test_params.get("xref_output", False) == True and x_ref_mdct is not None:
            save_img(dae.latents_to_img(x_ref_mdct), os.path.join(output_path, f"step_{last_global_step}_{filename.replace(file_ext, '_x_ref.png')}"))

        #save_img(format.mel_spec_to_img(input_mel_spec), os.path.join(output_path, f"step_{last_global_step}_{filename.replace(file_ext, '_input_mel_spec.png')}"))
        #save_img(format.mel_spec_to_img(ddec_cond), os.path.join(output_path, f"step_{last_global_step}_{filename.replace(file_ext, '_output_mel_spec.png')}"))
        #save_img(format.mel_spec_to_img(ddec_cond - input_mel_spec), os.path.join(output_path, f"step_{last_global_step}_{filename.replace(file_ext, '_error_mel_spec.png')}"))

        if output_raw is not None:
            output_flac_file_path = os.path.join(output_path, f"step_{last_global_step}_{filename.replace(file_ext, '_decoded.flac')}")
            save_audio(output_raw, sample_rate, output_flac_file_path, metadata=metadata, target_lufs=test_params["output_lufs"])
            print(f"Saved flac output to {output_flac_file_path}")

        if copy_sample_source_files == True:
            output_flac_file_path = os.path.join(output_path, f"step_{last_global_step}_{filename.replace(file_ext, '_source.flac')}")
            save_audio(source_raw_sample, sample_rate, output_flac_file_path, target_lufs=test_params["output_lufs"])
            print(f"Saved flac output to {output_flac_file_path}")

    print(f"\nFinished in: {datetime.datetime.now() - start_time}")
    if dae is not None:
        print(f"Latents avg mean: {avg_latents_mean / len(test_samples)}")
        print(f"Latents avg std: {avg_latents_std / len(test_samples)}")

    if collage_img is not None:
        save_img(collage_img, os.path.join(output_path, "1", f"step_{last_global_step}_collage.png"))
        print(f"Saved latents collage to {os.path.join(output_path, '1', f'_step_{last_global_step}_collage.png')}")

if __name__ == "__main__":

    init_cuda()
    dae_test()