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
from pipelines.dual_diffusion_pipeline import DualDiffusionPipeline, SampleParams
from modules.unets.unet_edm2_ddec_i1 import DDec_UNet_I1
from modules.embeddings.clap import CLAP_Embedding
from modules.daes.dae import DualDiffusionDAE
from modules.formats.mdct import MDCT_Format
from utils.dual_diffusion_utils import (
    init_cuda, normalize, save_audio, load_audio, load_safetensors,
    save_img, get_audio_info, dict_str, save_tensor_raw
)


@torch.inference_mode()
def dae_test() -> None:

    torch.manual_seed(0)

    test_params = config.load_json(
        os.path.join(config.CONFIG_PATH, "tests", "dae_test.json"))
    
    model_name = test_params["model_name"]
    model_load_options = test_params["model_load_options"]
    length = test_params["length"]

    model_path = os.path.join(config.MODELS_PATH, model_name)
    print(f"Loading DualDiffusion model from '{model_path}'...")
    pipeline = DualDiffusionPipeline.from_pretrained(model_path, **model_load_options)
    dae: DualDiffusionDAE = getattr(pipeline, "dae", None)
    ddec: DDec_UNet_I1 = getattr(pipeline, "ddec", None)
    format: MDCT_Format = pipeline.format
    embedding: CLAP_Embedding = pipeline.embedding

    sample_rate = format.config.sample_rate
    
    last_global_step = ddec.config.last_global_step
    output_path = os.path.join(model_path, "output", "ddec", f"step_{last_global_step}")
    os.makedirs(output_path, exist_ok=True)

    model_metadata = {"model_metadata": dict_str(pipeline.model_metadata)}
    print(f"{model_metadata['model_metadata']}\n")

    dataset_path = config.DATASET_PATH
    test_samples: list[str] = test_params["test_samples"] or []
    #sample_shape = pipeline.get_mel_spec_shape(raw_length=length)
    #latent_shape = pipeline.get_latent_shape(sample_shape)
    #print(f"Sample shape: {sample_shape}  Latent shape: {latent_shape}")

    if test_params.get("latents_img_use_pca", None) is not None:
        dae.config.latents_img_use_pca = test_params["latents_img_use_pca"]
    
    if test_params.get("latents_img_flip_stereo", None) is not None:
        dae.config.latents_img_flip_stereo = test_params["latents_img_flip_stereo"]

    start_time = datetime.datetime.now()
    avg_latents_mean = avg_latents_std = 0

    add_random_test_samples = test_params["add_random_test_samples"]
    if add_random_test_samples > 0:
        train_samples = config.load_json(os.path.join(config.DATASET_PATH, "train.jsonl"))
        test_samples += [sample["file_name"] for sample in random.sample(train_samples, add_random_test_samples)]
    copy_sample_source_files: bool = test_params["copy_sample_source_files"]

    for filename in test_samples:   
        
        print(f"\nfile: {filename}")
        file_ext = os.path.splitext(filename)[1]

        file_path = os.path.join(dataset_path, filename)
        if os.path.isfile(file_path) == False:
            file_path = os.path.join(config.DEBUG_PATH, filename)

        audio_len = get_audio_info(file_path).frames
        count = format.get_raw_crop_width(raw_length=min(length, audio_len))
        source_raw_sample = load_audio(file_path, count=count)
        input_mdct = format.raw_to_mdct(source_raw_sample.unsqueeze(0).to(format.device))

        safetensors_file_name = os.path.join(f"{os.path.splitext(filename)[0]}.safetensors")
        safetensors_file_path = os.path.join(dataset_path, safetensors_file_name)
        if os.path.isfile(safetensors_file_path) == False:
            safetensors_file_path = os.path.join(config.DEBUG_PATH, safetensors_file_name)
        if os.path.isfile(safetensors_file_path):
            latents_dict = load_safetensors(safetensors_file_path)
            audio_embedding = normalize(latents_dict["clap_audio_embeddings"].mean(dim=0, keepdim=True)).float()
        else:
            audio_embedding = normalize(embedding.encode_audio(
                input_mdct, sample_rate=format.config.sample_rate).mean(dim=0, keepdim=True)).float()

        filename = os.path.basename(filename)

        # ***************** dae stage *****************
        if test_params["dae_bypass"] == False and dae is not None:
            dae_embedding = dae.get_embeddings(audio_embedding)

            if test_params["latents_tiled_encode"] == True:
                raise NotImplementedError()
                latents = dae.tiled_encode(input_mel_spec.to(dtype=dae.dtype), dae_embedding,
                    max_chunk=test_params["latents_tiled_max_chunk_size"], overlap=test_params["latents_tiled_overlap"])
                
                output_embeddings = dae.decode(latents, dae_embedding)
            else:
                add_latents_noise = None#0.03 #None
                latents, output_embeddings, _ = dae.forward(input_mdct.to(dtype=dae.dtype), dae_embedding, add_latents_noise)

        else:
            latents = None
            output_embeddings = None

        print("latents freq stds:", latents.std(dim=(1,3)))
        print("latents freq means:", latents.mean(dim=3).abs().mean(dim=1))
        latents -= latents.mean(dim=(3), keepdim=True)
        latents /= latents.std(dim=(1,3), keepdim=True)
        
        latents = latents.to(dtype=torch.bfloat16)
        output_embeddings = output_embeddings.to(dtype=torch.bfloat16)


        #for i, embedding in enumerate(output_embeddings):
        #    output_embeddings embedding.to(torch.bfloat16)

        if ddec is not None and test_params["ddec_output"] != False:
            ddec_params = SampleParams(
                seed=5000,
                num_steps=100, length=audio_len, cfg_scale=1.5, input_perturbation=0, input_perturbation_offset=0,
                use_heun=True, schedule="edm2", rho=7, sigma_max=12, sigma_min=0.0001, stereo_fix=0
            )

            print("latents.shape:", latents.shape)
            print("get_mdct_shape:", format.get_mdct_shape(raw_length=count))
            print("input_mdct.shape:", input_mdct.shape)
            print("output_embeddings.shape:", output_embeddings.shape)

            output_ddec_mdct = pipeline.diffusion_decode(
                ddec_params, audio_embedding=output_embeddings,
                sample_shape=format.get_mdct_shape(raw_length=count),
                x_ref=None, module=ddec)
            
            output_raw = format.mdct_to_raw(output_ddec_mdct.float())
        else:
            ddec_params = output_ddec_mdct = ddec = output_raw = None

        print(f"input   mean/std: {input_mdct.mean().item():.4} {input_mdct.std().item():.4}")
        if output_ddec_mdct is not None:
            print(f"output  mean/std: {output_ddec_mdct.mean().item():.4} {output_ddec_mdct.std().item():.4}")
            
        if latents is not None:
            latents_mean = latents.mean().item()
            latents_std = latents.std().item()
            avg_latents_mean += latents_mean
            avg_latents_std += latents_std
            print(f"latents mean/std: {latents_mean:.4} {latents_std:.4}")
        
        metadata = {**model_metadata}
        metadata["ddec_metadata"] = dict_str(ddec_params.__dict__) if ddec is not None else "null"

        if latents is not None:
            save_img(dae.latents_to_img(latents), os.path.join(output_path, f"step_{last_global_step}_{filename.replace(file_ext, '_latents.png')}"))
            save_img(dae.latents_to_img(latents), os.path.join(output_path, "1", f"step_{last_global_step}_{filename.replace(file_ext, '_latents.png')}"))
            save_tensor_raw(latents.float().contiguous(), os.path.join(output_path, f"step_{last_global_step}_{filename.replace(file_ext, '_latents.raw')}"))

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


if __name__ == "__main__":

    init_cuda()
    dae_test()