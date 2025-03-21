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

from dataclasses import dataclass
from typing import Optional
import os
import datetime
import random
import glob

import torch

from pipelines.dual_diffusion_pipeline import DualDiffusionPipeline, SampleParams
from modules.unets.unet_edm2_ddec_mclt import DDec_MCLT_UNet
from modules.daes.dae import DualDiffusionDAE
from modules.formats.spectrogram import SpectrogramFormat
from modules.formats.mclt import DualMCLTFormat, DualMCLTFormatConfig
from utils.dual_diffusion_utils import (
    init_cuda, normalize, save_audio, load_audio, load_safetensors,
    save_img, dict_str, get_no_clobber_filepath, get_audio_metadata,
    tensor_to_img, tensor_info_str
)


@dataclass
class UNetTestConfig:

    model_name: str
    model_load_options: dict
    
    unet_params: SampleParams
    ddec_params: SampleParams

    output_lufs: float  = -16
    num_fgla_iters: int = 300
    skip_ddec: bool     = False

    copy_sample_source_files: bool          = True
    add_random_test_samples: int            = 0
    random_test_samples_seed: Optional[int] = None
    test_samples: Optional[list[int]]       = None

@torch.inference_mode()
def unet_test() -> None:

    torch.manual_seed(0)
    #os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:8192"

    cfg: UNetTestConfig = config.load_config(UNetTestConfig,
        os.path.join(config.CONFIG_PATH, "tests", "unet_test.json"), quiet=True)

    model_path = os.path.join(config.MODELS_PATH, cfg.model_name)
    print(f"Loading DualDiffusion model from '{model_path}'...")
    pipeline = DualDiffusionPipeline.from_pretrained(model_path, **cfg.model_load_options)
    dae: DualDiffusionDAE = getattr(pipeline, "dae", None)
    ddec: DDec_MCLT_UNet = pipeline.ddec
    format: SpectrogramFormat = pipeline.format
    mclt = DualMCLTFormat(DualMCLTFormatConfig())

    ddec.compile(fullgraph=True, dynamic=False)
    ddec.to(memory_format=torch.channels_last)
    pipeline.unet.compile(fullgraph=True, dynamic=False)
    pipeline.unet.to(memory_format=torch.channels_last)

    sample_rate = format.config.sample_rate
    crop_width = format.sample_raw_crop_width(length=cfg.unet_params.length)
    last_global_step = pipeline.unet.config.last_global_step

    if cfg.random_test_samples_seed is None:
        cfg.random_test_samples_seed = random.randint(1000, 9999)
    random.seed(cfg.random_test_samples_seed)
    print(f"Using random test samples seed: {cfg.random_test_samples_seed}")
    base_seed = cfg.unet_params.seed or cfg.random_test_samples_seed * 10

    model_metadata = {"model_metadata": dict_str(pipeline.model_metadata)}
    print(f"{model_metadata['model_metadata']}\n")

    cfg.test_samples = cfg.test_samples or []
    sample_shape = pipeline.get_sample_shape(length=cfg.unet_params.length)
    latent_shape = pipeline.get_latent_shape(sample_shape)
    print(f"Sample shape: {sample_shape}  Latent shape: {latent_shape}")
    print("UNet Params: ", dict_str(cfg.unet_params.__dict__))
    print("DDec Params: ", dict_str(cfg.ddec_params.__dict__))

    output_path = os.path.join(model_path, "output", f"step_{last_global_step}")
    os.makedirs(output_path, exist_ok=True)
    start_time = datetime.datetime.now()
    avg_latents_mean = avg_latents_std = 0

    # if the test sample path is a directory, instead add all the flac files in that directory
    expanded_test_samples = []
    for filename in cfg.test_samples:
        root = config.DATASET_PATH
        audio_full_path = os.path.join(config.DATASET_PATH, filename)
        if os.path.isdir(audio_full_path) == False:
            root = config.DEBUG_PATH
            audio_full_path = os.path.join(config.DEBUG_PATH, filename)
        if os.path.isdir(audio_full_path) == True:
            expanded_test_samples += [os.path.relpath(path, root) for path in glob.glob(f"{audio_full_path}/*.flac")]
        else:
            expanded_test_samples += [filename]
    cfg.test_samples = expanded_test_samples
    
    # add random test samples from the dataset train split
    if cfg.add_random_test_samples > 0:
        train_samples = config.load_json(os.path.join(config.DATASET_PATH, "train.jsonl"))
        cfg.test_samples += [sample["file_name"] for sample in random.sample(train_samples, cfg.add_random_test_samples)]

    for i, filename in enumerate(cfg.test_samples):
        
        print(f"\nfile: {filename}")

        safetensors_file_name = os.path.join(f"{os.path.splitext(filename)[0]}.safetensors")
        safetensors_full_path = os.path.join(config.DATASET_PATH, safetensors_file_name)
        if os.path.isfile(safetensors_full_path):
            latents_dict = load_safetensors(safetensors_full_path)
            clap_audio_embeddings = latents_dict["clap_audio_embeddings"]
        else:
            latents_dict = None
            clap_audio_embeddings = None

        audio_full_path = os.path.join(config.DATASET_PATH, filename)
        if os.path.isfile(audio_full_path) == False:
            audio_full_path = os.path.join(config.DEBUG_PATH, filename)
        if os.path.isfile(audio_full_path) == False:
            print(f"Error: Could not find {filename}, skipping...")
            continue
        input_audio, input_sample_rate = load_audio(audio_full_path, return_sample_rate=True, device=pipeline.embedding.device)
        input_audio_metadata = get_audio_metadata(audio_full_path)
        if clap_audio_embeddings is None:
            clap_audio_embeddings = pipeline.embedding.encode_audio(input_audio, sample_rate=input_sample_rate)
        input_sample = format.raw_to_sample(input_audio[:, :crop_width])

        audio_embedding = normalize(clap_audio_embeddings.mean(dim=0, keepdim=True)).float()
        if dae is not None:
            dae_embeddings = dae.get_embeddings(audio_embedding.to(dtype=dae.dtype, device=dae.device))
            sample_shape = latent_shape
        else:
            dae_embeddings = None

        cfg.unet_params.seed = base_seed + i
        cfg.unet_params.prompt = filename
        
        unet_output = pipeline.diffusion_decode(cfg.unet_params,
            audio_embedding=audio_embedding, module=pipeline.unet).float()
        if dae is not None:
            raise NotImplementedError("DAE not implemented")
            latents = unet_output
            output_sample = dae.decode(latents.to(dtype=dae.dtype), dae_embeddings)
        else:
            latents = None
            output_sample = unet_output
        
        if cfg.skip_ddec == False:
            x_ref = format.convert_to_abs_exp1(output_sample)
            x_ref = x_ref.to(dtype=ddec.dtype)

            print(f"x_ref   mean/std: {x_ref.mean().item():.4} {x_ref.std().item():.4}")

            cfg.ddec_params.seed = cfg.unet_params.seed
            output_mclt_sample = pipeline.diffusion_decode(cfg.ddec_params,
                audio_embedding=audio_embedding, x_ref=x_ref, module_name="ddec")
            
            output_raw_sample = mclt.sample_to_raw(output_mclt_sample.float())
        else:
            output_raw_sample = format.sample_to_raw(output_sample.float(), n_fgla_iters=cfg.num_fgla_iters)

        print(f"input   mean/std: {input_sample.mean().item():.4} {input_sample.std().item():.4}")
        print(f"output  mean/std: {output_sample.mean().item():.4} {output_sample.std().item():.4}")
        
        output_label = cfg.unet_params.get_label(pipeline.model_metadata)

        if latents is not None:
            print(f"latents mean/std: {latents.mean().item():.4} {latents.std().item():.4}")
            
            latents_mean = latents.mean().item()
            latents_std = latents.std().item()
            avg_latents_mean += latents_mean
            avg_latents_std += latents_std 

            if dae is not None:
                latents_img = dae.latents_to_img(latents)
            else:
                latents_img = tensor_to_img(latents, flip_y=True)
            output_latents_file_path = os.path.join(output_path, f"{output_label}_output_latents.png")
            output_latents_file_path = get_no_clobber_filepath(output_latents_file_path)
            save_img(latents_img, output_latents_file_path)

            input_latents = latents_dict["latents"][0:1, ..., :latents.shape[-1]] if latents_dict is not None else None
            if input_latents is not None and dae is not None:
                output_latents_file_path = os.path.join(output_path, f"{output_label}_input_latents.png")
                output_latents_file_path = get_no_clobber_filepath(output_latents_file_path)
                save_img(dae.latents_to_img(input_latents), output_latents_file_path)
                
        output_sample_file_path = os.path.join(output_path, f"{output_label}_output_sample.png")
        output_sample_file_path = get_no_clobber_filepath(output_sample_file_path)
        save_img(format.sample_to_img(output_sample), output_sample_file_path)

        output_sample_file_path = os.path.join(output_path, f"{output_label}_input_sample.png")
        output_sample_file_path = get_no_clobber_filepath(output_sample_file_path)
        save_img(format.sample_to_img(input_sample), output_sample_file_path)

        metadata = {**model_metadata, "diffusion_metadata": dict_str(cfg.unet_params.__dict__)}
        metadata["ddec_metadata"] = dict_str(cfg.ddec_params.__dict__) if cfg.skip_ddec == False else "null"

        output_flac_file_path = os.path.join(output_path, f"{output_label}.flac")
        output_flac_file_path = get_no_clobber_filepath(output_flac_file_path)
        save_audio(output_raw_sample, sample_rate, output_flac_file_path, metadata=metadata, target_lufs=cfg.output_lufs)
        print(f"Saved flac output to {output_flac_file_path}")

        if cfg.copy_sample_source_files == True:
            output_flac_file_path = os.path.join(output_path, f"{output_label}_input_prompt.flac")
            save_audio(input_audio, input_sample_rate, output_flac_file_path,
                       target_lufs=cfg.output_lufs, metadata=input_audio_metadata)
            print(f"Saved flac output to {output_flac_file_path}")

    print(f"\nFinished in: {datetime.datetime.now() - start_time}")
    print(f"Latents avg mean: {avg_latents_mean / len(cfg.test_samples)}")
    print(f"Latents avg std: {avg_latents_std / len(cfg.test_samples)}")

if __name__ == "__main__":

    init_cuda()
    unet_test()