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
import glob

import torch
import numpy as np

from modules.embeddings.clap import CLAP_Embedding
from pipelines.dual_diffusion_pipeline import DualDiffusionPipeline, SampleParams
from modules.unets.unet_edm2_q7_ddec_f9 import UNet
from modules.embeddings.clap import CLAP_Embedding
from modules.daes.dae_edm2_q7 import DAE
from modules.formats.ms_mdct_dual_9 import MS_MDCT_DualFormat
from modules.mp_tools import mp_sum
from utils.dual_diffusion_utils import (
    init_cuda, normalize, save_audio, load_audio, load_safetensors,
    save_img, get_audio_info, dict_str, tensor_to_img
)


@torch.inference_mode()
def dae_test() -> None:
    
    test_params = config.load_json(
        os.path.join(config.CONFIG_PATH, "tests", "dae_mdct_test.json"))
    
    torch.manual_seed(0)
    if test_params["random_test_samples_seed"] is not None:
        random.seed(test_params["random_test_samples_seed"])

    if test_params.get("models_path_override", None) is not None:
        config.MODELS_PATH = test_params["models_path_override"]
        
    model_name = test_params["model_name"]
    model_load_options = test_params["model_load_options"]
    length = test_params["length"]

    model_path = os.path.join(config.MODELS_PATH, model_name)
    print(f"Loading DualDiffusion model from '{model_path}'...")
    pipeline = DualDiffusionPipeline.from_pretrained(model_path, **model_load_options)
    dae: DAE = getattr(pipeline, "dae", None)
    ddecm: UNet = getattr(pipeline, "ddecm", None)
    ddecp: UNet = getattr(pipeline, "ddecp", None)
    format: MS_MDCT_DualFormat = pipeline.format
    embedding: CLAP_Embedding = pipeline.embedding
    
    print("latents ch mean:", dae.latents_stats_tracker.mean)
    print("latents ch var:", dae.latents_stats_tracker.var)

    sample_rate = format.config.sample_rate
    
    if test_params.get("ddec_output", False) != True:
        ddecm = ddecp = None
    if ddecp is not None and ddecp.config.last_global_step == 0:
        ddecp = None
    if ddecm is not None and ddecm.config.last_global_step == 0:
        ddecm = None

    if ddecm is None and ddecp is None:
        last_global_step = dae.config.last_global_step
        output_path = os.path.join(model_path, "output", "dae", f"step_{last_global_step}")
    else:
        if ddecm is not None: last_global_step = ddecm.config.last_global_step
        else: last_global_step = 0
        if ddecp is not None:
            last_global_step = last_global_step or ddecp.config.last_global_step
        
        output_path = os.path.join(model_path, "output", "ddec", f"step_{last_global_step}")

    #last_global_step = dae.config.last_global_step
    #output_path = os.path.join(model_path, "output", "dae", f"step_{last_global_step}")
    os.makedirs(output_path, exist_ok=True)

    model_metadata = {"model_metadata": dict_str(pipeline.model_metadata)}
    print(f"{model_metadata['model_metadata']}\n")

    dataset_path = config.DATASET_PATH
    test_samples: list[str] = test_params["test_samples"] or []
    length = length or format.config.default_raw_length
    if length >= 0:
        sample_shape = pipeline.get_mel_spec_shape(raw_length=length)
        latent_shape = pipeline.get_latent_shape(sample_shape)
        print(f"Sample shape: {sample_shape}  Latent shape: {latent_shape}")
    else:
        test_params["latents_img_save_collage"] = False

    if test_params.get("latents_img_use_pca", None) is not None:
        dae.config.latents_img_use_pca = test_params["latents_img_use_pca"]
        dae.config.latents_img_split_stereo = False

    if test_params.get("latents_img_split_stereo", None) is not None:
        dae.config.latents_img_split_stereo = test_params["latents_img_split_stereo"]
    
    if test_params.get("ms_img_show_center_channel", None) is not None:
        format.config.ms_img_show_center_channel = test_params["ms_img_show_center_channel"]
    
    start_time = datetime.datetime.now()
    avg_latents_mean = avg_latents_var = 0
    collage_img = None

    # if the test sample path is a directory, instead add all the flac files in that directory
    expanded_test_samples = []
    for filename in test_samples:
        root = config.DATASET_PATH
        audio_full_path = os.path.join(config.DATASET_PATH, filename)
        if os.path.isdir(audio_full_path) == False:
            root = config.DEBUG_PATH
            audio_full_path = os.path.join(config.DEBUG_PATH, filename)
        if os.path.isdir(audio_full_path) == True:
            expanded_test_samples += [os.path.relpath(path, root) for path in glob.glob(f"{audio_full_path}/*.flac")]
        else:
            expanded_test_samples += [filename]
    test_samples = expanded_test_samples

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
        if length >= 0:
            if audio_len < length:
                print(f"WARNING: audio length {audio_len} is shorter than the test length {length}, skipping...")
                continue
            count = format.get_raw_crop_width(raw_length=min(length, audio_len))
        else:
            count = format.get_raw_crop_width(raw_length=audio_len)
        source_raw_sample = load_audio(file_path, count=count)
        input_raw_sample = source_raw_sample.unsqueeze(0).to(format.device)
        input_mel_spec = format.mel_spec.raw_to_mel_spec(input_raw_sample)
        input_mdct_phase_psd = format.raw_to_mdct_phase_psd(input_raw_sample, level=-1)
        input_mdct_phase_psd = format.flatten_mdct_phase_psd(input_mdct_phase_psd)
        input_mdct_psd = format.raw_to_mdct_psd(input_raw_sample, level=0)
        input_ms_psds = format.raw_to_ms_psd(input_raw_sample, level=-1)
        #input_mel_spec = input_ms_psds[-1]

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

        full_filename = filename
        filename = os.path.basename(filename)

        # ***************** dae stage *****************
        if test_params.get("dae_bypass", False) == False:
            dae_input = input_ms_psds
            dae_embedding = dae.get_embeddings(audio_embedding.to(dtype=dae.dtype))

            if test_params["latents_tiled_encode"] == True:
                latents = dae.tiled_encode(dae_input, dae_embedding,
                    max_chunk=test_params["latents_tiled_max_chunk_size"], overlap=test_params["latents_tiled_overlap"])
            else:
                latents = dae.encode(dae_input, dae_embedding)
                latents: torch.Tensor = latents.float()

            if latents is not None:
                latents_mean = latents.mean().item()
                latents_var = latents.var().item()
                avg_latents_mean += latents_mean
                avg_latents_var += latents_var
                print(f"latents mean/var: {latents_mean:.4} {latents_var:.4}")

            if test_params.get("add_latents_noise", None) is not None:
                decode_latents = latents + torch.randn_like(latents) * test_params["add_latents_noise"]
            else:
                decode_latents = latents

            ddec_cond = dae.decode(decode_latents.to(dtype=dae.dtype), dae_embedding)
        else:
            latents = None
            ddec_cond = input_ms_psds

        # ***************** ddec stage *****************
        """
        if ddecm is not None:

            ddecm_x_ref = ddec_cond

            ddecm_params = SampleParams(
                seed=5000,
                num_steps=50, length=audio_len, cfg_scale=0, input_perturbation=0.5, input_perturbation_offset=-3,
                use_heun=False, schedule="ln_linear", rho=1, sigma_max=300, sigma_min=0.01, stereo_fix=0
            )

            output_ddecm = pipeline.diffusion_decode(
                ddecm_params, audio_embedding=audio_embedding,
                sample_shape=input_mdct_phase_psd.shape,
                x_ref=ddecm_x_ref, module=ddecm).float()

            ddec_cond = format.flatten_mdct_phase_psd(output_ddecm)
            
        else:
            output_ddecm = None
        """

        if ddecp is not None:
            
            #ddecp_x_ref = [(x + torch.randn_like(x) * 0.3) for x in ddec_cond]
            ddecp_x_ref = ddec_cond
            #ddecp_x_ref = format.ms_psd_to_psd_linear(ddec_cond)

            ddecp_params = SampleParams(
                seed=5000,
                num_steps=50, length=audio_len, cfg_scale=0, input_perturbation=1, input_perturbation_offset=100,
                use_heun=False, schedule="cos", rho=1, sigma_max=1000, sigma_min=0.001, stereo_fix=0
            )

            output_ddecp = pipeline.diffusion_decode(
                ddecp_params, audio_embedding=audio_embedding,
                sample_shape=input_mdct_phase_psd.shape,
                x_ref=ddecp_x_ref, module=ddecp).float()

            decode_level = -1
            if decode_level >= 0:
                output_mdct_phase_psd = format.unflatten_mdct_phase_psd(output_ddecp)[decode_level]
            else:
                output_mdct_phase_psd = format.unflatten_mdct_phase_psd(output_ddecp)
            output_raw = format.mdct_phase_psd_to_raw(output_mdct_phase_psd, level=decode_level)
            
            output_mel_spec = format.mel_spec.raw_to_mel_spec(output_raw)
            #output_mel_spec = format.raw_to_ms_psd(output_raw, level=format.config.num_ms_psds - 1)
            output_mdct_psd = format.raw_to_mdct_psd(output_raw, level=0)
        else:
            output_raw = output_mel_spec = output_ddecp = output_mdct_psd = ddecp_x_ref = decode_level = None
        
        metadata = {**model_metadata}
        metadata["test_config.json"] = dict_str(test_params)
        metadata["source"] = full_filename
        #metadata["ddecm_metadata"] = dict_str(ddecm_params.__dict__) if ddecm is not None else "null"
        metadata["ddecp_metadata"] = dict_str(ddecp_params.__dict__) if ddecp is not None else "null"
        metadata["decode_level"] = str(decode_level)

        if latents is not None:
            align_ref = (input_mdct_psd - input_mdct_psd.amin())
            #latents = torch.cat((latents[:, 0:4], latents[:, 4:]), dim=2)
            #latents = latents + torch.randn_like(latents) * 0.05
            latents_img = dae.latents_to_img(latents, align_ref=align_ref)
            save_img(latents_img, os.path.join(output_path, "1", f"step_{last_global_step}_{filename.replace(file_ext, '_latents.png')}"))

            if test_params.get("latents_img_save_collage", False) == True:
                if collage_img is None:
                    collage_img = latents_img
                else:
                    collage_img = np.concatenate([collage_img, latents_img], axis=0)
        
        #save_img(format.mel_spec.mel_spec_to_img(input_mel_spec), os.path.join(output_path, f"step_{last_global_step}_{filename.replace(file_ext, '_mel_spec_input.png')}"))
        save_img(format.ms_psd_to_img(input_mel_spec), os.path.join(output_path, f"step_{last_global_step}_{filename.replace(file_ext, '_mel_spec_input.png')}"))
        if output_mel_spec is not None:
            output_mel_spec.clip_(input_mel_spec.amin(), input_mel_spec.amax())
            output_mel_spec[0, 0, 0, 0] = input_mel_spec.amin(); output_mel_spec[0, 0, 0, 1] = input_mel_spec.amax()
            #save_img(format.mel_spec.mel_spec_to_img(output_mel_spec), os.path.join(output_path, f"step_{last_global_step}_{filename.replace(file_ext, '_mel_spec_output.png')}"))
            save_img(format.ms_psd_to_img(output_mel_spec), os.path.join(output_path, f"step_{last_global_step}_{filename.replace(file_ext, '_mel_spec_output.png')}"))

        if ddec_cond is not None:
            for i, cond in enumerate(ddec_cond):
                save_img(tensor_to_img(cond, flip_y=True), os.path.join(output_path, "2", f"step_{last_global_step}_{filename.replace(file_ext, f'_ms_psd{i}_output.png')}"))
                save_img(tensor_to_img(input_ms_psds[i], flip_y=True), os.path.join(output_path, "2", f"step_{last_global_step}_{filename.replace(file_ext, f'_ms_psd{i}_input.png')}"))

        if output_mdct_psd is not None:
            save_img(tensor_to_img(input_mdct_psd, flip_y=True),  os.path.join(output_path, f"step_{last_global_step}_{filename.replace(file_ext, '_mdct_psd_input.png')}"))
            save_img(tensor_to_img(output_mdct_psd, flip_y=True), os.path.join(output_path, f"step_{last_global_step}_{filename.replace(file_ext, '_mdct_psd_output.png')}"))

        if output_raw is not None:
            output_flac_file_path = os.path.join(output_path, f"step_{last_global_step}_{filename.replace(file_ext, '_decoded.flac')}")
            save_audio(output_raw, sample_rate, output_flac_file_path, metadata=metadata, target_lufs=test_params["output_lufs"])
            print(f"Saved flac output to {output_flac_file_path}")

        if copy_sample_source_files == True:
            output_flac_file_path = os.path.join(output_path, f"step_{last_global_step}_{filename.replace(file_ext, '_source.flac')}")
            save_audio(source_raw_sample, sample_rate, output_flac_file_path, metadata=metadata, target_lufs=test_params["output_lufs"])
            print(f"Saved flac output to {output_flac_file_path}")

    print(f"\nFinished in: {datetime.datetime.now() - start_time}")
    if dae is not None:
        print(f"Latents avg mean: {avg_latents_mean / len(test_samples)}")
        print(f"Latents avg var: {avg_latents_var / len(test_samples)}")

    if collage_img is not None:
        save_img(collage_img, os.path.join(output_path, "1", f"_step_{last_global_step}_collage.png"))
        print(f"Saved latents collage to {os.path.join(output_path, '1', f'_step_{last_global_step}_collage.png')}")

if __name__ == "__main__":

    init_cuda()
    dae_test()