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
import time
import datetime
import argparse

import numpy as np
import torch

from pipelines.dual_diffusion_pipeline import DualDiffusionPipeline, SampleParams
from utils.dual_diffusion_utils import init_cuda, save_audio, load_audio, dict_str


@torch.inference_mode()
def sample(args: argparse.Namespace) -> None:

    # load sampling config
    sampling_config = config.load_json(
        os.path.join(config.CONFIG_PATH, "sampling", "sampling.json"))
    
    model_name:str = sampling_config["model_name"]
    load_latest_checkpoints: bool = sampling_config["load_latest_checkpoints"]
    load_ema: str = sampling_config["load_ema"]
    device: torch.device = sampling_config["device"]
    fp16: bool = sampling_config["fp16"]
    show_debug_plots: bool = sampling_config["show_debug_plots"]
    web_server_port: int = sampling_config["web_server_port"]

    if args.interactive == True:
        raise NotImplementedError("Interactive sampling is not yet implemented.")
    if args.sample_cfg_file is None:
        raise ValueError("Must either launch with interactive enabled or provide a sampling configuration file.")
    
    # load sampling params
    sample_params_path = os.path.join(config.CONFIG_PATH, "sampling", args.sample_cfg_file)
    if not os.path.exists(sample_params_path):
        raise FileNotFoundError(f"Sampling configuration file '{sample_params_path}' not found.")
    sample_params = SampleParams(**config.load_json(sample_params_path))

    # load model
    model_dtype = torch.bfloat16 if fp16 else torch.float32
    model_path = os.path.join(config.MODELS_PATH, model_name)
    print(f"Loading DualDiffusion model from '{model_path}' (dtype={model_dtype}) (ema={load_ema})...")
    pipeline = DualDiffusionPipeline.from_pretrained(model_path,
                                                     torch_dtype=model_dtype,
                                                     device=device,
                                                     load_latest_checkpoints=load_latest_checkpoints,
                                                     load_emas={"unet": load_ema})
    last_global_step = pipeline.unet.config["last_global_step"]

    # validate / pre-process sampling params
    if sample_params.seed is None:
        sample_params.seed = np.random.randint(
            10000, 99999 - sample_params.num_batches * sample_params.batch_size)

    game_names = {
        pipeline.dataset_game_names[game_id]: sample_params.game_ids[game_id]
        for game_id in sample_params.game_ids.keys()}
    print("Game IDs:")
    for game_name, weight in game_names.items():
        print(f"{game_name:<{max(len(name) for name in game_names)}} : {weight}")
    sorted_game_ids = sorted(sampling_params.game_ids.items(), key=lambda x:x[1])[-1]

    if sample_params.img2img_input is not None:
        crop_width = pipeline.format.sample_raw_crop_width(length=sample_params.length)
        sample_params.img2img_input = load_audio(
            os.path.join(config.DATASET_PATH, sample_params.img2img_input), start=0, count=crop_width)

    # setup metadata to be saved in output
    sampling_params = {
        "steps": steps,
        "seed": seed,
        "batch_size": batch_size,
        "length": length,
        "cfg_scale": cfg_scale,
        "sigma_max": sigma_max,
        "sigma_min": sigma_min,
        "rho": rho,
        "slerp_cfg": slerp_cfg,
        "game_ids": game_ids,
        "use_midpoint_integration": use_midpoint_integration,
        "input_perturbation": input_perturbation,
        "img2img_strength": img2img_strength,
        "img2img_input": input_audio,
        "schedule": schedule,
        "show_debug_plots": show_debug_plots
    }
    metadata = sampling_params.copy()
    metadata["model_name"] = model_name
    metadata["ema_checkpoint"] = load_ema
    metadata["global_step"] = last_global_step
    metadata["fp16"] = fp16
    metadata["fgla_iterations"] = fgla_iterations
    metadata["img2img_input"] = img2img_input_path
    metadata["timestamp"] = datetime.datetime.now().strftime("%m/%d/%Y %I:%M:%S %p")
    metadata["game_names"] = game_names

    # sample

    output_path = os.path.join(model_path, f"output/step_{last_global_step}")

    start_time = datetime.datetime.now()
    for i in range(num_samples):
        print(f"Generating batch {i+1}/{num_samples}...")

        start = time.time()
        output = pipeline(**sampling_params)
        print(f"Time taken: {time.time()-start}")

        batch_output_path = os.path.join(output_path, f"step_{last_global_step}_{steps}_{'ema'+ema_std+'_' if load_ema else ''}{'s' if slerp_cfg else 'l'}cfg{cfg_scale}_sgm{sigma_max}-{sigma_min}_r{rho}_g{top_game_id}_s{seed}")
        for i, sample in enumerate(output.unbind(0)):
            output_flac_file_path = f"{batch_output_path}_b{i}.flac"
            save_audio(sample, pipeline.config["model_params"]["sample_rate"], output_flac_file_path, metadata={"diffusion_metadata": dict_str(metadata)})
            print(f"Saved flac output to {output_flac_file_path}")

        seed += batch_size

    print(f"Finished in: {datetime.datetime.now() - start_time}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DualDiffusion sampling script.")
    parser.add_argument(
        "--sample_cfg_file",
        type=str,
        required=False,
        help="Use a preset sampling configuration file in $CONFIG_PATH/sampling/*",
    )
    parser.add_argument(
        "--interactive",
        type=bool,
        default=False,
        help="Start a local web interface for interactive sampling (TBD)",
    )
    return parser.parse_args()


if __name__ == "__main__":

    init_cuda()
    args = parse_args()

    sample(args)