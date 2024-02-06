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
from dotenv import load_dotenv

import torch
from dual_diffusion_pipeline import DualDiffusionPipeline
from dual_diffusion_utils import dict_str

load_dotenv(override=True)

MODEL_NAME = "dualdiffusion2d_1000_7"
MODEL_SEED = 400

MODEL_PARAMS = {
    # dataset format params
    "sample_raw_channels": int(os.environ.get("DATASET_NUM_CHANNELS")),
    "sample_rate": int(os.environ.get("DATASET_SAMPLE_RATE")),

    # sample format params
    "sample_format": "mclt",
    "sample_raw_length": 65536*4,
    "num_chunks": 256,
    "u": 8000,
    "noise_floor": 1e-5,

    # diffusion unet training params
    "input_perturbation": 0.1,
    "snr_gamma": 5,

    # vae unet training params
    "kl_loss_weight": 1e-6,
    "format_real_loss_weight": 0,
    "format_imag_loss_weight": 0,

    "multiscale_spectral_loss": {
        "real_loss_weight": 0.04,
        "imag_loss_weight": 0.04,    
        "block_overlap": 8,
        "block_widths": [
            512,
            1024,
            2048,
            4096,
            8192,
            16384,
            32768,
            65536,
            131072
        ],
        "window_fn": "blackman_harris",
    },
}

SCHEDULER_PARAMS = {
    #"prediction_type": "sample",
    #"prediction_type": "epsilon",
    "prediction_type": "v_prediction",

    "beta_schedule": "scaled_linear",
    "beta_start": 0.00085,
    "beta_end": 0.012,

    #"beta_schedule": "linear", 
    #"beta_start" : 0.0001,
    #"beta_end" : 0.02,

    #"beta_schedule": "squaredcos_cap_v2",

    "rescale_betas_zero_snr": True,
    #"rescale_betas_zero_snr": False,
}

VAE_PARAMS = {
    "latent_channels": 2,
    "sample_size": (64, 2048),
    "act_fn": "silu",
    "conv_size": (3,3),

    "block_out_channels": (32, 96, 288),
    "layers_per_block": 3,

    "layers_per_mid_block": 2,
    "add_mid_attention": True,
    #"add_mid_attention": False,

    #"norm_num_groups": 32,
    #"norm_num_groups": (0, 0, 32, 32),
    "norm_num_groups": (0, 0, 32),

    "downsample_type": "conv",
    "upsample_type": "conv_transpose",
    #"upsample_type": "conv",
    "downsample_ratio": (2,2),

    "attention_num_heads": (8,8,8),
    "separate_attn_dim_mid": (2,3),
    "double_attention": False,
    "pre_attention": False,
    "add_attention": False,
    #"add_attention": True,

    "freq_embedding_dim": 0,
    "time_embedding_dim": 0,

    #"use_noise_channel": True,
    "use_noise_channel": False,

    "in_channels": DualDiffusionPipeline.get_sample_format(MODEL_PARAMS).get_num_channels(MODEL_PARAMS)[0],
    "out_channels": DualDiffusionPipeline.get_sample_format(MODEL_PARAMS).get_num_channels(MODEL_PARAMS)[1],
}

UNET_PARAMS = {
    "dropout": 0.,
    "act_fn": "silu",
    "conv_size": (3,3),

    #"attention_num_heads": 4,
    "attention_num_heads": (8,8,8),

    "separate_attn_dim_mid": (0,),
    "add_mid_attention": True,
    "layers_per_mid_block": 1,
    #"mid_block_bottleneck_channels": 32,

    "add_attention": True,
    #"double_attention": True,
    "double_attention": False,
    #"pre_attention": True,
    "pre_attention": False,
    
    #"separate_attn_dim_down": (2,3,),
    "separate_attn_dim_down": (0,0),
    
    #"separate_attn_dim_up": (3,2,3,),
    "separate_attn_dim_up": (0,0,0),
    
    #"freq_embedding_dim": 256,
    #"time_embedding_dim": 256,
    #"freq_embedding_dim": (512, 0, 0, 0,),
    #"time_embedding_dim": 0,
    "freq_embedding_dim": 256,
    "time_embedding_dim": 1024,

    #"downsample_type": "resnet",
    #"upsample_type": "resnet",
    "downsample_type": "conv",
    "upsample_type": "conv",

    "norm_num_groups": 32,
    #"norm_num_groups": (32, 64, 128, 128,),
    #"norm_num_groups": -4,

    #"layers_per_block": 1,
    "layers_per_block": 2,

    #"block_out_channels": (256, 384, 640, 1024), # 320
    "block_out_channels": (256, 416, 672), # 330
}


if __name__ == "__main__":

    torch.manual_seed(MODEL_SEED)

    print("Models Params:")
    print(dict_str(MODEL_PARAMS))

    if VAE_PARAMS is not None:
        UNET_PARAMS["in_channels"]  = VAE_PARAMS["latent_channels"]
        UNET_PARAMS["out_channels"] = VAE_PARAMS["latent_channels"]

        print("VAE Params:")
        print(dict_str(VAE_PARAMS))
    else:
        UNET_PARAMS["in_channels"]  = DualDiffusionPipeline.get_sample_format(MODEL_PARAMS).get_num_channels(MODEL_PARAMS)[0]
        UNET_PARAMS["out_channels"] = DualDiffusionPipeline.get_sample_format(MODEL_PARAMS).get_num_channels(MODEL_PARAMS)[1]
    
    print("UNET Params:")
    print(dict_str(UNET_PARAMS))
    print("Scheduler Params:")
    print(dict_str(SCHEDULER_PARAMS))
    print("")
    
    NEW_MODEL_PATH = os.path.join(os.environ.get("MODEL_PATH"), MODEL_NAME)

    if os.path.exists(NEW_MODEL_PATH):
        print(f"Warning: Output folder already exists '{NEW_MODEL_PATH}'")
        if input("Overwrite existing model? (y/n): ").lower() not in ["y","yes"]: exit()
    
    pipeline = DualDiffusionPipeline.create_new(MODEL_PARAMS, UNET_PARAMS, SCHEDULER_PARAMS, vae_params=VAE_PARAMS)
    pipeline.save_pretrained(NEW_MODEL_PATH, safe_serialization=True)
    print(f"Created new DualDiffusion model with config at '{NEW_MODEL_PATH}'")
