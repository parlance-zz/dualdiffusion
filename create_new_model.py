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

load_dotenv()
torch.manual_seed(200)

MODEL_NAME = "dualdiffusion2d_400_mclt_4vae_mssloss1_cepstrum_micro_noise_8"
MODEL_PARAMS = {
    #"prediction_type": "sample",
    #"prediction_type": "epsilon",
    "prediction_type": "v_prediction",

    #"beta_schedule": "trained_betas",
    #"beta_schedule": "linear", 
    "beta_schedule": "squaredcos_cap_v2", 
    "beta_start" : 0.0001,
    "beta_end" : 0.02,
    #"rescale_betas_zero_snr": True,
    "rescale_betas_zero_snr": False,

    "sample_raw_channels": int(os.environ.get("DATASET_NUM_CHANNELS")),
    "sample_rate": int(os.environ.get("DATASET_SAMPLE_RATE")),

    "sample_format": "mclt",
    "sample_raw_length": 65536*2,
    "num_chunks": 64,
    #"u": 8000,
    #"add_abs_input": True,
    "add_abs_input": False,
    "noise_octaves": 8,
}

#VAE_PARAMS = None
VAE_PARAMS = {
    "multiscale_spectral_loss": {
        "version": 1,
        "sample_block_width": 2*MODEL_PARAMS["num_chunks"],
        "block_widths": [
            67,
            127,
            257,
            509,
            1021,
            2053,
            4099,
            8191,
            16381,
            32771,
            65539,
        ],
        "block_offsets": [
            0,
            0.33333333333333333,
            0.66666666666666667,
        ],
        "u": 8000,
        "use_cepstrum": True,
    },
    
    #"multiscale_spectral_loss": {
    #    "version": 2,
    #    "sample_block_width": 2*MODEL_PARAMS["num_chunks"],
    #    "num_filters": 1024,
    #    "sample_rate": MODEL_PARAMS["sample_rate"],
    #    "min_freq": 0,
    #    "max_freq": 4000,
    #    "max_logvar": 15,
    #    "min_logvar": -2,
    #    "freq_scale": "mel",
    #    "normalize_amplitude": True,
    #    "u": 8000,
    #},

    "latent_channels": 4,
    "sample_size": (64, 2048),
    "act_fn": "silu",
    "conv_size": (3,3),

    #"block_out_channels": [128, 256, 512, 512],
    "block_out_channels": (16, 32, 64, 128),
    #"layers_per_block": 2,
    "layers_per_block": 3,

    "layers_per_mid_block": 2,
    #"add_mid_attention": True,
    "add_mid_attention": False,

    #"norm_num_groups": 32,
    "norm_num_groups": (0, 0, 32, 32),

    "downsample_type": "conv",
    "upsample_type": "conv_transpose",
    #"upsample_type": "conv",
    "downsample_ratio": (2,2),

    #"attention_num_heads": (2,4,8,16),

    #"attention_num_heads": (8,16,32,32),
    #"separate_attn_dim_down": (2,3),
    #"separate_attn_dim_up": (3,2,3),
    #"separate_attn_dim_down": (3,3),
    #"separate_attn_dim_up": (3,3,3),
    #"separate_attn_dim_mid": (3,),
    #"separate_attn_dim_mid": (0,),
    "double_attention": False,
    "pre_attention": False,
    "add_attention": False,
    #"add_attention": True,

    #"freq_embedding_dim": 256,
    #"time_embedding_dim": 256,
    #"freq_embedding_dim": 64,
    "freq_embedding_dim": 0,
    "time_embedding_dim": 0,
    "noise_embedding_dim": 0,

    "in_channels": DualDiffusionPipeline.get_sample_format(MODEL_PARAMS).get_num_channels(MODEL_PARAMS)[0],
    "out_channels": DualDiffusionPipeline.get_sample_format(MODEL_PARAMS).get_num_channels(MODEL_PARAMS)[1],
}

UNET_PARAMS = {
    #"dropout": (0, 0, 0, 0.1, 0.15, 0.25),
    "dropout": 0.0,

    "act_fn": "silu",

    #"conv_size": (1,3),
    "conv_size": (3,3),

    "use_skip_samples": True,
    #"use_skip_samples": False,

    #"attention_num_heads": 4,
    "attention_num_heads": (8,8,16,16),

    "separate_attn_dim_mid": (0,),
    "add_mid_attention": True,
    "layers_per_mid_block": 1,
    #"mid_block_bottleneck_channels": 32,

    #"double_attention": True,
    "double_attention": False,
    #"pre_attention": True,
    "pre_attention": False,
    #"no_conv_in": True,
    "no_conv_in": False,
    
    "separate_attn_dim_down": (2,3,),
    #"separate_attn_dim_down": (3,),
    
    "separate_attn_dim_up": (3,2,3,),
    #"separate_attn_dim_up": (2,3,),
    
    "freq_embedding_dim": 256,
    "time_embedding_dim": 256,
    #"freq_embedding_dim": (512, 0, 0, 0,),
    #"time_embedding_dim": 0,

    #"downsample_type": "resnet",
    #"upsample_type": "resnet",
    "downsample_type": "conv",
    "upsample_type": "conv",

    "norm_eps": 1e-05,
    "norm_num_groups": 32,
    #"norm_num_groups": (32, 64, 128, 128,),
    #"norm_num_groups": -4,

    #"layers_per_block": 1,
    "layers_per_block": 2,

    #"block_out_channels": (256, 384, 640, 1024), # 320
    "block_out_channels": (128, 256, 512, 512), # 330
}


if __name__ == "__main__":

    if VAE_PARAMS is not None:
        UNET_PARAMS["in_channels"]  = VAE_PARAMS["latent_channels"]
        UNET_PARAMS["out_channels"] = VAE_PARAMS["latent_channels"]
    else:
        UNET_PARAMS["in_channels"]  = DualDiffusionPipeline.get_sample_format(MODEL_PARAMS).get_num_channels(MODEL_PARAMS)[0]
        UNET_PARAMS["out_channels"] = DualDiffusionPipeline.get_sample_format(MODEL_PARAMS).get_num_channels(MODEL_PARAMS)[1]
        
    NEW_MODEL_PATH = os.path.join(os.environ.get("MODEL_PATH"), MODEL_NAME)

    if os.path.exists(NEW_MODEL_PATH):
        print(f"Warning: Output folder already exists '{NEW_MODEL_PATH}'")
        if input("Overwrite existing model? (y/n): ").lower() not in ["y","yes"]: exit()
    
    pipeline = DualDiffusionPipeline.create_new(MODEL_PARAMS, UNET_PARAMS, vae_params=VAE_PARAMS)
    pipeline.save_pretrained(NEW_MODEL_PATH, safe_serialization=True)
    print(f"Created new DualDiffusion model with config at '{NEW_MODEL_PATH}'")
