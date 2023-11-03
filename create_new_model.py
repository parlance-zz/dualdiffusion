import os
from dotenv import load_dotenv

import torch
from dual_diffusion_pipeline import DualDiffusionPipeline

load_dotenv()

torch.manual_seed(200)

MODEL_NAME = "dualdiffusion2d_211"
MODEL_PARAMS = {
    #"prediction_type": "sample",
    "prediction_type": "v_prediction",
    #"prediction_type": "epsilon",
    #"beta_schedule": "trained_betas",
    #"beta_schedule": "linear", 
    "beta_schedule": "squaredcos_cap_v2", 
    "beta_start" : 0.0001,
    "beta_end" : 0.02,
    #"rescale_betas_zero_snr": True,
    "rescale_betas_zero_snr": False,
    "sample_raw_length": 65536*2,
    #"sample_raw_length": 65536,
    "sample_raw_channels": int(os.environ.get("DATASET_NUM_CHANNELS")),
    #"num_chunks": 128, 
    "num_chunks": 256, 
    "sample_rate": int(os.environ.get("DATASET_SAMPLE_RATE")),
    #"freq_embedding_dim": 0,
    "freq_embedding_dim": 126,
    "spatial_window_length": 1024, #2048,
    #"sample_format": "overlapped",
    "sample_format": "embedding",
    #"fftshift": True,
    #"fftshift": False,

    #"sample_std": 0.021220825965105643,
    #"sample_format": "ln",
    #"ln_amplitude_floor": -12,
    #"ln_amplitude_mean": -6.1341057,
    #"ln_amplitude_std": 1.66477387,
    #"phase_integral_mean": 0,
    #"phase_integral_std": 4.32964091,
}

#VAE_PARAMS = {
#}
VAE_PARAMS = None

UNET_PARAMS = {
    #"dropout": (0, 0, 0, 0.1, 0.15, 0.25),
    "dropout": 0.0,
    "act_fn": "silu",
    "conv_size": (3,3),
    #"conv_size": (1,3),
    #"attention_num_heads": 8,
    #"attention_num_heads": (8,12,20,32,52,84),
    #"attention_num_heads": (6,12,24,48),
    "attention_num_heads": (2,4,8,16,),

    #"double_attention": False,
    #"pre_attention": True,
    #"separate_attn_dim_down": (3,2,3),
    #"separate_attn_dim_up": (3,2,2,3),
    #"downsample_type": "resnet",
    #"upsample_type": "resnet",

    #"use_separable_mid_block": False,
    "use_separable_mid_block": True,
    #"separate_attn_dim_mid": (0,),
    "separate_attn_dim_mid": (0,),
    "add_mid_attention": True,
    "layers_per_mid_block": 1,

    #"double_attention": True,
    "double_attention": False,
    #"pre_attention": False,
    "pre_attention": True,
    "no_conv_in": True,
    
    "separate_attn_dim_down": (3,2,3,),
    #"separate_attn_dim_down": (3,),
    
    "separate_attn_dim_up": (2,3,2,3,),
    #"separate_attn_dim_up": (2,3,),

    #"downsample_type": "resnet",
    #"upsample_type": "resnet",
    "downsample_type": "conv",
    "upsample_type": "conv",

    "norm_eps": 1e-05,
    "norm_num_groups": 32,

    "layers_per_block": 2,
    #"layers_per_block": 2,
    #"block_out_channels": (128, 192, 320, 512, 832, 1344),
    #"block_out_channels": (128, 192, 320, 512, 832),#, 832, 1344),
    "block_out_channels": (128, 256, 512, 1024),
    "down_block_types": (
        "SeparableAttnDownBlock2D",
        "SeparableAttnDownBlock2D",
        "SeparableAttnDownBlock2D",
        "SeparableAttnDownBlock2D",
    ),
    "up_block_types": (
        "SeparableAttnUpBlock2D",
        "SeparableAttnUpBlock2D",
        "SeparableAttnUpBlock2D",
        "SeparableAttnUpBlock2D",
    ),
    "in_channels": MODEL_PARAMS["sample_raw_channels"]*2 + MODEL_PARAMS["freq_embedding_dim"],
    "out_channels": MODEL_PARAMS["sample_raw_channels"]*2,
}

#UPSCALER_PARAMS = {
#}
UPSCALER_PARAMS = None

if __name__ == "__main__":

    NEW_MODEL_PATH = os.path.join(os.environ.get("MODEL_PATH"), MODEL_NAME)

    if os.path.exists(NEW_MODEL_PATH):
        print(f"Warning: Output folder already exists '{NEW_MODEL_PATH}'")
        if input("Overwrite existing model? (y/n): ").lower() not in ["y","yes"]: exit()
    
    pipeline = DualDiffusionPipeline.create_new(MODEL_PARAMS, UNET_PARAMS, vae_params=VAE_PARAMS, upscaler_params=UPSCALER_PARAMS)
    pipeline.save_pretrained(NEW_MODEL_PATH, safe_serialization=True)
    print(f"Created new DualDiffusion model with config at '{NEW_MODEL_PATH}'")
