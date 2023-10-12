import os
from dotenv import load_dotenv

import torch
from dual_diffusion_pipeline import DualDiffusionPipeline

load_dotenv()

torch.manual_seed(100)

MODEL_NAME = "dualdiffusion2d_113"
MODEL_PARAMS = {
    #"prediction_type": "sample",
    "prediction_type": "v_prediction",
    #"beta_schedule": "trained_betas",
    #"beta_schedule": "linear", 
    "beta_schedule": "squaredcos_cap_v2", 
    "beta_start" : 0.0001,
    "beta_end" : 0.02,
    #"rescale_betas_zero_snr": True,
    "rescale_betas_zero_snr": False,
    #"sample_raw_length": 65536*2,
    "sample_raw_length": 65536,
    #"sample_raw_channels": int(os.environ.get("DATASET_SAMPLE_CHANNELS")),
    "sample_raw_channels": 1,
    #"num_chunks": 256, 
    "num_chunks": 128, 
    #"sample_rate": int(os.environ.get("DATASET_SAMPLE_RATE")),
    "sample_rate": 8000,
    "freq_embedding_dim": 0,
    "last_global_step": 0,
    "spatial_window_length": 1024,
    "sample_format": "overlapped",
    "fftshift": True,
    #"sample_std": 0.021220825965105643,
    #"sample_format": "ln",
    #"ln_amplitude_floor": -12,
    #"ln_amplitude_mean": -6.1341057,
    #"ln_amplitude_std": 1.66477387,
    #"phase_integral_mean": 0,
    #"phase_integral_std": 4.32964091,
}

UNET_PARAMS = {
    #"dropout": (0, 0, 0, 0.1, 0.15, 0.25),
    "dropout": 0.0,
    "act_fn": "silu",
    #"conv_size": (3,3),

    #"attention_num_heads": 8,
    "attention_num_heads": (8,12,20,32,52,84),

    #"double_attention": False,
    #"pre_attention": True,
    #"separate_attn_dim_down": (3,2,3),
    #"separate_attn_dim_up": (3,2,2,3),
    #"downsample_type": "resnet",
    #"upsample_type": "resnet",

    "use_separable_mid_block": False,
    #"separate_attn_dim_mid": (2,3,2,3),
    "add_mid_attention": True,
    "layers_per_mid_block": 1,

    "double_attention": False,
    "pre_attention": False,
    "separate_attn_dim_down": (2,3),
    "separate_attn_dim_up": (3,2,3),

    #"downsample_type": "resnet",
    #"upsample_type": "resnet",
    "downsample_type": "conv",
    "upsample_type": "conv",

    "norm_eps": 1e-05,
    "norm_num_groups": 32,

    "layers_per_block": 2,
    "block_out_channels": (128, 192, 320, 512, 832, 1344),
    "down_block_types": (
        "SeparableAttnDownBlock2D",
        "SeparableAttnDownBlock2D",
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
        "SeparableAttnUpBlock2D",
        "SeparableAttnUpBlock2D",
    ),
    "in_channels": MODEL_PARAMS["sample_raw_channels"]*2 + MODEL_PARAMS["freq_embedding_dim"],
    "out_channels": MODEL_PARAMS["sample_raw_channels"]*2,
}

if __name__ == "__main__":

    NEW_MODEL_PATH = os.path.join(os.environ.get("MODEL_PATH"), MODEL_NAME)

    if os.path.exists(NEW_MODEL_PATH):
        print(f"Warning: Output folder already exists '{NEW_MODEL_PATH}'")
        if input("Overwrite existing model? (y/n): ").lower() not in ["y","yes"]: exit()
    
    pipeline = DualDiffusionPipeline.create_new(MODEL_PARAMS, UNET_PARAMS)
    pipeline.save_pretrained(NEW_MODEL_PATH, safe_serialization=True)
    print(f"Created new DualDiffusion model with config at '{NEW_MODEL_PATH}'")
