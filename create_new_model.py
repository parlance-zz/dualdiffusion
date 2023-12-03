import os
from dotenv import load_dotenv

import torch
from dual_diffusion_pipeline import DualDiffusionPipeline

load_dotenv()
torch.manual_seed(200)

MODEL_NAME = "dualdiffusion2d_330_mdct_complex_v8_256embed_2vae_mssloss"
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
    "sample_raw_channels": int(os.environ.get("DATASET_NUM_CHANNELS")),
    "sample_rate": int(os.environ.get("DATASET_SAMPLE_RATE")),
    "freq_embedding_dim": 0,
    "time_embedding_dim": 0,

    #"sample_format": "time_overlapped",
    #"sample_raw_length": 65536*2,
    #"num_chunks": 256, 
    #"rfft": True,

    #"sample_format": "overlapped",
    #"sample_raw_length": 65536*2,
    #"num_chunks": 128, 
    #"spatial_window_length": 512,
    #"rfft": True,
    #"ifft": True,
    #"fftshift": False,

    "sample_format": "mdct",
    "sample_raw_length": 65536*2,
    "num_chunks": 256,
    "complex": True,
    "multiscale_spectral_loss": {
        "num_filters": 12,
        "num_octaves": 12,
        "filter_std": torch.pi,
        "max_q": 1,
        "u": 22000,
    }
    
    #"sample_format": "mcltbce",
    #"sample_raw_length": 65536*2,
    #"num_chunks": 256,
    #"pos_weight": 2.129015562746317381309,

    #"sample_format": "normal",
    #"sample_raw_length": 65536*2,
    #"num_chunks": 256, 
    #"spatial_window_length": 512,
    #"rfft": True,
    #"ifft": True,

    #"sample_std": 0.021220825965105643,
    #"sample_format": "ln",
    #"ln_amplitude_floor": -12,
    #"ln_amplitude_mean": -6.1341057,
    #"ln_amplitude_std": 1.66477387,
    #"phase_integral_mean": 0,
    #"phase_integral_std": 4.32964091,
}


#VAE_PARAMS = None
VAE_PARAMS = {
  "latent_channels": 2,
  "sample_size": (256, 512),
  "act_fn": "silu",
  "conv_size": (3,3),
  
  "block_out_channels": [128, 256, 512, 512],
  "layers_per_block": 2,

  "layers_per_mid_block": 1,
  "add_mid_attention": True,

  "norm_num_groups": 32,

  "downsample_type": "conv",
  "upsample_type": "conv",

  "attention_num_heads": (8,16,32,32),
  "separate_attn_dim_down": (2,3),
  "separate_attn_dim_up": (3,2,3),    
  "separate_attn_dim_mid": (0,),
  "double_attention": False,
  "pre_attention": False,

  "freq_embedding_dim": 256,
  "time_embedding_dim": 256,

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

    #"use_separable_mid_block": False,
    "use_separable_mid_block": True,
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
}


#UPSCALER_PARAMS = {
#}
UPSCALER_PARAMS = None



if __name__ == "__main__":

    if VAE_PARAMS is not None:
        UNET_PARAMS["in_channels"]  = VAE_PARAMS["latent_channels"] + MODEL_PARAMS["freq_embedding_dim"] + MODEL_PARAMS["time_embedding_dim"]
        UNET_PARAMS["out_channels"] = VAE_PARAMS["latent_channels"]
    else:
        UNET_PARAMS["in_channels"]  = DualDiffusionPipeline.get_sample_format(MODEL_PARAMS).get_num_channels(MODEL_PARAMS)[0]
        UNET_PARAMS["out_channels"] = DualDiffusionPipeline.get_sample_format(MODEL_PARAMS).get_num_channels(MODEL_PARAMS)[1]
        
    NEW_MODEL_PATH = os.path.join(os.environ.get("MODEL_PATH"), MODEL_NAME)

    if os.path.exists(NEW_MODEL_PATH):
        print(f"Warning: Output folder already exists '{NEW_MODEL_PATH}'")
        if input("Overwrite existing model? (y/n): ").lower() not in ["y","yes"]: exit()
    
    pipeline = DualDiffusionPipeline.create_new(MODEL_PARAMS, UNET_PARAMS, vae_params=VAE_PARAMS, upscaler_params=UPSCALER_PARAMS)
    pipeline.save_pretrained(NEW_MODEL_PATH, safe_serialization=True)
    print(f"Created new DualDiffusion model with config at '{NEW_MODEL_PATH}'")
