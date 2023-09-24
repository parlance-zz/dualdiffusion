import os
from dotenv import load_dotenv

from dual_diffusion_pipeline import DualDiffusionPipeline

load_dotenv()

MODEL_NAME = "dualdiffusion2d_49"
MODEL_PARAMS = {
    "format": "pre_fft", #"no_pre_fft",
    "prediction_type": "v_prediction", #"sample",
    "beta_schedule": "squaredcos_cap_v2", #"linear",
    "beta_start" : 0.0001,
    "beta_end" : 0.02,
    "rescale_betas_zero_snr": False, #True
    "sample_raw_length": 65536*4,
    "sample_raw_channels": int(os.environ.get("DATASET_SAMPLE_CHANNELS")),
    "num_chunks": 256, 
    "sample_rate": int(os.environ.get("DATASET_SAMPLE_RATE")),
    "freq_embedding_dim": 0,
    "last_global_step": 0,
}

UNET_PARAMS = {
    #"dropout": 0.1,
    "dropout": 0.0,
    #"act_fn": "mish",
    "act_fn": "silu",
    #"attention_head_dim": (16, 32, 64),
    "attention_head_dim": 16,
    #"separate_attn_dim": (2,3),
    "separate_attn_dim": (3,2),
    #"positional_coding_dims": (3,), 
    "positional_coding_dims": (),
    #"reverse_separate_attn_dim": True,
    "reverse_separate_attn_dim": False,
    "double_attention": True,
    #"double_attention": False,
    "add_attention": True,
    #"add_attention": False,
    "norm_eps": 1e-05,
    #"norm_num_groups": 32,
    "norm_num_groups": 4,
    "layers_per_block": 2,
    "conv_size": (3,3),
    "downsample_type": "resnet",
    "upsample_type": "resnet",
    "block_out_channels": (64, 64, 64, 64, 64, 64, 64, 64),
    "down_block_types": (
        "SeparableAttnDownBlock2D",
        "SeparableAttnDownBlock2D",
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
        "SeparableAttnUpBlock2D",
        "SeparableAttnUpBlock2D",
    ),
    "in_channels": 2 + MODEL_PARAMS["freq_embedding_dim"],
    "out_channels": 2,
}

if __name__ == "__main__":

    NEW_MODEL_PATH = os.path.join(os.environ.get("MODEL_PATH"), MODEL_NAME)

    if os.path.exists(NEW_MODEL_PATH):
        print(f"Warning: Output folder already exists '{NEW_MODEL_PATH}'")
        if input("Overwrite existing model? (y/n): ").lower() not in ["y","yes"]: exit()
    
    pipeline = DualDiffusionPipeline.create_new(MODEL_PARAMS, UNET_PARAMS, NEW_MODEL_PATH)
    print(f"Created new DualDiffusion model with config at '{NEW_MODEL_PATH}'")
