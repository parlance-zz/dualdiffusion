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

MODEL_NAME = "edm2_vae_test7_3"
MODEL_SEED = 2000

MODEL_PARAMS = {
    # dataset format params
    "sample_raw_channels": int(os.environ.get("DATASET_NUM_CHANNELS")),
    "sample_rate": int(os.environ.get("DATASET_SAMPLE_RATE")),

    # sample format params
    "sample_format": "spectrogram",
    "sample_raw_length": 32000*45,
    "noise_floor": 2e-5,
    "t_scale": None, #3.5714285714, # scales the linear positional embedding for absolute time range within each sample, None disables t_range conditioning
    "vae_class": "AutoencoderKL_EDM2",

    # diffusion unet training params
    "unet_training_params": {
        "input_perturbation": 0,
        "sigma_ln_std": 1.,
        "sigma_ln_mean": -0.4,
        "stratified_sigma_sampling": True,
    },

    "spectrogram_params": {
        "abs_exponent": 0.25,

        # FFT parameters
        "step_size_ms": 8,

        # hann ** 40 window settings
        "window_duration_ms": 200,
        "padded_duration_ms": 200,
        "window_exponent": 32,
        "window_periodic": True,

        # hann ** 0.5 window settings
        #"window_exponent": 0.5,
        #"window_duration_ms": 30,
        #"padded_duration_ms": 200,

        # mel scale params
        "num_frequencies": 256,
        "min_frequency": 20,
        "max_frequency": 16000,
        "mel_scale_type": "htk",

        # phase recovery params
        "num_griffin_lim_iters": 400,
        "num_dm_iters": 0,
        "fgla_momentum": 0.99,
        "dm_beta": 1,
        "stereo_coherence": 0.67,
    },

    # vae training params
    "vae_training_params": {
        "point_loss_weight": 0,
        "channel_kl_loss_weight": 1.,
        "recon_loss_weight": 0.5,
        "imag_loss_weight": 1.,
        "block_overlap": 8,
        "block_widths": [
            8,
            16,
            32,
            64
        ],
    },
}

format = DualDiffusionPipeline.get_sample_format(MODEL_PARAMS)

VAE_PARAMS = {
    "latent_channels": 4,        # Number of channels in latent space.
    "target_snr": 2.,            # The learned latent snr will not exceed this snr
    "label_dim": 1612,           # Class label dimensionality. 0 = unconditional.
    "dropout": 0,                # Dropout rate for model blocks
    "model_channels": 96,        # Base multiplier for the number of channels.
    "channels_per_head": 64,     # Number of channels per attention head for blocks using self-attention
    "channel_mult": [1,2,3,5],   # Per-resolution multipliers for the number of channels.
    "channel_mult_emb": None,    # Multiplier for final embedding dimensionality. None = select based on channel_mult.
    "num_layers_per_block": 3,   # Number of residual blocks per resolution.
    "attn_levels": [],           # List of resolutions with self-attention.
    "midblock_decoder_attn": False, # Whether to use attention in the first layer of the decoder.
}

UNET_PARAMS = {
    "pos_channels": 0,           # Number of positional embedding channels for attention.
    "use_t_ranges": False,
    "label_dim": 1612,           # Class label dimensionality. 0 = unconditional.
    "label_dropout": 0.1,        # Dropout rate for the class embedding.
    "dropout": 0,                # Dropout rate for model blocks
    "model_channels": 192,       # Base multiplier for the number of channels.
    "channels_per_head": 64,     # Number of channels per attention head for blocks using self-attention
    "channel_mult": [1,2,3,4],   # Per-resolution multipliers for the number of channels.
    "channel_mult_noise": None,  # Multiplier for noise embedding dimensionality. None = select based on channel_mult.
    "channel_mult_emb": None,    # Multiplier for final embedding dimensionality. None = select based on channel_mult.
    "num_layers_per_block": 3,   # Number of residual blocks per resolution.
    "attn_levels": [2,3],        # List of resolutions with self-attention.
    "label_balance": 0.5,        # Balance between noise embedding (0) and class embedding (1).
    "concat_balance": 0.5,       # Balance between skip connections (0) and main path (1).
    "sigma_max": 80.,            # Expected max noise std
    "sigma_min": 0.002,          # Expected min noise std
    "sigma_data": 0.5,           # Expected data / input sample std
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
        UNET_PARAMS["in_channels"]  = format.get_num_channels()[0]
        UNET_PARAMS["out_channels"] = format.get_num_channels()[1]
    
    print("UNET Params:")
    print(dict_str(UNET_PARAMS))
    print("")
    
    NEW_MODEL_PATH = os.path.join(os.environ.get("MODEL_PATH"), MODEL_NAME)

    if os.path.exists(NEW_MODEL_PATH):
        print(f"Warning: Output folder already exists '{NEW_MODEL_PATH}'")
        if input("Overwrite existing model? (y/n): ").lower() not in ["y","yes"]: exit()
    
    pipeline = DualDiffusionPipeline.create_new(MODEL_PARAMS, UNET_PARAMS, vae_params=VAE_PARAMS)
    pipeline.save_pretrained(NEW_MODEL_PATH, safe_serialization=True)
    print(f"Created new DualDiffusion model with config at '{NEW_MODEL_PATH}'")
