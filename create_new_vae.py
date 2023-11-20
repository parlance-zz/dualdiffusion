import os
from dotenv import load_dotenv

from autoencoder_kl_dual import AutoencoderKLDual
#from diffusers.models import AutoencoderKL

load_dotenv()

MODEL_NAME = "dualvae_1"
VAE_PARAMS = {
  "in_channels": 2,
  "out_channels": 2,
  "latent_channels": 4,
  "sample_size": (512, 512),
  "act_fn": "silu",
  "conv_size": (3,3),

  "block_out_channels": [32, 64, 128, 256,],
  "layers_per_block": 2,

  "layers_per_mid_block": 1,
  "add_mid_attention": True,

  "norm_num_groups": 32,

  "downsample_type": "conv",
  "upsample_type": "conv",

  "attention_num_heads": (8,8,16,16),
  "separate_attn_dim_down": (2,3),
  "separate_attn_dim_up": (3,2,3),    
  "separate_attn_dim_mid": (0,),
  "double_attention": False,
  "pre_attention": False,

  "freq_embedding_dim": 0,
  "time_embedding_dim": 0,
}

if __name__ == "__main__":

    NEW_MODEL_PATH = os.path.join(os.environ.get("MODEL_PATH"), MODEL_NAME)

    # vae test
    vae = AutoencoderKLDual.from_config(VAE_PARAMS).to("cuda")
    #vae.enable_tiling()
    
    import torch
    data_shape = (1, VAE_PARAMS["in_channels"], VAE_PARAMS["sample_size"][0], VAE_PARAMS["sample_size"][1])
    data = torch.randn(data_shape).to("cuda")
    res = vae(data)
    print(res.sample.shape)
    exit()
    
    if os.path.exists(NEW_MODEL_PATH):
        print(f"Error: Output folder already exists '{NEW_MODEL_PATH}'")
        exit(1)
    
    AutoencoderKLDual.from_config(VAE_PARAMS).save_pretrained(NEW_MODEL_PATH)
    print(f"Created new DualVAE model with config at '{NEW_MODEL_PATH}'")
