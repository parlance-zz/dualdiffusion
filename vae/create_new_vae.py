import os
from dotenv import load_dotenv

#from autoencoder_kl_dual import AutoencoderKLDual
from diffusers.models import AutoencoderKL

load_dotenv()

MODEL_NAME = "dualvae_1"
VAE_PARAMS = {
  "act_fn": "silu",
  "block_out_channels": [
    32,
    64,
    128,
    256,
  ],
  "down_block_types": [
    "DownEncoderBlock2D",
    "DownEncoderBlock2D",
    "DownEncoderBlock2D",
    "DownEncoderBlock2D",
  ],
  "in_channels": 2,
  "latent_channels": 4,
  "layers_per_block": 2,
  "norm_num_groups": 32,
  "out_channels": 2,
  "sample_size": 1024,
  "scaling_factor": 0.13025,
  "up_block_types": [
    "UpDecoderBlock2D",
    "UpDecoderBlock2D",
    "UpDecoderBlock2D",
    "UpDecoderBlock2D",
  ]
}

if __name__ == "__main__":

    NEW_MODEL_PATH = os.path.join(os.environ.get("MODEL_PATH"), MODEL_NAME)

    # vae test
    #vae = AutoencoderKLDual.from_config(VAE_PARAMS).to("cuda")
    vae = AutoencoderKL.from_pretrained("d:/temp/sdxlvae").to("cuda")
    vae.enable_tiling()
    
    import torch
    data = torch.randn(1, 3, 1024, 1024).to("cuda")
    res = vae(data)
    print(res.sample.shape)
    exit()
    
    if os.path.exists(NEW_MODEL_PATH):
        print(f"Error: Output folder already exists '{NEW_MODEL_PATH}'")
        exit(1)
    
    AutoencoderKLDual.from_config(VAE_PARAMS).save_pretrained(NEW_MODEL_PATH)
    print(f"Created new DualVAE model with config at '{NEW_MODEL_PATH}'")
