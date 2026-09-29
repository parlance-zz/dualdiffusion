from utils import config

import os

import torch

from create_new_model import print_module_info
from modules.formats.ms_mdct_dual_9 import MS_MDCT_DualFormat

model_name = "edm2_dae_p11"
model_path = os.path.join(config.MODELS_PATH, model_name)

print(f"Saving new modules to {model_path}...")
"""
from modules.embeddings.clap import CLAP_Config, CLAP_Embedding
embedding = CLAP_Embedding(CLAP_Config())
embedding.save_pretrained(model_path, subfolder="embedding")
"""

from modules.daes.dae_edm2_q4112 import DAE, DAE_Config
dae = DAE(DAE_Config())
print_module_info(dae, "dae")

if input("Save module? (y/n) ").lower() == 'y':
    dae.save_pretrained(model_path, subfolder="dae")
    print(f"Saved model to {model_path}/dae")

from modules.unets.unet_edm2_q4112_ddec import UNet, UNetConfig
ddecp = UNet(UNetConfig(x_ref_noise_max_sigma=0.5, x_ref_noise_mel_density_pow=0))
print_module_info(ddecp, "ddecp")

format = MS_MDCT_DualFormat.from_pretrained(model_path, subfolder="format")
x_ref = []
for n_psd_freqs in ddecp.config.in_psd_num_freqs:
    x_ref.append(torch.randn(1, 3, n_psd_freqs, 64))
_, x_ref_sigma = ddecp.get_x_ref_noise(x_ref, format)
print("x_ref bin sigma:")
print(x_ref_sigma[0].flatten())

if input("Save module? (y/n) ").lower() == 'y':
    ddecp.save_pretrained(model_path, subfolder="ddecp")
    print(f"Saved model to {model_path}/ddecp")

from modules.unets.unet_edm2_p6 import UNet, UNetConfig
unet = UNet(UNetConfig(num_layers_per_block=16, in_channels=1024, out_channels=1024))
#unet = UNet(UNetConfig(model_channels=8192, mlp_groups=64, emb_linear_groups=64, num_layers_per_block=24, channel_mult_noise=0.125, in_channels=3072, out_channels=3072))
print_module_info(unet, "unet")

if input("Save module? (y/n) ").lower() == 'y':
    unet.save_pretrained(model_path, subfolder="unet")
    print(f"Saved model to {model_path}/unet")