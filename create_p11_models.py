from utils import config

import os

from create_new_model import print_module_info

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
ddecp = UNet(UNetConfig())
print_module_info(ddecp, "ddecp")

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