from utils import config

import os

import torch

from modules.daes.dae_edm2_p4 import DAE, DAE_Config
#from modules.daes.dae_edm2_p6 import DAE, DAE_Config
from modules.unets.unet_edm2_p4_ddec import UNet, UNetConfig
#from modules.unets.unet_edm2_p4_ddec import UNet, UNetConfig
#from modules.formats.mdct import MDCT_FormatConfig, MDCT_Format
#from modules.embeddings.clap import CLAP_Config, CLAP_Embedding
from create_new_model import print_module_info

model_name = "edm2_dae_p4"
model_path = os.path.join(config.MODELS_PATH, model_name)

"""
embedding = CLAP_Embedding(CLAP_Config())
embedding.save_pretrained(model_path, subfolder="embedding")

mdct_format = MDCT_Format(MDCT_FormatConfig())
mdct_format.save_pretrained(model_path, subfolder="format")
"""

dae = DAE(DAE_Config())
print_module_info(dae, "dae")

if input("Save module? (y/n) ").lower() == 'y':
    dae.save_pretrained(model_path, subfolder="dae")
    print(f"Saved model to {model_path}/dae")

ddecm = UNet(UNetConfig(mp_fourier_ln_sigma_offset=2))
print_module_info(ddecm, "ddecm")

if input("Save module? (y/n) ").lower() == 'y':
    ddecm.save_pretrained(model_path, subfolder="ddecm")
    print(f"Saved model to {model_path}/ddecm")

ddecp = UNet(UNetConfig(mp_fourier_ln_sigma_offset=0))
print_module_info(ddecp, "ddecp")

if input("Save module? (y/n) ").lower() == 'y':
    ddecp.save_pretrained(model_path, subfolder="ddecp")
    print(f"Saved model to {model_path}/ddecp")