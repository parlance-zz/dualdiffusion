from utils import config

import os

from create_new_model import print_module_info

model_name = "edm2_dae_p10"
model_path = os.path.join(config.MODELS_PATH, model_name)

print(f"Saving new modules to {model_path}...")
"""
from modules.embeddings.clap import CLAP_Config, CLAP_Embedding
embedding = CLAP_Embedding(CLAP_Config())
embedding.save_pretrained(model_path, subfolder="embedding")
"""

from modules.daes.dae_edm2_q7 import DAE, DAE_Config
dae = DAE(DAE_Config())
print_module_info(dae, "dae")

if input("Save module? (y/n) ").lower() == 'y':
    dae.save_pretrained(model_path, subfolder="dae")
    print(f"Saved model to {model_path}/dae")

"""
from modules.unets.unet_edm2_q7_ddec import UNet, UNetConfig
ddecm = UNet(UNetConfig())
print_module_info(ddecm, "ddecm")

if input("Save module? (y/n) ").lower() == 'y':
    ddecm.save_pretrained(model_path, subfolder="ddecm")
    print(f"Saved model to {model_path}/ddecm")
"""

from modules.unets.unet_edm2_q7_ddec import UNet, UNetConfig
ddecp = UNet(UNetConfig())
print_module_info(ddecp, "ddecp")

if input("Save module? (y/n) ").lower() == 'y':
    ddecp.save_pretrained(model_path, subfolder="ddecp")
    print(f"Saved model to {model_path}/ddecp")


from modules.unets.unet_edm2_p6 import UNet, UNetConfig
unet = UNet(UNetConfig(num_layers_per_block=8, block_kernel_size=1))
print_module_info(unet, "unet")

if input("Save module? (y/n) ").lower() == 'y':
    unet.save_pretrained(model_path, subfolder="unet")
    print(f"Saved model to {model_path}/unet")