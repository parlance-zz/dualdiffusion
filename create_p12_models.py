from utils import config

import os

from create_new_model import print_module_info, seed_and_get_module_serial_num


model_name = "edm2_dae_p12"
model_path = os.path.join(config.MODELS_PATH, model_name)

print(f"Saving new modules to {model_path}...")

#from modules.embeddings.clap import CLAP_Config, CLAP_Embedding
#embedding = CLAP_Embedding(CLAP_Config())
#embedding.save_pretrained(model_path, subfolder="embedding")

from modules.daes.dae_edm2_q432b import DAE, DAE_Config
from modules.unets.unet_edm2_p6 import UNetConfig
#unet_cfg = UNetConfig(num_layers_per_block=12, in_channels=512, out_channels=512)
unet_cfg=None
serial = seed_and_get_module_serial_num()
dae = DAE(DAE_Config(unet=unet_cfg))
dae.config.serial_num = serial
print_module_info(dae, "dae")

if input("Save module? (y/n) ").lower() == 'y':
    dae.save_pretrained(model_path, subfolder="dae")
    print(f"Saved model to {model_path}/dae")

from modules.unets.unet_edm2_q432_ddec import UNet, UNetConfig
serial = seed_and_get_module_serial_num()
ddecp = UNet(UNetConfig())
ddecp.config.serial_num = serial
print_module_info(ddecp, "ddecp")

#from modules.formats.ms_mdct_dual_10 import MS_MDCT_DualFormat
#format = MS_MDCT_DualFormat.from_pretrained(model_path, subfolder="format")
#x_ref_sigma = ddecp.config.x_ref_sigma_scale * format.get_mel_density(ddecp.config.in_num_freqs, pow=ddecp.config.x_ref_noise_mel_density_pow, normalize=True).float()
#print("x_ref sigma:")
#print(x_ref_sigma.flatten())
#exit()

if input("Save module? (y/n) ").lower() == 'y':
    ddecp.save_pretrained(model_path, subfolder="ddecp")
    print(f"Saved model to {model_path}/ddecp")

#from modules.unets.unet_edm2_p6 import UNet, UNetConfig
#serial = seed_and_get_module_serial_num()
#unet = UNet(UNetConfig(model_channels=8192, mlp_groups=64, emb_linear_groups=64, num_layers_per_block=24, channel_mult_noise=0.125, in_channels=3072, out_channels=3072))
#unet.config.serial_num = serial
#print_module_info(unet, "unet")

#if input("Save module? (y/n) ").lower() == 'y':
#    unet.save_pretrained(model_path, subfolder="unet")
#    print(f"Saved model to {model_path}/unet")