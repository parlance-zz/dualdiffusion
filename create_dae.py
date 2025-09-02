from modules.daes.dae_edm2_2psd_a1 import DAE_2PSD_A1, DAE_2PSD_A1_Config
from create_new_model import print_module_info

dae = DAE_2PSD_A1(DAE_2PSD_A1_Config())
print_module_info(dae, "dae")

if input("Save DAE model? (y/n): ").lower() not in ["y", "yes"]:
    exit()
    
dae.save_pretrained("/home/parlance/dualdiffusion/models/edm2_mdct_2psd_a1", subfolder="dae")
