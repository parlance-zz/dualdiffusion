from modules.cnets.cnet_edm2_2psd_a1 import CNet_2PSD_A1, CNet_2PSD_A1_Config
from create_new_model import print_module_info

cnet0 = CNet_2PSD_A1(CNet_2PSD_A1_Config(in_num_freqs=64))
cnet1 = CNet_2PSD_A1(CNet_2PSD_A1_Config(in_num_freqs=1024))
print_module_info(cnet0, "cnet0")

if input("Save CNet models? (y/n): ").lower() not in ["y", "yes"]:
    exit()

cnet0.save_pretrained("/home/parlance/dualdiffusion/models/edm2_mdct_2psd_a1", subfolder="cnet0")
cnet1.save_pretrained("/home/parlance/dualdiffusion/models/edm2_mdct_2psd_a1", subfolder="cnet1")

