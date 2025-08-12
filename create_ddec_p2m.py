from modules.unets.unet_edm2_ddec_mdct_p2 import DDec_MDCT_UNet_P2, DDec_MDCT_UNet_P2_Config
from create_new_model import print_module_info

ddec_p2m = DDec_MDCT_UNet_P2(DDec_MDCT_UNet_P2_Config())
print_module_info(ddec_p2m, "ddec_p2m")

if input("Save DDEC P2M model? (y/n): ").lower() not in ["y", "yes"]:
    exit()
    
ddec_p2m.save_pretrained("/home/parlance/dualdiffusion/models/edm2_dae_d3a_v2_nt", subfolder="ddec_p2m")
