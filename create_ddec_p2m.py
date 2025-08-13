from modules.unets.unet_edm2_ddec_mdct_p5 import DDec_MDCT_UNet_P5, DDec_MDCT_UNet_P5_Config
from create_new_model import print_module_info

ddec_p2m = DDec_MDCT_UNet_P5(DDec_MDCT_UNet_P5_Config())
print_module_info(ddec_p2m, "ddec")

if input("Save DDEC P2M model? (y/n): ").lower() not in ["y", "yes"]:
    exit()
    
ddec_p2m.save_pretrained("/home/parlance/dualdiffusion/models/edm2_mdct_p2m_nt", subfolder="ddec")
