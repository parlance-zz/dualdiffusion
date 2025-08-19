from modules.unets.unet_edm2_ddec_mdct_d2 import DDec_MDCT_UNet_D2, DDec_MDCT_UNet_D2_Config
from create_new_model import print_module_info

ddec = DDec_MDCT_UNet_D2(DDec_MDCT_UNet_D2_Config())
print_module_info(ddec, "ddec")

if input("Save DDEC model? (y/n): ").lower() not in ["y", "yes"]:
    exit()
    
ddec.save_pretrained("/home/parlance/dualdiffusion/models/edm2_mdct_p2m_nt", subfolder="ddec")
