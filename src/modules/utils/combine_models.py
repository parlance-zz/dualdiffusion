from utils import config

import os

from pipelines.dual_diffusion_pipeline import DualDiffusionPipeline

if __name__ == "__main__":

    model1_name = "edm2_vt7_12d_4"
    #model1_ema = {"unet": "ema_0.9999.safetensors"}
    model1_ema = True # setting this to True loads the longest ema, e.g. 0.99999 which may not be what you want

    model2_name = "edm2_vt7_c1a_1"
    #model2_ema = {"unet": "ema_0.9999.safetensors"}
    model2_ema = True
    
    blended_model_name = "my_blended_model"
    blend_t = 0.5 # closer to 0 means more of model1, closer to 1 means more of model2

    model1 = DualDiffusionPipeline.from_pretrained(os.path.join(config.MODELS_PATH, model1_name), load_checkpoints=True, load_emas=model1_ema)
    model2 = DualDiffusionPipeline.from_pretrained(os.path.join(config.MODELS_PATH, model2_name), load_checkpoints=True, load_emas=model2_ema)

    model1.unet.blend_weights(model2.unet, t=blend_t)
    model1.save_pretrained(os.path.join(config.MODELS_PATH, blended_model_name))