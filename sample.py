# MIT License
#
# Copyright (c) 2023 Christopher Friesen
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import time
from dotenv import load_dotenv

import numpy as np
import torch
import datetime

from dual_diffusion_pipeline import DualDiffusionPipeline
from dual_diffusion_utils import init_cuda, save_audio, load_audio

if __name__ == "__main__":

    init_cuda()
    load_dotenv(override=True)

    os.environ["MODEL_PATH"] = "Z:/dualdiffusion/models"; model_name = "edm2_vae_test7_2"
    #model_name = "edm2_vae7_v_pred"
    
    num_samples = 1
    batch_size = 2
    length = 0
    steps = 100
    cfg_scale = 5
    sigma_max = 80
    sigma_min = 0.002
    rho = 7
    slerp_cfg = False
    use_midpoint_integration = False
    input_perturbation = 0
    schedule = None #"cos"
    fgla_iterations = 200
    fp16 = True
    device = "cuda"
    show_debug_plots = True

    game_ids = []
    #game_ids += [785]   #megaman 7
    #game_ids += [1027]  #megaman 9
    #game_ids += [787]   #megaman x
    #game_ids += [788]   #megaman x2
    #game_ids += [789]   #megaman x3
    #game_ids += [152]   #breath of fire
    #game_ids += [153]   #breath of fire 2
    #game_ids += [213]   #chrono trigger
    #game_ids += [1190]  #star ocean
    #game_ids += [1400]  #tales of phantasia
    #game_ids += [1416]  #terranigma
    #game_ids += [1078]  #secret of mana
    #game_ids += [384]   #final fantasy mystic quest
    game_ids += [385]   #final fantasy 4
    #game_ids += [386]   #final fantasy 5
    #game_ids += [387]   #final fantasy 6
    #game_ids += [73]    #bahamut lagoon
    #game_ids += [1081]  #seiken densetsu 3
    #game_ids += [1302]  #super mario rpg
    #game_ids += [340]   #earthbound
    #game_ids += [705]   #zelda
    #game_ids += [1161]  #soul blazer
    #game_ids += [413]   #front mission
    #game_ids += [414]   #front mission gun hazard
    #game_ids += [230]   #contra
    #game_ids += [249]   #cybernator
    #game_ids += [1409]  #turtles in time
    #game_ids += [1099]  #gundam wing endless duel
    #game_ids += [1168]  #sparkster
    #game_ids += [1505]  #vortex
    #game_ids += [471]   #gradius 3
    #game_ids += [341]   #earthworm jim
    #game_ids += [342]   #earthworm jim 2
    #game_ids += [1187]  #star fox
    #game_ids += [1188]  #star fox 2
    #game_ids += [366]   #f-zero
    #game_ids += [944]   #pilotwings
    #game_ids += [1473]  #un squadron
    #game_ids += [70]    #axelay
    #game_ids += [1305]  #super metroid
    #game_ids += [1488]  #umihara kawase
    #game_ids += [669]   #kirby's dream course
    #game_ids += [666]   #kirby superstar
    #game_ids += [667]   #kirby's avalanche
    #game_ids += [670]   #kirby's dream land 3
    #game_ids += [896]   #cameltry (on the ball)
    #game_ids += [668]   #super puyo puyo
    #game_ids += [1130]  #sim city
    #game_ids += [107]   #battletoads & double dragon
    #game_ids += [108]   #battletoads in battlemaniacs
    #game_ids += [1494]  #uniracers
    #game_ids += [1024]  #rock n' roll racing
    #game_ids += [1449]  #top gear 3000
    #game_ids += [851]   #nba jam
    #game_ids += [851]   #nba jam tournament edition
    #game_ids += [950]   #plok!
    #game_ids += [1232]  #super bomberman
    #game_ids += [1234]  #super bomberman 2
    #game_ids += [1235]  #super bomberman 3
    #game_ids += [1236]  #super bomberman 4
    #game_ids += [1237]  #super bomberman 5
    #game_ids += [290]   #donkey kong country 1
    #game_ids += [291]   #donkey kong country 2
    #game_ids += [292]   #donkey kong country 3
    #game_ids += [1304]  #yoshi's island
    #game_ids += [1303]  #super mario world
    #game_ids += [1298]  #mario 1
    #game_ids += [1299]  #mario 2
    #game_ids += [1300]  #mario 3
    #game_ids += [765]   #mario paint
    #game_ids += [1301]  #mario kart
    #game_ids = [np.random.randint(0, 1612)]

    img2img_strength = 0.42 #0.425
    img2img_input_path = None
    #img2img_input_path = "1/Final Fantasy - Mystic Quest  [Mystic Quest Legend] - 07 Battle 1.flac"
    #img2img_input_path = "1/Final Fantasy VI - 104 Locke.flac"
    #img2img_input_path = "2/Mega Man X - 14 Spark Mandrill.flac"
    #img2img_input_path = "1/Final Fantasy V - 203 Battle with Gilgamesh.flac"
    #img2img_input_path = "1/Final Fantasy V - 120 The Dragon Spreads it's Wings.flac"
    #img2img_input_path = "2/Secret of Mana - 23 Eternal Recurrence.flac"
    #img2img_input_path = "1/Final Fantasy VI - 104 Locke.flac"
    #img2img_input_path = "2/U.N. Squadron - 04 Front Line Base.flac"
    #img2img_input_path = "2/Super Mario RPG - The Legend of the Seven Stars - 217 Weapons Factory.flac"
    #img2img_input_path = "2/Super Mario RPG - The Legend of the Seven Stars - 135 Welcome to Booster Tower.flac"
    #img2img_input_path = "2/Super Mario RPG - The Legend of the Seven Stars - 128 Beware the Forest's Mushrooms.flac"
    #img2img_input_path = "2/Mega Man X3 - 07 Neon Tiger.flac"
    #img2img_input_path = "2/Mega Man X2 - 23 Absolute Zero.flac"
    #img2img_input_path = "2/Super Mario All-Stars - 102 Overworld.flac"
    #img2img_input_path = "2/Mega Man X2 - 15 Dust Devil.flac"
    #img2img_input_path = "2/Mega Man X2 - 09 Panzer des Drachens.flac"
    #img2img_input_path = "2/Mega Man X2 - 11 Volcano's Fury.flac"
    #img2img_input_path = "2/Mario Paint - 09 Creative Exercise.flac"
    #img2img_input_path = "1/Contra III - The Alien Wars - 05 Neo Kobe Steel Factory.flac"
    #img2img_input_path = "2/Spindizzy Worlds - 06 Level Music 4.flac"
    #img2img_input_path = "1/Legend of Zelda, The - A Link to the Past - 04a Time of the Falling Rain.flac"

    seed = np.random.randint(10000, 99999-num_samples*batch_size)
    #seed = 53958 # good seed for x2
    #seed = 58367 # good seed for un squadron
    #seed = 97092 # good seed for gundam + x3 (4batch)
    #seed = 48146 # good seed for turtles in time (4batch)
    #seed = 74296 # good seed for contra + gundam wing (4batch), or umihara kawase (4batch)
    #seed = 2000
    #seed = 88446
    #seed = 46699
    #seed = 49171 # good seed for cybernator (2batch)
    #seed = 43820 # good seed for contra3 + cybernator (1batch)
    seed = 84512 # good seed for gundam (2batch)

    model_dtype = torch.bfloat16 if fp16 else torch.float32
    model_path = os.path.join(os.environ.get("MODEL_PATH", "./"), model_name)
    print(f"Loading DualDiffusion model from '{model_path}' (dtype={model_dtype})...")
    pipeline = DualDiffusionPipeline.from_pretrained(model_path,
                                                     torch_dtype=model_dtype,
                                                     load_latest_checkpoints=True,
                                                     device=device)
    last_global_step = pipeline.unet.config["last_global_step"]
    pipeline.format.spectrogram_params.num_griffin_lim_iters = fgla_iterations
    
    if img2img_input_path is not None:
        crop_width = pipeline.format.get_sample_crop_width(length=length)
        dataset_path = os.environ.get("DATASET_PATH", "./dataset/samples_hq")
        input_audio = load_audio(os.path.join(dataset_path, img2img_input_path), start=0, count=crop_width)
    else:
        input_audio = None

    output_path = os.path.join(model_path, "output")
    os.makedirs(output_path, exist_ok=True)

    start_time = datetime.datetime.now()

    """
    x = torch.linspace(1, 0, steps, dtype=pipeline.unet.dtype).to(pipeline.device)
    y = pipeline.unet.get_timestep_logvar(x)
    from dual_diffusion_utils import multi_plot
    multi_plot((y, "logvar"))
    exit()
    """

    for i in range(num_samples):
        print(f"Generating batch {i+1}/{num_samples}...")

        start = time.time()
        output = pipeline(steps=steps,
                          seed=seed,
                          batch_size=batch_size,
                          length=length,
                          cfg_scale=cfg_scale,
                          sigma_max=sigma_max,
                          sigma_min=sigma_min,
                          rho=rho,
                          slerp_cfg=slerp_cfg,
                          game_ids=game_ids,
                          use_midpoint_integration=use_midpoint_integration,
                          input_perturbation=input_perturbation,
                          img2img_strength=img2img_strength,
                          img2img_input=input_audio,
                          schedule=schedule,
                          show_debug_plots=show_debug_plots)
        print(f"Time taken: {time.time()-start}")

        output_path = os.path.join(output_path, f"step_{last_global_step}_{steps}_{'s' if slerp_cfg else 'l'}cfg{cfg_scale}_sgm{sigma_max}_p{input_perturbation}_g{game_ids[0]}_s{seed}")
        for i, sample in enumerate(output.unbind(0)):
            output_flac_file_path = f"{output_path}_b{i}.flac"
            save_audio(sample, pipeline.config["model_params"]["sample_rate"], output_flac_file_path)
            print(f"Saved flac output to {output_flac_file_path}")

        seed += batch_size
    
    if img2img_input_path is not None:
        output_flac_file_path = f"{output_path}_original.flac"
        save_audio(input_audio, pipeline.config["model_params"]["sample_rate"], output_flac_file_path)
        print(f"Saved flac output to {output_flac_file_path}")

    print(f"Finished in: {datetime.datetime.now() - start_time}")