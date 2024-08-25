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

from utils import config

import os
import time
import datetime
import json

import numpy as np
import torch

from pipelines.dual_diffusion_pipeline import DualDiffusionPipeline, SamplingParams
from utils.dual_diffusion_utils import init_cuda, save_audio, load_audio, dict_str


if __name__ == "__main__":

    init_cuda()

    #model_name = "edm2_vae_test7_6a"
    #model_name = "edm2_vae_test7_20"
    #model_name = "edm2_vae_test7_12"
    model_name = "edm2_vae_test7_12c"
    
    sampling_params = SamplingParams(
        steps = 100,
        seed  = None,
        batch_size = 2,
        length     = 0,
        cfg_scale  = 1.5,
        sigma_max  = 200.,
        sigma_min  = 0.03,
        rho        = 7.
        game_ids: game_ids,
        use_midpoint_integration: use_midpoint_integration,
        input_perturbation: input_perturbation,
        img2img_strength: img2img_strength,
        img2img_input: input_audio,
        schedule: schedule,
        show_debug_plots: show_debug_plots
    )

    num_samples = 1
    batch_size = 2
    length = 0#int(2 * (512/15.625*32000))
    steps = 100
    cfg_scale = 1.5#1.7071#1.618#1.5#2#2.3#2.718
    schedule = "edm2"
    sigma_max = 200
    #sigma_max = 100
    sigma_min = 0.03 #0.0022
    rho = 7
    slerp_cfg = False
    use_midpoint_integration = True
    input_perturbation = None
    schedule = None
    fgla_iterations = 200
    load_ema = None
    #load_ema = "pf_ema_std-0.003.safetensors"
    #load_ema = "pf_ema_std-0.005.safetensors"
    load_ema = "pf_ema_std-0.010.safetensors"
    #load_ema = "pf_ema_std-0.020.safetensors"
    fp16 = True
    device = "cuda"
    show_debug_plots = True

    game_ids = {}
    #game_ids[78] = 1    #ballz 3d
    #game_ids[997] = 1   #radical rex
    #game_ids[335] = 1   #evo
    #game_ids[26] = 0.3   #aerofighters
    #game_ids[785] = 0.3   #megaman 7
    #game_ids[1027] = 0.3  #megaman 9
    #game_ids[787] = 0.2   #megaman x
    #game_ids[788] = 0.3   #megaman x2
    #game_ids[789] = 0.1   #megaman x3
    #game_ids[152] = 1   #breath of fire
    #game_ids[153] = 1   #breath of fire 2
    #game_ids[213] = 0.1   #chrono trigger
    #game_ids[1190] = 0.1  #star ocean
    #game_ids[1400] = 0.3  #tales of phantasia
    #game_ids[1416] = 1  #terranigma
    #game_ids[1078] = 0.2  #secret of mana
    #game_ids[384] = 0.3   #final fantasy mystic quest
    #game_ids[385] = 0.1   #final fantasy 4
    #game_ids[386] = 0.5   #final fantasy 5
    #game_ids[387] = 0.1   #final fantasy 6
    #game_ids[73] = 0.4    #bahamut lagoon
    #game_ids[1081] = 0.3 #seiken densetsu 3
    #game_ids[1302] = 0.3  #super mario rpg
    #game_ids[340] = 1   #earthbound
    #game_ids[705] = 0.3   #zelda
    #game_ids[1161] = 1  #soul blazer
    #game_ids[413] = 0.2   #front mission 
    #game_ids[414] = 1   #front mission gun hazard
    #game_ids[230] = 0.3  #contra
    #game_ids[249] = 0.3   #cybernator
    #game_ids[1409] = 1 #turtles in time    
    game_ids[1099] = 0.2  #gundam wing endless duel
    #game_ids[813] = 0.2  #mighty morphin' power rangers : the movie (same authors as gundam wing)
    #game_ids[438] = 0.1   #gekisou sentai car rangers - zenkai! racer senshi  (same authors as gundam wing)
    #game_ids[1201] = 0.2  #street fighter alpha 2
    #game_ids[1202] = 1  #street fighter 2
    #game_ids[1203] = 0.3  #street fighter 2 turbo
    #game_ids[1168] = 0.1  #sparkster
    #game_ids[1505] = 0.1  #vortex
    #game_ids[471] = 0.3  #gradius 3
    #game_ids[341] = 1   #earthworm jim
    #game_ids[342] = 1   #earthworm jim 2
    #game_ids[1187] = 0.1  #star fox
    #game_ids[1188] = 0.2  #star fox 2
    #game_ids[366] = 0.2   #f-zero
    #game_ids[944] = 1   #pilotwings
    game_ids[1473] = 0.2  #un squadron
    #game_ids[70] = 0.1    #axelay
    #game_ids[1305] = 1  #super metroid
    #game_ids[1244] = 1 #super castlevania 4
    #game_ids[1488] = 0.3  #umihara kawase
    #game_ids[669] = 1   #kirby's dream course
    #game_ids[666] = 0.2   #kirby superstar
    #game_ids[667] = 1   #kirby's avalanche
    #game_ids[670] = 1   #kirby's dream land 3
    #game_ids[896] = 0.2   #cameltry (on the ball)
    #game_ids[668] = 1   #super puyo puyo
    #game_ids[1130] = 1  #sim city
    #game_ids[107] = 1   #battletoads & double dragon
    #game_ids[108] = 1   #battletoads in battlemaniacs
    #game_ids[1494] = 1  #uniracers
    #game_ids[1024] = 1  #rock n' roll racing
    #game_ids[1449] = 1  #top gear 3000
    #game_ids[851] = 1   #nba jam
    #game_ids[851] = 1   #nba jam tournament edition
    #game_ids[950] = 1   #plok!
    #game_ids[1232] = 1  #super bomberman
    #game_ids[1234] = 0.2  #super bomberman 2
    #game_ids[1235] = 0.2  #super bomberman 3
    #game_ids[1236] = 1  #super bomberman 4
    #game_ids[1237] = 1  #super bomberman 5
    #game_ids[290] = 1   #donkey kong country 1
    #game_ids[291] = 0.3   #donkey kong country 2
    #game_ids[292] = 0.3   #donkey kong country 3
    #game_ids[1304] = 1  #yoshi's island
    #game_ids[1303] = 1  #super mario world
    #game_ids[1298] = 1  #mario 1
    #game_ids[1299] = 1  #mario 2
    #game_ids[1300] = 0.5  #mario 3
    #game_ids[765] = 1   #mario paint
    #game_ids[1301] = 1 #mario kart
    #game_ids[np.random.randint(0, 1612)] = 0.1
    #for i in range(1612): game_ids[i] = game_ids.get(i, 0) + 2e-3
    #game_ids = { 1099: 1, 756: 1}
    #game_ids = {756: 1}
    #game_ids[np.random.randint(0, 1612)] = 0.1
    #game_ids[np.random.randint(0, 1612)] = 0.2
    #game_ids[np.random.randint(0, 1612)] = 0.3

    img2img_strength = 0.33 #0.75#0.75#0.46 #0.66 #0.42
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
    #seed = 49171 # good seed for cybernator (2batch)
    #seed = 43820 # good seed for contra3 + cybernator (1batch)
    #seed = 84512 # good seed for gundam (2batch), mm7 + mm9 (2batch), kirby superstar (2batch)
    #seed = 24012 # good seed for vortex (2batch)
    #seed = 19534 # good seed for ff5 (2batch)
    #seed = 66787 # good seed for gundam wing (2batch) and umihara kawase (2batch)
    #seed = 19905 # good seed for mm7 (2batch)
    #seed = 47588 # good seed for ff6 (1batch) or (gundam wing (1batch) img2img on spark mandrill with 0.75str) or (gundam wing + cybernator + x3 (2batch) img2img on spark mandrill with 0.75str)
    #seed = 22176 # good seed for ff6 (3batch) or bahamut lagoon (3batch)
    #seed = 58012 # good seed for gradius (2batch) or ff5 (2batch) or kirby superstar (2batch)
    #seed = 69084 # good seed for gundam + cybernator (4batch)
    #seed = 94535 # good seed for secret of mana (2batch)
    #seed = 53110 # good seed for mm7 (2batch) or umihara kawase (2batch) or both combined (2batch)
    #seed = 13598 # good seed for star fox (2batch)
    #seed = 56889 # good seed for umihara kawase (2batch) (img2img on spark mandrill with 0.8str)
    #seed = 94585 # great seed for 0.2 gradius 3 and 1. gundam wing (2batch)
    #seed = 98333 # good seed for ff4 (2batch)
    #seed = 85983 # good seed for ff6 (2batch)
    #seed = 84504 # good seed for ff mystic quest (2batch)
    #seed = 45299 # good seed for dkc1 (2batch) or gundam wing (2batch) or gundam wing x1 turtles in time x0.3 gradius 3 x0.2 (2batch)
    #seed = 71169 # good seed for gundam wing (2batch)
    #seed = 94317 # good seed for mario 3 (2batch)
    #seed = 65477 # good seed for mmx1 (2batch)
    #seed = 38739 # good seed for mystic quest + power rangers (2batch)
    #seed = 56534 # good seed for ff mq (1) + gundam wing (0.15) + mmx3 (0.15) (2batch)
    #seed = 17839 # good seed for mm7 (0.1) + power rangers (0.2) (2batch)
    #seed = 75567 # battle music for ff6 (0.3) + power rangers (0.01) (2batch)
    #seed = 80794 # boss music for star fox (0.1) + power rangers (0.01) (2batch)
    #seed = 21667 # good seed for ff6 (0.3) + power rangers (0.01) (2batch)
    #seed = 59683 # good seed for mmx2 (0.2) + power rangers (0.1) (2batch)
    #seed = 44418 # good seed for kirby superstar (2batch)
    #seed = 97680 # good seed for kirby superstar (2batch) (heart of nova and great cave offensive)
    #seed = 98323 # good seed for mmx 1 (0.3) and power rangers (0.1) (2batch)
    #seed = 19350 # good seed for dkc2 (2batch) (stickerbrush symphony)
    #seed = 91936 # good seed for ff5 (0.3) and ff mq (0.1) (2batch)



    model_dtype = torch.bfloat16 if fp16 else torch.float32
    model_path = os.path.join(config.MODELS_PATH, model_name)
    print(f"Loading DualDiffusion model from '{model_path}' (dtype={model_dtype}) (ema={load_ema})...")
    pipeline = DualDiffusionPipeline.from_pretrained(model_path,
                                                     torch_dtype=model_dtype,
                                                     load_latest_checkpoints=True,
                                                     load_ema=load_ema,
                                                     device=device)
    last_global_step = pipeline.unet.config["last_global_step"]
    pipeline.format.spectrogram_params.num_griffin_lim_iters = fgla_iterations
    
    if load_ema:
        ema_std = load_ema.replace("pf_ema_std-", "").replace(".safetensors", "")
        print(f"Using EMA checkpoint {load_ema}")

    if img2img_input_path is not None:
        crop_width = pipeline.format.sample_raw_crop_width(length=length)
        input_audio = load_audio(os.path.join(config.DATASET_PATH, img2img_input_path), start=0, count=crop_width)
    else:
        input_audio = None

    output_path = os.path.join(model_path, f"output/step_{last_global_step}")
    os.makedirs(output_path, exist_ok=True)

    dataset_infos_path = os.path.join(config.DATASET_PATH, "dataset_infos", "dataset_info.json")
    with open(dataset_infos_path, "r") as f:
        dataset_info = json.load(f)
    dataset_game_names = {value: key for key, value in dataset_info["games"].items()}
    game_names = {dataset_game_names[game_id]: game_ids[game_id] for game_id in game_ids.keys()}
    print("Game IDs:")
    for game_name, weight in game_names.items():
        print(f"{game_name:<{max(len(name) for name in game_names)}} : {weight}")

    top_game_id = sorted(game_ids.items(), key=lambda x:x[1])[-1][0]
    sampling_params = {
        "steps": steps,
        "seed": seed,
        "batch_size": batch_size,
        "length": length,
        "cfg_scale": cfg_scale,
        "sigma_max": sigma_max,
        "sigma_min": sigma_min,
        "rho": rho,
        "slerp_cfg": slerp_cfg,
        "game_ids": game_ids,
        "use_midpoint_integration": use_midpoint_integration,
        "input_perturbation": input_perturbation,
        "img2img_strength": img2img_strength,
        "img2img_input": input_audio,
        "schedule": schedule,
        "show_debug_plots": show_debug_plots
    }
    metadata = sampling_params.copy()
    metadata["model_name"] = model_name
    metadata["ema_checkpoint"] = load_ema
    metadata["global_step"] = last_global_step
    metadata["fp16"] = fp16
    metadata["fgla_iterations"] = fgla_iterations
    metadata["img2img_input"] = img2img_input_path
    metadata["timestamp"] = datetime.datetime.now().strftime("%m/%d/%Y %I:%M:%S %p")
    metadata["game_names"] = game_names

    start_time = datetime.datetime.now()
    for i in range(num_samples):
        print(f"Generating batch {i+1}/{num_samples}...")

        start = time.time()
        output = pipeline(**sampling_params)
        print(f"Time taken: {time.time()-start}")

        batch_output_path = os.path.join(output_path, f"step_{last_global_step}_{steps}_{'ema'+ema_std+'_' if load_ema else ''}{'s' if slerp_cfg else 'l'}cfg{cfg_scale}_sgm{sigma_max}-{sigma_min}_r{rho}_g{top_game_id}_s{seed}")
        for i, sample in enumerate(output.unbind(0)):
            output_flac_file_path = f"{batch_output_path}_b{i}.flac"
            save_audio(sample, pipeline.config["model_params"]["sample_rate"], output_flac_file_path, metadata={"diffusion_metadata": dict_str(metadata)})
            print(f"Saved flac output to {output_flac_file_path}")

        seed += batch_size

    print(f"Finished in: {datetime.datetime.now() - start_time}")