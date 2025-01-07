#!/bin/bash
accelerate launch --config_file "/home/parlance/dualdiffusion/models/edm2_vt7_12d_4_clap2/training/vae_accelerate.yaml" "/home/parlance/dualdiffusion/src/train.py" --model_path="/home/parlance/dualdiffusion/models/edm2_vt7_12d_4_clap2" --train_config_path="/home/parlance/dualdiffusion/models/edm2_vt7_12d_4_clap2/training/vae_train.json"
