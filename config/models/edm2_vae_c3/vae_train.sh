#!/bin/bash
accelerate launch --config_file "/home/parlance/dualdiffusion/models/edm2_vae_c1/training/vae_accelerate.yaml" "/home/parlance/dualdiffusion/src/train.py" --model_path="/home/parlance/dualdiffusion/models/edm2_vae_c1" --train_config_path="/home/parlance/dualdiffusion/models/edm2_vae_c1/training/vae_train.json"
