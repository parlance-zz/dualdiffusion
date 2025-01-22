#!/bin/bash
accelerate launch --config_file "/home/parlance/dualdiffusion/models/edm2_vae_d1_ddec/training/ddec_accelerate.yaml" "/home/parlance/dualdiffusion/src/train.py" --model_path="/home/parlance/dualdiffusion/models/edm2_vae_d1_ddec" --train_config_path="/home/parlance/dualdiffusion/models/edm2_vae_d1_ddec/training/ddec_train.json"
