#!/bin/bash
accelerate launch --config_file "/home/parlance/dualdiffusion/models/edm2_dae_aclap_1/training/dae_accelerate.yaml" "/home/parlance/dualdiffusion/src/train.py" --model_path="/home/parlance/dualdiffusion/models/edm2_dae_aclap_1" --train_config_path="/home/parlance/dualdiffusion/models/edm2_dae_aclap_1/training/dae_train.json"
