#!/bin/bash

accelerate launch \
  train.py \
  --mixed_precision="bf16" \
  --pretrained_model_name_or_path="/ephemeral/parlance/dualdiffusion/models/dualdiffusion2d_2000_6" \
  --module="unet" \
  --train_batch_size=12 \
  --num_train_epochs=5000 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-4 \
  --seed=400 \
  --lr_scheduler="constant_with_warmup" \
  --dataloader_num_workers=10 \
  --max_grad_norm=1 \
  --num_validation_samples=5 \
  --num_validation_epochs=25 \
  --num_validation_steps=100 \
  --dataset_name="parlance/spc_audio" \
  --cache_dir="/ephemeral/parlance/hf_cache"


#  --adam_weight_decay=0 \
# --use_ema