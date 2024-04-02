#!/bin/bash

accelerate launch \
  train.py \
  --mixed_precision="bf16" \
  --pretrained_model_name_or_path="/home/parlance/dualdiffusion/models/edm2_100_2" \
  --module="unet" \
  --train_batch_size=10 \
  --num_train_epochs=200 \
  --gradient_accumulation_steps=3 \
  --learning_rate=1e-2 \
  --seed=2000 \
  --lr_scheduler="edm2" \
  --lr_warmup_steps=20000 \
  --max_grad_norm=10 \
  --num_validation_samples=0 \
  --adam_weight_decay=0 \
  --adam_beta2=0.99 \
  --dataloader_num_workers=8



#  --adam_weight_decay=0 \
# --use_ema