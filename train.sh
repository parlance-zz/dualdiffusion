#!/bin/bash

accelerate launch \
  train.py \
  --mixed_precision="fp16" \
  --allow_tf32 \
  --snr_gamma=5 \
  --train_data_dir="/home/ubuntu/stor-lgdiffusion/dualdiffusion/dataset/samples" \
  --pretrained_model_name_or_path="/home/ubuntu/stor-lgdiffusion/dualdiffusion/models/new_lgdiffusion3" \
  --output_dir="/home/ubuntu/stor-lgdiffusion/dualdiffusion/models/new_lgdiffusion3" \
  --train_batch_size=8 \
  --num_train_epochs=500 \
  --checkpointing_steps=2278 \
  --checkpoints_total_limit=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-4 \
  --report_to="tensorboard" \
  --resume_from_checkpoint=latest \
  --seed=42
