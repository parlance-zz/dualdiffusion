#!/bin/bash

accelerate launch \
  train.py \
  --mixed_precision="fp16" \
  --allow_tf32 \
  --train_data_dir="/home/parlance/dualdiffusion/dataset/samples" \
  --raw_sample_format="int16" \
  --pretrained_model_name_or_path="/home/parlance/dualdiffusion/models/dualdiffusion2d_210" \
  --output_dir="/home/parlance/dualdiffusion/models/dualdiffusion2d_210" \
  --train_batch_size=4 \
  --num_train_epochs=500 \
  --checkpointing_steps=1235 \
  --checkpoints_total_limit=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1.25e-5 \
  --report_to="tensorboard" \
  --resume_from_checkpoint=latest \
  --seed=200 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=6175 \
  --num_validation_samples=10 \
  --num_validation_epochs=5 \
  --dataloader_num_workers=4

# --snr_gamma=1 \
# --input_perturbation=0.1
# --dropout=0.1