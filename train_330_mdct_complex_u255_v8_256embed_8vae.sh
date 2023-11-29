#!/bin/bash

accelerate launch \
  train.py \
  --mixed_precision="fp16" \
  --allow_tf32 \
  --train_data_dir="/home/parlance/dualdiffusion/dataset/samples" \
  --raw_sample_format="int16" \
  --pretrained_model_name_or_path="/home/parlance/dualdiffusion/models/dualdiffusion2d_330_mdct_complex_u255_v8_256embed_8vae" \
  --module="vae" \
  --train_batch_size=1 \
  --num_train_epochs=500 \
  --checkpointing_steps=2470 \
  --checkpoints_total_limit=1 \
  --gradient_accumulation_steps=8 \
  --learning_rate=1e-4 \
  --report_to="tensorboard" \
  --resume_from_checkpoint=latest \
  --seed=200 \
  --lr_scheduler="constant_with_warmup" \
  --num_validation_samples=5 \
  --num_validation_epochs=5

#  --max_grad_norm=50

# --gradient_checkpointing

# --snr_gamma=1 \
# --input_perturbation=0.1
# --dropout=0.1