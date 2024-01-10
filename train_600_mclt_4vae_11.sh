#!/bin/bash

accelerate launch \
  train.py \
  --mixed_precision="fp16" \
  --allow_tf32 \
  --train_data_dir="/home/parlance/dualdiffusion/dataset/samples" \
  --raw_sample_format="int16" \
  --pretrained_model_name_or_path="/home/parlance/dualdiffusion/models/dualdiffusion2d_600_mclt_4vae_11" \
  --module="vae" \
  --train_batch_size=16 \
  --num_train_epochs=5000 \
  --checkpointing_steps=1235 \
  --checkpoints_total_limit=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=3e-5 \
  --report_to="tensorboard" \
  --resume_from_checkpoint=latest \
  --seed=200 \
  --lr_scheduler="constant_with_warmup" \
  --num_validation_samples=5 \
  --num_validation_epochs=5 \
  --kl_loss_weight=1e-4 \
  --max_grad_norm=5

# --gradient_checkpointing

# --snr_gamma=1 \
# --input_perturbation=0.1
# --dropout=0.1