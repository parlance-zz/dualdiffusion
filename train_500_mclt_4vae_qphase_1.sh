#!/bin/bash

accelerate launch \
  train.py \
  --mixed_precision="fp16" \
  --allow_tf32 \
  --train_data_dir="/home/parlance/dualdiffusion/dataset/samples" \
  --raw_sample_format="int16" \
  --pretrained_model_name_or_path="/home/parlance/dualdiffusion/models/dualdiffusion2d_500_mclt_4vae_qphase_1" \
  --module="vae" \
  --train_batch_size=4 \
  --num_train_epochs=5000 \
  --checkpointing_steps=618 \
  --checkpoints_total_limit=1 \
  --gradient_accumulation_steps=8 \
  --learning_rate=1e-4 \
  --report_to="tensorboard" \
  --resume_from_checkpoint=latest \
  --seed=200 \
  --lr_scheduler="constant_with_warmup" \
  --num_validation_samples=5 \
  --num_validation_epochs=5 \
  --kl_loss_weight=1e-5 \
  --kl_loss_recon_weight=0 \
  --max_grad_norm=5

# --gradient_checkpointing

# --snr_gamma=1 \
# --input_perturbation=0.1
# --dropout=0.1