REM remember to run accelerate config before running this script
accelerate launch^
 --mixed_precision="fp16"^
 train.py^
 --mixed_precision="fp16"^
 --train_data_dir="./dataset/samples"^
 --model_config_name_or_path="./models/lgdiffusion_freq9"^
 --output_dir="./models/lgdiffusion_freq9"^
 --snr_gamma=5.0^
 --train_batch_size=20^
 --num_epochs=1000^
 --checkpointing_steps=1822^
 --checkpoints_total_limit=1^
 --gradient_accumulation_steps=1^
 --learning_rate=3e-5^
 --lr_warmup_steps=1822^
 --mixed_precision="fp16"^
 --logger="tensorboard"^
 --resume_from_checkpoint=latest
REM --use_ema
REM --resume_from_checkpoint=latest^