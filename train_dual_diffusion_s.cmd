REM remember to run accelerate config before running this script
accelerate launch^
 --mixed_precision="fp16"^
 train_dual_diffusion.py^
 --train_data_dir="./dataset/dual"^
 --model_config_name_or_path="./models/dualdiffusion_s"^
 --dual_training_mode="s"^
 --output_dir="./models/dualdiffusion_s"^
 --num_epochs=10000^
 --checkpointing_steps=100000000^
 --checkpoints_total_limit=10^
 --save_model_epochs=1^
 --gradient_accumulation_steps=1^
 --learning_rate=1e-4^
 --lr_warmup_steps=200^
 --mixed_precision="fp16"^
 --logger="tensorboard"^
 --use_ema