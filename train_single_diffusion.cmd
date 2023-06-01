REM remember to run accelerate config before running this script
accelerate launch^
 --mixed_precision="fp16"^
 train_single_diffusion.py^
 --train_data_dir="./dataset/single"^
 --model_config_name_or_path="./models/singlediffusion"^
 --output_dir="./models/singlediffusion"^
 --train_batch_size=75^
 --num_epochs=10000^
 --checkpointing_steps=100000000^
 --checkpoints_total_limit=10^
 --save_model_epochs=1^
 --gradient_accumulation_steps=3^
 --learning_rate=1e-4^
 --lr_warmup_steps=500^
 --mixed_precision="fp16"^
 --logger="tensorboard"^
 --use_ema