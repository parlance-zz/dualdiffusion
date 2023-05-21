REM remember to run accelerate config before running this script
accelerate launch^
 --mixed_precision="fp16"^
 train_dual_diffusion.py^
 --train_data_dir="./dataset/dual"^
 --model_config_name_or_path="./models/new_dualdiffusion"^
 --dual_training_mode="f"^
 --output_dir="./models/dualdiffusion"^
 --train_batch_size=128^
 --num_epochs=1000^
 --checkpointing_steps=5000
 --save_model_epochs=10
 --gradient_accumulation_steps=2^
 --use_ema^
 --learning_rate=1e-4^
 --lr_warmup_steps=500^
 --mixed_precision="fp16"^
 --logger="tensorboard"