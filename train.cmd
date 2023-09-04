REM remember to run accelerate config before running this script
accelerate launch^
 train.py^
 --mixed_precision="fp16"^
 --train_data_dir="./dataset/samples"^
 --model_config_name_or_path="./models/new_lgdiffusion5"^
 --output_dir="./models/new_lgdiffusion5"^
 --train_batch_size=4^
 --num_epochs=1000^
 --checkpointing_steps=6076^
 --checkpoints_total_limit=1^
 --gradient_accumulation_steps=1^
 --learning_rate=1e-4^
 --lr_warmup_steps=3644^
 --logger="tensorboard"^
 --resume_from_checkpoint=latest
REM --use_ema
REM --resume_from_checkpoint=latest^