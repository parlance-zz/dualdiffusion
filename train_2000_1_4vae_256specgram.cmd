REM remember to run accelerate config before running this script
accelerate launch^
 train.py^
 --mixed_precision="fp16"^
 --pretrained_model_name_or_path="D:/dualdiffusion/models/dualdiffusion2d_2000_1"^
 --module="vae"^
 --train_batch_size=2^
 --num_train_epochs=5000^
 --gradient_accumulation_steps=4^
 --learning_rate=1e-4^
 --seed=400^
 --lr_scheduler="constant_with_warmup"^
 --max_grad_norm=10^
 --dataloader_num_workers=4

REM --gradient_checkpointing
REM --use_ema