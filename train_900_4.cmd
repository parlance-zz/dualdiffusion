REM remember to run accelerate config before running this script
accelerate launch^
 train.py^
 --mixed_precision="fp16"^
 --allow_tf32^
 --train_data_dir="D:/dualdiffusion/dataset/samples"^
 --raw_sample_format="int16"^
 --pretrained_model_name_or_path="D:/dualdiffusion/models/dualdiffusion2d_900_4"^
 --module="unet"^
 --train_batch_size=18^
 --num_train_epochs=5000^
 --checkpointing_steps=1098^
 --checkpoints_total_limit=1^
 --gradient_accumulation_steps=1^
 --learning_rate=2e-4^
 --report_to="tensorboard"^
 --resume_from_checkpoint=latest^
 --seed=300^
 --lr_scheduler="constant_with_warmup"^
 --num_validation_samples=4^
 --num_validation_epochs=5^
 --kl_loss_weight=0^
 --max_grad_norm=1

REM --checkpointing_steps=2470^
REM --max_grad_norm=25
 
REM --phase_augmentation=False

REM --gradient_checkpointing


REM --lr_warmup_steps=2470^
REM --snr_gamma=1^
REM --input_perturbation=0.1

REM --dropout=0.1
REM --snr_gamma=5
REM --input_perturbation=0.1

REM --sample_format="float32"^
REM --train_data_dir="./dataset/samples_micro"^

REM --num_validation_samples=0
REM --use_ema