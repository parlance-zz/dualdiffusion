REM remember to run accelerate config before running this script
accelerate launch^
 train.py^
 --mixed_precision="fp16"^
 --allow_tf32^
 --train_data_dir="D:/dualdiffusion/dataset/samples"^
 --raw_sample_format="int16"^
 --pretrained_model_name_or_path="D:/dualdiffusion/models/dualdiffusion2d_312_v3_64embed"^
 --train_batch_size=3^
 --num_train_epochs=500^
 --checkpointing_steps=2196^
 --checkpoints_total_limit=1^
 --gradient_accumulation_steps=3^
 --learning_rate=1e-4^
 --report_to="tensorboard"^
 --resume_from_checkpoint=latest^
 --seed=200^
 --lr_scheduler="constant_with_warmup"^
 --lr_warmup_steps=2196^
 --num_validation_samples=5^
 --num_validation_epochs=5

REM --snr_gamma=1^
REM --input_perturbation=0.1

REM --dropout=0.1                        
REM --snr_gamma=5
REM --input_perturbation=0.1

REM --sample_format="float32"^
REM --train_data_dir="./dataset/samples_micro"^

REM --num_validation_samples=0
REM --use_ema