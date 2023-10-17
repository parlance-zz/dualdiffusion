REM remember to run accelerate config before running this script
accelerate launch^
 train.py^
 --mixed_precision="fp16"^
 --allow_tf32^
 --train_data_dir="./dataset/samples"^
 --raw_sample_format="int16"^
 --pretrained_model_name_or_path="./models/dualdiffusion2d_125"^
 --output_dir="./models/dualdiffusion2d_125"^
 --train_batch_size=2^
 --num_train_epochs=500^
 --checkpointing_steps=2470^
 --checkpoints_total_limit=1^
 --gradient_accumulation_steps=4^
 --learning_rate=1e-4^
 --report_to="tensorboard"^
 --resume_from_checkpoint=latest^
 --seed=100^
 --lr_scheduler="constant_with_warmup"^
 --lr_warmup_steps=2470^
 --num_validation_samples=10^
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