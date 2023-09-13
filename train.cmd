REM remember to run accelerate config before running this script
accelerate launch^
 train.py^
 --mixed_precision="fp16"^
 --allow_tf32^
 --train_data_dir="./dataset/samples"^
 --pretrained_model_name_or_path="./models/dualdiffusion2d_28"^
 --output_dir="./models/dualdiffusion2d_28"^
 --train_batch_size=12^
 --num_train_epochs=500000^
 --checkpointing_steps=3038^
 --checkpoints_total_limit=1^
 --gradient_accumulation_steps=1^
 --learning_rate=1e-4^
 --report_to="tensorboard"^
 --resume_from_checkpoint=latest^
 --seed=43

REM --sample_format="fp32"^
REM --train_data_dir="./dataset/samples_micro"^
REM --num_validation_samples=0
REM --input_perturbation=0.1
REM --snr_gamma=5^
REM --use_ema