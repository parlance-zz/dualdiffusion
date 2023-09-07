REM remember to run accelerate config before running this script
accelerate launch^
 train1d.py^
 --allow_tf32^
 --train_data_dir="./dataset/samples"^
 --pretrained_model_name_or_path="./models/dualdiffusion1d_14"^
 --output_dir="./models/dualdiffusion1d_14"^
 --train_batch_size=4^
 --num_train_epochs=500^
 --checkpointing_steps=4555^
 --checkpoints_total_limit=1^
 --gradient_accumulation_steps=1^
 --learning_rate=1e-4^
 --report_to="tensorboard"^
 --resume_from_checkpoint=latest^
 --seed=42^
 --num_validation_samples=0

REM --snr_gamma=5^
REM --use_ema