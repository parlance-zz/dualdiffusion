REM remember to run accelerate config before running this script
accelerate launch^
 train1d.py^
 --allow_tf32^
 --mixed_precision="fp16"^
 --train_data_dir="./dataset/samples"^
 --pretrained_model_name_or_path="./models/dualdiffusion1d_19"^
 --output_dir="./models/dualdiffusion1d_19"^
 --train_batch_size=16^
 --num_train_epochs=500^
 --checkpointing_steps=3417^
 --checkpoints_total_limit=1^
 --gradient_accumulation_steps=1^
 --learning_rate=1e-4^
 --report_to="tensorboard"^
 --resume_from_checkpoint=latest^
 --seed=42^
 --num_validation_samples=0^
 --snr_gamma=5^
 --input_perturbation=0.1

REM --snr_gamma=5^
REM --use_ema