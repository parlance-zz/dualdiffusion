#!/bin/bash
accelerate launch \
  --config_file "%MODEL_PATH%/config/accelerate.yaml" \
  "%TRAIN_SCRIPT_PATH%" \
  --model_path="%MODEL_PATH%" \
  --train_config_path="%MODEL_PATH%/config/train_unet.json"