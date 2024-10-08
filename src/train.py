# MIT License
#
# Copyright (c) 2023 Christopher Friesen
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import utils.config as config

import os
import argparse

from utils.dual_diffusion_utils import init_cuda
from training.trainer import DualDiffusionTrainer, DualDiffusionTrainerConfig

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DualDiffusion training script.")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to pretrained / new model",
    )
    parser.add_argument(
        "--train_config_path",
        type=str,
        required=True,
        help="Path to training configuration json file",
    )
    return parser.parse_args()
    
if __name__ == "__main__":

    init_cuda()
    args = parse_args()

    train_config = DualDiffusionTrainerConfig.from_json(args.train_config_path,
                                                        model_path=args.model_path,
                                                        model_name=os.path.basename(args.model_path),
                                                        model_src_path=config.SRC_PATH)
    trainer = DualDiffusionTrainer(train_config)
    trainer.train()