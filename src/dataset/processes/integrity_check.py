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

from utils import config

from typing import Optional, Union
from dataclasses import dataclass
import os
import logging
import subprocess

import torch

from dataset.dataset_processor import DatasetProcessor, DatasetProcessStage
from utils.dual_diffusion_utils import load_safetensors_ex


@dataclass
class IntegrityCheckProcessConfig:
    delete_corrupt_files: bool = False # delete any flac or safetensors files that fail integrity check

class IntegrityCheck(DatasetProcessStage): # please note to run this process you need the "flac" utility in your environment path
    
    def __init__(self, process_config: IntegrityCheckProcessConfig) -> None:
        self.process_config = process_config

    def info_banner(self, logger: logging.Logger) -> None:
        if self.process_config.delete_corrupt_files == True:
            logger.warning("WARNING: Automatic deletion of corrupt files is enabled")

    def summary_banner(self, logger: logging.Logger, completed: bool) -> None:
        logger.info(f"{self.output_queue.queue.qsize()} files ok, {self.error_queue.qsize()} files with errors")

    @torch.inference_mode()
    def process(self, input_dict: dict) -> Optional[Union[dict, list[dict]]]:

        file_path = input_dict["file_path"]
        file_ext = os.path.splitext(file_path)[1].lower()
        corrupt_file = False

        if file_ext == ".flac":
            result = subprocess.run(["flac", "-t", file_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            if result.returncode != 0:
                self.logger.error(f"error reading \"{file_path}\"")
                corrupt_file = True
            
        elif file_ext == ".safetensors":      
            try:
                load_safetensors_ex(file_path)
            except Exception as e:
                self.logger.error(f"error reading \"{file_path}\" ({e})")
                corrupt_file = True
        else:
            return None

        if corrupt_file == True:
            if self.process_config.delete_corrupt_files == True:
                if self.processor_config.test_mode == False:
                    os.remove(file_path)
                    self.logger.debug(f"deleted \"{file_path}\"")
                else:
                    self.logger.debug(f"(would have) deleted \"{file_path}\"")
        else:
            self.logger.debug(f"\"{file_path}\" ok")
            return {}
        
if __name__ == "__main__":

    process_config: IntegrityCheckProcessConfig = config.load_config(IntegrityCheckProcessConfig,
                        os.path.join(config.CONFIG_PATH, "dataset", "integrity_check.json"))
    
    dataset_processor = DatasetProcessor()
    dataset_processor.process(
        "IntegrityCheck",
        [IntegrityCheck(process_config)],
    )

    exit(0)