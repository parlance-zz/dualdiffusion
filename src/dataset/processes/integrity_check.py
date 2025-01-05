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

from typing import Optional, Union
import os
import logging
import subprocess

import torch

from dataset.dataset_processor import DatasetProcessor, DatasetProcessStage

class IntegrityCheck(DatasetProcessStage):
    
    def summary_banner(self, logger: logging.Logger) -> None:
        logger.info(f"{self.output_queue.queue.qsize()} files ok, {self.error_queue.qsize()} files with errors")

    @torch.inference_mode()
    def process(self, input_dict: dict) -> Optional[Union[dict, list[dict]]]:

        file_path = input_dict["file_path"]
        result = subprocess.run(["flac", "-t", file_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if result.returncode == 0:
            self.logger.debug(f"\"{file_path}\" ok")
            return {}
        else:
            self.logger.error(f"error reading \"{file_path}\"")
            return None

if __name__ == "__main__":

    dataset_processor = DatasetProcessor()
    dataset_processor.process(
        "IntegrityCheck",
        [IntegrityCheck()],
    )

    os._exit(0)