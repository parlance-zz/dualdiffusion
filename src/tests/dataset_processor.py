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
import time
import os

import torch

from dataset.dataset_processor import DatasetProcessor, DatasetProcessStage, WorkQueue


class TestStage(DatasetProcessStage):

    def info_banner(self, logger):
        logger.info("<Info Banner>")
    
    def summary_banner(self, logger):
        logger.info("<Summary Banner>")
    
    @torch.inference_mode()
    def start_process(self):
        self.test_errors = 0
        self.test_warnings = 0
    
    @torch.inference_mode()
    def process(self, input_dict: dict) -> Optional[Union[dict, list[dict]]]:
        self.logger.debug(input_dict)
        time.sleep(0.1)

        if self.test_errors < 5:
            self.test_errors += 1
            self.logger.error(f"<Test Error {self.test_errors}>")
        elif self.test_warnings < 10:
            self.test_warnings += 1
            self.logger.warning(f"<Test Warning {self.test_warnings}>")

        return None


if __name__ == "__main__":

    input_queue = WorkQueue()
    for i in range(50):
        input_queue.put({"test_data": i})

    DatasetProcessor().process(
        "TestProcess",
        [TestStage()],
        input=input_queue
    )

    os._exit(0)