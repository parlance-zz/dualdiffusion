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
import os
import re as regex
import shutil
import logging

import torch

from dataset.dataset_processor import DatasetProcessor, DatasetProcessStage

class Import(DatasetProcessStage):

    def info_banner(self, logger: logging.Logger) -> None:
        root_dst_path = self.processor_config.import_dst_path or config.DATASET_PATH
        logger.info(f"Importing files from: {self.processor_config.import_paths}")
        logger.info(f"Importing files to: {root_dst_path}")

    def summary_banner(self, logger: logging.Logger) -> None:
        verb = "Moved" if self.processor_config.import_move_no_copy == True else "Copied"
        if self.processor_config.test_mode == True:
            logger.info(f"(Would have) {verb} {self.output_queue.queue.qsize()} files.")
        else:
            logger.info(f"{verb} {self.output_queue.queue.qsize()} files.")

    @torch.inference_mode()
    def start_process(self):
        self.filter_pattern = regex.compile(self.processor_config.import_filter_regex)     

    @torch.inference_mode()
    def process(self, input_dict: dict) -> Optional[Union[dict, list[dict]]]:
        match = self.filter_pattern.match(os.path.basename(input_dict["file_path"]))

        if match is not None:
            match = match.group(self.processor_config.import_filter_group)
            root_dst_path = self.processor_config.import_dst_path or config.DATASET_PATH

            src_path = os.path.normpath(input_dict["file_path"])
            rel_path = os.path.relpath(input_dict["file_path"], input_dict["scan_path"])
            dst_path = os.path.normpath(os.path.join(root_dst_path, os.path.dirname(rel_path), match))

            src_size = os.path.getsize(src_path)
            dst_size = os.path.getsize(dst_path) if os.path.isfile(dst_path) else 0

            # only move/copy if the destination doesn't already exist, unless force_overwrite is eanabled
            if dst_size == 0 or self.processor_config.force_overwrite == True:
                
                with self.critical_lock: # task will not be terminated inside this block
                    self.logger.debug(f"\"{src_path}\" -> \"{dst_path}\"")

                    if self.processor_config.test_mode == False: # test mode disallows writing changes
                        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

                         # copy_on_write not needed for move as it is already atomic
                        if self.processor_config.import_move_no_copy == True:
                            shutil.move(src_path, dst_path)
                        
                        # alternatively copy_on_write uses a temporary file to guarantee integrity
                        elif self.processor_config.copy_on_write == True:
                            tmp_path = f"{dst_path}.tmp"
                            try:
                                shutil.copy2(src_path, tmp_path)
                                shutil.move(tmp_path, dst_path)
                                if os.path.isfile(tmp_path):
                                    os.remove(tmp_path)

                            except Exception as e:
                                try:
                                    if os.path.isfile(tmp_path):
                                        os.remove(tmp_path)
                                except: pass
                                raise e
                        else:
                            shutil.copy2(src_path, dst_path)
                    
                    return {} # return an empty dict to tally files moved/copied in summary

            elif src_size != dst_size and self.processor_config.import_warn_file_size_mismatch == True:
                self.logger.warning( # if the files in the destination are not expected to be modified this might indicate corruption
                    f"Warning: file size mismatch: \"{src_path}\" ({src_size}) -> \"{dst_path}\" ({dst_size})")
        
        return None


if __name__ == "__main__":

    dataset_processor = DatasetProcessor()
    dataset_processor.process(
        "Import",
        [Import()],
        input=dataset_processor.config.import_paths,
    )

    os._exit(0)