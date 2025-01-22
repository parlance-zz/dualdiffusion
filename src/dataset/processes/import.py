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
from utils.dual_diffusion_utils import get_audio_info

class Import(DatasetProcessStage):

    def info_banner(self, logger: logging.Logger) -> None:
        root_dst_path = self.processor_config.import_dst_path or config.DATASET_PATH
        logger.info(f"Importing files from: {self.processor_config.import_paths}")
        logger.info(f"Importing files to: {root_dst_path}")
        if self.processor_config.import_move_no_copy == True:
            logger.info("WARNING: Moving imported files instead of copying: enabled")
        if self.processor_config.import_overwrite_if_larger == True:
            logger.info("WARNING: Overwrite destination files if source file is larger: enabled")
        if self.processor_config.import_delete_short_samples == True:
            if self.processor_config.min_audio_length is not None:
                logger.info(f"WARNING: Deleting source files below minimum length ({self.processor_config.min_audio_length}s): enabled")

    def summary_banner(self, logger: logging.Logger, completed: bool) -> None:
        verb = "Moved" if self.processor_config.import_move_no_copy == True else "Copied"
        if self.processor_config.test_mode == True:
            logger.info(f"(Would have) {verb} {self.output_queue.queue.qsize()} files.")
        else:
            logger.info(f"{verb} {self.output_queue.queue.qsize()} files.")

    @torch.inference_mode()
    def start_process(self):
        if self.processor_config.import_filter_regex is not None:
            self.filter_pattern = regex.compile(self.processor_config.import_filter_regex)
        else: # default is *.flac, preserves original filename
            self.filter_pattern = "(?i)^.*\\.flac$"
            self.processor_config.import_filter_group = 0

    @torch.inference_mode()
    def process(self, input_dict: dict) -> Optional[Union[dict, list[dict]]]:
        match = self.filter_pattern.match(os.path.basename(input_dict["file_path"]))

        if match is not None:
            match = match.group(self.processor_config.import_filter_group)
            root_dst_path = self.processor_config.import_dst_path or config.DATASET_PATH

            src_path = os.path.normpath(input_dict["file_path"])
            src_size = os.path.getsize(src_path)
            src_rel_path = os.path.normpath(os.path.relpath(input_dict["file_path"], input_dict["scan_path"]))
            src_rel_dir: str = os.path.dirname(src_rel_path)

            # if sample is too short:
            audio_info = get_audio_info(src_path)
            min_length = self.processor_config.min_audio_length
            if min_length is not None and audio_info.duration < min_length:
                if self.processor_config.import_delete_short_samples != True:
                    return None # skip it
                else: # or delete it
                    self.logger.debug(f"Deleting \"{src_path}\" source file length ({audio_info.duration}s) is below config min length ({min_length}s)")
                    os.remove(src_path)
                    return None
        
            # relative path normalization
            src_rel_dir_parts = [folder for folder in src_rel_dir.split(os.sep) if folder]
            if src_rel_dir_parts == ["."]: src_rel_dir_parts = []

            if self.processor_config.import_min_tree_depth is not None:
                for i in range(self.processor_config.import_min_tree_depth - len(src_rel_dir_parts)):
                    src_rel_dir_parts += [f"_t_{i}"]

            dst_file_prefix = ""
            if self.processor_config.import_max_tree_depth is not None:
                dst_file_prefix = "_".join(src_rel_dir_parts[self.processor_config.import_max_tree_depth:])
                if len(dst_file_prefix) > 0:
                    dst_file_prefix = f"_t_{dst_file_prefix}_"
                src_rel_dir_parts = src_rel_dir_parts[:self.processor_config.import_max_tree_depth]

            # destination path construction
            dst_rel_dir = os.path.join(*src_rel_dir_parts) if len(src_rel_dir_parts) > 0 else ""
            dst_path = os.path.normpath(os.path.join(root_dst_path, dst_rel_dir, f"{dst_file_prefix}{match}"))
            dst_size = os.path.getsize(dst_path) if os.path.isfile(dst_path) else 0

            if dst_size != 0 and src_size != dst_size and self.processor_config.import_warn_file_size_mismatch == True:
                self.logger.warning( # if the files in the destination are not expected to be modified this might indicate corruption
                    f"Warning: file size mismatch: \"{src_path}\" ({src_size}) -> \"{dst_path}\" ({dst_size})")
                
            # only move/copy if the destination doesn't already exist, unless force_overwrite is enabled
            write_file = (dst_size == 0 or self.processor_config.force_overwrite == True)

            # OR src_size is larger than dst_size and import_overwrite_if_larger is enabled
            if dst_size != 0 and src_size > dst_size and self.processor_config.import_overwrite_if_larger == True:
                self.logger.warning(
                    f"Warning: overwriting smaller existing file: \"{src_path}\" ({src_size}) -> \"{dst_path}\" ({dst_size})")
                write_file = True

            if write_file == True:
                self.logger.debug(f"\"{src_path}\" -> \"{dst_path}\"")

                if self.processor_config.test_mode == False: # test mode disallows writing changes
                    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

                    # copy_on_write not needed for move as it is already atomic
                    if self.processor_config.import_move_no_copy == True:
                        try: os.rename(src_path, dst_path)
                        except: shutil.move(src_path, dst_path)
                    else:
                        # alternatively, copying uses a temporary file to guarantee integrity
                        tmp_path = f"{dst_path}.tmp"
                        try:
                            shutil.copy2(src_path, tmp_path)
                            os.rename(tmp_path, dst_path)
                            if os.path.isfile(tmp_path):
                                os.remove(tmp_path)

                        except Exception as e:
                            try:
                                if os.path.isfile(tmp_path):
                                    os.remove(tmp_path)
                            except: pass
                            raise e
                    
                    return {} # return an empty dict to tally files moved/copied in summary
        
        return None


if __name__ == "__main__":

    dataset_processor = DatasetProcessor()
    dataset_processor.process(
        "Import",
        [Import()],
        input=dataset_processor.config.import_paths,
    )

    exit(0)