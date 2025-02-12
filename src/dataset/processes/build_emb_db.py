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

from typing import Optional, Union, Any, Literal
from dataclasses import dataclass
import os
import logging

import torch
import safetensors.torch as safetensors

from dataset.dataset_processor import DatasetProcessor, DatasetProcessStage
from utils.dual_diffusion_utils import normalize, save_safetensors


@dataclass
class BuildEmbbedding_DB_ProcessConfig:
    pass

class BuildEmbeddingDB(DatasetProcessStage):

    def __init__(self, process_config: BuildEmbbedding_DB_ProcessConfig) -> None:
        self.process_config = process_config
        
    def get_stage_type(self) -> Literal["io", "cpu", "cuda"]:
        return "cpu"
    
    def info_banner(self, logger: logging.Logger):
        pass

    def summary_banner(self, logger: logging.Logger, completed: bool) -> None:
        
        # if the process was aborted don't write any emb db files
        if completed != True: return
        
        # aggregate samples from each worker process
        sample_audio_embs: dict[str, torch.Tensor] = {}
        sample_text_embs: dict[str, torch.Tensor] = {}
        
        i, num_worker_outputs = 1, self.output_queue.queue.qsize()
        while self.output_queue.queue.qsize() > 0:
            logger.info(f"Collecting data from worker processes ({i}/{num_worker_outputs})...")
            worker_output_dict: dict[str, dict] = self.output_queue.get()
            sample_audio_embs.update(**worker_output_dict["audio_embs"])
            sample_text_embs.update(**worker_output_dict["text_embs"])

            i += 1

        logger.info(f"Found {len(sample_audio_embs)} samples with audio embeddings")
        if len(sample_audio_embs) > 0 and self.processor_config.test_mode == False:
            audio_db_path = os.path.join(config.DATASET_PATH, "dataset_infos", "audio_emb_db.safetensors")
            save_safetensors(sample_audio_embs, audio_db_path, metadata={"emb_db_type": "clap_audio"}, copy_on_write=True)
            logger.info(f"Saved audio embeddings db at {audio_db_path}")

        logger.info(f"Found {len(sample_text_embs)} samples with text embeddings")
        if len(sample_text_embs) > 0 and self.processor_config.test_mode == False:
            text_db_path = os.path.join(config.DATASET_PATH, "dataset_infos", "text_emb_db.safetensors")
            save_safetensors(sample_text_embs, text_db_path, metadata={"emb_db_type": "clap_text"}, copy_on_write=True)
            logger.info(f"Saved text embeddings db at {text_db_path}")

        # fix stage processed/skipped stats
        total_processed = max(len(sample_audio_embs), len(sample_text_embs))
        self.input_queue.total_count.value = self.skip_counter.value
        self.input_queue.processed_count.value = total_processed
        self.skip_counter.value -= total_processed
        self.input_queue.total_count.value += self.skip_counter.value
        self.input_queue.processed_count.value += self.skip_counter.value

    @torch.inference_mode()
    def start_process(self) -> None:
        self.sample_audio_embs = {}
        self.sample_text_embs = {}

    @torch.inference_mode()
    def finish_process(self) -> None:
        self.output_queue.put(
            {
                "audio_embs": self.sample_audio_embs,
                "text_embs": self.sample_text_embs
            })

    @torch.inference_mode()
    def process(self, input_dict: dict) -> Optional[Union[dict, list[dict]]]:
        
        file_path = input_dict["file_path"]
        file_ext:str = os.path.splitext(file_path)[1]

        if file_ext == ".safetensors": # load latents metadata
    
            file_name = os.path.normpath(os.path.relpath(file_path, input_dict["scan_path"]))
            with safetensors.safe_open(file_path, framework="pt") as f:

                try:
                    audio_emb: torch.Tensor = f.get_slice("clap_audio_embeddings")[:]
                    self.logger.debug(f"\"{file_path}\" clap_audio_embeddings {audio_emb.shape}")
                    self.sample_audio_embs[file_name] = normalize(audio_emb.float().mean(dim=0)).to(dtype=torch.bfloat16)
                except:
                    pass

                try:
                    text_emb: torch.Tensor = f.get_slice("clap_text_embeddings")[:]
                    self.logger.debug(f"\"{file_path}\" clap_text_embeddings {text_emb.shape}")
                    self.sample_text_embs[file_name] = normalize(text_emb.float().mean(dim=0)).to(dtype=torch.bfloat16)
                except:
                    pass
        
        return None


if __name__ == "__main__":

    process_config: BuildEmbbedding_DB_ProcessConfig = config.load_config(BuildEmbbedding_DB_ProcessConfig,
                                             os.path.join(config.CONFIG_PATH, "dataset", "build_emb_db.json"))
    
    dataset_processor = DatasetProcessor()
    dataset_processor.process(
        "BuildEmbeddingDB",
        [BuildEmbeddingDB(process_config)],
        input=config.DATASET_PATH
    )

    exit(0)