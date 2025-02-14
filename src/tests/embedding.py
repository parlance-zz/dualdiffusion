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

from dataclasses import dataclass
import os
import random

import torch

from modules.embeddings.clap import CLAP_Config, CLAP_Embedding
from utils.dual_diffusion_utils import init_cuda, load_audio, tensor_info_str


@dataclass
class EmbeddingTestConfig:
    device: str = "cuda"
    add_random_test_samples: int = 4
    test_text: list[str] = ()

@torch.inference_mode()
def embedding_test():

    cfg: EmbeddingTestConfig = config.load_config(EmbeddingTestConfig,
        os.path.join(config.CONFIG_PATH, "tests", "embedding.json"))
    
    clap: CLAP_Embedding = CLAP_Embedding(CLAP_Config()).to(device=cfg.device)

    train_samples = config.load_json(os.path.join(config.DATASET_PATH, "train.jsonl"))
    test_samples = [sample["file_name"] for sample in random.sample(train_samples, cfg.add_random_test_samples)]
    
    for sample in test_samples:
        audio = load_audio(os.path.join(config.DATASET_PATH, sample)).to(device=cfg.device)
        audio_embeddings = clap.encode_audio(audio, sample_rate=32000)
        print(f"encoding audio: '{sample}'\n{tensor_info_str(audio_embeddings)}\n")

    if len(cfg.test_text) > 0:
        text_embeddings = clap.encode_text(cfg.test_text)
        print(f"encoding text: {cfg.test_text}\n{tensor_info_str(text_embeddings)}")


if __name__ == "__main__":

    init_cuda()
    embedding_test()