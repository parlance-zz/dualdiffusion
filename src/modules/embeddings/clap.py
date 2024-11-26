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

import torch
import laion_clap

from modules.embeddings.embedding import DualDiffusionEmbedding, DualDiffusionEmbeddingConfig
from utils.dual_diffusion_utils import normalize


@dataclass
class CLAP_Config(DualDiffusionEmbeddingConfig):
    enable_fusion: bool = False
    audio_encoder: str = "HTSAT-base"
    text_encoder: str = "roberta"
    clap_model_filename: str = "music_audioset_epoch_15_esc_90.14.pt"

class CLAP_Embedding(DualDiffusionEmbedding):

    has_trainable_parameters: bool = False
    supports_half_precision: bool = False
    supports_compile: bool = False

    def __init__(self, config: CLAP_Config) -> None:
        super().__init__()
        self.config = config
        self.clap_model = None

    def load_clap_model(self) -> None:
        if self.module_path is not None:
            clap_model_path = os.path.join(self.module_path, self.config.clap_model_filename)
            if not os.path.isfile(clap_model_path):
                clap_model_path = config.CLAP_MODEL_PATH
        else:
            clap_model_path = config.CLAP_MODEL_PATH

        if not os.path.isfile(clap_model_path):
            raise FileNotFoundError(f"CLAP model file not found")

        self.clap_model = laion_clap.CLAP_Module(device=self.device,
            enable_fusion=self.config.enable_fusion, amodel=self.config.audio_encoder, tmodel=self.config.text_encoder)
        self.clap_model.load_ckpt(clap_model_path, verbose=False)
        self.clap_model = self.clap_model.to(device=self.device, memory_format=self.memory_format)
        
        if self.use_compile == True:
            self.clap_model.get_audio_embedding_from_data = torch.compile(self.clap_model.get_audio_embedding_from_data, **self.compile_options)
            self.clap_model.get_text_embedding = torch.compile(self.clap_model.get_text_embedding, **self.compile_options)

    def encode_audio(self, audio: torch.Tensor, normalize_audio: bool = True) -> torch.Tensor:
        if audio.ndim == 3:
            audio = audio.mean(dim=1)  # downmix to mono
        elif audio.ndim == 1:
            audio = audio.unsqueeze(0) # add batch dimension
        elif audio.ndim != 2:
            raise ValueError("Tensor shape for encode_audio must be either (batch, channels, samples), (batch, samples), or (samples)")
        if self.clap_model is None:
            self.load_clap_model()

        audio = audio.to(device=self.device, dtype=torch.float32)
        if normalize_audio == True:
            audio = audio / (audio.abs().amax(dim=-1, keepdim=True) + 1e-4)
        
        audio_embeddings = torch.cat([self.clap_model.get_audio_embedding_from_data(chunk, use_tensor=True)
            for chunk in audio.split(self.config.max_audio_batch)], dim=0)
        return normalize(audio_embeddings).float()

    def encode_text(self, text: list[str]) -> torch.Tensor:
        if not isinstance(text, list):
            raise ValueError("Parameter text must be list[str]")
        if self.clap_model is None:
            self.load_clap_model()

        text_batches = [text[i:i + self.config.max_text_batch] for i in range(0, len(text), self.config.max_text_batch)]
        text_embeddings = torch.cat([self.clap_model.get_text_embedding(batch, use_tensor=True) for batch in text_batches], dim=0)
        return normalize(text_embeddings).float()