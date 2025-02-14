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
from typing import Literal, Union

import torch
import torchaudio

from modules.embeddings.embedding import DualDiffusionEmbedding, DualDiffusionEmbeddingConfig
from modules.mp_tools import normalize


@dataclass
class CLAP_Config(DualDiffusionEmbeddingConfig):

    sample_rate:           int = 48000
    sample_crop_width:     int = 480000
    sample_raw_channels:   int = 1

    enable_fusion: bool = False
    audio_encoder: str = "HTSAT-base"
    text_encoder: str = "roberta"

    @property
    def embedding_dim(self) -> int:
        return 1024

    @property
    def audio_embedding_duration(self) -> int:
        return self.sample_crop_width / self.sample_rate

class CLAP_Embedding(DualDiffusionEmbedding):

    has_trainable_parameters: bool = False
    supports_half_precision: bool = False
    supports_compile: bool = False

    def __init__(self, config: CLAP_Config) -> None:
        super().__init__()
        self.config = config

        self.clap_model1 = None
        self.clap_model2 = None
        self.clap_processor = None
        self.clap_tokenizer = None

    def load_clap_model(self) -> None:
        # larger_clap_music
        from transformers import ClapModel, ClapProcessor, AutoTokenizer
        self.clap_model1: ClapModel = ClapModel.from_pretrained(config.CLAP_MODEL1_PATH).to(device=self.device, memory_format=self.memory_format)
        self.clap_processor: ClapProcessor = ClapProcessor.from_pretrained(config.CLAP_MODEL1_PATH)
        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(config.CLAP_MODEL1_PATH)

        # laion-clap
        import laion_clap
        self.clap_model2 = laion_clap.CLAP_Module(device=self.device,
            enable_fusion=self.config.enable_fusion, amodel=self.config.audio_encoder, tmodel=self.config.text_encoder)
        self.clap_model2.load_ckpt(config.CLAP_MODEL2_PATH, verbose=False)
        self.clap_model2 = self.clap_model2.to(device=self.device, memory_format=self.memory_format)

    def encode_audio(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:

        if audio.ndim == 2:
            audio = audio.mean(dim=0) # downmix to mono
        elif audio.ndim == 3:
            audio = audio.mean(dim=1).squeeze(0)  # downmix to mono
        elif audio.ndim != 1:
            raise ValueError("Tensor shape for encode_audio must be either (batch, channels, samples), (batch, samples), or (samples)")
        if self.clap_model1 is None:
            self.load_clap_model()

        # move to model device and resample if needed
        audio = audio.to(device=self.device, dtype=torch.float32)
        if sample_rate != self.config.sample_rate:
            audio = torchaudio.functional.resample(audio, sample_rate, self.config.sample_rate)
    
        # chunkify embedding audio
        chunk_size = self.config.sample_crop_width
        if audio.shape[-1] < chunk_size:
            raise ValueError(f"Cannot encode audio embedding, audio too short (len: {audio.shape[-1]})")
        audio = audio[:audio.shape[0] // chunk_size * chunk_size].reshape(-1, chunk_size)
        
        audio_features: torch.Tensor = self.clap_processor(audios=[chunk.cpu().numpy() for chunk in audio.unbind()],
                                    return_tensors="pt", sampling_rate=self.config.sample_rate)["input_features"]
        audio_embeddings1 = normalize(self.clap_model1.get_audio_features(audio_features.to(self.device)))
        audio_embeddings2 = normalize(self.clap_model2.get_audio_embedding_from_data(audio, use_tensor=True))

        return torch.cat((audio_embeddings1, audio_embeddings2), dim=1)

    def encode_text(self, text: list[str]) -> torch.Tensor:
        if not isinstance(text, list):
            raise ValueError("Parameter text must be list[str]")
        if self.clap_model1 is None:
            self.load_clap_model()
        
        tokens = self.tokenizer(text, return_tensors="pt", padding=True).to(device=self.device)
        text_embeddings1 = normalize(self.clap_model1.get_text_features(**tokens))
        text_embeddings2 = normalize(self.clap_model2.get_text_embedding(text, use_tensor=True))

        return torch.cat((text_embeddings1, text_embeddings2), dim=1)
    
    def encode_labels(self, labels: Union[int, torch.Tensor, list[int], dict[str, float]]) -> torch.Tensor:
        raise NotImplementedError()
        """
        if self.dataset_embeddings is None:
            self.load_dataset_embeddings()

        if isinstance(labels, int):
            labels = {}
        if self.config.embedding_type == "sum":
            unconditional_embedding = normalize(self.dataset_embeddings["_unconditional_audio"]).float().to(device=self.device)
            sample_embeddings = torch.zeros(self.config.embedding_dim, device=self.device)
            for game_name, weight in params.prompt.items():
                sample_embeddings += self.dataset_embeddings[f"{game_name}_audio"].to(device=unet.device) * weight
                sample_embeddings += self.dataset_embeddings[f"{game_name}_text"].to(device=unet.device) * weight
            sample_embeddings = normalize(sample_embeddings).float()
            unet_class_embeddings = unet.get_clap_embeddings(sample_embeddings, unconditional_embedding, conditioning_mask)
        elif unet.config.label_dim == 1024:
            unconditional_audio_embedding = normalize(self.dataset_embeddings["_unconditional_audio"]).float().to(device=unet.device)
            unconditional_text_embedding = normalize(self.dataset_embeddings["_unconditional_text"]).float().to(device=unet.device)
            unconditional_embedding = torch.cat((unconditional_audio_embedding, unconditional_text_embedding))
            sample_embeddings = torch.zeros(unet.config.label_dim, device=unet.device)
            for game_name, weight in params.prompt.items():
                sample_embeddings += torch.cat((self.dataset_embeddings[f"{game_name}_audio"].to(device=unet.device) * weight,
                                                self.dataset_embeddings[f"{game_name}_text"].to(device=unet.device) * weight))
            sample_embeddings = normalize(sample_embeddings).float()
            unet_class_embeddings = unet.get_clap_embeddings(sample_embeddings, unconditional_embedding, conditioning_mask)
        """