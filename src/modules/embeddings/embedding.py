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
from abc import ABC, abstractmethod
from dataclasses import dataclass
import os

import torch

from modules.module import DualDiffusionModule, DualDiffusionModuleConfig


@dataclass
class DualDiffusionEmbeddingConfig(DualDiffusionModuleConfig, ABC):
    
    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        pass


class DualDiffusionEmbedding(DualDiffusionModule, ABC):

    module_name: str = "embedding"
    
    def __init__(self):
        super().__init__()

        self.dataset_info = None
        self.dataset_game_ids = None
        self.dataset_game_names = None

    def load_dataset_info(self) -> None:
        if self.module_path is not None:
            dataset_info_path = os.path.join(self.module_path, "dataset_info.json")
            if not os.path.isfile(dataset_info_path):
                dataset_info_path = os.path.join(config.DATASET_PATH, "dataset_infos", "dataset_info.json")
        else:
            dataset_info_path = os.path.join(config.DATASET_PATH, "dataset_infos", "dataset_info.json")
        if not os.path.isfile(dataset_info_path):
            raise FileNotFoundError(f"Could not find dataset info at '{dataset_info_path}'")

        self.dataset_info = config.load_json(dataset_info_path)
        self.dataset_game_ids = self.dataset_info["game_id"]
        self.dataset_game_names = {value: key for key, value in self.dataset_info["game_id"].items()}

    # input shape: (batch, emb_dim) output shapes: (batch,), (emb_dim, emb_dim), (batch, n_components)
    def get_pca(self, embeddings: torch.Tensor, n_components: Optional[int] = None) -> torch.Tensor:
        if embeddings.ndim != 2:
            raise ValueError("Parameter embeddings for get_embeddings_pca must have shape (batch, embedding_dim)")
        if n_components is not None and (n_components < 1 or n_components > embeddings.shape[1]):
            raise ValueError("Parameter n_components for get_embeddings_pca must be between 1 and embedding_dim")
        covariance_matrix = embeddings.T @ embeddings / (embeddings.shape[0] - 1)
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix)
        principal_components = embeddings @ eigenvectors[:, :n_components or embeddings.shape[1]]

        return eigenvalues, eigenvectors, principal_components
    
    # calculates cosine similarity between each pair of vectors in x and y (assumes x and y have unit variance)
    # input shapes: (batch_x, emb_dim), (batch_y, emb_dim) output shape: (batch_x, batch_y)
    def get_cosine_similarity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1: x = x.unsqueeze(0) # add missing batch dimension
        if y.ndim == 1: y = y.unsqueeze(0)
        if x.ndim != 2 or y.ndim != 2:
            raise ValueError("Both x and y must have shape (batch, embedding_dim) or (embedding_dim)")
        return torch.mm(x / x.shape[1]**0.5, y.T / y.shape[1]**0.5).clip(-1, 1) 
    
    # calculates the convolution of x and y where batch is the convolution dimension
    # input shapes: (batch_x, emb_dim), (batch_y, emb_dim) output shape: (batch_x,)
    def conv1d(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor: # (useful for duplicate detection)
        if x.ndim == 1: x = x.unsqueeze(0) # add missing batch dimension
        if y.ndim == 1: y = y.unsqueeze(0)
        if x.ndim != 2 or y.ndim != 2:
            raise ValueError("Both x and y must have shape (batch, embedding_dim) or (embedding_dim)")
        return torch.nn.functional.conv1d(x.T.unsqueeze(0), y.T.unsqueeze(0), padding="same").squeeze(0).T

    @abstractmethod
    def encode_audio(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        pass
    
    @abstractmethod
    def encode_text(self, text: list[str]) -> torch.Tensor:
        pass

    @abstractmethod
    def encode_labels(self, labels: Union[int, torch.Tensor, list[int], dict[str, float]]) -> torch.Tensor:
        pass

    def compile(self, **kwargs) -> None:
        if type(self).supports_compile == True:
            super().compile(**kwargs)
            self.encode_audio = torch.compile(self.encode_audio, **kwargs)
            self.encode_text = torch.compile(self.encode_text, **kwargs)
            self.encode_labels = torch.compile(self.encode_labels, **kwargs)