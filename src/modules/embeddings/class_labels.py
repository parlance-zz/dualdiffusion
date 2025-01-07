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

from dataclasses import dataclass
from typing import Union

import torch

from modules.embeddings.embedding import DualDiffusionEmbedding, DualDiffusionEmbeddingConfig


@dataclass
class ClassLabels_Config(DualDiffusionEmbeddingConfig):
    
    label_dim: int

    @property
    def embedding_dim(self) -> int:
        return self.label_dim
    

class ClassLabels_Embedding(DualDiffusionEmbedding):

    has_trainable_parameters: bool = False
    supports_half_precision: bool = False
    supports_compile: bool = False

    def __init__(self, config: ClassLabels_Config) -> None:
        super().__init__()
        self.config = config

    def encode_audio(self, audio: torch.Tensor, normalize_audio: bool = True) -> torch.Tensor:
        raise NotImplementedError("Method encode_audio is not implemented for ClassLabels_Embedding")

    def encode_text(self, text: list[str]) -> torch.Tensor:
        raise NotImplementedError("Method encode_text is not implemented for ClassLabels_Embedding")
    
    def get_class_label(self, label_name: Union[str, int]) -> int:

        if isinstance(label_name, int):
            return label_name
        elif isinstance(label_name, str):
            if self.dataset_game_ids is None:
                self.load_dataset_info()            
            return self.dataset_game_ids[label_name]
        else:
            raise ValueError(f"Unknown label type '{type(label_name)}'")
        
    @torch.no_grad()
    def encode_labels(self, labels: Union[int, torch.Tensor, list[int], dict[str, float]]) -> torch.Tensor:

        if isinstance(labels, int):
            labels = torch.tensor([labels])
        
        if isinstance(labels, torch.Tensor):
            if labels.ndim < 1: labels = labels.unsqueeze(0)
            class_labels = torch.nn.functional.one_hot(labels, num_classes=self.config.label_dim)
        
        elif isinstance(labels, list):
            class_labels = torch.zeros((1, self.config.label_dim))
            class_labels = class_labels.index_fill_(1, torch.tensor(labels, dtype=torch.long), 1)
        
        elif isinstance(labels, dict):
            _labels = {self.get_class_label(l): w for l, w in labels.items()}

            class_ids = torch.tensor(list(_labels.keys()), dtype=torch.long)
            weights = torch.tensor(list(_labels.values())).float()
            class_labels = torch.zeros((1, self.config.label_dim)).scatter_(1, class_ids.unsqueeze(0), weights.unsqueeze(0))
        else:
            raise ValueError(f"Unknown labels dtype '{type(labels)}'")
        
        return class_labels.to(device=self.device, dtype=self.dtype)