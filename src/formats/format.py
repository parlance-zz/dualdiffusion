from typing import Optional, Union
from abc import ABC, abstractmethod

import torch

class DualDiffusionFormat(ABC):

    @abstractmethod
    def get_raw_crop_width(self, length: Optional[int] = None) -> int:
        pass
    
    @abstractmethod
    def get_num_channels(self) -> tuple[int, int]:
        pass
    
    @abstractmethod
    def get_sample_shape(self, bsz: int = 1, length: Optional[int] = None) -> tuple:
        pass

    @abstractmethod
    def raw_to_sample(self, raw_samples: torch.Tensor, return_dict: bool = False) -> Union[torch.Tensor, dict]:
        pass

    @abstractmethod
    def sample_to_raw(self, samples: torch.Tensor, return_dict: bool = False) -> Union[torch.Tensor, dict]:
        pass
    
    @abstractmethod
    def get_ln_freqs(self, x: torch.Tensor) -> torch.Tensor:
        pass