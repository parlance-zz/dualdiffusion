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
from typing import Union, Optional

import torch

from training.trainer import DualDiffusionTrainer
from training.module_trainers.module_trainer import ModuleTrainer
from training.module_trainers.unet_trainer_p4 import UNetTrainerConfig, UNetTrainer
from modules.unets.unet_edm2_p4 import UNet
from modules.mp_tools import normalize


@dataclass
class UNetTrainer_LDM_Config(UNetTrainerConfig):
    pass

class UNetTrainer_LDM(ModuleTrainer):
    
    @torch.no_grad()
    def __init__(self, config: UNetTrainer_LDM_Config, trainer: DualDiffusionTrainer) -> None:

        self.config = config
        self.trainer = trainer
        self.logger = trainer.logger

        self.unet: UNet = trainer.get_train_module("unet")

        if trainer.config.enable_model_compilation:
            self.unet.compile(**trainer.config.compile_params)

        self.logger.info(f"Training LDM: {trainer.config.train_modules}")
        self.unet_trainer = UNetTrainer(UNetTrainerConfig(**config.__dict__), trainer, self.unet, "unet")

        #self.trainer.optimizer.optimizer.zero_momentum()

    @torch.no_grad()
    def init_batch(self, validation: bool = False) -> Optional[dict[str, Union[torch.Tensor, float]]]:
        return self.unet_trainer.init_batch(validation)
    
    def train_batch(self, batch: dict) -> Optional[dict[str, Union[torch.Tensor, float]]]:

        latents: torch.Tensor = batch["latents"].float().clone()
        audio_embeddings = normalize(batch["audio_embeddings"]).float().clone().detach()
        
        logs = self.unet_trainer.train_batch(latents, embeddings=audio_embeddings)
        logs["loss"] = logs["loss/unet"]

        logs.update({
            "io_stats/latents_var": latents.var(dim=(1,2,3)),
            "io_stats/latents_mean": latents.mean(dim=(1,2,3))
        })

        if self.trainer.config.enable_debug_mode == True:
            print("latents.shape:", latents.shape)
            print("audio_embeddings.shape:", audio_embeddings.shape)

        return logs
      
    @torch.no_grad()
    def finish_batch(self) -> Optional[dict[str, Union[torch.Tensor, float]]]:
        return self.unet_trainer.finish_batch()