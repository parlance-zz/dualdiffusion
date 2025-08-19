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
from training.module_trainers.unet_trainer import UNetTrainer, UNetTrainerConfig
from modules.unets.unet_edm2_ddec_mdct_p3 import DDec_MDCT_UNet_P3
from modules.formats.mdct_psd import MDCT_PSD_Format
from modules.mp_tools import normalize


def random_stereo_augmentation(x: torch.Tensor) -> torch.Tensor:
    
    output = x.clone()
    flip_mask = (torch.rand(x.shape[0]) > 0.5).to(x.device)
    output[flip_mask] = output[flip_mask].flip(dims=(1,))
    
    return output

@dataclass
class DiffusionDecoder_Trainer_Config(UNetTrainerConfig):

    loss_weight_pow: float = 0.25
    loss_weight_min: float = 0.15

    crop_edges: int = 2 # used to avoid artifacts due to mdct lapped blocks at beginning and end of sample

class DiffusionDecoder_Trainer(UNetTrainer):
    
    @torch.no_grad()
    def __init__(self, config: DiffusionDecoder_Trainer_Config, trainer: DualDiffusionTrainer) -> None:

        self.config = config
        self.trainer = trainer
        self.logger = trainer.logger

        self.ddec: DDec_MDCT_UNet_P3 = trainer.get_train_module("ddec")
        self.format: MDCT_PSD_Format = trainer.pipeline.format.to(self.trainer.accelerator.device)

        if trainer.config.enable_model_compilation:
            self.ddec.compile(**trainer.config.compile_params)
            self.format.compile(**trainer.config.compile_params)

        self.logger.info(f"Training module: {trainer.config.train_modules}")
        self.logger.info(f"Loss weight power: {self.config.loss_weight_pow}")
        self.logger.info(f"Loss weight min: {self.config.loss_weight_min}")
        self.logger.info(f"Crop edges: {self.config.crop_edges}")

        self.unet = self.ddec
        self.unet_trainer_init(crop_edges=config.crop_edges)

        """
        def reset_adam_optimizer_state(optimizer):
            # reset ema of momentum and variance
            for state in optimizer.state.values():
                if isinstance(state, dict):
                    state["exp_avg"].zero_()
                    state["exp_avg_sq"].fill_(1)
        
        reset_adam_optimizer_state(trainer.optimizer)
        """

    def train_batch(self, batch: dict) -> Optional[dict[str, Union[torch.Tensor, float]]]:

        # prepare model inputs
        if "audio_embeddings" in batch:
            audio_embeddings = normalize(batch["audio_embeddings"]).detach()
        else:
            audio_embeddings = None

        #if self.config.random_stereo_augmentation == True:
        #    raw_samples = random_stereo_augmentation(batch["audio"])
        #else:
        #    raw_samples = batch["audio"]
        raw_samples = batch["audio"]

        with torch.no_grad():
            
            mdct: torch.Tensor = self.format.raw_to_mdct(raw_samples, random_phase_augmentation=True).detach()
            mdct_psd: torch.Tensor = self.format.raw_to_mdct_psd(raw_samples).requires_grad_(False).detach()
            mdct_scaled = self.format.scale_mdct_from_psd(mdct, mdct_psd).detach()
            #p2m: torch.Tensor = self.format.mdct_to_p2m(mdct).detach()
            #p2m_psd = self.format.mdct_to_p2m_psd(mdct).requires_grad_(False).detach()
            #p2m_scaled = self.format.scale_p2m_from_psd(p2m, p2m_psd).detach()
        
        #loss_weight = ((p2m_psd.clip(min=0) * self.format.p2m_mel_density) ** self.config.loss_weight_pow).requires_grad_(False).detach()
        loss_weight = ((mdct_psd.clip(min=0) * self.format.mdct_mel_density) ** self.config.loss_weight_pow).requires_grad_(False).detach()

        if self.trainer.config.enable_debug_mode == True:
            print("loss_weight min:", loss_weight.mean(dim=(1,2,3)).amin().item())
            print("loss_weight avg:", loss_weight.mean().item())

        loss_weight = (loss_weight / loss_weight.mean(dim=(1,2,3), keepdim=True).clip(min=self.config.loss_weight_min)).requires_grad_(False).detach()
        #return self.unet_train_batch(p2m_scaled, audio_embeddings, p2m_psd, loss_weight=loss_weight)
        return self.unet_train_batch(mdct_scaled, audio_embeddings, mdct_psd, loss_weight=loss_weight)