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
from typing import Union, Optional, Any

import torch

from training.trainer import DualDiffusionTrainer
from training.module_trainers.module_trainer import ModuleTrainer, ModuleTrainerConfig
from training.module_trainers.unet_trainer_p4 import UNetTrainerConfig, UNetTrainer
from modules.unets.unet_edm2_p4_ddec import UNet
from modules.formats.ms_mdct_dual_2 import MS_MDCT_DualFormat
from modules.mp_tools import normalize


@torch.no_grad()
def random_stereo_augmentation(x: torch.Tensor) -> torch.Tensor:
    
    output = x.clone()
    flip_mask = (torch.rand(x.shape[0]) > 0.5).to(x.device)
    output[flip_mask] = output[flip_mask].flip(dims=(1,))
    
    return output

@dataclass
class DiffusionDecoder_Trainer_Config(ModuleTrainerConfig):

    ddecmp: dict[str, Any]

    random_stereo_augmentation: bool = True
    random_phase_augmentation: bool  = True

    crop_edges: int = 4 # used to avoid artifacts due to mdct lapped blocks at beginning and end of sample

class DiffusionDecoder_Trainer(ModuleTrainer):
    
    @torch.no_grad()
    def __init__(self, config: DiffusionDecoder_Trainer_Config, trainer: DualDiffusionTrainer) -> None:

        self.config = config
        self.trainer = trainer
        self.logger = trainer.logger

        self.ddecmp: UNet = trainer.get_train_module("ddecmp")
        self.format: MS_MDCT_DualFormat = trainer.pipeline.format.to(self.trainer.accelerator.device)

        if trainer.config.enable_model_compilation:
            self.ddecmp.compile(**trainer.config.compile_params)
            self.format.compile(**trainer.config.compile_params)

        if self.config.random_stereo_augmentation == True:
            self.logger.info("Using random stereo augmentation")
        else: self.logger.info("Random stereo augmentation is disabled")

        if self.config.random_phase_augmentation == True:
            self.logger.info("Using random phase augmentation")
        else: self.logger.info("Random phase augmentation is disabled")

        self.logger.info(f"Crop edges: {self.config.crop_edges}")

        self.logger.info("DDECMP Trainer:")
        self.ddecmp_trainer = UNetTrainer(UNetTrainerConfig(**config.ddecmp), trainer, self.ddecmp, "ddecmp")
           
    @torch.no_grad()
    def init_batch(self, validation: bool = False) -> Optional[dict[str, Union[torch.Tensor, float]]]:
        return self.ddecmp_trainer.init_batch(validation)
    
    def train_batch(self, batch: dict) -> Optional[dict[str, Union[torch.Tensor, float]]]:

        # prepare model inputs
        if "audio_embeddings" in batch:
            audio_embeddings = normalize(batch["audio_embeddings"]).detach()
        else:
            audio_embeddings = None

        if self.config.random_stereo_augmentation == True:
            raw_samples = random_stereo_augmentation(batch["audio"])
        else:
            raw_samples = batch["audio"]

        mdct = self.format.raw_to_mdct(raw_samples, random_phase_augmentation=self.config.random_phase_augmentation)
        raw_samples = self.format.mdct_to_raw(mdct)

        mel_spec = self.format.raw_to_mel_spec(raw_samples)
        mel_spec = mel_spec[..., self.config.crop_edges:-self.config.crop_edges]
        mel_spec_linear = self.format.mel_spec_to_linear(mel_spec)

        mdct = mdct[..., self.config.crop_edges:-self.config.crop_edges]
        
        logs = {
            "io_stats/mel_spec_var": mel_spec.var(dim=(1,2,3)),
            "io_stats/mel_spec_mean": mel_spec.mean(dim=(1,2,3)),
            "io_stats/mel_spec_linear_var": mel_spec_linear.var(dim=(1,2,3)),
            "io_stats/mel_spec_linear_mean": mel_spec_linear.mean(dim=(1,2,3)),
            "io_stats/mel_spec_linear_mean_square": mel_spec_linear.pow(2).mean(dim=(1,2,3)),
            "io_stats/mdct_var": mdct.var(dim=(1,2,3)),
        }

        logs.update(self.ddecmp_trainer.train_batch(mdct, audio_embeddings, mel_spec_linear))
        logs["loss"] = logs["loss/ddecmp"]

        if self.trainer.config.enable_debug_mode == True:
            print("mdct.shape:", mdct.shape)
            print("mel_spec.shape:", mel_spec.shape)
            print("mel_spec_linear.shape:", mel_spec_linear.shape)

        return logs
      
    @torch.no_grad()
    def finish_batch(self) -> Optional[dict[str, Union[torch.Tensor, float]]]:
        return self.ddecmp_trainer.finish_batch()