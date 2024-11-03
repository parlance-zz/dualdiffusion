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

#from typing import Optional, Union
#from dataclasses import dataclass
from datetime import datetime
import multiprocessing.managers
import asyncio
import multiprocessing
import traceback
import logging
import os

import torch
#import numpy as np

from pipelines.dual_diffusion_pipeline import DualDiffusionPipeline, SampleParams
from utils.dual_diffusion_utils import (
    init_cuda, dict_str, get_available_torch_devices, move_tensors_to_cpu
)

class ModelServer:

    def __init__(self, model_server_state: multiprocessing.managers.DictProxy) -> None:

        self.init_logging()
        self.model_server_state = model_server_state

    def init_logging(self) -> None:
        self.logger = logging.getLogger(name="model_server")
        self.logger.setLevel(logging.DEBUG)

        if config.DEBUG_PATH is not None:
            logging_dir = os.path.join(config.DEBUG_PATH, "model_server")
            os.makedirs(logging_dir, exist_ok=True)

            datetime_str = datetime.now().strftime(r"%Y-%m-%d_%H_%M_%S")
            self.log_path = os.path.join(logging_dir, f"model_server_{datetime_str}.log")

            logging.basicConfig(
                handlers=[
                    logging.FileHandler(self.log_path),
                    logging.StreamHandler(),
                ],
                format=r"ModelServer: %(message)s",
            )
            self.logger.info(f"\nStarted model_server at {datetime_str}")
            self.logger.info(f"Logging to {self.log_path}")
        else:
            self.log_path = None
            self.logger.warning("WARNING: DEBUG_PATH not defined, logging to file disabled")

    async def cmd_get_available_torch_devices(self) -> None:
        self.model_server_state["available_torch_devices"] = get_available_torch_devices()
        self.logger.debug(f"Available torch devices: {self.model_server_state['available_torch_devices']}")
    
    async def cmd_load_model(self) -> None:
        
        model_path = os.path.join(config.MODELS_PATH, self.model_server_state["model_name"])
        self.logger.info(f"Loading DualDiffusion model from '{model_path}'...")
        self.pipeline = DualDiffusionPipeline.from_pretrained(model_path, **self.model_server_state["model_load_options"])
        self.logger.debug(f"Model metadata:\n{dict_str(self.pipeline.model_metadata)}")

        # setup dataset games list
        for game_name, count in self.pipeline.dataset_info["game_train_sample_counts"].items():
            if count == 0: self.pipeline.dataset_game_ids.pop(game_name)
        self.dataset_games_dict = {} # keys are actual game names, values are display strings
        for game_name in self.pipeline.dataset_game_ids.keys():
            self.dataset_games_dict[game_name] = f"({self.pipeline.dataset_info['game_train_sample_counts'][game_name]}) {game_name}"
        self.dataset_games_dict = dict(sorted(self.dataset_games_dict.items()))

        self.model_server_state["model_metadata"] = self.pipeline.model_metadata
        self.model_server_state["format_config"] = self.pipeline.format.config.__dict__
        self.model_server_state["dataset_games_dict"] = self.dataset_games_dict
        self.model_server_state["dataset_game_ids"] = self.pipeline.dataset_game_ids

    async def cmd_compile_model(self) -> None:
        
        self.logger.info("Compiling default path...")
        dummy_params = SampleParams(num_steps=1, use_heun=False, num_fgla_iters=1)
        sample_output = move_tensors_to_cpu(self.pipeline(dummy_params, quiet=True))

        self.logger.info("Compiling inpainting path...")
        dummy_params.input_audio = torch.randn_like(sample_output.raw_sample)
        dummy_params.inpainting_mask = torch.ones_like(sample_output.latents[:, :1])
        self.pipeline(dummy_params, quiet=True)

        self.logger.info("Compilation complete")

    async def cmd_generate(self) -> None:
        sample_output = self.pipeline(self.model_server_state["sample_params"], self.model_server_state)
        self.model_server_state["generate_output"] = move_tensors_to_cpu(sample_output) if sample_output is not None else None

    async def cmd_get_module_state_dict(self) -> None:
        module_name = self.model_server_state.get("module_name", "unet")
        self.logger.info(f"Loading module state dict ({module_name})...")
        module = getattr(self.pipeline, module_name)
        self.model_server_state["module_state_dict"] = {k: v.cpu() for k,v in module.state_dict().items()}

    async def cmd_get_latent_shape(self) -> None:
        audio_length = self.model_server_state["audio_length"]
        sample_shape = self.pipeline.get_sample_shape(bsz=1, length=audio_length)
        self.model_server_state["latent_shape"] = tuple(self.pipeline.get_latent_shape(sample_shape))

    async def run(self):
        while True:
            cmd = self.model_server_state.get("cmd", None)
            if cmd is None: await asyncio.sleep(0.1)
            else:
                try:
                    self.logger.debug(f"Processing command '{cmd}'")
                    await getattr(self, f"cmd_{cmd}")()
                    self.model_server_state["error"] = None
                except Exception as e:
                    error_str = f"Error processing command '{cmd}': {e}"
                    self.logger.error(error_str)
                    self.model_server_state["error"] = error_str
                    traceback.print_exception(type(e), e, e.__traceback__)  
                finally:
                    self.model_server_state["cmd"] = None

    @staticmethod
    def start_server(model_server_state: multiprocessing.managers.DictProxy) -> None:
        init_cuda()
        asyncio.run(ModelServer(model_server_state).run())