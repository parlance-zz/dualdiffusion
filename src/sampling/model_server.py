import multiprocessing.managers
from utils import config

from typing import Optional, Union
from dataclasses import dataclass
from datetime import datetime
import asyncio
import multiprocessing
import logging
import os

import torch
import numpy as np

from pipelines.dual_diffusion_pipeline import DualDiffusionPipeline, SampleParams, SampleOutput

from utils.dual_diffusion_utils import (
    init_cuda, save_audio, load_audio, dict_str, tensor_to_img,
    get_available_torch_devices
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

    async def get_available_torch_devices(self) -> list[str]:
        return get_available_torch_devices()
    
    async def load_model(self) -> dict:
        
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

    async def compile_model(self) -> None:
        pass
        # todo: run a single batch of default size / shape to trigger compilation

    async def abort(self) -> None:
        pass

    async def generate(self) -> None:
        sample_output = self.pipeline(self.model_server_state["sample_params"], self.model_server_state)
        self.model_server_state["generate_output"] = sample_output.cpu() if sample_output is not None else None

    async def run(self):
        while True:
            cmd = self.model_server_state.get("cmd", None)
            if cmd is None: await asyncio.sleep(0.1)
            else:
                try:
                    self.logger.debug(f"Processing command '{cmd}'")
                    await getattr(self, cmd)()
                    self.model_server_state["error"] = None
                except Exception as e:
                    error_str = f"Error processing command '{cmd}': {e}"
                    self.logger.error(error_str)
                    self.model_server_state["error"] = error_str
                finally:
                    self.model_server_state["cmd"] = None

    @staticmethod
    def start_server(model_server_state: multiprocessing.managers.DictProxy) -> None:
        init_cuda()
        asyncio.run(ModelServer(model_server_state).run())