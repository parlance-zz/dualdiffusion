from utils import config

from typing import Optional
from dataclasses import dataclass
from datetime import datetime
import threading
import logging
import random
import time
import os

import torch
import numpy as np
from nicegui import ui

from utils.dual_diffusion_utils import (
    init_cuda, save_audio, load_audio, dict_str,
    get_available_torch_devices, sanitize_filename
)
from pipelines.dual_diffusion_pipeline import DualDiffusionPipeline


@dataclass
class NiceGUIAppConfig:
    model_name: str
    model_load_options: dict

    web_server_host: Optional[str] = None
    web_server_port: int = 3001
    web_server_share: bool = False

    enable_dark_mode: bool = True
    enable_debug_logging: bool = False

class NiceGUIApp:

    def __init__(self) -> None:

        self.config = NiceGUIAppConfig(**config.load_json(
            os.path.join(config.CONFIG_PATH, "sampling", "nicegui_app.json")))
        
        self.init_logging()
        self.logger.debug(f"GradioAppConfig:\n{dict_str(self.config.__dict__)}")

        """
        # load model
        model_path = os.path.join(config.MODELS_PATH, self.config.model_name)
        model_load_options = self.config.model_load_options

        self.logger.info(f"Loading DualDiffusion model from '{model_path}'...")
        self.pipeline = DualDiffusionPipeline.from_pretrained(model_path, **model_load_options)
        self.logger.debug(f"Model metadata:\n{dict_str(self.pipeline.model_metadata)}")

        # remove games with no samples in training set
        for game_name, count in self.pipeline.dataset_info["game_train_sample_counts"].items():
            if count == 0: self.pipeline.dataset_game_ids.pop(game_name)

        """

        with ui.tabs().classes("w-full") as interface_tabs:
            generation_tab = ui.tab("Generation")
            model_settings_tab = ui.tab("Model Settings")
            debug_logs_tab = ui.tab("Debug Logs")

        with ui.tab_panels(interface_tabs, value=generation_tab).classes("w-full"):
            with ui.tab_panel(generation_tab).style("justify-content: center"):
                self.create_layout()
            with ui.tab_panel(model_settings_tab):
                ui.label("model settings stuff")
            with ui.tab_panel(debug_logs_tab):
                ui.label("debug logs stuff")    

    def init_logging(self) -> None:

        self.logger = logging.getLogger(name="nicegui_app")

        if self.config.enable_debug_logging:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)

        if config.DEBUG_PATH is not None:
            logging_dir = os.path.join(config.DEBUG_PATH, "nicegui_app")
            os.makedirs(logging_dir, exist_ok=True)

            datetime_str = datetime.now().strftime(r"%Y-%m-%d_%H_%M_%S")
            self.log_path = os.path.join(logging_dir, f"nicegui_app_{datetime_str}.log")
            
            logging.basicConfig(
                handlers=[
                    logging.FileHandler(self.log_path),
                    logging.StreamHandler()
                ],
                format="",
            )
            self.logger.info(f"\nStarted nicegui_app at {datetime_str}")
            self.logger.info(f"Logging to {self.log_path}")
        else:
            self.log_path = None
            self.logger.warning("WARNING: DEBUG_PATH not defined, logging disabled")

    def create_layout(self) -> None:

        with ui.row():
            with ui.column():
                with ui.row(): # gen param controls
                    with ui.card():
                        seed = ui.number(label="Seed", value=42, min=0, max=1000000, step=1).style("width: 100%")
                        auto_increment_seed = ui.checkbox("Auto Increment Seed", value=True)
                        ui.button("Randomize Seed").style("width: 100%").on_click(lambda: seed.set_value(random.randint(0, 99999)))
                        generate_button = ui.button("Generate").style("width: 100%")

                    with ui.card():
                        num_steps = ui.number(label="Number of Steps", value="100", min=1, max=1000, step=1).style("width: 100%")
                        cfg_scale = ui.number(label="CFG Scale", value=1.5, min=0, max=10, step=0.1).style("width: 100%")
                        use_heun = ui.checkbox("Use Heun's Method", value=True)
                        num_fgla_iters = ui.number(label="Number of FGLA Iterations", value=250, min=50, max=1000, step=50).style("width: 100%")

                    with ui.card().style("width: 200px"):
                        sigma_max = ui.number(label="Sigma Max", value=200, min=10, max=1000, step=10).style("width: 100%")
                        sigma_min = ui.number(label="Sigma Min", value=0.15, min=0.05, max=2, step=0.05).style("width: 100%")
                        rho = ui.number(label="Rho", value=7, min=0.05, max=1000, precision=2, step=0.05).style("width: 100%")

                        input_perturbation_label = ui.number(label="Input Perturbation", value=1, min=0, max=1, step=0.05).style("width: 100%")
                        input_perturbation = ui.slider(value=1, min=0, max=1, step=0.05).style("width: 100%")
                        input_perturbation.bind_value_from(input_perturbation_label, "value")
                        input_perturbation_label.bind_value_from(input_perturbation, "value")

                with ui.card(): # preset editor
                    
                    with ui.row():
                        with ui.column().style("width: 100%"):
                            ui.select(label="Select a Preset - (loaded preset: default)",
                                options=["default", "preset1", "megaman x"], value="default", new_value_mode="add-unique").style("width: 100%")

                        with ui.column():
                            ui.button("Save Changes")
                            ui.button("Load Preset")
                            ui.button("Delete Preset")
                    
            with ui.card().style("width: 600px"): # input audio controls
                
                
                with ui.row():
                    ui.label("Audio Input Mode")
                with ui.row():
                    with ui.tabs().classes("w-full").style("width: 500px") as input_audio_tabs:
                        no_input_audio = ui.tab("None")
                        img2img = ui.tab("Img2Img")
                        inpaint = ui.tab("Inpaint")
                        outpaint = ui.tab("Outpaint")

                    with ui.tab_panels(input_audio_tabs, value=no_input_audio).classes("w-full"):
                        with ui.tab_panel(img2img):
                            ui.label("img2img stuff")
                        with ui.tab_panel(inpaint):
                            ui.label("inpaint stuff")
                        with ui.tab_panel(outpaint):
                            ui.label("outpaint stuff")

        with ui.row(): # prompt editor
            with ui.column():
                ui.label("Select a game")
                ui.select(options=["(10) spc/3 Ninjas Kick Back"], value="(10) spc/3 Ninjas Kick Back")

            with ui.column():
                ui.number(label="Weight", value=1, min=0, max=100, step=1)

            with ui.column():
                ui.button("Add Game")

        ui.separator()

    def run(self) -> None:
        ui.run(dark=self.config.enable_dark_mode, width=1920, height=1080)


if __name__ in {"__main__", "__mp_main__"}:

    init_cuda()
    NiceGUIApp().run()