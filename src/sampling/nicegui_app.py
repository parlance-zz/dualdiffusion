from utils import config

from typing import Optional
from dataclasses import dataclass
from datetime import datetime
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
from sampling.schedule import SamplingSchedule

@dataclass
class NiceGUIAppConfig:
    model_name: str
    model_load_options: dict

    web_server_host: Optional[str] = None
    web_server_port: int = 3001

    enable_dark_mode: bool = True
    enable_debug_logging: bool = False

class NiceGUILogHandler(logging.Handler):
    def __init__(self, log_control: Optional[ui.log] = None) -> None:
        super().__init__()
        self.log_control = log_control
        self.buffered_messages = []

    def set_log_control(self, log_control: Optional[ui.log] = None) -> None:
        self.log_control = log_control

        for message in self.buffered_messages:
            self.log_control.push(message)
        self.buffered_messages.clear()

    def emit(self, record: logging.LogRecord) -> None:
        if self.log_control is not None:
            self.log_control.push(record.getMessage())
        else:
            self.buffered_messages.append(record.getMessage())

class NiceGUIApp:

    def __init__(self) -> None:

        self.config = NiceGUIAppConfig(**config.load_json(
            os.path.join(config.CONFIG_PATH, "sampling", "nicegui_app.json")))
        
        self.init_logging()
        self.logger.debug(f"NiceGUIAppConfig:\n{dict_str(self.config.__dict__)}")

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

        self.init_layout()

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
            self.log_handler = NiceGUILogHandler()

            logging.basicConfig(
                handlers=[
                    logging.FileHandler(self.log_path),
                    logging.StreamHandler(),
                    self.log_handler
                ],
                format="",
            )
            self.logger.info(f"\nStarted nicegui_app at {datetime_str}")
            self.logger.info(f"Logging to {self.log_path}")
        else:
            self.log_path = None
            self.logger.warning("WARNING: DEBUG_PATH not defined, logging to file disabled")

    def init_layout(self) -> None:

        with ui.tabs() as self.interface_tabs:
            self.generation_tab = ui.tab("Generation")
            self.model_settings_tab = ui.tab("Model Settings")
            self.debug_logs_tab = ui.tab("Debug Logs")

        with ui.tab_panels(self.interface_tabs, value=self.generation_tab).classes("w-full"):
            with ui.tab_panel(self.generation_tab):
                self.init_generation_layout()
            with ui.tab_panel(self.model_settings_tab):
                self.init_model_settings_layout()
            with ui.tab_panel(self.debug_logs_tab):
                self.init_debug_logs_layout()

    def init_model_settings_layout(self) -> None:
        ui.label("model settings stuff")

    def init_debug_logs_layout(self) -> None:
        ui.label("Debug Log:")
        self.debug_log = ui.log(max_lines=50).style("height: 500px")
        self.log_handler.set_log_control(self.debug_log)

    def init_generation_layout(self) -> None:
            
        with ui.row().classes("w-full"): # params, preset, and prompt editor
            with ui.card().classes("flex-grow-[1]"): # params and preset editor
                with ui.row().classes("w-full"): # gen param controls
                    with ui.card().classes("flex-grow-[1]"): # seed params
                        ui.label("General:")
                        self.seed = ui.number(label="Seed", value=10042, min=10000, max=99999, step=1).classes("w-full")
                        self.seed.on("wheel", lambda: None)
                        self.auto_increment_seed = ui.checkbox("Auto Increment Seed", value=True).classes("w-full")
                        ui.button("Randomize Seed").classes("w-full").on_click(lambda: self.seed.set_value(random.randint(0, 99999)))
                        self.generate_button = ui.button("Generate").classes("w-full")

                    with ui.card().classes("flex-grow-[5]"): # gen params
                        ui.label("Parameters:")
                        with ui.grid(columns=2).classes("w-full") as self.gen_params:
                            self.num_steps = ui.number(label="Number of Steps", value=100, min=10, max=1000, precision=0, step=10).classes("w-full")
                            self.cfg_scale = ui.number(label="CFG Scale", value=1.5, min=0, max=10, step=0.1).classes("w-full")
                            self.use_heun = ui.checkbox("Use Heun's Method", value=True).classes("w-full")
                            self.num_fgla_iters = ui.number(label="Number of FGLA Iterations", value=250, min=50, max=1000, precision=0, step=50).classes("w-full")

                            self.sigma_max = ui.number(label="Sigma Max", value=200, min=10, max=1000, step=10).classes("w-full")
                            self.sigma_min = ui.number(label="Sigma Min", value=0.15, min=0.05, max=2, step=0.05).classes("w-full")
                            self.rho = ui.number(label="Rho", value=7, min=0.5, max=1000, precision=2, step=0.5).classes("w-full")
                            self.input_perturbation = ui.number(label="Input Perturbation", value=1, min=0, max=1, step=0.05).classes("w-full")
                            
                            self.show_schedule_button = ui.button("Show Schedule").classes("w-full")

                        def on_params_changed():
                            self.preset_select._props['label']=f"Select a Preset - (loaded preset: {self.last_loaded_preset}*)"
                            self.preset_select.update()
                            self.preset_load_button.enable()
                            self.preset_save_button.enable()

                        for param in self.gen_params:
                            if not isinstance(param, ui.button):
                                param.on_value_change(on_params_changed)
                            if isinstance(param, ui.number):
                                param.on("wheel", lambda: None)

                with ui.card().classes("w-full"): # preset editor
                    ui.label("Preset Editor:")
                    with ui.row().classes("w-full"):
                        with ui.column().classes("flex-grow-[4]"):
                            self.last_loaded_preset = "default"
                            self.preset_select = ui.select(
                                label=f"Select a Preset - (loaded preset: {self.last_loaded_preset})",
                                options=["default", "preset1", "megaman x"],
                                value="default", new_value_mode="add-unique").classes("w-full")

                        with ui.column().classes("flex-grow-[1] flex items-center"):
                            self.preset_load_button = ui.button("Load Preset").classes("w-full")
                            self.preset_save_button = ui.button("Save Changes").classes("w-full")
                            self.preset_delete_button = ui.button("Delete Preset").classes("w-full")

                            self.preset_load_button.disable()
                            self.preset_save_button.disable()
                            self.preset_delete_button.disable()

            with ui.card().classes("flex-grow-[10]"): # prompt editor
                ui.label("Prompt Editor:")
                with ui.row().classes("w-full flex items-center"):
                    self.game_select = ui.select(label="Select a game",
                        options=["(10) spc/3 Ninjas Kick Back"],
                        value="(10) spc/3 Ninjas Kick Back").classes("flex-grow-[1000]")
                    self.game_weight = ui.number(label="Weight", value=1, min=0, max=100, step=1).classes("flex-grow-[1]")
                    self.game_weight.on("wheel", lambda: None)
                    self.game_add_button = ui.button("Add Game").classes("flex-grow-[1]")

                ui.separator()

                # selected games go here

            with ui.card().classes("w-1/4"): # audio input editor
                with ui.row().classes("w-full"):
                    ui.label("Audio Input Mode:")
                with ui.row().classes("w-full"):
                    with ui.tabs().classes("w-full") as input_audio_tabs:
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
        
        ui.separator()

        # queued / output samples go here

        self.sigma_schedule_dialog = ui.dialog()

        def show_sigma_schedule_dialog():

            self.sigma_schedule_dialog.clear()
            with self.sigma_schedule_dialog, ui.card():
                ui.label("Sigma Schedule:")
                sigma_schedule = SamplingSchedule.get_schedule(
                    "edm2", int(self.num_steps.value) + 1,
                    sigma_max=self.sigma_max.value, sigma_min=self.sigma_min.value, rho=self.rho.value).log()

                x = np.arange(int(self.num_steps.value) + 1)
                y = sigma_schedule.log().numpy()
                
                with ui.matplotlib(figsize=(5, 4)).figure as fig:
                    ax = fig.gca()
                    ax.plot(x, y, '-')
                    ax.set_xlabel("step")
                    ax.set_ylabel("ln(sigma)")

                self.sigma_schedule_dialog.open()
                ui.button("Close").classes("ml-auto").on_click(lambda: self.sigma_schedule_dialog.close())
        
        self.show_schedule_button.on_click(show_sigma_schedule_dialog)
            
    def get_saved_presets(self) -> None:
        preset_files = os.listdir(os.path.join(config.CONFIG_PATH, "sampling", "presets"))
        saved_presets = []
        for file in preset_files:
            if os.path.splitext(file)[1] == ".json":
                saved_presets.append(os.path.splitext(file)[0])
        saved_presets = sorted(saved_presets)
        self.logger.debug(f"Found saved presets: {saved_presets}")
        return saved_presets

    def save_preset(self, preset_name: str) -> None:
        save_preset_path = os.path.join(
            config.CONFIG_PATH, "sampling", "presets", f"{sanitize_filename(preset_name)}.json")
        self.logger.debug(f"Saving preset '{save_preset_path}'")
        config.save_json({"prompt": self.prompt, "gen_params": self.gen_params}, save_preset_path)
    
    def load_preset(self, preset_name: str) -> None:
        load_preset_path = os.path.join(
            config.CONFIG_PATH, "sampling", "presets", f"{sanitize_filename(preset_name)}.json")
        self.logger.debug(f"Loading preset '{load_preset_path}'")

        loaded_preset_dict = config.load_json(load_preset_path)
        self.prompt = loaded_preset_dict["prompt"]
        self.gen_params = loaded_preset_dict["gen_params"]
        #...
    
    def delete_preset(self, preset_name: str):
        delete_preset_path = os.path.join(
            config.CONFIG_PATH, "sampling", "presets", f"{sanitize_filename(preset_name)}.json")
        self.logger.debug(f"Deleting preset '{delete_preset_path}'")
        os.remove(delete_preset_path)

    def run(self) -> None:
        ui.run(dark=self.config.enable_dark_mode, title="Dual-Diffusion WebUI",
            host=self.config.web_server_host, port=self.config.web_server_port)


if __name__ in {"__main__", "__mp_main__"}:

    init_cuda()
    NiceGUIApp().run()