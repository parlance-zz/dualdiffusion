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

from typing import Optional
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from PIL import Image
import asyncio
import multiprocessing
import logging
import random
import os

import numpy as np
from nicegui import ui, app

from utils.dual_diffusion_utils import (
    dict_str, tensor_to_img
)
from sampling.model_server import ModelServer
from sampling.schedule import SamplingSchedule
from sampling.nicegui_elements import (
    EditableSelect, LockButton, ScrollableNumber, PromptEditor,
    PresetEditor, OutputEditor
)


@dataclass
class NiceGUIAppConfig:
    model_name: str
    model_load_options: dict
    save_output_latents: bool = True
    hide_latents_after_generation: bool = True

    web_server_host: Optional[str] = None
    web_server_port: int = 3001

    enable_dark_mode: bool = True
    enable_debug_logging: bool = False
    max_debug_log_length: int = 10000

class NiceGUILogHandler(logging.Handler):
    def __init__(self, log_element: Optional[ui.log] = None) -> None:
        super().__init__()
        self.log_element = log_element
        self.buffered_messages = []

    def set_log_element(self, log_control: Optional[ui.log] = None) -> None:
        self.log_element = log_control

        for message in self.buffered_messages:
            self.log_element.push(message)
        self.buffered_messages.clear()

    def emit(self, record: logging.LogRecord) -> None:
        if self.log_element is not None:
            self.log_element.push(record.getMessage())
        else:
            self.buffered_messages.append(record.getMessage())

class NiceGUIApp:

    def __init__(self) -> None:

        self.config = NiceGUIAppConfig(**config.load_json(
            os.path.join(config.CONFIG_PATH, "sampling", "nicegui_app.json")))
        
        self.init_logging()
        self.logger.debug(f"NiceGUIAppConfig:\n{dict_str(self.config.__dict__)}")

        self.mp_manager = multiprocessing.Manager()
        self.model_server_state = self.mp_manager.dict()
        self.model_server_process = multiprocessing.Process(daemon=False, name="model_server",
            target=ModelServer.start_server, args=(self.model_server_state,))
        self.model_server_process.start()
        self.gpu_lock = asyncio.Semaphore()

        self.init_layout()
        app.on_startup(partial(self.on_startup_app))

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
                format=r"NiceGUIApp: %(message)s",
            )
            self.logger.info(f"\nStarted nicegui_app at {datetime_str}")
            self.logger.info(f"Logging to {self.log_path}")
        else:
            self.log_path = None
            self.logger.warning("WARNING: DEBUG_PATH not defined, logging to file disabled")

    def init_layout(self) -> None:
        
        # default element props / classes
        ui.tooltip.default_props("delay=1000")
        ui.select.default_props("options-dense")
        EditableSelect.default_props("options-dense")

        # main layout is split into 3 tabs
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
        debug_log_element = ui.log(max_lines=self.config.max_debug_log_length).style("height: calc(100vh - 200px)")
        self.interface_tabs.on_value_change(
            lambda e: ui.run_javascript(
                f'getElement({debug_log_element.id}).lastChild.scrollIntoView()') if e.value == "Debug Logs" else None)
        self.log_handler.set_log_element(debug_log_element)

    def init_generation_layout(self) -> None:
                
        with ui.row().classes("w-full"): # gen params, preset, and prompt editor
            with ui.card().classes("flex-grow-[1]"):
                with ui.row().classes("w-full"): # gen params and seed
                    with ui.card().classes("flex-grow-[1] items-center"): # general (non-saved) params
                        self.generate_length = ScrollableNumber(label="Length (seconds)", value=0, min=0, max=300, precision=0, step=5).classes("w-full")
                        self.seed = ScrollableNumber(label="Seed", value=10042, min=10000, max=99999, precision=0, step=1).classes("w-full")
                        self.auto_increment_seed = ui.checkbox("Auto Increment Seed", value=True).classes("w-full")

                        with ui.column().classes("w-full gap-0 items-center"):
                            with ui.button("Randomize Seed", icon="casino", on_click=lambda: self.seed.set_value(random.randint(10000, 99999))).classes("w-52 rounded-b-none"):
                                ui.tooltip("Choose new seed at random")
                            self.generate_button = ui.button("Generate", icon="audiotrack", color="green", on_click=partial(self.on_click_generate_button)).classes("w-52 rounded-none")
                            with self.generate_button:
                                ui.tooltip("Generate new sample with current settings")

                    with ui.card().classes("flex-grow-[3]"): # gen params
                        self.gen_param_elements = {}
                        with ui.grid(columns=2).classes("w-full items-center"):                 
                            with LockButton() as self.gen_params_lock_button:
                                ui.tooltip("Lock to freeze parameters when loading presets")

                            self.gen_param_elements["num_steps"] = ScrollableNumber(label="Number of Steps", value=100, min=10, max=1000, precision=0, step=10).classes("w-full")
                            self.gen_param_elements["cfg_scale"] = ScrollableNumber(label="CFG Scale", value=1.5, min=0, max=10, step=0.1).classes("w-full")
                            self.gen_param_elements["use_heun"] = ui.checkbox("Use Heun's Method", value=True).classes("w-full")
                            self.gen_param_elements["num_fgla_iters"] = ScrollableNumber(label="Number of FGLA Iterations", value=250, min=50, max=1000, precision=0, step=50).classes("w-full")

                            self.gen_param_elements["sigma_max"] = ScrollableNumber(label="Sigma Max", value=200, min=10, max=1000, step=10).classes("w-full")
                            self.gen_param_elements["sigma_min"] = ScrollableNumber(label="Sigma Min", value=0.15, min=0.05, max=2, step=0.05).classes("w-full")
                            self.gen_param_elements["rho"] = ScrollableNumber(label="Rho", value=7, min=0.5, max=1000, precision=2, step=0.5).classes("w-full")
                            self.gen_param_elements["input_perturbation"] = ScrollableNumber(label="Input Perturbation", value=1, min=0, max=1, step=0.05).classes("w-full")
                            
                            self.gen_param_elements["schedule"] = ui.select(label="Σ Schedule", options=SamplingSchedule.get_schedules_list(), value="edm2").classes("w-full")
                            self.sigma_schedule_dialog = ui.dialog()
                            self.show_schedule_button = ui.button("Show σ Schedule", on_click=lambda: self.on_click_show_schedule_button()).classes("w-44 items-center")
                            with self.show_schedule_button:
                                ui.tooltip("Show noise schedule with current settings")

                        self.gen_params = {}
                        for param_name, param_element in self.gen_param_elements.items():
                            self.gen_params[param_name] = param_element.value
                            param_element.bind_value(self.gen_params, param_name)
                            param_element.on_value_change(lambda: self.on_change_gen_param())

                self.preset_editor = PresetEditor()
                self.preset_editor.get_prompt_editor = lambda: self.prompt_editor
                self.preset_editor.get_gen_params = lambda: self.gen_params
                self.preset_editor.get_gen_params_locked = lambda: self.gen_params_lock_button.is_locked

            self.prompt_editor = PromptEditor()
            self.prompt_editor.on_prompt_change = lambda: self.on_change_gen_param()

        ui.separator().classes("bg-primary").style("height: 3px")

        self.output_editor = OutputEditor()
        self.output_editor.get_app_config = lambda: self.config

    async def wait_for_model_server(self) -> str:
        while self.model_server_state.get("cmd", None) is not None:
            await asyncio.sleep(0.1)
        error = self.model_server_state.get("error", None)
        if error is not None:
            self.logger.error(f"wait_for_model_server error: {error}")
        return error
    
    # wait for model server to be idle, then send args and command
    async def model_server_cmd(self, cmd: str, **kwargs) -> None:
        await self.wait_for_model_server()
        self.model_server_state.update(kwargs)
        self.model_server_state["cmd"] = cmd

    # load a new model on the model server and retrieve updated model / dataset metadata
    async def load_model(self, model_name: str, model_load_options: dict) -> None:
        async with self.gpu_lock:
            await self.model_server_cmd("load_model", model_name=model_name, model_load_options=model_load_options)
            self.logger.info(f"Loading model: {model_name}...")
            loading_notification = ui.notification(timeout=None)
            loading_notification.message = f"Loading model: {model_name}..."
            loading_notification.spinner = True

            error = await self.wait_for_model_server()
            if error is not None:
                self.logger.error(f"Error loading model: {self.model_server_state['error']}")
                loading_notification.dismiss()
                ui.notify(f"Error loading model: {self.model_server_state['error']}",
                    type="error", color="red", close_button=True)
                return

            self.logger.info(f"Loaded model: {model_name} successfully")
            loading_notification.message = f"Loaded model {model_name} successfully"
            loading_notification.spinner = False
            await asyncio.sleep(0.5)
            loading_notification.dismiss()
            
            self.output_editor.update_model_info(model_name,
                self.model_server_state["model_metadata"],
                self.model_server_state["format_config"],
                self.model_server_state["dataset_game_ids"])
            self.prompt_editor.update_dataset_games_dict(self.model_server_state["dataset_games_dict"])
            
    async def compile_model(self, model_name: str) -> None:
        async with self.gpu_lock:
            await self.model_server_cmd("compile_model")
            self.logger.info(f"Compiling model: {model_name}...")
            compiling_notification = ui.notification(timeout=None)
            compiling_notification.message = f"Compiling model: {model_name}..."
            compiling_notification.spinner = True

            error = await self.wait_for_model_server()
            if error is not None:
                self.logger.error(f"Error compiling model: {self.model_server_state['error']}")
                compiling_notification.dismiss()
                ui.notify(f"Error compiling model: {self.model_server_state['error']}",
                    type="error", color="red", close_button=True)
            else:
                self.logger.info(f"Compiled model: {model_name} successfully")
                compiling_notification.message = f"Compiled model {model_name} successfully"
                compiling_notification.spinner = False
                await asyncio.sleep(0.5)
                compiling_notification.dismiss()
    
    async def on_startup_app(self) -> None:
        await self.model_server_cmd("get_available_torch_devices")
        await self.load_model(self.config.model_name, self.config.model_load_options)
        self.preset_editor.load_preset()

        if self.config.model_load_options["compile_options"] is not None:
            await self.compile_model(self.config.model_name)

    async def on_click_generate_button(self) -> None:
        
        if len(self.prompt_editor.prompt) == 0: # abort if no prompt games selected
            self.logger.error("No prompt games selected")
            ui.notify("No prompt games selected", type="warning", color="red", close_button=True)
            return
        
        # queue new output sample and auto-increment seed
        output_sample = self.output_editor.add_output_sample(int(self.generate_length.value),
            int(self.seed.value), self.prompt_editor.prompt.copy(), self.gen_params.copy())
        if self.auto_increment_seed.value == True:
            self.seed.set_value(self.seed.value + 1)
        
        # this lock holds output samples in queue while generation is in progress
        async with self.gpu_lock:
            if output_sample not in self.output_editor.output_samples:
                return # handle early removal from queue
            
            # reset abort state and send generate command to model server
            self.model_server_state["generate_abort"] = False
            await self.model_server_cmd("generate", sample_params=output_sample.sample_params)
            while self.model_server_state.get("cmd", None) is not None:

                if output_sample not in self.output_editor.output_samples:
                    self.model_server_state["generate_abort"] = True
                    return # abort in progress
                
                # update linear progress
                step = self.model_server_state.get("generate_step", None)
                if step is not None:
                    output_sample.sampling_progress_element.set_value(
                        f"{int(step/output_sample.sample_params.num_steps*100)}%")

                # update latents image preview
                latents = self.model_server_state.get("generate_latents", None)
                if latents is not None:
                    latents_image = tensor_to_img(latents, flip_y=True)
                    latents_image = Image.fromarray(latents_image)
                    output_sample.latents_image_element.set_source(latents_image)
                
                # required to keep the interface responsive
                await asyncio.sleep(0.1)

            # if any error occurred on the model server, display it and remove from workspace
            error = self.model_server_state.get("error", None)
            if error is not None:
                self.logger.error(f"on_click_generate_button error: {error}")
                ui.notify(f"Error generating sample: {error}", type="error", color="red", close_button=True)
                self.output_editor.remove_output_sample(output_sample)
                return
            
            # retrieve sampling output from model server
            output_sample.sample_output = self.model_server_state["generate_output"]
        
        if output_sample not in self.output_editor.output_samples:
            return # handle late abort 

        # update output sample elements with completed sample output   
        self.output_editor.on_output_sample_generated(output_sample)
    
    def on_click_show_schedule_button(self) -> None:

        self.sigma_schedule_dialog.clear()
        with self.sigma_schedule_dialog, ui.card():
            ui.label("Sigma Schedule:")
            sigma_schedule = SamplingSchedule.get_schedule(
                self.gen_params["schedule"], int(self.gen_params["num_steps"]),
                sigma_max=self.gen_params["sigma_max"],
                sigma_min=self.gen_params["sigma_min"],
                rho=self.gen_params["rho"])

            x = np.arange(int(self.gen_params["num_steps"]) + 1)
            y = sigma_schedule.log().numpy()
            
            with ui.matplotlib(figsize=(5, 4)).figure as fig:
                ax = fig.gca()
                ax.plot(x, y, "-")
                ax.set_xlabel("step")
                ax.set_ylabel("ln(sigma)")

            self.sigma_schedule_dialog.open()
            ui.button("Close").classes("ml-auto").on_click(lambda: self.sigma_schedule_dialog.close())
            
    def on_change_gen_param(self) -> None:
        self.preset_editor.on_preset_modified()

    def run(self) -> None:
        on_air_token = os.getenv("ON_AIR_TOKEN", None)
        ui.run(dark=self.config.enable_dark_mode, title="Dual-Diffusion WebUI", reload=False, show=False,
            host=self.config.web_server_host, port=self.config.web_server_port, on_air=on_air_token)


if __name__ in {"__main__", "__mp_main__"}:
    
    if os.getenv("_IS_NICEGUI_SUBPROCESS", None) is None: # ugly hack for windows
        os.environ["_IS_NICEGUI_SUBPROCESS"] = "1"
        NiceGUIApp().run()