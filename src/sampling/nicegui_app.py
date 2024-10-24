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
import matplotlib.pyplot as plt
import asyncio
import multiprocessing
import logging
import os

from nicegui import ui, app

from utils.dual_diffusion_utils import (
    dict_str, tensor_to_img
)
from sampling.model_server import ModelServer
from sampling.nicegui_elements import (
    EditableSelect, PromptEditor,
    PresetEditor, OutputEditor, GenParamsEditor
)


@dataclass
class NiceGUIAppConfig:
    model_name: str
    model_load_options: dict
    save_output_latents: bool = True
    hide_latents_after_generation: bool = True
    use_verbose_labels: bool = False

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
        self.interface_tabs.on_value_change(lambda e: ui.run_javascript( # fix to scroll to bottom when viewing log
            f'getElement({debug_log_element.id}).lastChild.scrollIntoView()') if e.value == "Debug Logs" else None)
        self.log_handler.set_log_element(debug_log_element)

    def init_generation_layout(self) -> None:
                
        with ui.row().classes("w-full"): # gen params, preset, and prompt editor
            with ui.card():
                self.gen_params_editor = GenParamsEditor()
                self.preset_editor = PresetEditor()
                self.preset_editor.get_prompt_editor = lambda: self.prompt_editor
                self.preset_editor.get_gen_params_editor = lambda: self.gen_params_editor
                self.gen_params_editor.on_change_gen_param = lambda: self.preset_editor.on_preset_modified()

            self.prompt_editor = PromptEditor()
            self.prompt_editor.on_change_prompt = lambda: self.preset_editor.on_preset_modified()

        ui.separator().classes("bg-primary").style("height: 3px")

        self.output_editor = OutputEditor()
        self.output_editor.get_app_config = lambda: self.config
        self.output_editor.get_prompt_editor = lambda: self.prompt_editor
        self.output_editor.get_gen_params_editor = lambda: self.gen_params_editor
        self.output_editor.generate_button.on_click(partial(self.on_click_generate_button))

    # wait for model server idle, then send new command and parameters
    async def model_server_cmd(self, cmd: str, **kwargs) -> None:
        await self.wait_for_model_server()
        self.model_server_state.update(kwargs)
        self.model_server_state["cmd"] = cmd

    # wait for model server idle, returns error from last command if any
    async def wait_for_model_server(self) -> str:
        while self.model_server_state.get("cmd", None) is not None:
            await asyncio.sleep(0.1)
        error = self.model_server_state.get("error", None)
        if error is not None:
            self.logger.error(f"wait_for_model_server error: {error}")
        return error

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
    
    # trigger torch.compile on model server and show progress notifications
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
    
    # on startup load configured model, default preset, and trigger compilation if enabled
    async def on_startup_app(self) -> None:
        await self.model_server_cmd("get_available_torch_devices")
        await self.load_model(self.config.model_name, self.config.model_load_options)
        self.preset_editor.load_preset()

        if self.config.model_load_options["compile_options"] is not None:
            await self.compile_model(self.config.model_name)

        # set matplotlibs to use dark theme
        plt.style.use("dark_background")

    # queues a new output sample for generation, then proceeds with generation when ready
    async def on_click_generate_button(self) -> None:
        
        if len(self.prompt_editor.prompt) == 0: # abort if no prompt games selected
            self.logger.error("No prompt games selected")
            ui.notify("No prompt games selected", type="warning", color="red", close_button=True)
            return
        
        # queue new output sample and auto-increment seed
        output_sample = self.output_editor.add_output_sample(int(self.gen_params_editor.generate_length.value),
            int(self.gen_params_editor.seed.value), self.prompt_editor.prompt.copy(), self.gen_params_editor.gen_params.copy())
        if self.gen_params_editor.auto_increment_seed.value == True:
            self.gen_params_editor.seed.set_value(self.gen_params_editor.seed.value + 1)
        
        # this lock holds output samples in queue while generation is in progress
        async with self.gpu_lock:
            if output_sample not in self.output_editor.output_samples:
                return # handle early removal from queue
            
            # reset abort state and send generate command to model server
            self.model_server_state["generate_abort"] = False
            toggled_show_latents = False
            await self.model_server_cmd("generate", sample_params=output_sample.sample_params)
            while self.model_server_state.get("cmd", None) is not None:
                
                # abort in progress if output sample was removed from workspace
                if output_sample not in self.output_editor.output_samples:
                    self.model_server_state["generate_abort"] = True
                    await self.wait_for_model_server() # we need this otherwise there is a race condition
                    return
                
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

                    # only show latents after we have an image, otherwise it messes up positioning
                    if output_sample.toggle_show_latents_button.is_toggled == False and toggled_show_latents == False:
                        output_sample.toggle_show_latents_button.toggle(is_toggled=True)
                        toggled_show_latents = True # only force show latents when starting generation, they can be hidden again by user
                
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
            
    def run(self) -> None:
        on_air_token = os.getenv("ON_AIR_TOKEN", None)
        ui.run(dark=self.config.enable_dark_mode, title="Dual-Diffusion WebUI", reload=False, show=False,
            host=self.config.web_server_host, port=self.config.web_server_port, on_air=on_air_token)


if __name__ in {"__main__", "__mp_main__"}:
    
    if os.getenv("_IS_NICEGUI_SUBPROCESS", None) is None: # ugly hack for windows
        os.environ["_IS_NICEGUI_SUBPROCESS"] = "1"
        NiceGUIApp().run()