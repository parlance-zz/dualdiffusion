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

import cv2
import torch
import numpy as np
from nicegui import ui, app

from utils.dual_diffusion_utils import (
    save_audio, load_audio, save_safetensors, load_safetensors, dict_str,
    sanitize_filename, tensor_to_img
)
from pipelines.dual_diffusion_pipeline import SampleParams, SampleOutput
from sampling.model_server import ModelServer
from sampling.schedule import SamplingSchedule


@dataclass
class OutputSample:
    name: str
    seed: int
    prompt: dict
    gen_params: dict
    sample_params: SampleParams
    sample_output: Optional[SampleOutput] = None
    audio_path: Optional[str] = None
    latents_path: Optional[str] = None

    card_element: Optional[ui.card] = None
    name_label_element: Optional[ui.label] = None
    sampling_progress_element: Optional[ui.linear_progress] = None
    latents_image_element: Optional[ui.interactive_image] = None
    spectrogram_image_element: Optional[ui.interactive_image] = None
    audio_element: Optional[ui.audio] = None
    use_as_input_button: Optional[ui.button] = None
    select_range: Optional[ui.range] = None
    move_up_button: Optional[ui.button] = None
    move_down_button: Optional[ui.button] = None

@dataclass
class NiceGUIAppConfig:
    model_name: str
    model_load_options: dict
    save_output_latents: bool = True

    web_server_host: Optional[str] = None
    web_server_port: int = 3001

    enable_dark_mode: bool = True
    enable_debug_logging: bool = False
    max_debug_log_length: int = 10000

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

        self.mp_manager = multiprocessing.Manager()
        self.model_server_state = self.mp_manager.dict()
        
        self.model_server_process = multiprocessing.Process(daemon=False, name="model_server",
            target=ModelServer.start_server, args=(self.model_server_state,))
        self.model_server_process.start()

        self.gpu_lock = asyncio.Semaphore()
        self.input_output_sample: OutputSample = None

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
        self.debug_log = ui.log(max_lines=self.config.max_debug_log_length).style("height: calc(100vh - 200px)")
        self.log_handler.set_log_control(self.debug_log)
        self.interface_tabs.on_value_change(
            lambda e: ui.run_javascript(f'getElement({self.debug_log.id}).lastChild.scrollIntoView()') if e.value == "Debug Logs" else None)

    def init_generation_layout(self) -> None:
            
        with ui.row().classes("w-full"): # gen params, preset, and prompt editor
            with ui.card().classes("flex-grow-[1]"):
                with ui.row().classes("w-full"): # gen params and seed
                    with ui.card().classes("flex-grow-[1] items-center"): # general (non-saved) params
                        self.generate_length = ui.number(label="Length (seconds)", value=0, min=0, max=300, precision=0, step=5).classes("w-full")
                        self.generate_length.on("wheel", lambda: None)
                        self.seed = ui.number(label="Seed", value=10042, min=10000, max=99999, precision=0, step=1).classes("w-full")
                        self.seed.on("wheel", lambda: None)
                        self.auto_increment_seed = ui.checkbox("Auto Increment Seed", value=True).classes("w-full")

                        with ui.column().classes("w-full gap-0 items-center"):
                            with ui.button("Randomize Seed", icon="casino", on_click=lambda: self.seed.set_value(random.randint(10000, 99999))).classes("w-52 rounded-b-none"):
                                ui.tooltip("Choose new seed at random").props('delay=1000')
                            self.generate_button = ui.button("Generate", icon="audiotrack", color="green", on_click=partial(self.on_click_generate_button)).classes("w-52 rounded-none")
                            with self.generate_button:
                                ui.tooltip("Generate new sample with current settings").props('delay=1000')
                            self.clear_output_button = ui.button("Clear Outputs", icon="delete", color="red", on_click=lambda: self.clear_output_samples()).classes("w-52 rounded-t-none")
                            self.clear_output_button.disable()
                            with self.clear_output_button:
                                ui.tooltip("Clear all output samples in workspace").props('delay=1000')

                    with ui.card().classes("flex-grow-[3]"): # gen params
                        self.gen_param_elements = {}
                        with ui.grid(columns=2).classes("w-full items-center"):

                            self.gen_params_lock_button = ui.button(icon="lock_open", on_click=lambda: self.toggle_gen_params_lock()).style(
                                'position: absolute; top: 5px; right: 5px; z-index: 2; background-color: transparent; '
                                'border: none; width: 20px; height: 20px; padding: 0;'
                            ).classes("bg-transparent")
                            with self.gen_params_lock_button:
                                ui.tooltip("Lock to freeze parameters when loading presets").props('delay=1000')

                            self.gen_param_elements["num_steps"] = ui.number(label="Number of Steps", value=100, min=10, max=1000, precision=0, step=10).classes("w-full")
                            self.gen_param_elements["cfg_scale"] = ui.number(label="CFG Scale", value=1.5, min=0, max=10, step=0.1).classes("w-full")
                            self.gen_param_elements["use_heun"] = ui.checkbox("Use Heun's Method", value=True).classes("w-full")
                            self.gen_param_elements["num_fgla_iters"] = ui.number(label="Number of FGLA Iterations", value=250, min=50, max=1000, precision=0, step=50).classes("w-full")

                            self.gen_param_elements["sigma_max"] = ui.number(label="Sigma Max", value=200, min=10, max=1000, step=10).classes("w-full")
                            self.gen_param_elements["sigma_min"] = ui.number(label="Sigma Min", value=0.15, min=0.05, max=2, step=0.05).classes("w-full")
                            self.gen_param_elements["rho"] = ui.number(label="Rho", value=7, min=0.5, max=1000, precision=2, step=0.5).classes("w-full")
                            self.gen_param_elements["input_perturbation"] = ui.number(label="Input Perturbation", value=1, min=0, max=1, step=0.05).classes("w-full")
                            
                            self.gen_param_elements["schedule"] = ui.select(label="Σ Schedule", options=SamplingSchedule.get_schedules_list(), value="edm2").classes("w-full")
                            self.sigma_schedule_dialog = ui.dialog()
                            self.show_schedule_button = ui.button("Show σ Schedule", on_click=lambda: self.on_click_show_schedule_button()).classes("w-44 items-center")
                            with self.show_schedule_button:
                                ui.tooltip("Show noise schedule with current settings").props('delay=1000')

                        self.gen_params = {}
                        for param_name, param_element in self.gen_param_elements.items():
                            self.gen_params[param_name] = param_element.value
                            param_element.bind_value(self.gen_params, param_name)
                            param_element.on_value_change(lambda: self.on_change_gen_param())
                            if isinstance(param_element, ui.number):
                                param_element.on("wheel", lambda: None)

                with ui.card().classes("w-full"): # preset editor
                    with ui.row().classes("w-full"):
                        with ui.column().classes("flex-grow-[4]"):

                            self.last_loaded_preset = "default"
                            self.new_preset_name = ""
                            self.loading_preset = False
                            self.saved_preset_list = self.get_saved_presets()

                            self.preset_select = ui.select(
                                label=f"Select a Preset - (loaded preset: {self.last_loaded_preset})",
                                options=self.saved_preset_list,
                                value="default", with_input=True).classes("w-full")
                            
                            self.preset_select.on("input-value", lambda e: self.on_input_value_preset_select(e.args))
                            self.preset_select.on("blur", lambda e: self.on_blur_preset_select(e))
                            self.preset_select.on_value_change(lambda e: self.on_value_change_preset_select(e.value))     

                        with ui.column().classes("gap-0 items-center"):
                            self.preset_load_button = ui.button("Load", icon="source", on_click=lambda: self.load_preset()).classes("w-36 rounded-b-none")
                            self.preset_save_button = ui.button("Save", icon="save", color="green", on_click=lambda: self.save_preset()).classes("w-36 rounded-none")
                            self.preset_delete_button = ui.button("Delete", icon="delete", color="red", on_click=lambda: self.delete_preset()).classes("w-36 rounded-t-none")
                            with self.preset_load_button:
                                ui.tooltip("Load selected preset").props('delay=1000')
                            with self.preset_save_button:
                                ui.tooltip("Save current parameters to selected preset").props('delay=1000')
                            with self.preset_delete_button:
                                ui.tooltip("Delete selected preset").props('delay=1000')

                        self.preset_load_button.disable()
                        self.preset_save_button.disable()
                        self.preset_delete_button.disable()

            with ui.card().classes("flex-grow-[50]"): # prompt editor                    
                self.prompt = {}
                with ui.row().classes("w-full h-12 flex items-center"):
                    self.game_select = ui.select(label="Select a game", with_input=True,
                        options={}).classes("flex-grow-[1000]")
                    self.game_weight = ui.number(label="Weight", value=10, min=-100, max=100, step=1).classes("flex-grow-[1]")
                    self.game_weight.on("wheel", lambda: None)
                    self.game_add_button = ui.button(icon='add', color='green').classes("w-1")
                    self.game_add_button.on_click(lambda: self.on_click_game_add_button())
                    with self.game_add_button:
                        ui.tooltip("Add selected game to prompt").props('delay=1000')

                ui.separator().classes("bg-primary").style("height: 3px")
                with ui.column().classes("w-full") as self.prompt_games_column:
                    pass # added prompt game elements will be created in this container

        ui.separator().classes("bg-primary").style("height: 3px")

        # queued / output samples go here
        self.output_samples: list[OutputSample] = []
        self.output_samples_column = ui.column().classes("w-full")      

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
            ui.notify(f"Loading model: {model_name}", type="info", color="blue", close_button=True)
            error = await self.wait_for_model_server()

            if error is not None:
                ui.notify(f"Error loading model: {self.model_server_state['error']}", type="error", color="red", close_button=True)
            else:
                ui.notify(f"Loaded model {model_name} successfully", type="success", color="green", close_button=True)
                self.model_metadata = self.model_server_state["model_metadata"]
                self.format_config = self.model_server_state["format_config"]
                self.dataset_games_dict = self.model_server_state["dataset_games_dict"]
                self.dataset_game_ids = self.model_server_state["dataset_game_ids"]

                self.game_select.options = self.dataset_games_dict
                self.game_select.value = next(iter(self.dataset_games_dict))
                self.game_select.update()
                self.refresh_game_prompt_elements()

                if model_load_options["compile_options"] is not None:
                    await self.model_server_cmd("compile_model")
                    ui.notify("Compiling model...", type="info", color="blue", close_button=True)
                    error = await self.wait_for_model_server()
                    if error is not None:
                        ui.notify(f"Error compiling model: {self.model_server_state['error']}", type="error", color="red", close_button=True)
                    else:
                        ui.notify("Model compilation complete", type="success", color="green", close_button=True)
    
    async def on_startup_app(self) -> None:
        await self.model_server_cmd("get_available_torch_devices")
        await self.load_model(self.config.model_name, self.config.model_load_options)
        self.load_preset()

    def save_output_sample(self, sample_output: SampleOutput) -> tuple[str, Optional[str]]:
        metadata = {"diffusion_metadata": dict_str(sample_output.params.get_metadata())}
        last_global_step = self.model_metadata["last_global_step"]["unet"]
        audio_output_filename = f"{sample_output.params.get_label(self.model_metadata, self.dataset_game_ids)}.flac"
        audio_output_path = os.path.join(
            config.MODELS_PATH, self.config.model_name, "output", f"step_{last_global_step}", audio_output_filename)
        
        audio_output_path = save_audio(sample_output.raw_sample.squeeze(0),
            self.format_config["sample_rate"], audio_output_path, metadata=metadata, no_clobber=True)
        self.logger.info(f"Saved audio output to {audio_output_path}")

        if self.config.save_output_latents and sample_output.latents is not None:
            latents_output_path = os.path.join(os.path.dirname(audio_output_path), "latents",
                f"{os.path.splitext(os.path.basename(audio_output_path))[0]}.safetensors")
            save_safetensors({"latents": sample_output.latents}, latents_output_path, metadata=metadata)
            self.logger.info(f"Saved latents to {latents_output_path}")
        else:
            latents_output_path = None

        return audio_output_path, latents_output_path

    async def on_click_generate_button(self) -> None:
        
        if len(self.prompt) == 0: # abort if no prompt games selected
            self.logger.error("No prompt games selected")
            ui.notify("No prompt games selected", type="warning", color="red", close_button=True)
            return
        
        # setup sample params and auto-increment seed
        sample_params: SampleParams = SampleParams(seed=self.seed.value,
            length=int(self.generate_length.value) * self.format_config["sample_rate"],
            prompt={**self.prompt}, **self.gen_params)
        if self.auto_increment_seed.value == True: self.seed.set_value(self.seed.value + 1)
        self.logger.info(f"on_click_generate_button - params:{dict_str(sample_params.__dict__)}")

        if self.input_output_sample is not None: # setup inpainting input
            sample_params.input_audio = self.input_output_sample.sample_output.latents
            sample_params.input_audio_pre_encoded = True
            sample_params.inpainting_mask = torch.zeros_like(sample_params.input_audio[:, 0:1])
            sample_params.inpainting_mask[..., self.input_output_sample.select_range.value["min"]:self.input_output_sample.select_range.value["max"]] = 1.
        
        # get name / label and add output sample to workspace
        output_sample = OutputSample(name=f"{sample_params.get_label(self.model_metadata, self.dataset_game_ids)}",
            seed=sample_params.seed, prompt=sample_params.prompt, gen_params={**self.gen_params}, sample_params=sample_params)
        self.add_output_sample(output_sample)

        # this lock holds output samples in queue while generation is in progress
        async with self.gpu_lock:
            if output_sample not in self.output_samples:
                return # handle early removal from queue
            
            # reset abort state and send generate command to model server
            self.model_server_state["generate_abort"] = False
            await self.model_server_cmd("generate", sample_params=sample_params)
            while self.model_server_state.get("cmd", None) is not None:

                if output_sample not in self.output_samples:
                    self.model_server_state["generate_abort"] = True
                    return # abort in progress
                
                # update linear progress
                step = self.model_server_state.get("generate_step", None)
                if step is not None:
                    output_sample.sampling_progress_element.set_value(f"{int(step/sample_params.num_steps*100)}%")

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
                self.remove_output_sample(output_sample)
                return
            
            # retrieve sampling output from model server
            output_sample.sample_output = self.model_server_state["generate_output"]
        
        if output_sample not in self.output_samples:
            return # handle late abort

        # save output sample audio / latents
        output_sample.audio_path, output_sample.latents_path = self.save_output_sample(output_sample.sample_output)

        # set output sample name label
        output_sample.name = os.path.splitext(os.path.basename(output_sample.audio_path))[0]
        output_sample.name_label_element.set_text(output_sample.name)

        # set spectrogram image
        spectrogram_image = output_sample.sample_output.spectrogram.mean(dim=(0,1))
        spectrogram_image = tensor_to_img(spectrogram_image, colormap=True, flip_y=True)
        spectrogram_image = cv2.resize(
            spectrogram_image, (spectrogram_image.shape[1]//2, spectrogram_image.shape[0]), interpolation=cv2.INTER_AREA)
        spectrogram_image = Image.fromarray(spectrogram_image)
        output_sample.spectrogram_image_element.set_source(spectrogram_image)
        output_sample.spectrogram_image_element.set_visibility(True)

        # set audio element
        output_sample.audio_element.set_source(output_sample.audio_path)
        output_sample.audio_element.set_visibility(True)

        # hide progress and setup inpainting range select element
        output_sample.sampling_progress_element.set_visibility(False)
        output_sample.select_range.max = output_sample.sample_output.latents.shape[-1]
        output_sample.select_range.value = {
            "min": output_sample.select_range.max//2 - output_sample.select_range.max//4,
            "max": output_sample.select_range.max//2 + output_sample.select_range.max//4}
        output_sample.select_range.update()
        output_sample.use_as_input_button.enable()

        #output_sample.audio_element.on("timeupdate", lambda e: self.logger.debug(e))
        #output_sample.audio_element.seek(10)
        #output_sample.audio_element.play()
    
    def refresh_output_samples(self) -> None:

        if len(self.output_samples) == 0:
            self.clear_output_button.disable()
        else:
            self.clear_output_button.enable()

        for i, output_sample in enumerate(self.output_samples):
            if i == 0: output_sample.move_up_button.disable()
            else: output_sample.move_up_button.enable()
            if i == len(self.output_samples) - 1: output_sample.move_down_button.disable()
            else: output_sample.move_down_button.enable()

            if self.input_output_sample != output_sample:
                output_sample.use_as_input_button.classes(remove="border-4", add="border-none")
                output_sample.select_range.set_visibility(False)
            else:
                output_sample.use_as_input_button.classes(remove="border-none", add="border-4")
                output_sample.select_range.set_visibility(True)
        
    def add_output_sample(self, output_sample: OutputSample) -> None:
        
        def move_output_sample(output_sample: OutputSample, direction: int) -> None:
            current_index = output_sample.card_element.parent_slot.children.index(output_sample.card_element)
            new_index = min(max(current_index + direction, 0), len(output_sample.card_element.parent_slot.children) - 1)
            if new_index != current_index:
                output_sample.card_element.move(self.output_samples_column, target_index=new_index)
                self.output_samples.insert(new_index, self.output_samples.pop(current_index))
                self.refresh_output_samples()

        def use_output_sample_as_input(output_sample: OutputSample) -> None:
            if self.input_output_sample == output_sample:
                self.input_output_sample = None
            else:
                self.input_output_sample = output_sample
            self.refresh_output_samples()

        with ui.card().classes("w-full") as output_sample.card_element:
            with ui.column().classes("w-full gap-0"):
                with ui.row().classes("h-10 justify-between gap-0 w-full"):

                    with ui.row(): # output sample name label
                        output_sample.name_label_element = ui.label(output_sample.name).classes("p-2").style(
                            "border: 1px solid grey; border-bottom: none; border-radius: 10px 10px 0 0;")
                        
                    with ui.button_group().classes("h-10 gap-0"): # output sample icon toolbar
                        output_sample.use_as_input_button = ui.button(
                            icon="format_color_fill", color="orange", on_click=lambda: use_output_sample_as_input(output_sample)).classes("w-1 border-none border-double")
                        with output_sample.use_as_input_button:
                            ui.tooltip("Use this sample as inpainting input").props('delay=1000')
                        output_sample.use_as_input_button.disable()
                        output_sample.move_up_button = ui.button('▲', on_click=lambda: move_output_sample(output_sample, direction=-1)).classes("w-1")
                        with output_sample.move_up_button:
                            ui.tooltip("Move sample up").props('delay=1000')
                        output_sample.move_down_button = ui.button('▼', on_click=lambda: move_output_sample(output_sample, direction=1)).classes("w-1")
                        with output_sample.move_down_button:
                            ui.tooltip("Move sample down").props('delay=1000')
                        with ui.button('✕', color="red", on_click=lambda s=output_sample: self.remove_output_sample(s)).classes("w-1"):
                            ui.tooltip("Remove sample from workspace").props('delay=1000')

                        #ui.label("Rating:") 
                        #ui.slider(min=0, max=5, step=1, value=0).props("label-always").classes("h-10 top-0")

                output_sample.latents_image_element = ui.interactive_image().classes(
                    "w-full gap-0").style("image-rendering: pixelated; width: 100%; height: auto;").props("fit=scale-down")
                output_sample.sampling_progress_element = ui.linear_progress(
                    value="0%").classes("w-full font-bold gap-0").props("instant-feedback")
                output_sample.spectrogram_image_element = ui.interactive_image(
                    #cross="white", cross_horizontal=False).classes("w-full gap-0").props(add="fit=fill")
                    cross="white").classes("w-full gap-0").props(add="fit=fill")
                output_sample.spectrogram_image_element.set_visibility(False)

                output_sample.select_range = ui.range(min=0, max=688, step=1, value={"min": 0, "max": 0}).classes("w-full").props("step snap color='orange' label='Inpaint Selection'")
                output_sample.select_range.set_visibility(False)

                output_sample.audio_element = ui.audio("").classes("w-full").props("preload='auto'").style("filter: invert(1) hue-rotate(180deg);")
                output_sample.audio_element.set_visibility(False)

        self.output_samples.insert(0, output_sample)
        output_sample.card_element.move(self.output_samples_column, target_index=0)
        self.refresh_output_samples()

    def clear_output_samples(self) -> None:
        self.output_samples_column.clear()
        self.output_samples.clear()
        self.clear_output_button.disable()
        self.input_output_sample = None

    def remove_output_sample(self, output_sample: OutputSample) -> None:
        if self.input_output_sample == output_sample:
            self.input_output_sample = None
        self.output_samples_column.remove(output_sample.card_element)
        self.output_samples.remove(output_sample)
        self.refresh_output_samples()
        
    def refresh_game_prompt_elements(self) -> None:

        def on_game_select_change(new_game_name: str, old_game_name: str) -> None:
            # ugly hack to replace dict key while preserving order
            prompt_list = list(self.prompt.items())
            index = prompt_list.index((old_game_name, self.prompt[old_game_name]))
            prompt_list[index] = (new_game_name, self.prompt[old_game_name])
            self.prompt = dict(prompt_list)

            self.on_change_gen_param()
            self.refresh_game_prompt_elements()
            
        self.prompt_games_column.clear()
        with self.prompt_games_column:
            for game_name, game_weight in self.prompt.items():
    
                if game_name not in self.dataset_games_dict:
                    ui.notify(f"Error '{game_name}' not found in dataset_games_dict", type="error", color="red", close_button=True)
                    continue

                with ui.row().classes("w-full h-10 flex items-center"):
                    game_select_element = ui.select(value=game_name, with_input=True, options=self.dataset_games_dict).classes("flex-grow-[1000]")
                    weight_element = ui.number(label="Weight", value=game_weight, min=-100, max=100, step=1,
                        on_change=lambda: self.on_change_gen_param()).classes("flex-grow-[1]").on("wheel", lambda: None)
                    weight_element.bind_value(self.prompt, game_name)
                    game_select_element.on_value_change(
                        lambda e,g=game_name: on_game_select_change(new_game_name=e.value, old_game_name=g))
                    with ui.button(icon="remove", on_click=lambda g=game_name: self.on_click_game_remove_button(g)).classes("w-1 top-0 right-0").props("color='red'"):
                        ui.tooltip("Remove game from prompt").props('delay=1000')

            ui.separator().classes("bg-transparent")

    def on_click_game_remove_button(self, game_name: str) -> None:
        self.prompt.pop(game_name)
        self.refresh_game_prompt_elements()
        self.on_change_gen_param()
        
    def on_click_game_add_button(self):
        self.prompt.update({self.game_select.value: self.game_weight.value})
        self.refresh_game_prompt_elements()
        self.on_change_gen_param()

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
    
    def toggle_gen_params_lock(self):
        if self.gen_params_lock_button.icon == "lock":
            self.gen_params_lock_button.icon = "lock_open"
        else:
            self.gen_params_lock_button.icon = "lock"
            
    def on_change_gen_param(self) -> None:
        if self.loading_preset == False:
            self.preset_select._props["label"] = f"Select a Preset - (loaded preset: {self.last_loaded_preset}*)"
            self.preset_select.update()
            self.preset_load_button.enable()
            self.preset_save_button.enable()
    
    def on_input_value_preset_select(self, preset_name: str) -> None:
        self.new_preset_name = preset_name
        if preset_name != "default":
            self.preset_save_button.enable()

    def on_blur_preset_select(self, _) -> None:
        if self.new_preset_name != "" and self.new_preset_name not in self.saved_preset_list:
            self.preset_select.options = self.saved_preset_list + [self.new_preset_name]
            self.preset_select.set_value(self.new_preset_name)
            
    def on_value_change_preset_select(self, preset_name: str) -> None:
        self.preset_save_button.enable()
        if preset_name in self.saved_preset_list:
            self.preset_load_button.enable()
            self.preset_select.set_options(self.saved_preset_list)
            if preset_name != "default":
                self.preset_delete_button.enable()
            else:
                self.preset_delete_button.disable()
        else:
            self.preset_delete_button.disable()
            self.preset_load_button.disable()
            
    def get_saved_presets(self) -> list[str]:
        preset_files = os.listdir(os.path.join(config.CONFIG_PATH, "sampling", "presets"))
        saved_presets = []
        for file in preset_files:
            if os.path.splitext(file)[1] == ".json":
                saved_presets.append(os.path.splitext(file)[0])
        saved_presets = sorted(saved_presets)
        self.logger.debug(f"Found saved presets: {saved_presets}")
        self.saved_preset_list = saved_presets
        return saved_presets

    def save_preset(self) -> None:
        preset_name = self.preset_select.value
        preset_filename = f"{sanitize_filename(preset_name)}.json"
        save_preset_path = os.path.join(
            config.CONFIG_PATH, "sampling", "presets", preset_filename)
        preset_name = os.path.splitext(os.path.basename(save_preset_path))[0]
        self.logger.debug(f"Saving preset '{save_preset_path}'")

        save_preset_dict = {"prompt": self.prompt, "gen_params": self.gen_params}
        config.save_json(save_preset_dict, save_preset_path)
        self.logger.info(f"Saved preset {preset_name}: {dict_str(save_preset_dict)}")

        self.preset_select.options = self.get_saved_presets()
        self.last_loaded_preset = preset_name
        self.preset_select.value = preset_name
        self.preset_select._props["label"] = f"Select a Preset - (loaded preset: {self.last_loaded_preset})"
        self.preset_select.update()
        self.preset_load_button.disable()
        self.preset_save_button.disable()
        if preset_name != "default":
            self.preset_delete_button.enable()
    
    async def reset_preset_loading_state(self) -> None:
        await asyncio.sleep(0.25)
        self.loading_preset = False

    def load_preset(self) -> None:
        preset_name = self.preset_select.value
        load_preset_path = os.path.join(
            config.CONFIG_PATH, "sampling", "presets", f"{sanitize_filename(preset_name)}.json")

        loaded_preset_dict = config.load_json(load_preset_path)
        self.logger.info(f"Loaded preset {preset_name}: {dict_str(loaded_preset_dict)}")

        self.last_loaded_preset = preset_name

        if self.gen_params_lock_button.icon == "lock_open":
            self.gen_params.update(loaded_preset_dict["gen_params"])
            self.preset_load_button.disable()
            self.preset_save_button.disable()
            self.preset_select._props["label"] = f"Select a Preset - (loaded preset: {self.last_loaded_preset})"
            self.loading_preset = True
            asyncio.create_task(self.reset_preset_loading_state())
        else:
            # check if loaded_preset_dict matches current gen_params, even if locked
            if all([self.gen_params[p] == loaded_preset_dict["gen_params"][p] for p in self.gen_params.keys()]) == True:
                self.preset_load_button.disable()
                self.preset_save_button.disable()
                self.preset_select._props["label"] = f"Select a Preset - (loaded preset: {self.last_loaded_preset})"
                self.loading_preset = True
                asyncio.create_task(self.reset_preset_loading_state())
            else:
                self.preset_select._props["label"] = f"Select a Preset - (loaded preset: {self.last_loaded_preset}*)"

        self.preset_select.update()
        self.prompt.clear()
        self.prompt.update(loaded_preset_dict["prompt"])
        self.refresh_game_prompt_elements()
    
    def delete_preset(self) -> None:
        preset_name = self.preset_select.value
        delete_preset_path = os.path.join(
            config.CONFIG_PATH, "sampling", "presets", f"{sanitize_filename(preset_name)}.json")
        self.logger.info(f"Deleting preset '{delete_preset_path}'")

        os.remove(delete_preset_path)
        self.preset_select.options = self.get_saved_presets()
        self.preset_select.set_value("default")
        self.preset_load_button.enable()
        self.preset_save_button.enable()
        self.preset_delete_button.disable()

    def run(self) -> None:
        on_air_token = os.getenv("ON_AIR_TOKEN", None)
        ui.run(dark=self.config.enable_dark_mode, title="Dual-Diffusion WebUI", reload=False, show=False,
            host=self.config.web_server_host, port=self.config.web_server_port, on_air=on_air_token)


if __name__ in {"__main__", "__mp_main__"}:
    
    if os.getenv("_IS_NICEGUI_SUBPROCESS", None) is None: # ugly hack
        os.environ["_IS_NICEGUI_SUBPROCESS"] = "1"
        NiceGUIApp().run()