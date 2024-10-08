from utils import config

from typing import Optional, Union
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from subprocess import Popen
from PIL import Image
import asyncio
import logging
import random
import os

import torch
import numpy as np
from nicegui import ui, app

from utils.dual_diffusion_utils import (
    init_cuda, save_audio, load_audio, dict_str, tensor_to_img,
    get_available_torch_devices, sanitize_filename
)
from pipelines.dual_diffusion_pipeline import DualDiffusionPipeline, SampleParams, SampleOutput
from sampling.schedule import SamplingSchedule
from sampling.scrollable_log import ScrollableLog

@dataclass
class OutputSample:
    name: str
    seed: int
    prompt: dict
    gen_params: dict
    sample_params: SampleParams
    sample_output: Optional[SampleOutput] = None
    audio_path: Optional[str] = None
    is_input_audio: bool = False

    card_element: Optional[ui.card] = None
    name_label_element: Optional[ui.label] = None
    sampling_progress_element: Optional[ui.linear_progress] = None
    latents_image_element: Optional[ui.interactive_image] = None
    spectrogram_image_element: Optional[ui.image] = None
    audio_element: Optional[ui.audio] = None
    use_as_input_button: Optional[ui.button] = None
    select_range: Optional[ui.range] = None

@dataclass
class NiceGUIAppConfig:
    model_name: str
    model_load_options: dict
    max_gpu_concurrency: int = 1

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

        # load model
        model_path = os.path.join(config.MODELS_PATH, self.config.model_name)
        model_load_options = self.config.model_load_options

        self.logger.info(f"Loading DualDiffusion model from '{model_path}'...")
        self.pipeline = DualDiffusionPipeline.from_pretrained(model_path, **model_load_options)
        self.logger.debug(f"Model metadata:\n{dict_str(self.pipeline.model_metadata)}")

        # setup dataset games list
        for game_name, count in self.pipeline.dataset_info["game_train_sample_counts"].items():
            if count == 0: self.pipeline.dataset_game_ids.pop(game_name)
        self.dataset_games_dict = {} # keys are actual game names, values are display strings
        for game_name in self.pipeline.dataset_game_ids.keys():
            self.dataset_games_dict[game_name] = f"({self.pipeline.dataset_info['game_train_sample_counts'][game_name]}) {game_name}"
        self.dataset_games_dict = dict(sorted(self.dataset_games_dict.items()))

        self.gpu_lock = asyncio.Semaphore(self.config.max_gpu_concurrency)
        self.input_audio_sample: OutputSample = None

        self.init_layout()
        app.on_startup(lambda: self.on_startup_app())

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
                # todo: switching tabs currently scrolls the log back up to the top, try to fix this
                """
                def on_debug_logs_tab_focus():  
                    with self.log_handler.log_control:
                        tmp_label = ui.label("")
                    self.log_handler.log_control.remove(tmp_label)
                self.debug_logs_tab.on("focus", on_debug_logs_tab_focus)
                """

    def init_model_settings_layout(self) -> None:
        ui.label("model settings stuff")

    def init_debug_logs_layout(self) -> None:
        ui.label("Debug Log:")
        #self.debug_log = ui.log(max_lines=self.config.max_debug_log_length).style("height: 500px")
        self.debug_log = ScrollableLog(max_lines=self.config.max_debug_log_length).style("height: 500px")
        self.log_handler.set_log_control(self.debug_log)

    def init_generation_layout(self) -> None:
            
        with ui.row().classes("w-full"): # gen params, preset, and prompt editor
            with ui.card().classes("flex-grow-[1]"):
                with ui.row().classes("w-full"): # gen params and seed
                    with ui.card().classes("flex-grow-[1]"): # seed params
                        self.seed = ui.number(label="Seed", value=10042, min=10000, max=99999, precision=0, step=1).classes("w-full")
                        self.seed.on("wheel", lambda: None)
                        self.auto_increment_seed = ui.checkbox("Auto Increment Seed", value=True).classes("w-full")
                        with ui.button("Randomize Seed", icon="casino", on_click=lambda: self.seed.set_value(random.randint(10000, 99999))).classes("w-full"):
                            ui.tooltip("Choose new seed at random").props('delay=1000')
                        self.generate_length = ui.number(label="Length (seconds)", value=0, min=0, max=300, precision=0, step=5).classes("w-full")
                        self.generate_button = ui.button("Generate", icon="audiotrack", color="green", on_click=partial(self.on_click_generate_button)).classes("w-full")
                        with self.generate_button:
                            ui.tooltip("Generate new sample with current settings").props('delay=1000')
                        self.clear_output_button = ui.button("Clear Outputs", icon="delete", color="red", on_click=lambda: self.clear_output_samples()).classes("w-full")
                        self.clear_output_button.disable()
                        with self.clear_output_button:
                            ui.tooltip("Clear all output samples in workspace").props('delay=1000')

                    with ui.card().classes("flex-grow-[3]"): # gen params
                        self.gen_param_elements = {}
                        with ui.grid(columns=2).classes("w-full items-center"):
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
                            self.show_schedule_button = ui.button("Show σ Schedule", on_click=lambda: self.on_click_show_schedule_button()).classes("w-full h-1")
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

                        with ui.column().classes("items-center"):
                            #with ui.button_group().classes():
                            with ui.column().classes("gap=0"):
                                self.preset_load_button = ui.button("Load", icon="source", on_click=lambda: self.load_preset()).classes("w-full gap-0")
                                self.preset_save_button = ui.button("Save", icon="save", color="green", on_click=lambda: self.save_preset()).classes("w-full gap-0")
                                self.preset_delete_button = ui.button("Delete", icon="delete", color="red", on_click=lambda: self.delete_preset()).classes("w-full gap-0")
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
                with ui.row().classes("w-full flex items-center"):
                    self.game_select = ui.select(label="Select a game", value=next(iter(self.dataset_games_dict)), with_input=True,
                        options=self.dataset_games_dict).classes("flex-grow-[1000]")
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
                        
    def on_startup_app(self) -> None:
        self.load_preset()
        #app.add_static_files(...)
        ui.add_body_html('''
        <script type="module">
            import WaveSurfer from 'https://cdn.jsdelivr.net/npm/wavesurfer.js@7/dist/wavesurfer.esm.js'
            import Spectrogram from 'https://cdn.jsdelivr.net/npm/wavesurfer.js@7.8/dist/plugins/spectrogram.esm.js'
            import TimelinePlugin from 'https://cdn.jsdelivr.net/npm/wavesurfer.js@7.8/dist/plugins/timeline.esm.js'
            import RegionsPlugin from 'https://cdn.jsdelivr.net/npm/wavesurfer.js@7.8/dist/plugins/regions.esm.js'
            window.WaveSurfer = WaveSurfer;
            window.Spectrogram = Spectrogram;
            window.TimelinePlugin = TimelinePlugin;
            window.RegionsPlugin = RegionsPlugin;
        </script>
        ''')

    """                      
    const wavesurfer = WaveSurfer.create({
    container: '#waveform',
    waveColor: '#4F4A85',
    progressColor: '#383351',
    url: '/audio.flac',
    sampleRate: 32000,
    height: 0,
    cursorColor: 'white',
    dragToSeek: true,
    })

    // Create a timeline plugin instance with custom options
    const topTimeline = TimelinePlugin.create({
    height: 20,
    timeInterval: 0.2,
    primaryLabelInterval: 5,
    //secondaryLabelInterval: 1,
    style: {
        fontSize: '10px',
        color: '#FFFFFF',
    },
    })
    wavesurfer.registerPlugin(topTimeline)

    /*
    // Initialize the Regions plugin
    const regions = RegionsPlugin.create()
    wavesurfer.registerPlugin(regions)

    // Create some regions at specific time ranges
    wavesurfer.on('decode', () => {
    regions.addRegion({
        start: 9,
        end: 10,
        content: 'Cramped region',
        color: randomColor(),
        minLength: 1,
        maxLength: 10,
    })
    })
    */

    // Initialize the Spectrogram plugin
    wavesurfer.registerPlugin(
    Spectrogram.create({
        labels: false,
        height: 200,
        splitChannels: false,
        scale: 'mel',
        frequencyMax: 16000,
        windowFunc: 'blackman',
    }),
    )

    wavesurfer.on('interaction', () => {
    wavesurfer.playPause()
    })
    """

    def save_output_sample(self, sample_output: SampleOutput) -> str:
        metadata = {"diffusion_metadata": dict_str(sample_output.params.get_metadata())}
        last_global_step = self.pipeline.model_metadata["last_global_step"]["unet"]
        audio_output_filename = f"{sample_output.params.get_label(self.pipeline)}.flac"
        audio_output_path = os.path.join(
            config.MODELS_PATH, self.config.model_name, "output", f"step_{last_global_step}", audio_output_filename)
        
        audio_output_path = save_audio(sample_output.raw_sample.squeeze(0),
            self.pipeline.format.config.sample_rate, audio_output_path, metadata=metadata, no_clobber=True)
        self.logger.info(f"Saved audio output to {audio_output_path}")
        return audio_output_path

    async def on_click_generate_button(self) -> None:
        
        if len(self.prompt) == 0:
            self.logger.error("No prompt games selected")
            ui.notify("No prompt games selected", type="warning", color="red", close_button=True)
            return
        
        sample_params: SampleParams = SampleParams(seed=self.seed.value,
            length=int(self.generate_length.value) * self.pipeline.format.config.sample_rate,
            prompt={**self.prompt}, **self.gen_params)
        if self.auto_increment_seed.value == True: self.seed.set_value(self.seed.value + 1)
        self.logger.info(f"on_click_generate_button - params:{dict_str(sample_params.__dict__)}")

        if self.input_audio_sample is not None: # setup inpainting input
            sample_params.input_audio = self.input_audio_sample.sample_output.latents
            sample_params.input_audio_pre_encoded = True
            sample_params.inpainting_mask = torch.zeros_like(sample_params.input_audio[:, 0:1])
            sample_params.inpainting_mask[..., self.input_audio_sample.select_range.value["min"]:self.input_audio_sample.select_range.value["max"]] = 1.

        output_sample = OutputSample(name=f"{sample_params.get_label(self.pipeline)}",
            seed=sample_params.seed, prompt=sample_params.prompt, gen_params={**self.gen_params}, sample_params=sample_params)
        self.add_output_sample(output_sample)

        await asyncio.sleep(0.1) # ensure the seed increment event is processed before beginning sampling

        def sampling_progress_callback(stage: str, progress: Union[float, torch.Tensor]) -> bool:
            if output_sample not in self.output_samples:
                return False
            
            if stage == "sampling":
                output_sample.sampling_progress_element.set_value(f"{int(progress*100)}%")
            elif stage == "latents":
                latents_image = tensor_to_img(progress, flip_y=True)
                latents_image = Image.fromarray(latents_image)
                output_sample.latents_image_element.set_source(latents_image)
            return True

        async with self.gpu_lock:
            if output_sample not in self.output_samples: return # handle removal from queue
            output_sample.sample_output = await self.pipeline(sample_params, lambda stage,
                progress: sampling_progress_callback(stage, progress))
        if output_sample not in self.output_samples: return # handle abort in progress
        output_sample.audio_path = self.save_output_sample(output_sample.sample_output)

        output_sample.name = os.path.splitext(os.path.basename(output_sample.audio_path))[0]
        output_sample.name_label_element.set_text(output_sample.name)
        spectrogram_image = output_sample.sample_output.spectrogram.mean(dim=(0,1))
        spectrogram_image = tensor_to_img(spectrogram_image, colormap=True, flip_y=True)
        spectrogram_image = Image.fromarray(spectrogram_image)
        output_sample.spectrogram_image_element.set_source(spectrogram_image)
        output_sample.spectrogram_image_element.set_visibility(True)
        output_sample.audio_element.set_source(output_sample.audio_path)
        output_sample.audio_element.set_visibility(True)
        output_sample.sampling_progress_element.set_visibility(False)
        output_sample.select_range.max = output_sample.sample_output.latents.shape[-1]
        output_sample.select_range.update()
        output_sample.use_as_input_button.enable()

        # todo: button to open output sample in configured daw
        #Popen(["c:/program files/audacity/audacity.exe", audio_output_path])
    
    def add_output_sample(self, output_sample: OutputSample) -> None:
        
        def remove_output_sample(output_sample: OutputSample) -> None:
            if self.input_audio_sample == output_sample:
                self.input_audio_sample = None
            self.output_samples_column.remove(output_sample.card_element)
            self.output_samples.remove(output_sample)
            if len(self.output_samples) == 0:
                self.clear_output_button.disable()

        def move_output_sample(output_sample: OutputSample, direction: int) -> None:
            current_index = output_sample.card_element.parent_slot.children.index(output_sample.card_element)
            new_index = min(max(current_index + direction, 0), len(output_sample.card_element.parent_slot.children) - 1)
            if new_index != current_index:
                output_sample.card_element.move(self.output_samples_column, target_index=new_index)

        def use_output_sample_as_input(output_sample: OutputSample) -> None:
            if self.input_audio_sample == output_sample:
                self.input_audio_sample = None
                output_sample.is_input_audio = False
                output_sample.use_as_input_button.classes(remove="border-4", add="border-none")
                output_sample.select_range.set_visibility(False)
                self.logger.debug("use_output_sample_as_input - removed input")
            elif self.input_audio_sample is not None:
                self.input_audio_sample.use_as_input_button.classes(remove="border-4", add="border-none")
                self.input_audio_sample.select_range.set_visibility(False)
                self.input_audio_sample = output_sample
                output_sample.is_input_audio = True
                output_sample.use_as_input_button.classes(remove="border-none", add="border-4")
                output_sample.select_range.set_visibility(True)
                self.logger.debug("use_output_sample_as_input - changed input")
            else:
                self.input_audio_sample = output_sample
                output_sample.is_input_audio = True
                output_sample.use_as_input_button.classes(add="border-4", remove="border-none")
                output_sample.select_range.set_visibility(True)
                self.logger.debug("use_output_sample_as_input - set input")

        self.output_samples.insert(0, output_sample)
        self.clear_output_button.enable()

        with ui.card().classes("w-full") as output_sample.card_element:
            with ui.row().classes("w-full"):
                with ui.column().classes("flex-grow-[50] gap-0"):
                    with ui.row().classes("gap-0 items-center"):
                        output_sample.name_label_element = ui.label(output_sample.name).classes("p-2").style(
                            "border: 1px solid grey; border-bottom: none; border-radius: 10px 10px 0 0;")
                        #ui.label("Rating:") 
                        #ui.slider(min=0, max=5, step=1, value=0).props("label-always").classes("h-10 top-0")
                    output_sample.latents_image_element = ui.interactive_image().classes(
                        "w-full gap-0").style("image-rendering: pixelated; width: 100%; height: auto;").props("fit=scale-down")
                    output_sample.sampling_progress_element = ui.linear_progress(
                        value="0%").classes("w-full font-bold gap-0").props("instant-feedback")
                    output_sample.spectrogram_image_element = ui.image().classes("w-full gap-0").style("width: 100%; height: 200px;").props("fit=fill")
                    output_sample.spectrogram_image_element.set_visibility(False)

                    output_sample.select_range = ui.range(min=0, max=688, step=1, value={"min": 0, "max": 0}).classes("w-full").props("step snap color='orange' label='Inpaint Selection'")
                    output_sample.select_range.set_visibility(False)

                    dummy_audio_path = os.path.join(config.DEBUG_PATH, "nicegui_app", "test_audio.flac")
                    output_sample.audio_element = ui.audio(dummy_audio_path).classes("w-full").props("preload='auto'").style("filter: invert(1) hue-rotate(180deg);")
                    output_sample.audio_element.set_visibility(False)
                    """
                    ui.html("<div id='waveform'></div>")
                    ui.run_javascript('''
                    const wavesurfer = window.WaveSurfer.create({
                    container: '#waveform',
                    waveColor: '#4F4A85',
                    progressColor: '#383351',
                    url: '/audio.flac',
                    sampleRate: 32000,
                    height: 0,
                    cursorColor: 'white',
                    dragToSeek: true,
                    })
                    ''')
                    """

                with ui.column().classes("w-8 gap-0 items-center"):
                    with ui.button('✕', on_click=lambda: remove_output_sample(output_sample)).classes("w-1 rounded-b-none").props("color='red'"):
                        ui.tooltip("Remove sample from workspace").props('delay=1000')
                    output_sample.use_as_input_button = ui.button(
                        icon="format_color_fill", color="orange", on_click=lambda: use_output_sample_as_input(output_sample)).classes("w-1 rounded-none border-none border-double")
                    with output_sample.use_as_input_button:
                        ui.tooltip("Use this sample as inpainting input").props('delay=1000')
                    output_sample.use_as_input_button.disable()
                    with ui.button('▲', on_click=lambda: move_output_sample(output_sample, direction=-1)).classes("w-1 rounded-none"):
                        ui.tooltip("Move sample up").props('delay=1000')
                    with ui.button('▼', on_click=lambda: move_output_sample(output_sample, direction=1)).classes("w-1 rounded-t-none"):
                        ui.tooltip("Move sample down").props('delay=1000')

        output_sample.card_element.move(self.output_samples_column, target_index=0)

    def clear_output_samples(self) -> None:
        self.output_samples_column.clear()
        self.output_samples.clear()
        self.clear_output_button.disable()
        self.input_audio_sample = None

    def refresh_game_prompt_elements(self) -> None:

        def on_game_select_change(new_game_name: str, old_game_name: str, weight_element: ui.number) -> None:
            self.logger.debug(f"game_select changed {old_game_name} to {new_game_name}")
            old_weight = self.prompt[old_game_name]
            self.prompt[new_game_name] = old_weight
            weight_element.bind_value(self.prompt, new_game_name)
            self.prompt.pop(old_game_name)
            self.on_change_gen_param()
            
        self.logger.debug(f"refresh_game_prompt_elements: {dict_str(self.prompt)}")
        self.prompt_games_column.clear()
        with self.prompt_games_column:
            for game_name, game_weight in self.prompt.items():
                with ui.row().classes("w-full flex items-center"):
                    game_select_element = ui.select(value=game_name, with_input=True, options=self.dataset_games_dict).classes("flex-grow-[1000]")
                    weight_element = ui.number(label="Weight", value=game_weight, min=-100, max=100, step=1,
                        on_change=lambda: self.on_change_gen_param()).classes("flex-grow-[1]").on("wheel", lambda: None)
                    game_select_element.on_value_change(
                        lambda e: on_game_select_change(new_game_name=e.value, old_game_name=game_name, weight_element=weight_element))
                    weight_element.bind_value(self.prompt, game_name)
                    with ui.button(icon="remove").classes("w-1 top-0 right-0").props("color='red'").on_click(lambda g=game_name: self.on_click_game_remove_button(g)):
                        ui.tooltip("Remove game from prompt").props('delay=1000')

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
            
    def on_change_gen_param(self) -> None:
        if self.loading_preset == False:
            self.preset_select._props["label"] = f"Select a Preset - (loaded preset: {self.last_loaded_preset}*)"
            self.preset_select.update()
            self.preset_load_button.enable()
            self.preset_save_button.enable()
        self.logger.debug(f"updated gen_params loading_preset: {self.loading_preset}")
    
    def on_input_value_preset_select(self, preset_name: str) -> None:
        self.new_preset_name = preset_name
        if preset_name != "default":
            self.preset_save_button.enable()

    def on_blur_preset_select(self, _) -> None:
        if self.new_preset_name != "" and self.new_preset_name not in self.saved_preset_list:
            self.preset_select.options = self.saved_preset_list + [self.new_preset_name]
            self.preset_select.set_value(self.new_preset_name)
            
    def on_value_change_preset_select(self, preset_name: str) -> None:
        self.logger.debug(f"Selected preset: {preset_name}")
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
        self.logger.debug("reset loading state")

    def load_preset(self) -> None:
        preset_name = self.preset_select.value
        load_preset_path = os.path.join(
            config.CONFIG_PATH, "sampling", "presets", f"{sanitize_filename(preset_name)}.json")
        self.logger.debug(f"Loading preset '{load_preset_path}'")

        loaded_preset_dict = config.load_json(load_preset_path)
        self.logger.info(f"Loaded preset {preset_name}: {dict_str(loaded_preset_dict)}")
        self.loading_preset = True
        asyncio.create_task(self.reset_preset_loading_state())

        self.prompt.clear()
        self.prompt.update(loaded_preset_dict["prompt"])
        self.refresh_game_prompt_elements()
        self.gen_params.update(loaded_preset_dict["gen_params"])
        self.preset_load_button.disable()
        self.preset_save_button.disable()
        self.last_loaded_preset = preset_name
        self.preset_select._props["label"] = f"Select a Preset - (loaded preset: {self.last_loaded_preset})"
        self.preset_select.update()
    
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
        ui.run(dark=self.config.enable_dark_mode, title="Dual-Diffusion WebUI",
            host=self.config.web_server_host, port=self.config.web_server_port, on_air=on_air_token)


if __name__ in {"__main__", "__mp_main__"}:

    init_cuda()
    NiceGUIApp().run()