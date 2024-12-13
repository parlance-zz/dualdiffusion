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

from typing import Optional, Union, Callable, Literal, Any
from dataclasses import dataclass
from copy import deepcopy
from PIL import Image
import os
import asyncio
import logging
import random

import numpy as np
import torch
import torchaudio
import cv2
import nicegui
from nicegui import ui

from sampling.schedule import SamplingSchedule
from sampling.nicegui_audio_editor import AudioEditor
from pipelines.dual_diffusion_pipeline import SampleParams, SampleOutput
from utils.dual_diffusion_utils import (
    sanitize_filename, dict_str, save_safetensors, load_safetensors,
    save_audio, load_audio, tensor_to_img, update_audio_metadata, update_safetensors_metadata
)

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

    editing_mode: Literal["normal", "inpaint", "extend"] = "normal"
    highlight_start: int = 0    # in units of latent pixels
    highlight_duration: int = 0 # in units of latent pixels
    card_element: Optional[ui.card] = None
    name_row_element: Optional[ui.row] = None
    name_label_element: Optional[ui.label] = None
    rating_element: Optional["StarRating"] = None
    sampling_progress_element: Optional[ui.linear_progress] = None
    latents_image_element: Optional[ui.interactive_image] = None
    audio_editor_element: Optional["AudioEditor"] = None
    toggle_show_latents_button: Optional["ToggleButton"] = None
    toggle_show_params_button: Optional["ToggleButton"] = None
    toggle_show_debug_button: Optional["ToggleButton"] = None
    inpaint_button: Optional[ui.button] = None
    select_range: Optional[ui.range] = None
    extend_button: Optional[ui.button] = None
    extend_mode_radio: Optional[ui.radio] = None
    move_up_button: Optional[ui.button] = None
    move_down_button: Optional[ui.button] = None
    show_parameters_row_element: Optional[ui.row] = None
    show_debug_plots_row_element: Optional[ui.row] = None

class ToggleButton(ui.button):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.is_toggled = False
        self.on_toggle = lambda _: None
        self.classes("border-none border-double")

        self.on("click", lambda: self.toggle())
            
    def toggle(self, is_toggled: Optional[bool] = None) -> bool:
        if is_toggled is not None:
            self.is_toggled = is_toggled
        else:
            self.is_toggled = not self.is_toggled

        if self.is_toggled == True:
            self.classes(remove="border-none", add="border-4")
        else:
            self.classes(remove="border-4", add="border-none")
        self.on_toggle(self.is_toggled)

class StarRating(ui.row):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.on_rating_change = lambda _: None
        self.is_disabled = False
        self.rating = 0

        self.stars: list[ui.icon] = []
        with self.classes("gap-0"):
            self.remove_icon = ui.icon("block", color="red").style("cursor: pointer")
            self.remove_icon.on("click", lambda: self.set_rating(0))
            for i in range(0, 5):
                star = ui.icon("star_rate", color="darkgray").style("cursor: pointer")
                star.on("click", lambda i=i: self.set_rating(i+1))
                self.stars.append(star)
    
    def disable(self) -> None:
        self.is_disabled = True
        self.remove_icon.style("cursor: not-allowed")
        for star in self.stars:
            star.style("cursor: not-allowed")

    def enable(self) -> None:
        self.is_disabled = False
        self.remove_icon.style("cursor: pointer")
        for star in self.stars:
            star.style("cursor: pointer")

    def set_rating(self, rating: int) -> None:
        if self.is_disabled == True:
            return
        
        self.rating = rating
        for i, star in enumerate(self.stars):
            if rating == 0:
                star.style("color: darkgray")
            elif i <= rating - 1:
                star.style("color: gold")
            else:
                star.style("color: lightgray")  

        self.on_rating_change(rating)

class EditableSelect(ui.select): # unfortunate necessity due to select element behavior

    def __init__(self, *args, **kwargs) -> None:
        kwargs["with_input"] = True
        kwargs["new_value_mode"] = None
        super().__init__(*args, **kwargs)

        self.new_value = self.value
        self.on("input-value", lambda e: self.on_input_value_self(e.args))
        self.on("blur", lambda: self.on_blur_self())
        self.on_value_change(lambda e: self.on_value_change_self(e.value))

        if not isinstance(self.options, list):
            raise ValueError("EditableSelect options must be a list")
        
    def set_original_options(self, original_options: Optional[list] = None) -> None:
        new_original_options = original_options or self.options
        if not isinstance(new_original_options, list):
            raise ValueError("EditableSelect options must be a list")
        
        self.original_options = deepcopy(new_original_options)
        self.options = deepcopy(new_original_options)

        self.on_blur_self()

    def on_input_value_self(self, value: str) -> None:
        self.new_value = value

    def on_value_change_self(self, value: str) -> None:
        self.new_value = value
        if self.new_value in self.original_options:
            self.set_options(options=self.original_options, value=self.new_value)

    def set_value(self, value: str) -> None:
        self.new_value = value
        super().set_value(value)

    def on_blur_self(self) -> None:
        if self.new_value != "":
            if self.new_value not in self.original_options:
                self.set_options(options=self.original_options + [self.new_value], value=self.new_value)
            else:
                super().set_value(self.new_value)

class LockButton(ui.button):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(icon="lock_open", *args, **kwargs)     
        self.style("position: absolute; top: 5px; right: 5px; z-index: 2; background-color: transparent;"
                   "border: none; width: 20px; height: 20px; padding: 0;").classes("bg-transparent z-10")
        self.on_click(lambda: self.toggle_lock())

    def toggle_lock(self) -> bool:
        self.is_locked = not self.is_locked
        return self.is_locked

    @property
    def is_locked(self) -> bool:
        return self.icon == "lock"
    
    @is_locked.setter
    def is_locked(self, locked: bool) -> None:
        self.icon = "lock" if locked == True else "lock_open"

class CopyButton(ui.button):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(icon="content_copy", *args, **kwargs)     

        self.icon="content_copy"
        self.style("position: absolute; top: 5px; right: 5px; z-index: 2; background-color: transparent;"
            "border: none; width: 20px; height: 20px; padding: 0;").classes("bg-transparent z-10")

class ScrollableNumber(ui.number): # same as ui.number but works with mouse wheel

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # curiously, this enables values scrolling with the mouse wheel
        self.on("wheel", lambda: None)

class GenParamsEditor(ui.row):

    def __init__(self, *args, gen_params: Optional[dict[str, Union[float, int, str]]] = None, seed: Optional[int] = None, 
                 seamless_loop: Optional[bool] = None, length: Optional[int] = None, read_only: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        seed = seed or 10042
        length = length or 0
        seamless_loop = seamless_loop or False
        self.gen_params: dict[str, Union[float, int, str]] = gen_params or {}
        self.param_elements: dict[str, ui.element] = {}
        self.read_only = read_only
        self.on_change_gen_param = lambda: None

        with self.classes("w-full"):
            with ui.card().classes("w-48"):

                self.generate_length = ScrollableNumber(label="Length (seconds)", value=length, min=0, max=150, precision=0, step=5).classes("w-full")
                self.seamless_loop = ui.checkbox("Seamless Loop", value=seamless_loop).classes("w-full")
                with ui.row().classes("w-full items-center"):
                    self.seed = ScrollableNumber(label="Seed", value=seed, min=10000, max=99999, precision=0, step=1)
                    if read_only == False:
                        self.seed.classes("w-32")
                        with ui.button(icon="casino", on_click=lambda: self.seed.set_value(random.randint(10000, 99999))).classes("right-0 w-8").style("position: absolute; right: 5px;"
                            "border: none; width: 40px; height: 40px; padding: 0; font-size: 20px").classes("bg-transparent z-10"):
                            ui.tooltip("Choose new seed at random")
                    else:
                        self.seed.classes("w-full")
                
                if read_only == True:
                    self.generate_length.disable()
                    self.seamless_loop.disable()
                    self.seed.disable()
                else:
                    self.auto_increment_seed = ui.checkbox("Auto Increment Seed", value=True).classes("w-full")

            with ui.card().classes("w-[88]"):
                with ui.grid(columns=2).classes("w-full items-center"):
                    if read_only == False:
                        with LockButton() as self.lock_button:
                            ui.tooltip("Lock to freeze parameters when loading presets")

                    self.param_elements["num_steps"] = ScrollableNumber(label="Number of Steps", value=100, min=10, max=1000, precision=0, step=10).classes("w-full")
                    self.param_elements["cfg_scale"] = ScrollableNumber(label="CFG Scale", value=1.5, min=0, max=10, step=0.1).classes("w-full")
                    self.param_elements["use_heun"] = ui.checkbox("Use Heun's Method", value=True).classes("w-full")
                    self.param_elements["num_fgla_iters"] = ScrollableNumber(label="Number of FGLA Iterations", value=250, min=50, max=1000, precision=0, step=50).classes("w-full")

                    self.param_elements["input_perturbation"] = ScrollableNumber(label="Input Perturbation", value=1, min=0, max=1, step=0.05).classes("w-full")
                    self.param_elements["input_perturbation_offset"] = ScrollableNumber(label="Input Perturbation Offset", value=0, min=-10, max=10, step=0.05).classes("w-full")

                    self.param_elements["sigma_max"] = ScrollableNumber(label="Sigma Max", value=200, min=10, max=1000, step=10).classes("w-full")
                    self.param_elements["sigma_min"] = ScrollableNumber(label="Sigma Min", value=0.15, min=0.05, max=2, step=0.05).classes("w-full")
                    self.param_elements["schedule"] = ui.select(label="Σ Schedule", options=SamplingSchedule.get_schedules_list(), value="edm2").classes("w-full")
                    self.param_elements["rho"] = ScrollableNumber(label="Rho", value=7, min=0.5, max=1000, precision=2, step=0.5).classes("w-full")
                    
                    self.sigma_schedule_dialog = ui.dialog()
                    self.show_schedule_button = ui.button("Show σ Schedule", color="gray", on_click=lambda: self.on_click_show_schedule_button()).classes("w-full items-center")
                    with self.show_schedule_button:
                        ui.tooltip("Show noise schedule with current settings")

        for param_name, param_element in self.param_elements.items():
            if read_only == False:
                if param_name not in self.gen_params:
                    self.gen_params[param_name] = param_element.value
                param_element.bind_value(self.gen_params, param_name)
                param_element.on_value_change(lambda: self.on_change_gen_param())
            else:
                param_element.value = self.gen_params.get(param_name, param_element.value)
                param_element.disable()

    def on_click_show_schedule_button(self) -> None:

        self.sigma_schedule_dialog.clear()
        with self.sigma_schedule_dialog, ui.card():

            sigma_schedule = SamplingSchedule.get_schedule(
                self.gen_params["schedule"], int(self.gen_params["num_steps"]),
                sigma_max=self.gen_params["sigma_max"],
                sigma_min=self.gen_params["sigma_min"],
                rho=self.gen_params["rho"])

            x = np.arange(int(self.gen_params["num_steps"]) + 1)
            y = sigma_schedule.log().numpy()
            
            with ui.matplotlib(figsize=(5, 4)).classes("border-gray-600 border-2").figure as fig:
                #fig.patch.set_facecolor("#1d1d1d") # todo: doesn't do anything :(
                ax = fig.gca()
                ax.grid(color="#666666", linewidth=0.5, linestyle="--")
                ax.set_facecolor("#1d1d1d")
                ax.set_title("Sigma Schedule")
                ax.plot(x, y, "-", color="#5898d4")
                ax.set_xlabel("step")
                ax.set_ylabel("ln(sigma)")

            self.sigma_schedule_dialog.open()
            ui.button("Close").classes("ml-auto").on_click(lambda: self.sigma_schedule_dialog.close())

class PromptEditor(ui.card):

    def __init__(self, *args, prompt: Optional[dict[str, float]] = None, read_only: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.prompt: dict[str, float] = prompt or {}
        self.dataset_games_dict: dict[str, str] = {}
        self.read_only = read_only
        self.on_change_prompt = lambda: None

        with self.classes("flex-grow-[50]"):
            #with ui.row().classes("w-full h-0 gap-0"):
            #    with LockButton() as self.prompt_lock_button:
            #        ui.tooltip("Lock to freeze prompt when loading presets")
            if read_only == False:
                with ui.row().classes("w-full h-12 flex items-center"):
                    self.game_select = ui.select(label="Select a game", with_input=True, options={}).classes("flex-grow-[1000]")
                    self.game_weight = ScrollableNumber(label="Weight", value=10, min=-100, max=100, step=1).classes("flex-grow-[1]")
                    self.game_add_button = ui.button(icon="add", color="green", on_click=lambda: self.on_click_game_add_button()).classes("w-1")
                    with self.game_add_button:
                        ui.tooltip("Add selected game to prompt")
                ui.separator().classes("bg-primary").style("height: 3px")

            with ui.column().classes("w-full") as self.prompt_games_column:
                pass # added prompt game elements will be created in this container
        
        if read_only == True:
            self.refresh_game_prompt_elements()

    def update_dataset_games_dict(self, dataset_games_dict: dict[str, str]) -> None:
        self.dataset_games_dict = dataset_games_dict
        if self.read_only == False:
            self.game_select.options = dataset_games_dict
            self.game_select.value = next(iter(dataset_games_dict))
            self.game_select.update()
        self.refresh_game_prompt_elements()

    def update_prompt(self, new_prompt: dict[str, float]) -> None:
        self.prompt = {**new_prompt}
        self.refresh_game_prompt_elements()

    def on_click_game_remove_button(self, game_name: str) -> None:
        self.prompt.pop(game_name)
        self.refresh_game_prompt_elements()
        self.on_change_prompt()
        
    def on_click_game_add_button(self) -> None:
        self.prompt.update({self.game_select.value: self.game_weight.value})
        self.refresh_game_prompt_elements()
        self.on_change_prompt()
        
    def refresh_game_prompt_elements(self) -> None:

        def on_game_select_change(new_game_name: str, old_game_name: str) -> None:
            # ugly hack to replace dict key while preserving order
            prompt_list = list(self.prompt.items())
            index = prompt_list.index((old_game_name, self.prompt[old_game_name]))
            prompt_list[index] = (new_game_name, self.prompt[old_game_name])
            self.prompt = dict(prompt_list)

            self.on_change_prompt()
            self.refresh_game_prompt_elements()
            
        self.prompt_games_column.clear()
        with self.prompt_games_column:
            for game_name, game_weight in self.prompt.items():
                
                if game_name not in self.dataset_games_dict:
                    if self.read_only == False:
                        ui.notify(f"Error '{game_name}' not found in dataset_games_dict", type="error", color="red", close_button=True)
                        continue
                    else:
                        self.dataset_games_dict[game_name] = game_name
                    
                with ui.row().classes("w-full h-10 flex items-center"):
                    game_select_element = ui.select(value=game_name, with_input=True, options=self.dataset_games_dict).classes("flex-grow-[1000]")                    
                    weight_element = ScrollableNumber(label="Weight", value=game_weight, min=-100, max=100, step=1, on_change=self.on_change_prompt).classes("flex-grow-[1]")
                    weight_element.bind_value(self.prompt, game_name)

                    if self.read_only == False:
                        game_select_element.on_value_change(
                            lambda event, game_name=game_name: on_game_select_change(new_game_name=event.value, old_game_name=game_name))
                        with ui.button(icon="remove", on_click=lambda g=game_name: self.on_click_game_remove_button(g)).classes("w-1 top-0 right-0").props("color='red'"):
                            ui.tooltip("Remove game from prompt")
                    else:
                        game_select_element.disable()
                        weight_element.disable()

            ui.separator().classes("bg-transparent")

class PresetEditor(ui.card):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.logger = logging.getLogger(name="nicegui_app")
        self.last_loaded_preset = "default"
        self.new_preset_name = ""
        self.loading_preset = False
        self.saved_preset_list = self.get_saved_presets()

        self.get_prompt_editor: Callable[[], PromptEditor] = lambda: None
        self.get_gen_params_editor: Callable[[], GenParamsEditor] = lambda: None

        with self.classes("w-full"):
            with ui.row().classes("w-full"):
                with ui.column().classes("flex-grow-[4]"):
                          
                    self.preset_select = EditableSelect(
                        label=f"Select a Preset - (loaded preset: {self.last_loaded_preset})",
                        options=self.saved_preset_list, value="default").classes("w-full")
                    
                    self.preset_select.on("input-value", lambda e: self.on_value_change_preset_select(e.args))
                    self.preset_select.on_value_change(lambda e: self.on_value_change_preset_select(e.value))     

                with ui.column().classes("gap-0 items-center"):
                    self.preset_load_button = ui.button("Load", icon="source", on_click=lambda: self.load_preset()).classes("w-36 rounded-b-none")
                    self.preset_save_button = ui.button("Save", icon="save", color="green", on_click=lambda: self.save_preset()).classes("w-36 rounded-none")
                    self.preset_delete_button = ui.button("Delete", icon="delete", color="red", on_click=lambda: self.delete_preset()).classes("w-36 rounded-t-none")
                    with self.preset_load_button:
                        ui.tooltip("Load selected preset")
                    with self.preset_save_button:
                        ui.tooltip("Save current parameters to selected preset")
                    with self.preset_delete_button:
                        ui.tooltip("Delete selected preset")

                self.preset_load_button.disable()
                self.preset_save_button.disable()
                self.preset_delete_button.disable()
    
    def on_preset_modified(self) -> None:
        if self.loading_preset == False:
            self.preset_select._props["label"] = f"Select a Preset - (loaded preset: {self.last_loaded_preset}*)"
            self.preset_select.update()
            self.preset_save_button.enable()
            if self.preset_select.new_value in self.saved_preset_list:
                self.preset_load_button.enable()
        
    def on_value_change_preset_select(self, preset_name: str) -> None:
        self.preset_save_button.enable()
        if preset_name in self.saved_preset_list:
            self.preset_load_button.enable()
            if preset_name != "default": self.preset_delete_button.enable()
            else: self.preset_delete_button.disable()
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
        self.logger.debug(dict_str({'Found saved presets': saved_presets}))
        self.saved_preset_list = saved_presets
        return saved_presets

    def save_preset(self) -> None:
        preset_name = self.preset_select.value
        preset_filename = f"{sanitize_filename(preset_name)}.json"
        save_preset_path = os.path.join(
            config.CONFIG_PATH, "sampling", "presets", preset_filename)
        preset_name = os.path.splitext(os.path.basename(save_preset_path))[0]
        self.logger.debug(f"Saving preset '{save_preset_path}'")

        save_preset_dict = {
            "prompt": self.get_prompt_editor().prompt,
            "gen_params": self.get_gen_params_editor().gen_params
        }
        config.save_json(save_preset_dict, save_preset_path)
        self.logger.info(f"Saved preset {preset_name}: {dict_str(save_preset_dict)}")

        self.preset_select.set_original_options(self.get_saved_presets())
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

        if self.get_gen_params_editor().lock_button.is_locked == False:
            gen_params = {k: v for k,v in loaded_preset_dict["gen_params"].items() if k in self.get_gen_params_editor().gen_params}
            self.get_gen_params_editor().gen_params.update(gen_params)
            self.preset_load_button.disable()
            self.preset_save_button.disable()
            self.preset_select._props["label"] = f"Select a Preset - (loaded preset: {self.last_loaded_preset})"
            self.preset_select.set_value(self.last_loaded_preset)
            self.loading_preset = True
            asyncio.create_task(self.reset_preset_loading_state())
        else:
            # check if loaded_preset_dict matches current gen_params, even if locked
            gen_params = {k: v for k,v in self.get_gen_params_editor().gen_params.items() if k in loaded_preset_dict["gen_params"]}
            if all([gen_params[p] == loaded_preset_dict["gen_params"][p] for p in gen_params]) == True:
                self.preset_load_button.disable()
                self.preset_save_button.disable()
                self.preset_select._props["label"] = f"Select a Preset - (loaded preset: {self.last_loaded_preset})"
                self.preset_select.set_value(self.last_loaded_preset)
                self.loading_preset = True
                asyncio.create_task(self.reset_preset_loading_state())
            else:
                self.preset_select._props["label"] = f"Select a Preset - (loaded preset: {self.last_loaded_preset}*)"

        self.preset_select.update()
        self.get_prompt_editor().update_prompt(loaded_preset_dict["prompt"])
    
    def delete_preset(self) -> None:
        preset_name = self.preset_select.value
        delete_preset_path = os.path.join(
            config.CONFIG_PATH, "sampling", "presets", f"{sanitize_filename(preset_name)}.json")
        self.logger.info(f"Deleting preset '{delete_preset_path}'")

        os.remove(delete_preset_path)
        self.preset_select.set_original_options(self.get_saved_presets())
        self.preset_select.set_value("default")
        self.preset_load_button.enable()
        self.preset_save_button.enable()
        self.preset_delete_button.disable()

class OutputEditor(ui.column):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.classes("w-full")

        self.logger = logging.getLogger(name="nicegui_app")
        self.output_samples: list[OutputSample] = []
        self.input_output_sample: OutputSample = None

        self.get_app_config = lambda: None
        self.get_prompt_editor: Callable[[], PromptEditor] = lambda: None
        self.get_gen_params_editor: Callable[[], GenParamsEditor] = lambda: None
        self.get_latent_shape: Callable[[int], tuple[int, int, int, int]] = lambda: None

        with ui.row().classes("w-full"):
            with ui.button_group().classes("h-10 gap-0"): # output editor top toolbar
                self.generate_button = ui.button("Generate", icon="audiotrack", color="green")
                with self.generate_button:
                    ui.tooltip("Generate new sample with current settings")
                self.clear_output_button = ui.button("Clear Outputs", icon="delete", color="red", on_click=lambda: self.clear_output_samples())
                self.clear_output_button.disable()
                with self.clear_output_button:
                    ui.tooltip("Clear all output samples in workspace")
                self.load_sample_button = ui.button("Upload Sample", icon="upload")
                with self.load_sample_button:
                    ui.tooltip("Upload an existing sample from audio or latents file")

            with ui.row().classes("absolute right-4 items-center gap-2"): # output editor right toolbar
                ui.icon("volume_off", size="1.75rem")
                self.audio_volume_slider = ui.slider(value=1, min=0, max=1, step=0.01, on_change=lambda v: self.set_audio_volume(v.value)).classes("w-24")
                ui.icon("volume_up", size="1.75rem")

        self.output_samples_column = ui.column().classes("w-full") # output samples container

    def set_audio_volume(self, volume: float) -> None:
        for output_sample in self.output_samples:
            output_sample.audio_editor_element.set_volume(volume)

    def update_model_info(self, model_name: str,
                                model_metadata: dict[str, Any],
                                format_config: dict[str, Any],
                                dataset_game_ids: dict[str, int]) -> None:
        
        self.model_name = model_name
        self.model_metadata = model_metadata
        self.format_config = format_config
        self.dataset_game_ids = dataset_game_ids

    def save_output_sample(self, sample_output: SampleOutput) -> tuple[str, Optional[str]]:
        metadata = {
            "diffusion_metadata": dict_str(sample_output.params.get_metadata()),
            "model_metadata": dict_str(self.model_metadata),
        }

        last_global_step = self.model_metadata["last_global_step"]["unet"]
        audio_format = self.get_app_config().output_audio_format.lower()
        audio_output_filename = f"{sample_output.params.get_label(self.model_metadata, self.dataset_game_ids, verbose=self.get_app_config().use_verbose_labels)}{audio_format}"
        audio_output_path = os.path.join(
            config.MODELS_PATH, self.model_name, "output", f"step_{last_global_step}", audio_output_filename)

        if audio_format == ".mp3":
            bit_rate = self.get_app_config().output_audio_bitrate or -1
            qscale = self.get_app_config().output_audio_qscale or 9
            compression_level = self.get_app_config().output_audio_compression_level or -1
            compression = torchaudio.io.CodecConfig(
                bit_rate=bit_rate, compression_level=compression_level, qscale=qscale)
        else:
            compression = None
            
        audio_output_path = save_audio(sample_output.raw_sample.squeeze(0),
            self.format_config["sample_rate"], audio_output_path, metadata=metadata, no_clobber=True, compression=compression)
        self.logger.info(f"Saved audio output to {audio_output_path}")

        if self.get_app_config().save_output_latents == True and sample_output.latents is not None:
            latents_output_path = os.path.join(os.path.dirname(audio_output_path), "latents",
                f"{os.path.splitext(os.path.basename(audio_output_path))[0]}.safetensors")
            save_safetensors({"latents": sample_output.latents}, latents_output_path, metadata=metadata)
            self.logger.info(f"Saved latents to {latents_output_path}")
        else:
            latents_output_path = None

        return audio_output_path, latents_output_path

    # updates various elements after changes to output samples state or order
    def refresh_output_samples(self) -> None:

        if len(self.output_samples) == 0:
            self.clear_output_button.disable()
        else:
            self.clear_output_button.enable()

        # if there is a selected input sample change the generate button color accordingly
        if self.input_output_sample is not None:
            self.generate_button.props(add="color='orange'", remove="color='green'")
        else:
            self.generate_button.props(add="color='green'", remove="color='orange'")

        for i, output_sample in enumerate(self.output_samples):

            # enable/disable positioning buttons based on output sample position
            if i == 0: output_sample.move_up_button.disable()
            else: output_sample.move_up_button.enable()
            if i == len(self.output_samples) - 1: output_sample.move_down_button.disable()
            else: output_sample.move_down_button.enable()

            # enable/disable the highlighted border and select-range overlay if the output sample is selected as input
            if self.input_output_sample == output_sample:
                output_sample.card_element.style(replace="outline: 2px solid #ff9800;")
            else:
                output_sample.card_element.style(replace="outline: 2px solid #505050;")

            if output_sample.editing_mode == "inpaint":
                output_sample.inpaint_button.classes(remove="border-none", add="border-4")
                output_sample.select_range.set_visibility(True)
                output_sample.audio_editor_element.set_select_range_visibility(True)
            else:
                output_sample.inpaint_button.classes(remove="border-4", add="border-none")
                output_sample.select_range.set_visibility(False)
                output_sample.audio_editor_element.set_select_range_visibility(False)

            if output_sample.editing_mode == "extend":
                output_sample.extend_button.classes(remove="border-none", add="border-4")
                output_sample.extend_mode_radio.set_visibility(True)
            else:
                output_sample.extend_button.classes(remove="border-4", add="border-none")
                output_sample.extend_mode_radio.set_visibility(False)

            output_sample.audio_editor_element.update()
                
    # adds a new output sample to the top of the workspace, queued initially
    async def add_output_sample(self, length: int, seed: int, seamless_loop: bool,
            prompt: dict[str, float], gen_params: dict[str, Any]) -> OutputSample:
        
        # setup SampleParams input for actual pipeline generation call
        sample_params = SampleParams(seed=seed, length=length * self.format_config["sample_rate"],
            seamless_loop=seamless_loop, prompt=prompt.copy(), **gen_params)
        self.logger.info(f"OutputEditor.add_output_sample - params:{dict_str(sample_params.__dict__)}")

        # setup inpainting input
        if self.input_output_sample is not None:
            sample_params.input_audio = self.input_output_sample.sample_output.latents
            sample_params.input_audio_pre_encoded = True
            if self.input_output_sample.editing_mode == "inpaint":
                sample_params.length = self.input_output_sample.sample_params.length
                sample_params.inpainting_mask = torch.zeros_like(sample_params.input_audio[:, 0:1])
                sample_params.inpainting_mask[...,
                    self.input_output_sample.select_range.value["min"]:self.input_output_sample.select_range.value["max"]] = 1.
            elif self.input_output_sample.editing_mode == "extend":
                latent_shape = await self.get_latent_shape(sample_params.length)
                sample_params.inpainting_mask = torch.zeros(size=(1, 1, *latent_shape[2:]))

                input_latents_length = self.input_output_sample.sample_output.latents.shape[-1]
                if latent_shape[-1] <= input_latents_length:
                    ui.notify("Cannot extend, output length <= input length", type="error", color="red", close_button=True)
                    raise ValueError("Cannot extend, output length <= input length")
                
                if self.input_output_sample.extend_mode_radio.value == "Prepend":
                    sample_params.input_audio = torch.nn.functional.pad(sample_params.input_audio, (latent_shape[-1] - input_latents_length, 0))
                    sample_params.inpainting_mask[..., :latent_shape[-1] - input_latents_length] = 1
                else:
                    sample_params.input_audio = torch.nn.functional.pad(sample_params.input_audio, (0, latent_shape[-1] - input_latents_length))
                    sample_params.inpainting_mask[..., input_latents_length:] = 1
        
        # get name / label and add output sample to workspace
        output_sample = OutputSample(name=f"{sample_params.get_label(self.model_metadata, self.dataset_game_ids, verbose=self.get_app_config().use_verbose_labels)}",
            seed=sample_params.seed, prompt=sample_params.prompt, gen_params=gen_params, sample_params=sample_params)

        # if inpainting/outpainting set the highlight range to show the generated region
        if self.input_output_sample is not None:
            if self.input_output_sample.editing_mode == "inpaint":
                output_sample.highlight_start = self.input_output_sample.select_range.value["min"]
                output_sample.highlight_duration = self.input_output_sample.select_range.value["max"] - self.input_output_sample.select_range.value["min"]
            else:
                if self.input_output_sample.extend_mode_radio.value == "Prepend":
                    output_sample.highlight_start = 0
                    output_sample.highlight_duration = latent_shape[-1] - input_latents_length
                else:
                    output_sample.highlight_start = input_latents_length
                    output_sample.highlight_duration = latent_shape[-1] - input_latents_length

        # moves output sample up or down in the workspace
        def move_output_sample(output_sample: OutputSample, direction: int) -> None:
            current_index = output_sample.card_element.parent_slot.children.index(output_sample.card_element)
            new_index = min(max(current_index + direction, 0), len(output_sample.card_element.parent_slot.children) - 1)
            if new_index != current_index:
                output_sample.card_element.move(self.output_samples_column, target_index=new_index)
                self.output_samples.insert(new_index, self.output_samples.pop(current_index))
                self.refresh_output_samples()

        # update the select-range overlay in the AudioEditor element
        def set_select_range(output_sample: OutputSample) -> None:
            seconds_per_latent_pixel = output_sample.audio_editor_element._props["duration"] / output_sample.sample_output.latents.shape[-1]
            duration = (output_sample.select_range.value["max"] - output_sample.select_range.value["min"]) * seconds_per_latent_pixel
            output_sample.audio_editor_element.set_select_range(
                output_sample.select_range.value["min"] * seconds_per_latent_pixel, duration)
            
        # selects the chosen output sample as current input sample
        def use_output_sample_as_input(output_sample: OutputSample, mode: str) -> None:
            if self.input_output_sample == output_sample and self.input_output_sample.editing_mode == mode:
                self.input_output_sample.editing_mode = "normal"
                self.input_output_sample = None
            else:
                if self.input_output_sample is not None:
                    self.input_output_sample.editing_mode = "normal"
                self.input_output_sample = output_sample
                self.input_output_sample.editing_mode = mode
            self.refresh_output_samples()

        def on_click_inpaint_button(output_sample: OutputSample) -> None:
            use_output_sample_as_input(output_sample, "inpaint")
            set_select_range(output_sample) # required to refresh select range

        def on_click_extend_button(output_sample: OutputSample) -> None:
            use_output_sample_as_input(output_sample, "extend")
        
        # pauses all other output samples when a new one is played
        def on_play_audio(output_sample: OutputSample) -> None:
            for sample in self.output_samples:
                if sample != output_sample: sample.audio_editor_element.pause()

        # pause all other output samples when right-clicking on any of them
        def on_audio_editor_mouse(output_sample: OutputSample, e: nicegui.events.MouseEventArguments) -> None:
            if e.type == "mousedown":
                if e.button != 0:
                    for sample in self.output_samples:
                        if sample != output_sample: sample.audio_editor_element.pause()

        # updates rating metadata in output sample audio and latents files
        def change_output_rating(output_sample: OutputSample, rating: int) -> None:
            try: update_audio_metadata(output_sample.audio_path, rating=rating)
            except Exception as e:
                self.logger.error(f"Error updating audio metadata in {output_sample.audio_path}: {e}")
            try: update_safetensors_metadata(output_sample.latents_path, {"rating": str(rating)})
            except Exception as e:
                self.logger.error(f"Error updating latents metadata in {output_sample.latents_path}: {e}")

        # toggle visibility of optional elements in output sample card
        def on_toggle_show_latents(output_sample: OutputSample, is_toggled: bool) -> None:
            output_sample.latents_image_element.set_visibility(is_toggled)
        def on_toggle_show_params(output_sample: OutputSample, is_toggled: bool) -> None:
            output_sample.show_parameters_row_element.set_visibility(is_toggled)
        def on_toggle_show_debug(output_sample: OutputSample, is_toggled: bool) -> None:
            output_sample.show_debug_plots_row_element.set_visibility(is_toggled)

        # param copy buttons
        def on_click_copy_gen_params_button(output_sample: OutputSample) -> None:
            self.get_gen_params_editor().gen_params.update(output_sample.gen_params)
            self.get_gen_params_editor().seed.set_value(output_sample.seed)
            self.get_gen_params_editor().generate_length.set_value(output_sample.sample_params.length)
            self.get_gen_params_editor().seamless_loop.set_value(output_sample.sample_params.seamless_loop)
            ui.notification("Copied all parameters!", timeout=1, icon="content_copy")
        def on_click_copy_prompt_button(output_sample: OutputSample) -> None:
            self.get_prompt_editor().update_prompt(output_sample.prompt)
            ui.notification("Copied prompt!", timeout=1, icon="content_copy")
        def on_click_copy_all_button(output_sample: OutputSample) -> None:
            self.get_gen_params_editor().gen_params.update(output_sample.gen_params)
            self.get_gen_params_editor().seed.set_value(output_sample.seed)
            self.get_gen_params_editor().generate_length.set_value(output_sample.sample_params.length)
            self.get_gen_params_editor().seamless_loop.set_value(output_sample.sample_params.seamless_loop)
            self.get_prompt_editor().update_prompt(output_sample.prompt)
            ui.notification("Copied all parameters and prompt!", timeout=1, icon="content_copy")
        
        with ui.card().classes("w-full p-0") as output_sample.card_element:
            with ui.column().classes("w-full gap-0 m-0 p-0"):

                # setup output sample name label and star rating
                with ui.row().classes("h-min items-center no-wrap gap-0 absolute left-0 top-0") as output_sample.name_row_element:
                    output_sample.name_label_element = ui.label(output_sample.name).classes("z-10 ml-1").classes("shadow-lg")
                    with StarRating().classes("z-10 ml-3") as output_sample.rating_element:
                        ui.tooltip("Rate this sample")
                    output_sample.rating_element.on_rating_change = lambda rating: change_output_rating(output_sample, rating)
                    output_sample.rating_element.set_visibility(False)
                
                with ui.row().classes("w-full gap-0, m-0 p-0 no-wrap"):
                    with ui.column().classes("flex-grow gap-0 m-0 p-0"):
                        # setup generation progress and latent image elements
                        output_sample.sampling_progress_element = ui.linear_progress(
                            value=0., show_value=False).classes("w-full font-bold gap-0 h-6").props("instant-feedback")
                        with output_sample.sampling_progress_element:
                            progress_label = ui.label().classes("absolute-center text-sm text-white")
                            output_sample.sampling_progress_element.on_value_change(lambda v: progress_label.set_text(f"{v.value:.1%}"))
                        
                        output_sample.latents_image_element = ui.interactive_image().classes(
                            "w-full gap-0 m-0").style("image-rendering: pixelated; width: 100%; height: auto;").props("fit=scale-down")
                        
                        # setup audio editor element, initially hidden while waiting for generation
                        output_sample.audio_editor_element = AudioEditor().classes("w-full gap-0 m-0").props(add="fit=fill")
                        output_sample.audio_editor_element.audio_element.on("play", lambda: on_play_audio(output_sample))
                        output_sample.audio_editor_element.on_mouse(lambda e: on_audio_editor_mouse(output_sample, e))
                        output_sample.audio_editor_element.set_visibility(False)
                            
                        # setup inpainting range select element, extend/prepend radio group
                        output_sample.select_range = ui.range(min=0, max=0, step=1, value={"min": 0, "max": 0}).classes("w-full m-0").props("step snap color='orange' label='Inpaint Selection'")
                        output_sample.select_range.on_value_change(lambda: set_select_range(output_sample))
                        output_sample.select_range.set_visibility(False)
                        
                        output_sample.extend_mode_radio = ui.radio(["Prepend", "Extend"], value="Extend").props("color='orange' inline")
                        output_sample.extend_mode_radio.set_value("Extend")
                        output_sample.extend_mode_radio.set_visibility(False)

                    # setup output sample toolbar
                    with ui.column().classes("gap-0 p-0 m-0 z-10 w-min box-border").style("margin-left: -16px;"):
                        
                        button_classes = "w-8 gap-0 m-0 p-0"
                        with ui.button('✕', color="red", on_click=lambda s=output_sample: self.remove_output_sample(s)).classes(f"{button_classes} rounded-b-none rounded-l-none"):
                            ui.tooltip("Remove sample from workspace")
                        output_sample.move_up_button = ui.button('▲', color="gray",
                            on_click=lambda: move_output_sample(output_sample, direction=-1)).classes(f"{button_classes} rounded-none")
                        with output_sample.move_up_button:
                            ui.tooltip("Move sample up")
                        output_sample.move_down_button = ui.button('▼', color="gray",
                            on_click=lambda: move_output_sample(output_sample, direction=1)).classes(f"{button_classes} rounded-none")
                        with output_sample.move_down_button:
                            ui.tooltip("Move sample down")

                        with ui.button(icon="content_copy",
                            on_click=lambda: on_click_copy_all_button(output_sample)).classes(f"{button_classes} rounded-none"):
                            ui.tooltip("Copy all parameters to current settings")

                        output_sample.inpaint_button = ui.button(icon="format_color_fill", color="orange",
                            on_click=lambda: on_click_inpaint_button(output_sample)).classes(f"{button_classes} border-none border-double rounded-none")
                        with output_sample.inpaint_button:
                            ui.tooltip("Use this sample as inpainting input")
                        output_sample.inpaint_button.disable()
                        output_sample.extend_button = ui.button(icon="swap_horiz", color="orange",
                            on_click=lambda: on_click_extend_button(output_sample)).classes(f"{button_classes} border-none border-double rounded-none")
                        with output_sample.extend_button:
                            ui.tooltip("Change sample length")
                        output_sample.extend_button.disable()

                        with ToggleButton(icon="tune", color="gray").classes(f"{button_classes} rounded-none") as output_sample.toggle_show_params_button:
                            ui.tooltip("Toggle parameters visibility")
                        output_sample.toggle_show_params_button.on_toggle = lambda is_toggled: on_toggle_show_params(output_sample, is_toggled)
                        with ToggleButton(icon="query_stats", color="gray").classes(f"{button_classes} rounded-none") as output_sample.toggle_show_debug_button:
                            ui.tooltip("Toggle debug plot visibility")
                        output_sample.toggle_show_debug_button.on_toggle = lambda is_toggled: on_toggle_show_debug(output_sample, is_toggled)
                        output_sample.toggle_show_debug_button.disable()
                        with ToggleButton(icon="gradient", color="gray").classes(f"{button_classes} rounded-t-none rounded-l-none") as output_sample.toggle_show_latents_button:
                            ui.tooltip("Toggle latents visibility")
                        output_sample.toggle_show_latents_button.on_toggle = lambda is_toggled: on_toggle_show_latents(output_sample, is_toggled)
                        output_sample.toggle_show_latents_button.toggle(is_toggled=False)

                # re-use gen param and prompt editors in read-only mode to show sample params
                with ui.row().classes("w-full") as output_sample.show_parameters_row_element:
                    ui.separator().classes("bg-transparent")
                    with ui.card():
                        with GenParamsEditor(gen_params=gen_params, seed=seed, seamless_loop=seamless_loop, length=length, read_only=True):
                            with CopyButton().on_click(lambda: on_click_copy_gen_params_button(output_sample)):
                                ui.tooltip("Copy to current parameters")
                    with PromptEditor(prompt=prompt, read_only=True):
                        with CopyButton().on_click(lambda: on_click_copy_prompt_button(output_sample)):
                            ui.tooltip("Copy to current prompt")
                output_sample.toggle_show_params_button.toggle(is_toggled=False)

                # setup debug info and plots display
                with ui.row().classes("w-full") as output_sample.show_debug_plots_row_element:
                    ui.separator().classes("bg-transparent")
                    # debug plots and info elements will be added here
                output_sample.toggle_show_debug_button.toggle(is_toggled=False)
                output_sample.toggle_show_debug_button.disable()

        # add to output samples list and insert root card element at top of workspace
        self.output_samples.insert(0, output_sample)
        output_sample.card_element.move(self.output_samples_column, target_index=0)
        self.refresh_output_samples()

        return output_sample

    # called when generation / pipeline call is completed for this output sample
    def on_output_sample_generated(self, output_sample: OutputSample) -> None:

        # save output sample audio / latents, hide generation progress element
        output_sample.audio_path, output_sample.latents_path = self.save_output_sample(output_sample.sample_output)
        output_sample.sampling_progress_element.set_visibility(False)

        # set output sample name label to match audio filename and enable rating
        output_sample.name = os.path.splitext(os.path.basename(output_sample.audio_path))[0]
        output_sample.name_label_element.set_text(output_sample.name)
        output_sample.rating_element.set_visibility(True)

        # setup audio editor element
        spectrogram_image = output_sample.sample_output.spectrogram.mean(dim=(0,1))
        spectrogram_image = tensor_to_img(spectrogram_image, colormap=True, flip_y=True)
        spectrogram_image = cv2.resize(
            spectrogram_image, (int(spectrogram_image.shape[1]//4/0.9), spectrogram_image.shape[0]), interpolation=cv2.INTER_AREA)
        spectrogram_image = Image.fromarray(spectrogram_image)
        
        output_sample.audio_editor_element.set_source(spectrogram_image)
        if self.get_app_config().hide_latents_after_generation == True:
            output_sample.toggle_show_latents_button.toggle(is_toggled=False)
        audio_duration = output_sample.sample_output.raw_sample.shape[-1] / self.format_config["sample_rate"]
        output_sample.audio_editor_element.set_audio_source(output_sample.audio_path, duration=audio_duration)
        output_sample.audio_editor_element.set_visibility(True)
        output_sample.audio_editor_element.set_volume(self.audio_volume_slider.value)
        output_sample.audio_editor_element.set_looping(output_sample.sample_params.seamless_loop)

        # setup inpainting range select element and enable use as input button        
        output_sample.select_range.max = output_sample.sample_output.latents.shape[-1]
        output_sample.select_range.set_value({
            "min": output_sample.select_range.max//2 - output_sample.select_range.max//4,
            "max": output_sample.select_range.max//2 + output_sample.select_range.max//4})
        output_sample.inpaint_button.enable()
        output_sample.extend_button.enable()
        
        # set highlight range to show the generated region if inpainting/outpainting
        seconds_per_latent_pixel = audio_duration / output_sample.sample_output.latents.shape[-1]
        if output_sample.highlight_duration > 0:
            output_sample.audio_editor_element.set_highlight_range(
                output_sample.highlight_start * seconds_per_latent_pixel,
                output_sample.highlight_duration * seconds_per_latent_pixel)
            output_sample.audio_editor_element.set_highlight_range_visibility(True)
        
        # debug info and plots display
        output_sample.toggle_show_debug_button.enable()
        with output_sample.show_debug_plots_row_element:

            # show scalar debug values first
            debug_val_columns = [
                {"name": "name", "label": "Name", "field": "name", "required": True, "align": "left"},
                {"name": "value", "label": "Value", "field": "value"},
            ]
            debug_val_rows = []
            for name, value in output_sample.sample_output.debug_info.items():
                if isinstance(value, list) or isinstance(value, torch.Tensor):
                    pass
                elif isinstance(value, (int, float, bool, str, tuple)):
                    debug_val_rows.append({"name": name, "value": str(value)})
                else:
                    self.logger.error(f"Unsupported type in sample_output.debug_info: {name}={value}")
            ui.table(rows=debug_val_rows, columns=debug_val_columns, row_key="name", title="Debug Info")

            # show debug lists as plot images
            for name, value in output_sample.sample_output.debug_info.items():
                if isinstance(value, list):
                    x = np.arange(len(value))
                    y = np.array(value)
                    with ui.matplotlib(figsize=(4.75, 4.05)).classes("border-gray-600 border-2").figure as fig:
                        #fig.patch.set_facecolor("#1d1d1d") # todo: doesn't do anything :(
                        ax = fig.gca()
                        ax.grid(color="#666666", linewidth=0.5, linestyle="--")
                        ax.set_facecolor("#1d1d1d")
                        ax.set_title(name)
                        ax.plot(x, y, "-", color="#5898d4")
                        ax.set_xlabel("step")
                        # todo: add units metadata to debug_info
                        if "curvature" in name:
                            ax.set_ylabel("radians")

                elif isinstance(value, torch.Tensor):
                    
                    # if tensor is 2d show as image
                    if len(value.shape) == 4:
                        value:torch.Tensor = value.mean(dim=0)
                        for i, tensor in enumerate(value.unbind(0)):
                            with ui.matplotlib(figsize=(tensor.shape[1]*0.03, tensor.shape[0]*0.03)).classes("border-gray-600 border-2").figure as fig:
                                ax = fig.gca()
                                min_val = tensor.amin().item()
                                max_val = tensor.amax().item()
                                tensor_img = tensor_to_img(tensor, colormap=True, flip_y=True)
                                ax.imshow(tensor_img, aspect="auto")
                                ax.set_title(f"{name}_c{i} min: {min_val:.2f} max: {max_val:.2f} mean: {tensor.mean().item():.2f} std: {tensor.std().item():.2f}")
                                fig.subplots_adjust(left=12/tensor.shape[1], right=1 - 12/tensor.shape[1], top=1 - 10/tensor.shape[0], bottom=10/tensor.shape[0])
                                fig.patch.set_facecolor("#1d1d1d")

    # removes all output samples in workspace (except the selected input sample)
    def clear_output_samples(self) -> None:
        for output_sample in [*self.output_samples]:
            if self.input_output_sample != output_sample:
                self.output_samples_column.remove(output_sample.card_element)
                self.output_samples.remove(output_sample)
        self.refresh_output_samples()

    # remove chosen output sample (unless it is the selected input sample)
    def remove_output_sample(self, output_sample: OutputSample) -> None:
        if self.input_output_sample == output_sample: # prevent removing input sample while selected
            ui.notify("Cannot remove input sample", type="error", color="red", close_button=True)
        else:
            self.output_samples_column.remove(output_sample.card_element)
            self.output_samples.remove(output_sample)
            self.refresh_output_samples()
