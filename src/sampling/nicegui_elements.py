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

from typing import Optional, Any
from dataclasses import dataclass
from copy import deepcopy
from PIL import Image
import os
import asyncio
import logging

import torch
import cv2
from nicegui import ui

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

    card_element: Optional[ui.card] = None
    name_label_element: Optional[ui.label] = None
    rating_element: Optional["StarRating"] = None
    sampling_progress_element: Optional[ui.linear_progress] = None
    latents_image_element: Optional[ui.interactive_image] = None
    spectrogram_image_element: Optional[ui.interactive_image] = None
    audio_element: Optional[ui.audio] = None
    toggle_show_latents_button: Optional["ToggleButton"] = None
    toggle_show_spectrogram_button: Optional["ToggleButton"] = None
    toggle_show_params_button: Optional["ToggleButton"] = None
    toggle_show_debug_button: Optional["ToggleButton"] = None
    use_as_input_button: Optional[ui.button] = None
    select_range: Optional[ui.range] = None
    move_up_button: Optional[ui.button] = None
    move_down_button: Optional[ui.button] = None

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

    def on_blur_self(self) -> None:
        if self.new_value != "":
            if self.new_value not in self.original_options:
                self.set_options(options=self.original_options + [self.new_value], value=self.new_value)
            else:
                self.set_value(self.new_value)

class LockButton(ui.button):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(icon="lock_open", *args, **kwargs)     
        self.style("position: absolute; top: 5px; right: 5px; z-index: 2; background-color: transparent;"
                   "border: none; width: 20px; height: 20px; padding: 0;").classes("bg-transparent")
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

class ScrollableNumber(ui.number): # same as ui.number but works with mouse wheel

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # curiously, this enables values scrolling with the mouse wheel
        self.on("wheel", lambda: None)

class PromptEditor(ui.card):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.prompt = {}
        self.dataset_games_dict = {}
        self.on_prompt_change = lambda: None

        with self.classes("flex-grow-[50]"):
            #with ui.row().classes("w-full h-0 gap-0"):
            #    with LockButton() as self.prompt_lock_button:
            #        ui.tooltip("Lock to freeze prompt when loading presets")
            with ui.row().classes("w-full h-12 flex items-center"):

                self.game_select = ui.select(label="Select a game", with_input=True, options={}).classes("flex-grow-[1000]")
                self.game_weight = ScrollableNumber(label="Weight", value=10, min=-100, max=100, step=1).classes("flex-grow-[1]")
                self.game_add_button = ui.button(icon="add", color="green", on_click=lambda: self.on_click_game_add_button()).classes("w-1")
                with self.game_add_button:
                    ui.tooltip("Add selected game to prompt")

            ui.separator().classes("bg-primary").style("height: 3px")
            with ui.column().classes("w-full") as self.prompt_games_column:
                pass # added prompt game elements will be created in this container
    
    def update_dataset_games_dict(self, dataset_games_dict: dict[str, str]) -> None:
        self.dataset_games_dict = dataset_games_dict
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
        self.on_prompt_change()
        
    def on_click_game_add_button(self) -> None:
        self.prompt.update({self.game_select.value: self.game_weight.value})
        self.refresh_game_prompt_elements()
        self.on_prompt_change()
        
    def refresh_game_prompt_elements(self) -> None:

        def on_game_select_change(new_game_name: str, old_game_name: str) -> None:
            # ugly hack to replace dict key while preserving order
            prompt_list = list(self.prompt.items())
            index = prompt_list.index((old_game_name, self.prompt[old_game_name]))
            prompt_list[index] = (new_game_name, self.prompt[old_game_name])
            self.prompt = dict(prompt_list)

            self.on_prompt_change()
            self.refresh_game_prompt_elements()
            
        self.prompt_games_column.clear()
        with self.prompt_games_column:
            for game_name, game_weight in self.prompt.items():

                if game_name not in self.dataset_games_dict:
                    ui.notify(f"Error '{game_name}' not found in dataset_games_dict", type="error", color="red", close_button=True)
                    continue

                with ui.row().classes("w-full h-10 flex items-center"):
                    game_select_element = ui.select(value=game_name, with_input=True, options=self.dataset_games_dict).classes("flex-grow-[1000]")
                    weight_element = ScrollableNumber(label="Weight", value=game_weight, min=-100, max=100, step=1, on_change=self.on_prompt_change).classes("flex-grow-[1]")
                    weight_element.bind_value(self.prompt, game_name)
                    game_select_element.on_value_change(
                        lambda event, game_name=game_name: on_game_select_change(new_game_name=event.value, old_game_name=game_name))
                    with ui.button(icon="remove", on_click=lambda g=game_name: self.on_click_game_remove_button(g)).classes("w-1 top-0 right-0").props("color='red'"):
                        ui.tooltip("Remove game from prompt")

            ui.separator().classes("bg-transparent")

class PresetEditor(ui.card):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.logger = logging.getLogger(name="nicegui_app")
        self.last_loaded_preset = "default"
        self.new_preset_name = ""
        self.loading_preset = False
        self.saved_preset_list = self.get_saved_presets()

        self.get_prompt_editor = lambda: None
        self.get_gen_params = lambda: None
        self.get_gen_params_locked = lambda: None

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

        save_preset_dict = {"prompt": self.get_prompt_editor().prompt, "gen_params": self.get_gen_params()}
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

        if not self.get_gen_params_locked():
            self.get_gen_params().update(loaded_preset_dict["gen_params"])
            self.preset_load_button.disable()
            self.preset_save_button.disable()
            self.preset_select._props["label"] = f"Select a Preset - (loaded preset: {self.last_loaded_preset})"
            self.loading_preset = True
            asyncio.create_task(self.reset_preset_loading_state())
        else:
            # check if loaded_preset_dict matches current gen_params, even if locked
            gen_params = self.get_gen_params()
            if all([gen_params[p] == loaded_preset_dict["gen_params"][p] for p in gen_params]) == True:
                self.preset_load_button.disable()
                self.preset_save_button.disable()
                self.preset_select._props["label"] = f"Select a Preset - (loaded preset: {self.last_loaded_preset})"
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

        with ui.button_group().classes("h-10 gap-0"): # output samples toolbar
            self.clear_output_button = ui.button("Clear Outputs", icon="delete", color="red", on_click=lambda: self.clear_output_samples())
            self.clear_output_button.disable()
            with self.clear_output_button:
                ui.tooltip("Clear all output samples in workspace")
            self.load_sample_button = ui.button("Upload Sample", icon="upload")
            with self.load_sample_button:
                ui.tooltip("Upload an existing sample from audio or latents file")

        self.output_samples_column = ui.column().classes("w-full") # output samples container

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
        audio_output_filename = f"{sample_output.params.get_label(self.model_metadata, self.dataset_game_ids)}.flac"
        audio_output_path = os.path.join(
            config.MODELS_PATH, self.model_name, "output", f"step_{last_global_step}", audio_output_filename)
        
        audio_output_path = save_audio(sample_output.raw_sample.squeeze(0),
            self.format_config["sample_rate"], audio_output_path, metadata=metadata, no_clobber=True)
        self.logger.info(f"Saved audio output to {audio_output_path}")

        if self.get_app_config().save_output_latents == True and sample_output.latents is not None:
            latents_output_path = os.path.join(os.path.dirname(audio_output_path), "latents",
                f"{os.path.splitext(os.path.basename(audio_output_path))[0]}.safetensors")
            save_safetensors({"latents": sample_output.latents}, latents_output_path, metadata=metadata)
            self.logger.info(f"Saved latents to {latents_output_path}")
        else:
            latents_output_path = None

        return audio_output_path, latents_output_path

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
    
    def add_output_sample(self, length: int, seed: int,
            prompt: dict[str, float], gen_params: dict[str, Any]) -> OutputSample:
        
        # setup sample input params
        sample_params = SampleParams(seed=seed, length=length * self.format_config["sample_rate"],
            prompt=prompt.copy(), **gen_params)
        self.logger.info(f"OutputEditor.add_output_sample - params:{dict_str(sample_params.__dict__)}")

        if self.input_output_sample is not None: # setup inpainting input
            sample_params.input_audio = self.input_output_sample.sample_output.latents
            sample_params.input_audio_pre_encoded = True
            sample_params.inpainting_mask = torch.zeros_like(sample_params.input_audio[:, 0:1])
            sample_params.inpainting_mask[..., self.input_output_sample.select_range.value["min"]:self.input_output_sample.select_range.value["max"]] = 1.
        
        # get name / label and add output sample to workspace
        output_sample = OutputSample(name=f"{sample_params.get_label(self.model_metadata, self.dataset_game_ids)}",
            seed=sample_params.seed, prompt=sample_params.prompt, gen_params=gen_params, sample_params=sample_params)

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

        def change_output_rating(output_sample: OutputSample, rating: int) -> None:
            try: update_audio_metadata(output_sample.audio_path, rating=rating)
            except Exception as e:
                self.logger.error(f"Error updating audio metadata in {output_sample.audio_path}: {e}")
            try: update_safetensors_metadata(output_sample.latents_path, {"rating": str(rating)})
            except Exception as e:
                self.logger.error(f"Error updating latents metadata in {output_sample.latents_path}: {e}")

        def on_toggle_show_latents(output_sample: OutputSample, is_toggled: bool) -> None:
            output_sample.latents_image_element.set_visibility(is_toggled)
        def on_toggle_show_spectrogram(output_sample: OutputSample, is_toggled: bool) -> None:
            output_sample.spectrogram_image_element.set_visibility(is_toggled)
        def on_toggle_show_params(output_sample: OutputSample, is_toggled: bool) -> None:
            pass
        def on_toggle_show_debug(output_sample: OutputSample, is_toggled: bool) -> None:
            pass

        with ui.card().classes("w-full") as output_sample.card_element:
            with ui.column().classes("w-full gap-0"):
                with ui.row().classes("h-10 justify-between gap-0 w-full"):

                    with ui.row().classes("items-center"): # output sample name label
                        output_sample.name_label_element = ui.label(output_sample.name).classes("p-2").style(
                            "border: 1px solid grey; border-bottom: none; border-radius: 10px 10px 0 0;")
                        with StarRating() as output_sample.rating_element: # and star rating
                            ui.tooltip("Rate this sample")
                        output_sample.rating_element.disable()
                        output_sample.rating_element.classes("p-2").style(
                            "border: 1px solid grey; border-bottom: none; border-radius: 10px 10px 0 0;")
                        output_sample.rating_element.on_rating_change = lambda rating: change_output_rating(output_sample, rating)

                    with ui.button_group().classes("h-10 gap-0"): # output sample icon toolbar
                        with ToggleButton(icon="gradient", color="gray").classes("w-1") as output_sample.toggle_show_latents_button:
                            ui.tooltip("Toggle latents visibility")
                        output_sample.toggle_show_latents_button.on_toggle = lambda is_toggled: on_toggle_show_latents(output_sample, is_toggled)
                        with ToggleButton(icon="queue_music", color="gray").classes("w-1") as output_sample.toggle_show_spectrogram_button:
                            ui.tooltip("Toggle spectrogram visibility")
                        output_sample.toggle_show_spectrogram_button.on_toggle = lambda is_toggled: on_toggle_show_spectrogram(output_sample, is_toggled)
                        with ToggleButton(icon="tune", color="gray").classes("w-1") as output_sample.toggle_show_params_button:
                            ui.tooltip("Toggle parameters visibility")
                        output_sample.toggle_show_params_button.on_toggle = lambda is_toggled: on_toggle_show_params(output_sample, is_toggled)
                        with ToggleButton(icon="query_stats", color="gray").classes("w-1") as output_sample.toggle_show_debug_button:
                            ui.tooltip("Toggle debug plot visibility")
                        output_sample.toggle_show_debug_button.on_toggle = lambda is_toggled: on_toggle_show_debug(output_sample, is_toggled)

                        output_sample.toggle_show_spectrogram_button.disable()
                        output_sample.toggle_show_debug_button.disable()
                        
                        output_sample.use_as_input_button = ui.button(
                            icon="format_color_fill", color="orange", on_click=lambda: use_output_sample_as_input(output_sample)).classes("w-1 border-none border-double")
                        with output_sample.use_as_input_button:
                            ui.tooltip("Use this sample as inpainting input")
                        output_sample.use_as_input_button.disable()
                        output_sample.move_up_button = ui.button('▲', on_click=lambda: move_output_sample(output_sample, direction=-1)).classes("w-1")
                        with output_sample.move_up_button:
                            ui.tooltip("Move sample up")
                        output_sample.move_down_button = ui.button('▼', on_click=lambda: move_output_sample(output_sample, direction=1)).classes("w-1")
                        with output_sample.move_down_button:
                            ui.tooltip("Move sample down")
                        with ui.button('✕', color="red", on_click=lambda s=output_sample: self.remove_output_sample(s)).classes("w-1"):
                            ui.tooltip("Remove sample from workspace")

                output_sample.latents_image_element = ui.interactive_image().classes(
                    "w-full gap-0").style("image-rendering: pixelated; width: 100%; height: auto;").props("fit=scale-down")
                output_sample.sampling_progress_element = ui.linear_progress(
                    value="0%").classes("w-full font-bold gap-0").props("instant-feedback")
                output_sample.spectrogram_image_element = ui.interactive_image(
                    cross="white").classes("w-full gap-0").props(add="fit=fill")
                
                output_sample.toggle_show_latents_button.toggle(is_toggled=False)
                output_sample.toggle_show_spectrogram_button.toggle(is_toggled=False)

                output_sample.select_range = ui.range(min=0, max=688, step=1, value={"min": 0, "max": 0}).classes("w-full").props("step snap color='orange' label='Inpaint Selection'")
                output_sample.select_range.set_visibility(False)

                output_sample.audio_element = ui.audio("").classes("w-full").props("preload='auto'").style("filter: invert(1) hue-rotate(180deg);")
                output_sample.audio_element.set_visibility(False)

        self.output_samples.insert(0, output_sample)
        output_sample.card_element.move(self.output_samples_column, target_index=0)
        self.refresh_output_samples()

        return output_sample

    def on_output_sample_generated(self, output_sample: OutputSample) -> None:

        # save output sample audio / latents
        output_sample.audio_path, output_sample.latents_path = self.save_output_sample(output_sample.sample_output)

        # set output sample name label to match audio filename and enable rating
        output_sample.name = os.path.splitext(os.path.basename(output_sample.audio_path))[0]
        output_sample.name_label_element.set_text(output_sample.name)
        output_sample.rating_element.enable()

        # set spectrogram image
        spectrogram_image = output_sample.sample_output.spectrogram.mean(dim=(0,1))
        spectrogram_image = tensor_to_img(spectrogram_image, colormap=True, flip_y=True)
        spectrogram_image = cv2.resize(
            spectrogram_image, (spectrogram_image.shape[1]//4, spectrogram_image.shape[0]), interpolation=cv2.INTER_AREA)
        spectrogram_image = Image.fromarray(spectrogram_image)
        output_sample.spectrogram_image_element.set_source(spectrogram_image)
        if self.get_app_config().hide_latents_after_generation == True:
            output_sample.toggle_show_latents_button.toggle(is_toggled=False)

        output_sample.toggle_show_spectrogram_button.toggle(is_toggled=True)

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
        output_sample.toggle_show_spectrogram_button.enable()
        output_sample.toggle_show_debug_button.enable()

        #output_sample.audio_element.on("timeupdate", lambda e: self.logger.debug(e))
        #output_sample.audio_element.seek(10)
        #output_sample.audio_element.play()

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
