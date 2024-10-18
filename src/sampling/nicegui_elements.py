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
from copy import deepcopy
import os
import asyncio
import logging

from nicegui import ui

from utils.dual_diffusion_utils import sanitize_filename, dict_str


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

# ...