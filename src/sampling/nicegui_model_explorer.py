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

from PIL import Image
from datetime import datetime
import logging
import asyncio

import numpy as np
import torch
import cv2
from nicegui import ui

from utils.dual_diffusion_utils import tensor_to_img


class ModelExplorer(ui.column):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.logger = logging.getLogger(name="nicegui_app")
        self.generating_visualization = asyncio.Semaphore()
        self.loading_notification = None
        self.get_app = lambda: None
        self.selected_keys = set()

        with self:
            self.loaded_status_label = ui.label("Loaded Model: None").classes("w-full")
            with ui.row().classes("w-full items-center"):
                self.module_select = ui.select(label="Module", options=[]).classes("w-48")
                self.module_select.disable()
                self.load_module_button = ui.button("Load Module", on_click=self.on_click_load_module_button)
                self.load_module_button.disable()
            ui.separator().classes("w-full")
            self.details_column = ui.column().classes("w-full")

    async def on_tab_change(self):
        app = self.get_app()
        if app.last_loaded_model_name is None:
            self.loaded_status_label.set_text("Loaded Model: None")
            self.module_select.set_options([])
            self.module_select.disable()
            self.load_module_button.disable()
            return ui.notify("Error: No model loaded", type="error", color="red", close_button=True)
        
        self.module_select.enable()
        self.load_module_button.enable()

        self.model_metadata = app.model_server_state["model_metadata"]
        module_names = list(self.model_metadata["model_module_classes"].keys())
        self.module_select.set_options(module_names)
        if self.module_select.value is None:
            self.module_select.set_value(module_names[0])

    async def on_click_load_module_button(self) -> None:
        app = self.get_app()     
        module_state_dict = await app.get_module_state_dict(self.module_select.value)
        if module_state_dict is None: return     
        self.loaded_status_label.set_text(f"Loaded Model: {app.last_loaded_model_name} ({self.module_select.value})")
        self.module_state_dict = {k.removesuffix(".weight"): v for k, v in module_state_dict.items()}

        self.details_column.clear()
        self.selected_keys.clear()

        with self.details_column:       
            with ui.row().classes("w-full items-center"):
                self.filter_input = ui.input("Filter").classes("w-64")
                expand_button = ui.button('+ Expand All')
                collapse_button = ui.button('- Collapse All')
                uncheck_button = ui.button('Uncheck All', color="red")

            def build_tree(data):
                tree = {}

                for path in data:
                    parts = path.split(".")
                    current_level = tree

                    for part in parts:
                        if part not in current_level:
                            current_level[part] = {}
                        current_level = current_level[part]

                def convert_tree(tree, full_path=""):
                    result = []
                    for key, subtree in tree.items():
                        current_path = f"{full_path}.{key}" if full_path else key
                        children = convert_tree(subtree, current_path)
                        if len(children) > 0:
                            label = key
                        else:
                            tensor = self.module_state_dict[current_path].data
                            if tensor.numel() > 1:
                                label = f"{key} - Shape: {tuple(tensor.shape)} Mean: {tensor.mean().item()} Std: {tensor.std().item()}"
                            else:
                                label = f"{key} - Value: {tensor.item()}"
                        result.append({
                            "id": current_path,
                            "label": label,
                            "body": current_path,
                            "children": children,
                        })
                    return result

                return convert_tree(tree)

            self.module_tree = ui.tree(build_tree(list(self.module_state_dict.keys())),
                on_tick=lambda e: self.on_tick(e.value), tick_strategy="leaf")

            self.filter_input.bind_value_to(self.module_tree, "filter")
            expand_button.on_click(lambda: self.module_tree.expand())
            collapse_button.on_click(lambda: self.module_tree.collapse())

            async def uncheck_all():
                self.module_tree._props.setdefault('ticked', [])
                self.module_tree._props['ticked'][:] = []
                self.module_tree.update()
                await self.on_tick([])
            uncheck_button.on_click(uncheck_all)

    async def on_tick(self, keys: list[str]) -> None:
        
        async with self.generating_visualization:
            
            if self.loading_notification is None:
                self.loading_notification = ui.notification(timeout=None)
                self.loading_notification.message = f"Generating visualization..."
                self.loading_notification.spinner = True
                await asyncio.sleep(0.1)
            
            last_sleep = datetime.now()

            keys = set(keys)
            deselected_keys = self.selected_keys - keys
            new_selected_keys = keys - self.selected_keys

            for key in deselected_keys:
                self.module_tree.slots.pop(f"body-{key}", None)
            
            for key in new_selected_keys:

                tensor = self.module_state_dict[key].data
                if tensor.numel() == 1: continue

                now = datetime.now()
                if (now - last_sleep).total_seconds() > 1:
                    await asyncio.sleep(0.1)
                    last_sleep = now

                with self.module_tree.add_slot(f"body-{key}"):
                    
                    self.logger.debug(f"model_explorer: Generating visualization for {key}...")
                    if tensor.ndim >= 2: tensor = tensor.transpose(0, 1)
                    else:
                        # todo: 1d visualization
                        continue

                    while tensor.ndim < 4:
                        tensor.unsqueeze_(-1)

                    with ui.grid(columns=tensor.shape[-1], rows=tensor.shape[-2]).classes("gap-0"):
                        for i in range(tensor.shape[-2]):
                            for j in range(tensor.shape[-1]):                
                                image_array = tensor_to_img(tensor[:, :, i, j], colormap=True)
                                if image_array.ndim == 4: # ?????
                                    image_array = image_array[:, :, 0, :]

                                image = Image.fromarray(image_array)
                                ui.interactive_image(source=image)

            self.module_tree.update()
            self.selected_keys = keys
            self.loading_notification.dismiss()
            self.loading_notification = None