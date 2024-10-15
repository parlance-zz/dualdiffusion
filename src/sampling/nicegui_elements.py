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

from nicegui import ui


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

class ScrollableNumber(ui.number):

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
            #        ui.tooltip("Lock to freeze prompt when generating")
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
        
    def on_click_game_add_button(self):
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

