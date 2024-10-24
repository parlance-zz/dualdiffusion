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

import logging

from nicegui import ui
import nicegui.events

from sampling.custom_audio import CustomAudio

# *****************************************************************************
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union, cast

from typing_extensions import Self

from nicegui import optional_features
from nicegui.events import GenericEventArguments, Handler, MouseEventArguments, handle_event
from nicegui.elements.image import pil_to_base64
from nicegui.elements.mixins.content_element import ContentElement
from nicegui.elements.mixins.source_element import SourceElement

try:
    from PIL.Image import Image as PIL_Image
    optional_features.register('pillow')
except ImportError:
    pass


class AudioPlayer(SourceElement, ContentElement, component='nicegui_audio_editor.js'):
    CONTENT_PROP = 'content'
    PIL_CONVERT_FORMAT = 'PNG'

    def __init__(self,
                 source: Union[str, Path, 'PIL_Image'] = '', *,  # noqa: UP037
                 content: str = '',
                 size: Optional[Tuple[float, float]] = None,
                 on_mouse: Optional[Handler[MouseEventArguments]] = None,
                 events: List[str] = ['click'],  # noqa: B006
                 cross: Union[bool, str] = False,
                 ) -> None:

        super().__init__(source=source, content=content)
        self._props['events'] = events[:]
        self._props['cross'] = cross
        self._props['size'] = size

        if on_mouse:
            self.on_mouse(on_mouse)

    def set_source(self, source: Union[str, Path, 'PIL_Image']) -> None:  # noqa: UP037
        return super().set_source(source)

    def on_mouse(self, on_mouse: Handler[MouseEventArguments]) -> Self:
        """Add a callback to be invoked when a mouse event occurs."""
        def handle_mouse(e: GenericEventArguments) -> None:
            args = cast(dict, e.args)
            arguments = MouseEventArguments(
                sender=self,
                client=self.client,
                type=args.get('mouse_event_type', ''),
                image_x=args.get('image_x', 0.0),
                image_y=args.get('image_y', 0.0),
                button=args.get('button', 0),
                buttons=args.get('buttons', 0),
                alt=args.get('altKey', False),
                ctrl=args.get('ctrlKey', False),
                meta=args.get('metaKey', False),
                shift=args.get('shiftKey', False),
            )
            handle_event(on_mouse, arguments)
        self.on('mouse', handle_mouse)
        return self

    def _set_props(self, source: Union[str, Path, 'PIL_Image']) -> None:  # noqa: UP037
        if optional_features.has('pillow') and isinstance(source, PIL_Image):
            source = pil_to_base64(source, self.PIL_CONVERT_FORMAT)
        super()._set_props(source)

    def force_reload(self) -> None:
        """Force the image to reload from the source."""
        self._props['t'] = time.time()
        self.update()

# *****************************************************************************

class AudioEditor(AudioPlayer):

    def __init__(self, *args, **kwargs) -> None:
        audio_source = kwargs.pop("audio_source", None)
        kwargs["events"] = ["mousedown", "mouseup"]
        super().__init__(*args, **kwargs)

        self.logger = logging.getLogger(name="nicegui_app")
        self.playing = False
        self.duration = 0
        self.add_slot('cross', '<line :x1="props.x" y1="0" :x2="props.x" y2="100%" stroke="white" />')
        
        with self:
            self.audio_element = CustomAudio("", controls=False).props("preload='auto'").classes("w-full")
            self.audio_element.on("event", lambda e: self.time_update(e))
        
        if audio_source is not None:
            self.set_audio_source(audio_source)

        self.on_mouse(lambda e: self.handle_mouse(e))

    def handle_mouse(self, e: nicegui.events.MouseEventArguments) -> None:
        if e.type == "mousedown":
            if self.playing == False:
                print(e)
                self.seek(e.image_x * self.duration)
            self.play_pause()

    def play_pause(self) -> None:
        if self.playing == True:
            self.pause()
        else:
            self.play()
            
    def play(self) -> None:
        self.playing = True
        self.run_method("play", True)
        self.audio_element.play()

    def pause(self) -> None:
        self.playing = False
        self.run_method("play", False)
        self.audio_element.pause()

    def seek(self, seconds: float) -> None:
        self.audio_element.seek(seconds)

    def time_update(self, e) -> None:
        self.run_method("set_time", e.args["time"])

    def set_audio_source(self, audio_path: str, duration: float) -> None:
        self.audio_element.set_source(audio_path)
        self.props(f"duration='{duration}'")
        self.duration = duration