# MIT License
#
# Copyright (c) 2021 Zauberzeug GmbH
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
#SOFTWARE.

# Modifications under MIT License
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

import nicegui.events

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

# lightly modified copy of nicegui.elements.audio, needed for custom js
class CustomAudio(SourceElement, component='nicegui_custom_audio.js'):
    SOURCE_IS_MEDIA_FILE = True

    def __init__(self, src: Union[str, Path], *,
                 controls: bool = True,
                 autoplay: bool = False,
                 muted: bool = False,
                 loop: bool = False,
                 ) -> None:
        """Audio

        Displays an audio player.

        :param src: URL or local file path of the audio source
        :param controls: whether to show the audio controls, like play, pause, and volume (default: `True`)
        :param autoplay: whether to start playing the audio automatically (default: `False`)
        :param muted: whether the audio should be initially muted (default: `False`)
        :param loop: whether the audio should loop (default: `False`)

        See `here <https://developer.mozilla.org/en-US/docs/Web/HTML/Element/audio#events>`_
        for a list of events you can subscribe to using the generic event subscription `on()`.
        """
        super().__init__(source=src)
        self._props['controls'] = controls
        self._props['autoplay'] = autoplay
        self._props['muted'] = muted
        self._props['loop'] = loop

    def set_source(self, source: Union[str, Path]) -> None:
        return super().set_source(source)

    def seek(self, seconds: float) -> None:
        """Seek to a specific position in the audio.

        :param seconds: the position in seconds
        """
        self.run_method('seek', seconds)

    def play(self) -> None:
        """Play audio."""
        self.run_method('play')

    def pause(self) -> None:
        """Pause audio."""
        self.run_method('pause')

# lightly modified copy of nicegui.elements.interactive_image, needed for custom js
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
        self._props['duration'] = 0
        self._props['select_range'] = False

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

# custom element combining a spectrogram image and interactive audio playback / editing
class AudioEditor(AudioPlayer):

    def __init__(self, *args, **kwargs) -> None:
        audio_source = kwargs.pop("audio_source", None)
        kwargs["events"] = ["mousedown", "mouseup"]
        super().__init__(*args, **kwargs)

        self.logger = logging.getLogger(name="nicegui_app")
        self.playing = False
        self.duration = 0
        self.add_slot("cross", '<line :x1="props.x" y1="0" :x2="props.x" y2="100%" stroke="white" />')
        self.on_mouse(lambda e: self._on_mouse(e))

        with self:
            self.audio_element = CustomAudio("", controls=False).props("preload='auto'").classes("w-full")
            self.audio_element.on("time_update", lambda e: self.on_time_update(e))
            self.audio_element.on("duration_change", lambda e: self.on_duration_change(e))
            self.audio_element.on("play", lambda: self.on_play())
            self.audio_element.on("pause", lambda: self.on_pause())
            
        if audio_source is not None:
            self.set_audio_source(audio_source)

    def play(self) -> None:
        self.audio_element.play()
    def pause(self) -> None:
        self.audio_element.pause()
    def play_pause(self) -> None:
        if self.playing == True: self.pause()
        else: self.play()
    def seek(self, seconds: float) -> None:
        self.audio_element.seek(seconds)

    def on_play(self) -> None:
        self.playing = True
        self.run_method("play", True)
    def on_pause(self) -> None:
        self.playing = False
        self.run_method("play", False)
    def on_time_update(self, e) -> None:
        self.run_method("set_time", e.args["time"])
    def on_duration_change(self, e) -> None:
        self.duration = e.args["duration"]
        self.props(f"duration='{self.duration}'")

    def set_select_range(self, start: float, duration: float) -> None:
        self.run_method("set_select_range", start, duration)
    def set_audio_source(self, audio_path: str) -> None:
        self.audio_element.set_source(audio_path)
    def _on_mouse(self, e: nicegui.events.MouseEventArguments) -> None:
        if e.type == "mousedown":
            if self.playing == False:
                self.seek(e.image_x * self.duration)
            self.play_pause()