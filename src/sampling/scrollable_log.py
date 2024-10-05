from typing import Literal, Any, Optional

from nicegui.element import Element
from nicegui.elements.label import Label

class ScrollableLog(Element):

    def __init__(self, max_lines: Optional[int] = None) -> None:
        """Log View

        Create a log view that allows to add new lines without re-transmitting the whole history to the client.

        :param max_lines: maximum number of lines before dropping oldest ones (default: `None`)
        """
        super().__init__()
        self.max_lines = max_lines
        self._classes.append('nicegui-log')
        #self._classes.append('nicegui-scroll-area')

    def push(self, line: Any) -> None:
        """Add a new line to the log.

        :param line: the line to add (can contain line breaks)
        """
        for text in str(line).splitlines():
            with self:
                Label(text)
        while self.max_lines is not None and len(self.default_slot.children) > self.max_lines:
            self.remove(0)

    def scroll_to(self, *,
                  pixels: Optional[float] = None,
                  percent: Optional[float] = None,
                  axis: Literal['vertical', 'horizontal'] = 'vertical',
                  duration: float = 0.0,
                  ) -> None:
        """Set the scroll area position in percentage (float) or pixel number (int).

        You can add a delay to the actual scroll action with the `duration_ms` parameter.

        :param pixels: scroll position offset from top in pixels
        :param percent: scroll position offset from top in percentage of the total scrolling size
        :param axis: scroll axis
        :param duration: animation duration (in seconds, default: 0.0 means no animation)
        """
        if pixels is not None and percent is not None:
            raise ValueError('You can only specify one of pixels or percent')
        if pixels is not None:
            self.run_method('setScrollPosition', axis, pixels, 1000 * duration)
        elif percent is not None:
            self.run_method('setScrollPercentage', axis, percent, 1000 * duration)
        else:
            raise ValueError('You must specify one of pixels or percent')