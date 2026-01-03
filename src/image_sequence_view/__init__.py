# Copyright (C) 2026  Max Wiklund
#
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
from typing import Callable, Optional

from image_sequence import ImageSequence
from Qt import QtCore, QtGui, QtWidgets

import image_sequence_view.icons.qresource
from image_sequence_view._timeline_view import TimelineViewer
from image_sequence_view._viewer import ImageSequenceView
from image_sequence_view.constants import (
    DEFAULT_END_FRAME,
    DEFAULT_START_FRAME,
    FPS_RATES,
)

try:
    from PySide6.QtGui import QSurfaceFormat
except ImportError:
    from PyQt6.QtGui import QSurfaceFormat


class ImageSequenceWidget(QtWidgets.QWidget):
    """Widget to play image sequence."""

    def __init__(self, fps: float = 24, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        # Widgets:
        self._viewport = ImageSequenceView(fps=fps, parent=self)
        self._time_line_view = TimelineViewer(fps=fps, parent=self)

        self.set_image_sequence = self._viewport.set_image_sequence
        self.set_view_colorspace = self._viewport.set_view_colorspace

        # Layout:
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._viewport, 1)
        layout.addWidget(self._time_line_view)
        self.setLayout(layout)

        # Signals:
        self._viewport.frameChanged.connect(self._time_line_view.set_current_frame)
        self._viewport.frameChanged.connect(self._time_line_view.set_current_frame)
        self._viewport.rangeChanged.connect(self._time_line_view.set_range)
        self._viewport.playForward.connect(self._time_line_view.toggle_forward_callback)
        self._viewport.playBackward.connect(
            self._time_line_view.toggle_backward_callback
        )
        self._viewport.cachedFrame.connect(self._time_line_view.set_cached_frame)
        self._viewport.cachedChannelFrames.connect(
            self._time_line_view.set_cached_frames
        )

        self._time_line_view.fpsChanged.connect(self._viewport.set_fps)
        self._time_line_view.jumpToEnd.connect(self._viewport.jumpToEnd)
        self._time_line_view.jumpToStart.connect(self._viewport.jumpToStart)
        self._time_line_view.nextFrame.connect(self._viewport.next_frame_callback)
        self._time_line_view.previousFrame.connect(
            self._viewport.previous_frame_callback
        )
        self._time_line_view.frameChanged.connect(self._viewport.set_frame)
        self._time_line_view.playForward.connect(self._viewport.toggle_forward)
        self._time_line_view.playBackward.connect(self._viewport.toggle_backward)

    def set_ocio_input_colorspace_hook(self, hook: Callable[[str], str]) -> None:
        """Set function to call to figure out what the input colorspace is.

        Args:
            hook: Function to figure out input colorspace.

        **Examples**::

            def guess_input(image_path: str) -> str:
                return "raw"

            view = ImageSequenceWidget(...)
            view.set_ocio_input_colorspace_hook(guess_input)

        """
        self._viewport.viewport.set_ocio_input_colorspace_hook(hook)


def __test() -> None:
    """Test function."""

    def create_dark_palette():
        dark_palette = QtGui.QPalette()

        # Base colors
        dark_color = QtGui.QColor(45, 45, 45)
        disabled_color = QtGui.QColor(127, 127, 127)
        light_text_color = QtGui.QColor(220, 220, 220)

        # Set dark theme colors
        dark_palette.setColor(QtGui.QPalette.Window, dark_color)
        dark_palette.setColor(QtGui.QPalette.WindowText, light_text_color)
        dark_palette.setColor(QtGui.QPalette.Base, QtGui.QColor(30, 30, 30))
        dark_palette.setColor(QtGui.QPalette.AlternateBase, dark_color)
        dark_palette.setColor(QtGui.QPalette.ToolTipBase, light_text_color)
        dark_palette.setColor(QtGui.QPalette.ToolTipText, light_text_color)
        dark_palette.setColor(QtGui.QPalette.Text, light_text_color)
        dark_palette.setColor(QtGui.QPalette.Button, dark_color)
        dark_palette.setColor(QtGui.QPalette.ButtonText, light_text_color)
        dark_palette.setColor(QtGui.QPalette.BrightText, QtCore.Qt.red)

        dark_palette.setColor(QtGui.QPalette.Link, QtGui.QColor(42, 130, 218))

        # Disabled state colors
        dark_palette.setColor(
            QtGui.QPalette.Disabled, QtGui.QPalette.Text, disabled_color
        )
        dark_palette.setColor(
            QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, disabled_color
        )
        dark_palette.setColor(
            QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, disabled_color
        )

        return dark_palette

    src = os.path.join(os.path.dirname(__file__), "../../images/plate.1001.exr")
    seq = ImageSequence(src)
    seq.find_frames_on_disk()

    # Configure OpenGL
    fmt = QSurfaceFormat()
    fmt.setVersion(4, 1)  # Request OpenGL 4.1
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QtWidgets.QApplication(sys.argv)
    app.setPalette(create_dark_palette())
    app.setStyle("Fusion")
    window = ImageSequenceWidget()
    window.resize(1200, 1000)
    window.show()
    window.set_image_sequence(seq)
    window.set_view_colorspace("Rec.709")
    sys.exit(app.exec_())


if __name__ == "__main__":
    __test()
