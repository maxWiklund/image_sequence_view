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
import sys
from typing import Optional, Set

from Qt import QtCore, QtGui, QtWidgets

import image_sequence_view.icons.qresource
from image_sequence_view.constants import (
    DEFAULT_END_FRAME,
    DEFAULT_START_FRAME,
    FPS_RATES,
)


class ToolButton(QtWidgets.QToolButton):
    """ToolButton with larger size."""

    def minimumSizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(30, 30)


class CurrentFrameLabel(QtWidgets.QLabel):
    """Label to display current frame."""

    frameEdited = QtCore.Signal(int)

    def __init__(self):
        super(CurrentFrameLabel, self).__init__(str(DEFAULT_START_FRAME))
        self._editor = QtWidgets.QLineEdit(self)
        self._editor.hide()
        font = QtGui.QFont()
        font.setPointSize(16)
        self.setFont(font)
        self.setStyleSheet(
            """QLabel{ color: #4ac26c; background: #3a3a3a; border: black; border-radius: 5px;}"""
        )
        self.setAlignment(QtCore.Qt.AlignCenter)

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        """Open _editor to allow user to set current frame.

        Args:
            event: Event to process.

        """

        self._editor.setText(self.text())
        self._editor.setGeometry(self.rect())
        self._editor.setFrame(False)
        self._editor.setAlignment(self.alignment())

        self._editor.editingFinished.connect(self._finish_editing_callback)

        self._editor.show()
        self._editor.setFocus()
        self._editor.selectAll()

    def _finish_editing_callback(self) -> None:
        """Callback when user is done editing."""
        new_text = self._editor.text()
        try:
            frame = int(new_text)
        except ValueError:
            if new_text == "":
                self._editor.hide()
            return

        self.setText(new_text)
        self.frameEdited.emit(frame)
        self._editor.hide()


class TimelineRangeWidget(QtWidgets.QWidget):
    """Widget with timeline (no play-button)."""

    frameChanged = QtCore.Signal(int)

    def __init__(
        self,
        start: int = DEFAULT_START_FRAME,
        end: int = DEFAULT_END_FRAME,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent)
        self.start_frame = start
        self.end_frame = end
        self.current_frame = start
        self.dragging = False
        self._cached_frames = set()

        self.setMouseTracking(True)

    def set_cached_frame(self, frame: int) -> None:
        """Set frame marked as cached in time

        Args:
            frame: Frame to view as cached.

        """
        self._cached_frames.add(frame)

    def set_cached_frames(self, frames: Set[int]) -> None:
        """Provide list with frames that should be cached. Any frame missing will be set
        as not cached.

        Args:
            frames: List of cached frames.

        """
        self._cached_frames = set(frames)
        self.update()

    def _timeline_width(self) -> int:
        """Get width of timeline area.

        Returns:
            Width of the timeline to paint frames on.

        """
        return self.width() - (self._start_offset() + self._end_offset())

    def _start_offset(self) -> int:
        """Offset for the start frame label in pixels.

        Returns:
            Number of pixels to offset the timeline area (from the start).

        """
        return self.fontMetrics().horizontalAdvance(str(self.start_frame)) + 12

    def _end_offset(self) -> int:
        """Offset for the end frame label in pixels.

        Returns:
            Number of pixels to offset the timeline area (from the end).

        """
        return self.fontMetrics().horizontalAdvance(str(self.end_frame)) + 12

    def set_range(self, start: int, end: int) -> None:
        """Set the frame range for the timeline to display.

        Args:
            start: Start frame number.
            end: End frame number.

        """
        end = end if end != start else end + 1
        self.start_frame = start
        self.end_frame = max(end, 10)
        self._cached_frames = set()
        self.set_current_frame(frame=start)
        self.update()

    def set_current_frame(self, frame: int) -> None:
        """Set the current frame in the timeline.

        Args:
            frame: Frame number to set as the current.

        """
        self.current_frame = max(self.start_frame, min(frame, self.end_frame))
        self.frameChanged.emit(self.current_frame)
        self.update()

    def frame_to_pos(self, frame: int) -> QtCore.QPoint:
        """Get the position (local space) for the frame.

        Args:
            frame: Frame to get position for.

        Returns:
            Position for frame.

        """
        x = self._start_offset() + int(
            (frame - self.start_frame)
            / (self.end_frame - self.start_frame)
            * self._timeline_width()
        )
        return QtCore.QPoint(x, 0)

    def pos_to_frame(self, pos: QtCore.QPoint) -> int:
        """Get the frame number for the position.

        Args:
            pos: Position on timeline to get frame for (position is in local space).

        Returns:
            Frame number from position.

        """
        ratio = (pos.x() - self._start_offset()) / self._timeline_width()
        return int(self.start_frame + ratio * (self.end_frame - self.start_frame))

    @staticmethod
    def _compute_best_step(
        start_frame: int,
        end_frame: int,
        widget_width: int,
        font_metrics: QtGui.QFontMetrics,
    ) -> int:
        """Get the steps (the distance between each fram number) that is painted.

        Args:
            start_frame: Start frame in sequence.
            end_frame: End frame in sequence.
            widget_width: The width of the widget.
            font_metrics: Font metrics to use.

        Returns:
            Then frame step.

        """
        max_number = max(range(start_frame, end_frame + 1))
        label_width = (
            font_metrics.horizontalAdvance(str(max_number)) + 30
        )  # Conservative label estimate
        frame_range = end_frame - start_frame

        for step in range(1, frame_range + 1):
            tick_count = ((frame_range) // step) + 1
            required_width = tick_count * label_width
            if required_width <= widget_width:
                return step  # Smallest step that fits.
        return frame_range

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHints(
            QtGui.QPainter.Antialiasing | QtGui.QPainter.TextAntialiasing
        )
        # Draw frame ticks and numbers
        tick_color = QtGui.QColor(150, 150, 150)
        painter.setPen(tick_color)

        font_metrics = painter.fontMetrics()

        step = self._compute_best_step(
            self.start_frame,
            self.end_frame,
            self._timeline_width(),
            painter.fontMetrics(),
        )
        tick_frames = list(range(self.start_frame, self.end_frame + 1, step))

        bottom = font_metrics.height()

        step_height = self.height() - bottom

        if self.end_frame not in tick_frames:
            tick_frames.append(self.end_frame)

        for i, f in enumerate(sorted(tick_frames)):
            x = self.frame_to_pos(f).x()
            painter.drawLine(x, 0, x, step_height)

            for sub_frame in range(f + 1, f + step):
                x_ = self.frame_to_pos(sub_frame).x()
                painter.drawLine(x_, step_height // 2, x_, step_height)

            frame_label = str(f)
            text_width = font_metrics.horizontalAdvance(frame_label)
            h_layout = QtCore.Qt.AlignHCenter

            if i == len(tick_frames) - 1:
                # Don't draw the last frame.
                continue
            elif i != 0:
                x = x - text_width / 2

            frame_label = str(f)
            text_rect = QtCore.QRect(x, step_height + 2, text_width, 20)
            painter.drawText(text_rect, h_layout | QtCore.Qt.AlignTop, frame_label)

        number_of_frames = self.end_frame - self.start_frame
        frame_size = self._timeline_width() / number_of_frames
        for frame in self._cached_frames:
            pos = self.frame_to_pos(frame)
            cache_rect = QtCore.QRectF(pos.x() - 1, step_height - 1, frame_size + 1, 3)
            painter.fillRect(cache_rect, QtGui.QColor("#426b8f"))

        start_frame_rect = QtCore.QRect(
            0, 1, self._start_offset() - 1, self.height() - 1
        )
        end_frame_rect = QtCore.QRect(
            self.width() - self._end_offset() + 2,
            0,
            self._end_offset() - 1,
            self.height(),
        )

        painter.save()
        painter.setPen(QtGui.QPen(QtCore.Qt.lightGray))

        for rect, frame in (
            (start_frame_rect, self.start_frame),
            (end_frame_rect, self.end_frame),
        ):
            path = QtGui.QPainterPath()
            path.addRoundedRect(rect, 5, 5)
            painter.fillPath(path, QtGui.QColor("#4a4a4a"))
            painter.drawText(rect, QtCore.Qt.AlignCenter, str(frame))

        painter.restore()

        # Draw play-head
        self._draw_play_head(
            painter, QtGui.QColor("#4ac26c"), self.current_frame, step_height
        )

        time_line_rect = self.rect().adjusted(
            self._start_offset(), 0, -self._end_offset(), 0
        )

        pos = self.mapFromGlobal(QtGui.QCursor.pos())
        if not time_line_rect.contains(pos):
            return
        cursor_frame = self.pos_to_frame(pos)

        if cursor_frame != self.current_frame:
            color = QtGui.QColor("#4ac26c")
            color.setAlphaF(0.5)
            self._draw_play_head(painter, color, cursor_frame, step_height)

    def _draw_play_head(
        self, painter: QtGui.QPainter, color: QtGui.QColor, frame: int, step_height
    ) -> None:
        """Draw play-head triangle

        Args:
            painter: Painter to use.
            color: Color to draw play head with.
            frame: Frame to draw play head above.

        """
        play_head_x = self.frame_to_pos(frame).x()
        triangle_height = 7
        triangle_width = 15
        triangle_tip = step_height

        # Define the triangle points
        triangle = QtGui.QPolygon(
            [
                QtCore.QPoint(play_head_x, triangle_tip),  # Top point
                QtCore.QPoint(
                    play_head_x - triangle_width // 2, triangle_tip - triangle_height
                ),
                QtCore.QPoint(
                    play_head_x + triangle_width // 2, triangle_tip - triangle_height
                ),
            ]
        )

        painter.save()
        font = QtGui.QFont()
        font.setPointSize(14)
        painter.setFont(font)
        painter.setBrush(color)  # Fill color
        painter.setPen(QtCore.Qt.NoPen)  # Optional: No outline
        painter.drawPolygon(triangle)

        painter.setPen(color)
        painter.drawText(
            QtCore.QRect(play_head_x - 20, triangle_tip - 22, 40, 20),
            QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop,
            str(frame),
        )
        painter.restore()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        """Process mouse press events.

        Args:
            event: Event to process.

        """
        if event.button() == QtCore.Qt.LeftButton:
            self.dragging = True
            self.set_current_frame(self.pos_to_frame(self.local_cursor()))

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        """Process mouse move events.

        Args:
            event: Event to process.

        """
        if self.dragging:
            self.set_current_frame(self.pos_to_frame(self.local_cursor()))

        self.update()

    def leaveEvent(self, event: QtGui.QMouseEvent) -> None:
        """Process mouse enter events.

        Args:
            event: Event to process.

        """
        self.update()
        super().leaveEvent(event)

    def enterEvent(self, event: QtGui.QMouseEvent) -> None:
        """Process mouse enter events.

        Args:
            event: Event to process.

        """
        super().enterEvent(event)
        self.update()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
        self.dragging = False
        event.accept()

    def minimumSizeHint(self) -> QtCore.QSize:
        """Get minimum size for widget.

        Returns:
            minimum size for widget.

        """
        return QtCore.QSize(0, 35)

    def local_cursor(self) -> QtCore.QPoint:
        """Get cursor in local space.

        Returns:
            Cursor in local space.

        """
        return self.mapFromGlobal(QtGui.QCursor.pos())


class TimelineViewer(QtWidgets.QWidget):
    """Widget with timeline and buttons."""

    jumpToEnd = QtCore.Signal()
    jumpToStart = QtCore.Signal()
    previousFrame = QtCore.Signal()
    nextFrame = QtCore.Signal()
    fpsChanged = QtCore.Signal(float)
    playForward = QtCore.Signal()
    playBackward = QtCore.Signal()

    def __init__(
        self,
        start: int = DEFAULT_START_FRAME,
        end: int = DEFAULT_END_FRAME,
        fps: float = 25.0,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super(TimelineViewer, self).__init__(parent=parent)
        self._time_slider = TimelineRangeWidget(parent=self)
        self._time_slider.set_range(start, end)
        self._fps_combobox = QtWidgets.QComboBox()
        self._fps_combobox.setFocusPolicy(QtCore.Qt.NoFocus)
        self._fps_combobox.setEditable(True)
        self._fps_combobox.addItems(list(FPS_RATES))

        self._fps_combobox.setCurrentText(str(fps))

        self._current_frame_label = CurrentFrameLabel()
        self._next_frame_button = ToolButton()
        self._previous_frame_button = ToolButton()
        self._play_forward_button = ToolButton()
        self._play_back_button = ToolButton()
        self._jump_to_end_button = ToolButton()
        self._jump_to_start_button = ToolButton()

        self._next_frame_button.setIcon(QtGui.QIcon(":jumpForward.svg"))
        self._previous_frame_button.setIcon(QtGui.QIcon(":jumpBackward.svg"))
        self._jump_to_end_button.setIcon(QtGui.QIcon(":next.svg"))
        self._jump_to_start_button.setIcon(QtGui.QIcon(":previous.svg"))
        self._play_forward_button.setIcon(QtGui.QIcon(":playForward.svg"))
        self._play_back_button.setIcon(QtGui.QIcon(":playBackwards.svg"))

        footer_layout = QtWidgets.QHBoxLayout()
        footer_layout.setSpacing(2)
        footer_layout.addWidget(self._fps_combobox)
        footer_layout.addStretch(1)
        footer_layout.addWidget(self._jump_to_start_button)
        footer_layout.addWidget(self._previous_frame_button)
        footer_layout.addWidget(self._play_back_button)
        footer_layout.addWidget(self._current_frame_label)
        footer_layout.addWidget(self._play_forward_button)
        footer_layout.addWidget(self._next_frame_button)
        footer_layout.addWidget(self._jump_to_end_button)
        footer_layout.addStretch(1)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self._time_slider)
        layout.addLayout(footer_layout)
        self.setLayout(layout)

        self.set_current_frame = self._time_slider.set_current_frame
        self.set_cached_frame = self._time_slider.set_cached_frame
        self.set_cached_frames = self._time_slider.set_cached_frames
        self.frameChanged = self._time_slider.frameChanged

        # Signals:
        self._time_slider.frameChanged.connect(
            lambda value: self._current_frame_label.setText(str(value))
        )
        self._fps_combobox.currentTextChanged.connect(
            lambda value: self.fpsChanged.emit(float(value))
        )
        self._current_frame_label.frameEdited.connect(
            self._time_slider.set_current_frame
        )
        self._play_back_button.clicked.connect(self.playBackward.emit)
        self._play_forward_button.clicked.connect(self.playForward.emit)
        self._previous_frame_button.clicked.connect(self.previousFrame.emit)
        self._next_frame_button.clicked.connect(self.nextFrame.emit)
        self._jump_to_end_button.clicked.connect(self.jumpToEnd.emit)
        self._jump_to_start_button.clicked.connect(self.jumpToStart.emit)

    def set_range(self, start: int, end: int) -> None:
        """Set the frame range for the timeline to display.

        Args:
            start: Start frame number.
            end: End frame number.

        """
        self._time_slider.set_range(start, end)
        fm = QtGui.QFontMetrics(self._current_frame_label.font())
        self._current_frame_label.setMinimumWidth(fm.horizontalAdvance(str(end)) + 20)

    def toggle_forward_callback(self, state: bool) -> None:
        path = ":stop.svg" if state else ":playForward.svg"
        self._play_forward_button.setIcon(QtGui.QIcon(path))
        self._play_back_button.setIcon(QtGui.QIcon(":playBackwards.svg"))

    def toggle_backward_callback(self, state: bool) -> None:
        path = ":stop.svg" if state else ":playBackwards.svg"
        self._play_back_button.setIcon(QtGui.QIcon(path))
        self._play_forward_button.setIcon(QtGui.QIcon(":playForward.svg"))


def __test() -> None:
    """Test function"""
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")

    app.setStyleSheet(
        """QWidget{
    background: #323232;}"""
    )
    timeline = TimelineViewer(1001, 1159)
    timeline.set_current_frame(1002)
    timeline.frameChanged.connect(lambda f: print("Frame:", f))
    timeline.show()
    timeline.resize(1200, 180)
    sys.exit(app.exec())


if __name__ == "__main__":
    __test()
