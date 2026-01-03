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
import logging

import difflib
import os
import sys
from enum import Enum, auto
from typing import Callable, Optional

import numpy as np
import OpenImageIO as oiio
import PyOpenColorIO as ocio
from image_sequence import ImageSequence
from OpenGL import GL
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from Qt import QtCore, QtGui, QtWidgets

from image_sequence_view.constants import LOG

_RGB_ORDER = {"R": 0, "G": 1, "B": 2, "A": 3}


class PlayMode(Enum):
    Forward = auto()
    Backward = auto()


def get_channel_suffix(channel_name: str) -> str:
    """Get channel suffix (last character R, G, B).

    Args:
        channel_name: Channel name to get suffix from.

    Returns:
        Suffix diffuse.A -> A

    """
    return channel_name.split(".").pop(-1)


config = ocio.GetCurrentConfig()


DEFAULT_CHANNEL_NAME = "rgba"

VERT_SHADER = """#version 410 core
layout(location = 0) in vec3 in_position;
layout(location = 1) in vec2 in_texCoord;

uniform mat4 mvpMat;
out vec2 vert_texCoord;

void main() {
    vert_texCoord = in_texCoord;
    gl_Position = mvpMat * vec4(in_position, 1.0);
}
"""

_FRAG_SHADER_BASE = """#version 410 core
uniform sampler2D imageTex;
in vec2 vert_texCoord;
out vec4 frag_color;

void main() {
    frag_color = texture(imageTex, vert_texCoord);
}
"""

_FRAG_SHADER_OCIO = """#version 410 core
uniform sampler2D imageTex;
in vec2 vert_texCoord;
out vec4 frag_color;

{ocio_src}

void main() {{
    vec4 inColor = texture(imageTex, vert_texCoord);
    vec4 outColor = OCIOMain(inColor);
    frag_color = outColor;
}}
"""


def guess_colorspace(file_path: str) -> str:
    buf = oiio.ImageBuf(file_path)
    target = buf.spec().get_string_attribute("oiio:ColorSpace")
    if target:
        all_spaces = [cs.getName() for cs in config.getColorSpaces()]
        return next(
            iter(difflib.get_close_matches(target, all_spaces)), ocio.ROLE_DEFAULT
        )

    buf = oiio.ImageBufAlgo.channels(
        buf, tuple([0, 1, 2]), newchannelnames=("R", "G", "B")
    )
    data = buf.get_pixels(oiio.FLOAT)
    max_val = data.max()
    pct_above_1 = (data > 1.0).sum() / data.size * 100
    if max_val > 1.5 or pct_above_1 > 1.0:
        LOG.debug(f"{file_path} likely linear (HDR values present)")
        return ocio.ROLE_DEFAULT
    elif max_val <= 1.0:
        LOG.debug(f"{file_path} possibly gamma-encoded (clamped to 0-1)")
        return "Utility - Linear - Rec.709"
    else:
        LOG.debug(f"{file_path} uncertain colorspace.")
        return ocio.ROLE_DEFAULT


class GLImagePlane(QOpenGLWidget):
    channelsUpdated = QtCore.Signal(object)
    # Emit list of channels names

    cachedFrame = QtCore.Signal(int)

    frameChanged = QtCore.Signal(int)
    rangeChanged = QtCore.Signal(int, int)
    cachedChannelFrames = QtCore.Signal(object)
    playForward = QtCore.Signal(bool)
    playBackward = QtCore.Signal(bool)

    def __init__(self, fps=24, parent=None):
        super().__init__(parent=parent)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

        # Image data
        self.image_data = None
        self.image_width = 1
        self.image_height = 1

        # OpenGL resources
        self.gl_initialized = False
        self.vao = None
        self.vbo_pos = None
        self.vbo_tex = None
        self.vbo_idx = None
        self.image_tex = None
        self.shader_program = None
        self.vert_shader = None
        self.frag_shader = None

        # View parameters
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0

        # Viewport variables:
        self._last_mouse_position = QtGui.QCursor.pos()
        self._channel_targets = {DEFAULT_CHANNEL_NAME: ["R", "G", "B", "A"]}
        self._channel_cache = {}

        self.view_channel = DEFAULT_CHANNEL_NAME
        self._current_index = 0
        self._file_paths_to_load = []
        self._cached_channels = {}
        self._fps = fps
        self._start_frame = 0
        self._end_frame = 0

        self._play_state = PlayMode.Forward
        self._play_forward = False
        self._play_backward = False
        self._timer = QtCore.QTimer(self)

        self._fps_timer = QtCore.QElapsedTimer()
        self._fps_timer.start()
        self._fps_frame_count = 0
        self._fps_value = 0.0

        # OCIO
        self._ocio_input_cs = ocio.ROLE_DEFAULT
        self._ocio_display = config.getDefaultDisplay()
        self.ocio_view_color_space = config.getDefaultView(self._ocio_display)
        self._ocio_exposure = 0.0
        self._ocio_gamma = 1.0
        self._ocio_channel_hot = [1, 1, 1, 1]
        self._ocio_shader_desc = None
        self._ocio_shader_cache_id = None
        self._ocio_tex_ids = []

        self._ocio_guess_function = guess_colorspace

        # Shortcuts
        self._setup_shortcuts()

        self.setMouseTracking(True)
        self.setAcceptDrops(True)

    def _setup_shortcuts(self):
        """Setup keyboard shortcuts for channel view."""
        for i, key in enumerate(["R", "G", "B", "A"]):
            shortcut = QtWidgets.QShortcut(QtGui.QKeySequence(key), self)
            shortcut.activated.connect(lambda ch=i: self._toggle_channel(ch))

        reset_view_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("f"), self)
        reset_view_shortcut.activated.connect(self.reset_zoom_and_pan)

        toggle_play_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence(" "), self)
        toggle_play_shortcut.activated.connect(self._toggle_play_callback)

        step_forward_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("."), self)
        step_forward_shortcut.activated.connect(self._next_frame_callback)

        step_backwards_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence(","), self)
        step_backwards_shortcut.activated.connect(self._previous_frame_callback)

    def _toggle_play_callback(self) -> None:
        if self._play_state == PlayMode.Forward:
            self.toggle_forward()
        else:
            self.toggle_backward()

    def reset_zoom_and_pan(self) -> None:
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self.update()

    def _toggle_channel(self, channel: int) -> None:
        """Toggle channel isolation.

        Args:
            channel: Channel number to isolate (0-3) e.g RGBA.

        """
        if channel < 4 and (
            all(self._ocio_channel_hot) or not self._ocio_channel_hot[channel]
        ):
            self._ocio_channel_hot = [1 if i == channel else 0 for i in range(4)]
        else:
            self._ocio_channel_hot = [1, 1, 1, 1]
        self.update_ocio_processor()

    def set_ocio_input_colorspace_hook(self, hook: Callable[[str], str]) -> None:
        """Set function to call to figure out what the input colorspace is.

        Args:
            hook: Function to figure out input colorspace.

        **Examples**::

            def guess_input(image_path: str) -> str:
                return "raw"

            view = GLImagePlane(...)
            view.set_ocio_input_colorspace_hook(guess_input)

        """
        self._ocio_guess_function = hook

    def initializeGL(self) -> None:
        """Initialize OpenGL. MUST NOT raise exceptions."""
        GL.glClearColor(0.0, 0.0, 0.0, 1.0)
        # GL.glEnable(GL.GL_FRAMEBUFFER_SRGB)
        if not self._init_image_texture():
            return
        if not self._init_geometry():
            return
        if not self._build_shader_program():
            return

        self.gl_initialized = True

    def _init_image_texture(self) -> bool:
        """Initialize empty image texture."""
        try:
            self.image_tex = GL.glGenTextures(1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.image_tex)
            GL.glTexParameteri(
                GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE
            )
            GL.glTexParameteri(
                GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE
            )
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)

            # Create empty texture
            GL.glTexImage2D(
                GL.GL_TEXTURE_2D,
                0,
                GL.GL_RGBA32F,
                1,
                1,
                0,
                GL.GL_RGBA,
                GL.GL_FLOAT,
                None,
            )
            return True
        except Exception as e:
            LOG.error(f"Failed to init texture: {e}")
            return False

    def _init_geometry(self) -> bool:
        """Initialize quad geometry."""
        try:
            # Positions
            positions = np.array(
                [
                    -0.5,
                    0.5,
                    0.0,  # top-left
                    0.5,
                    0.5,
                    0.0,  # top-right
                    0.5,
                    -0.5,
                    0.0,  # bottom-right
                    -0.5,
                    -0.5,
                    0.0,  # bottom-left
                ],
                dtype=np.float32,
            )

            # Texture coordinates
            tex_coords = np.array(
                [
                    0.0,
                    1.0,  # top-left
                    1.0,
                    1.0,  # top-right
                    1.0,
                    0.0,  # bottom-right
                    0.0,
                    0.0,  # bottom-left
                ],
                dtype=np.float32,
            )

            # Indices
            indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)

            self.vao = GL.glGenVertexArrays(1)
            GL.glBindVertexArray(self.vao)

            # Position buffer
            self.vbo_pos = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo_pos)
            GL.glBufferData(
                GL.GL_ARRAY_BUFFER, positions.nbytes, positions, GL.GL_STATIC_DRAW
            )
            GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
            GL.glEnableVertexAttribArray(0)

            # Texture coord buffer
            self.vbo_tex = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo_tex)
            GL.glBufferData(
                GL.GL_ARRAY_BUFFER, tex_coords.nbytes, tex_coords, GL.GL_STATIC_DRAW
            )
            GL.glVertexAttribPointer(1, 2, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
            GL.glEnableVertexAttribArray(1)

            # Index buffer
            self.vbo_idx = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.vbo_idx)
            GL.glBufferData(
                GL.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL.GL_STATIC_DRAW
            )

            return True
        except Exception as e:
            LOG.error(f"Failed to init geometry: {e}")
            return False

    def _compile_shader(
        self, source: str, shader_type: GL.constant.IntConstant
    ) -> Optional[int]:
        """Compile shader.

        Args:
            source: Source code glsl to compile.
            shader_type: Type of shader to compile.

        Returns:
            Pointer to compiled shader.

        """
        try:
            shader = GL.glCreateShader(shader_type)
            GL.glShaderSource(shader, source)
            GL.glCompileShader(shader)

            if not GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS):
                error = GL.glGetShaderInfoLog(shader)
                LOG.error(f"Shader compilation failed: {error}")
                return None
            return shader
        except Exception as e:
            LOG.error(f"Shader compile error: {e}")
            return None

    def _build_shader_program(self, force=False):
        """Build or rebuild shader program."""
        try:
            if not force and self.shader_program and self._ocio_shader_desc:
                cache_id = self._ocio_shader_desc.getCacheID()
                if cache_id == self._ocio_shader_cache_id:
                    return True

            # Create program if needed
            if not self.shader_program:
                self.shader_program = GL.glCreateProgram()

            # Compile vertex shader (once)
            if not self.vert_shader:
                self.vert_shader = self._compile_shader(
                    VERT_SHADER, GL.GL_VERTEX_SHADER
                )
                if not self.vert_shader:
                    return False
                GL.glAttachShader(self.shader_program, self.vert_shader)

            # Recompile fragment shader
            if self.frag_shader:
                GL.glDetachShader(self.shader_program, self.frag_shader)
                GL.glDeleteShader(self.frag_shader)

            # Choose fragment shader source
            if self._ocio_shader_desc:
                frag_src = _FRAG_SHADER_OCIO.format(
                    ocio_src=self._ocio_shader_desc.getShaderText()
                )
            else:
                frag_src = _FRAG_SHADER_BASE

            self.frag_shader = self._compile_shader(frag_src, GL.GL_FRAGMENT_SHADER)
            if not self.frag_shader:
                return False

            GL.glAttachShader(self.shader_program, self.frag_shader)

            # Link program
            GL.glBindAttribLocation(self.shader_program, 0, "in_position")
            GL.glBindAttribLocation(self.shader_program, 1, "in_texCoord")
            GL.glLinkProgram(self.shader_program)

            if not GL.glGetProgramiv(self.shader_program, GL.GL_LINK_STATUS):
                error = GL.glGetProgramInfoLog(self.shader_program)
                LOG.error(f"Program link failed: {error}")
                return False

            if self._ocio_shader_desc:
                self._ocio_shader_cache_id = self._ocio_shader_desc.getCacheID()

            return True
        except Exception as e:
            LOG.error(f"Shader program build failed: {e}")
            return False

    def load_image_from_disk(
        self, image_path: str, new_sequence: bool = False
    ) -> np.ndarray:
        """Load image from disk with the set channel.

        Args:
            image_path: File path to image to load.
            new_sequence:
                True if sequence is new (This is only called on the first frame when
                a new sequence is set.

        Returns:
            Array with data for channel or empty.

        """
        channel = self.view_channel
        buf = self._channel_cache.get(channel, {}).get(self._current_index)
        if buf is not None:
            return buf

        if not os.path.exists(image_path):
            return np.zeros((self.image_height, self.image_width, 4), dtype=np.float32)

        try:
            buf = oiio.ImageBuf(image_path)
            spec = buf.spec()
            if new_sequence:
                self._ocio_input_cs = self._ocio_guess_function(image_path)
                self._load_channel_targets(spec)

            target_names = self._channel_targets.get(
                self.view_channel, ["R", "G", "B", "A"]
            )
            existing = [spec.channelindex(n) for n in target_names]
            channel_indices = (existing + [0.0] * (3 - len(existing)) + [1.0])[:4]

            buf = oiio.ImageBufAlgo.channels(
                buf, tuple(channel_indices), newchannelnames=("R", "G", "B", "A")
            )

            self.cachedFrame.emit(self._start_frame + self._current_index)
            data = buf.get_pixels(oiio.FLOAT)
            self.image_width = spec.width
            self.image_height = spec.height
            data = np.flipud(data)
            self._channel_cache.setdefault(channel, {})[self._current_index] = data
            self._cached_channels.setdefault(channel, set()).add(
                self._start_frame + self._current_index
            )
            return data
        except Exception as e:
            LOG.exception(f"Failed to load image: {e}")
            return np.zeros((self.image_height, self.image_width, 4), dtype=np.float32)

    def setup_frame_for_rendering(self) -> None:
        """Setup current frame for rendering and trigger event."""
        data = None
        if self._current_index < len(self._file_paths_to_load):
            data = self.load_image_from_disk(
                self._file_paths_to_load[self._current_index]
            )

        if data is None:
            data = np.zeros((self.image_height, self.image_width, 4), dtype=np.float32)

        self.makeCurrent()
        self.frameChanged.emit(self._start_frame + self._current_index)

        # Upload to texture
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.image_tex)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_RGBA32F,
            self.image_width,
            self.image_height,
            0,
            GL.GL_RGBA,
            GL.GL_FLOAT,
            data.ravel(),
        )
        self.update()

    def _load_channel_targets(self, spec: oiio.ImageSpec) -> None:
        """Load all channels on the first image."""
        channels = spec.channelnames
        grouped_channels = {}
        for channel in channels:
            if channel in ("RGBA"):
                grouped_channels.setdefault(DEFAULT_CHANNEL_NAME, []).append(channel)
            elif "." not in channel:
                grouped_channels[channel] = [channel, channel, channel]
            else:
                key = channel.split(".").pop(0)
                grouped_channels.setdefault(key, []).append(channel)

        # Sort each list in the dictionary
        self._channel_targets = {
            k: sorted(v, key=lambda x: _RGB_ORDER.get(get_channel_suffix(x), 99))
            for k, v in grouped_channels.items()
        }
        self.channelsUpdated.emit(list(self._channel_targets))

    def set_image_sequence(self, seq: ImageSequence) -> None:
        """Set image sequence to play on viewer.

        Args:
            seq: Image sequence to play.

        """
        self._channel_cache = {}
        self.rangeChanged.emit(int(float(str(seq.start()))), int(float(str(seq.end()))))

        self._current_index = 0

        # List of image file paths that will be displayed while playing the sequence.
        self._file_paths_to_load = seq.get_paths()

        # Reset caches.
        self._cached_channels = {}

        self._start_frame = int(float(str(seq.start())))
        self._end_frame = int(float(str(seq.end())))

        self.load_image_from_disk(self._file_paths_to_load[self._current_index], True)
        # Update OCIO
        self.update_ocio_processor()
        self.setup_frame_for_rendering()
        self.reset_zoom_and_pan()

    def set_view_channel(self, text: str) -> None:
        """Set channel to view in player e.g rgba or specular...

        Args:
            text: Channel name.

        """
        self.view_channel = text
        frames = list(self._cached_channels.get(self.view_channel, {}))
        self.cachedChannelFrames.emit(frames)
        self.setup_frame_for_rendering()

    def update_ocio_processor(self):
        """Update OCIO processor and rebuild shaders."""
        try:
            if LOG.level == logging.DEBUG:
                print("Updating OCIO processor...")
                print(f"  Input CS: {self._ocio_input_cs}")
                print(f"  Display: {self._ocio_display}")
                print(f"  View: {self.ocio_view_color_space}")

            # Build viewing pipeline
            exposure_tr = ocio.ExposureContrastTransform(
                exposure=self._ocio_exposure, dynamicExposure=True
            )
            channel_tr = ocio.MatrixTransform.View(
                channelHot=self._ocio_channel_hot, lumaCoef=config.getDefaultLumaCoefs()
            )
            display_tr = ocio.DisplayViewTransform()
            display_tr.setSrc(self._ocio_input_cs)
            display_tr.setDisplay(self._ocio_display)
            display_tr.setView(self.ocio_view_color_space)

            gamma_tr = ocio.ExposureContrastTransform(
                gamma=self._ocio_gamma, pivot=1.0, dynamicGamma=True
            )

            pipeline = ocio.LegacyViewingPipeline()
            pipeline.setLinearCC(exposure_tr)
            pipeline.setChannelView(channel_tr)
            pipeline.setDisplayViewTransform(display_tr)
            pipeline.setDisplayCC(gamma_tr)

            proc = pipeline.getProcessor(config)
            LOG.debug(f"Processor created successfully")

            # Extract GPU shader
            self._ocio_shader_desc = ocio.GpuShaderDesc.CreateShaderDesc(
                language=ocio.GPU_LANGUAGE_GLSL_4_0
            )
            gpu_proc = proc.getDefaultGPUProcessor()
            gpu_proc.extractGpuShaderInfo(self._ocio_shader_desc)
            LOG.debug(f"GPU shader extracted")
            self._allocate_ocio_textures()

            # Rebuild shader program
            if not self._build_shader_program():
                LOG.error(f"Failed to rebuild shader program")
                return

            # Set initial dynamic properties
            self._update_dynamic_property(
                ocio.DYNAMIC_PROPERTY_EXPOSURE, self._ocio_exposure
            )
            self._update_dynamic_property(ocio.DYNAMIC_PROPERTY_GAMMA, self._ocio_gamma)
            LOG.debug(f"Dynamic properties set")

            self.update()

        except Exception as e:
            LOG.exception(f"Failed to update OCIO processor: {e}")

    def _allocate_ocio_textures(self):
        """Allocate textures needed by OCIO."""
        # Delete old textures
        for tex, _, _, _, _ in self._ocio_tex_ids:
            GL.glDeleteTextures([tex])
        self._ocio_tex_ids.clear()

        tex_index = 1  # Start after image texture

        # 3D textures
        for tex_info in self._ocio_shader_desc.get3DTextures():
            # DEBUG: Check LUT data
            tex = GL.glGenTextures(1)
            GL.glActiveTexture(GL.GL_TEXTURE0 + tex_index)
            GL.glBindTexture(GL.GL_TEXTURE_3D, tex)
            GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(
                GL.GL_TEXTURE_3D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE
            )
            GL.glTexParameteri(
                GL.GL_TEXTURE_3D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE
            )
            GL.glTexParameteri(
                GL.GL_TEXTURE_3D, GL.GL_TEXTURE_WRAP_R, GL.GL_CLAMP_TO_EDGE
            )

            GL.glTexImage3D(
                GL.GL_TEXTURE_3D,
                0,
                GL.GL_RGB32F,
                tex_info.edgeLen,
                tex_info.edgeLen,
                tex_info.edgeLen,
                0,
                GL.GL_RGB,
                GL.GL_FLOAT,
                tex_info.getValues(),
            )

            self._ocio_tex_ids.append(
                (
                    tex,
                    tex_info.textureName,
                    tex_info.samplerName,
                    GL.GL_TEXTURE_3D,
                    tex_index,
                )
            )
            tex_index += 1

        # 1D/2D textures
        for tex_info in self._ocio_shader_desc.getTextures():
            tex = GL.glGenTextures(1)
            GL.glActiveTexture(GL.GL_TEXTURE0 + tex_index)

            internal_fmt = GL.GL_RGB32F
            fmt = GL.GL_RGB
            if tex_info.channel == self._ocio_shader_desc.TEXTURE_RED_CHANNEL:
                internal_fmt = GL.GL_R32F
                fmt = GL.GL_RED

            if tex_info.dimensions == self._ocio_shader_desc.TEXTURE_2D:
                GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
                GL.glTexParameteri(
                    GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR
                )
                GL.glTexParameteri(
                    GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR
                )
                GL.glTexImage2D(
                    GL.GL_TEXTURE_2D,
                    0,
                    internal_fmt,
                    tex_info.width,
                    tex_info.height,
                    0,
                    fmt,
                    GL.GL_FLOAT,
                    tex_info.getValues(),
                )
                tex_type = GL.GL_TEXTURE_2D
            else:
                GL.glBindTexture(GL.GL_TEXTURE_1D, tex)
                GL.glTexParameteri(
                    GL.GL_TEXTURE_1D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR
                )
                GL.glTexParameteri(
                    GL.GL_TEXTURE_1D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR
                )
                GL.glTexImage1D(
                    GL.GL_TEXTURE_1D,
                    0,
                    internal_fmt,
                    tex_info.width,
                    0,
                    fmt,
                    GL.GL_FLOAT,
                    tex_info.getValues(),
                )
                tex_type = GL.GL_TEXTURE_1D

            self._ocio_tex_ids.append(
                (
                    tex,
                    tex_info.textureName,
                    tex_info.samplerName,
                    tex_type,
                    tex_index,
                )
            )
            tex_index += 1

    def _update_dynamic_property(self, prop_type, value):
        """Update OCIO dynamic property."""
        try:
            if self._ocio_shader_desc and self._ocio_shader_desc.hasDynamicProperty(
                prop_type
            ):
                dyn_prop = self._ocio_shader_desc.getDynamicProperty(prop_type)
                dyn_prop.setDouble(value)
        except Exception as e:
            LOG.error(f"Failed to update dynamic property: {e}")

    def update_exposure(self, value: float) -> None:
        """Update exposure.

        Args:
            value: Exposure value to view.

        """
        self._ocio_exposure = value
        self._update_dynamic_property(ocio.DYNAMIC_PROPERTY_EXPOSURE, value)
        self.setup_frame_for_rendering()

    def update_gamma(self, value: float) -> None:
        """Update gamma value.

        Args:
            value: Gamma correction to apply to view.

        """
        self._ocio_gamma = 1.0 / max(0.001, value)
        self._update_dynamic_property(ocio.DYNAMIC_PROPERTY_GAMMA, self._ocio_gamma)
        self.setup_frame_for_rendering()

    def resizeGL(self, w: int, h: int) -> None:
        """Handle resize."""
        GL.glViewport(0, 0, w, h)

    def paintGL(self):
        """Render image."""
        try:
            GL.glClear(GL.GL_COLOR_BUFFER_BIT)
            if not self.gl_initialized or not self.shader_program:
                return

            GL.glUseProgram(self.shader_program)

            # Calculate MVP matrix
            mvp = self._calculate_mvp()
            mvp_loc = GL.glGetUniformLocation(self.shader_program, "mvpMat")
            if mvp_loc < 0:
                LOG.warning("mvpMat uniform not found")
            GL.glUniformMatrix4fv(mvp_loc, 1, GL.GL_FALSE, mvp)

            # Bind image texture
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.image_tex)
            img_tex_loc = GL.glGetUniformLocation(self.shader_program, "imageTex")
            if img_tex_loc < 0:
                LOG.warning("imageTex uniform not found")
            GL.glUniform1i(img_tex_loc, 0)

            # Bind OCIO textures
            for tex, tex_name, sampler_name, tex_type, tex_idx in self._ocio_tex_ids:
                GL.glActiveTexture(GL.GL_TEXTURE0 + tex_idx)
                GL.glBindTexture(tex_type, tex)
                sampler_loc = GL.glGetUniformLocation(self.shader_program, sampler_name)
                if sampler_loc >= 0:
                    GL.glUniform1i(sampler_loc, tex_idx)

            # Bind OCIO uniforms
            if self._ocio_shader_desc:
                for name, uniform_data in self._ocio_shader_desc.getUniforms():
                    if uniform_data.type == ocio.UNIFORM_DOUBLE:
                        uid = GL.glGetUniformLocation(self.shader_program, name)
                        if uid >= 0:
                            GL.glUniform1f(uid, uniform_data.getDouble())

            # Draw
            GL.glBindVertexArray(self.vao)
            GL.glDrawElements(GL.GL_TRIANGLES, 6, GL.GL_UNSIGNED_INT, None)
            GL.glBindVertexArray(0)

            GL.glFlush()

            self._fps_frame_count += 1
            elapsed_ms = self._fps_timer.elapsed()

            if elapsed_ms >= 500:  # update twice per second
                self._fps_value = self._fps_frame_count * 1000.0 / elapsed_ms
                self._fps_frame_count = 0
                self._fps_timer.restart()
            if self._timer.isActive():
                painter = QtGui.QPainter(self)
                painter.setRenderHint(QtGui.QPainter.TextAntialiasing)
                # Text
                painter.setPen(QtGui.QColor(0, 255, 0))
                painter.drawText(15, 28, f"{self._fps_value:.1f} FPS")
                painter.end()

        except Exception as e:
            LOG.critical(f"paintGL failed: {e}")

    def wheelEvent(self, event):
        """Handle _zoom."""
        delta = event.angleDelta().y()
        factor = 1.05 if delta > 0 else 0.95
        self._zoom *= factor
        self.update()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent, /) -> None:
        current_pos = QtGui.QCursor.pos()
        delta = current_pos - self._last_mouse_position
        self._last_mouse_position = current_pos
        if event.buttons() == QtCore.Qt.MiddleButton:
            self._pan_x += delta.x()
            self._pan_y -= delta.y()
            self.update()
            event.accept()
            return
        super(GLImagePlane, self).mouseMoveEvent(event)

    def _calculate_mvp(self) -> np.ndarray:
        """Calculate model-view-projection matrix.

        Returns:
             Model view projection.

        """
        # Fit image to window
        win_w, win_h = float(self.width()), float(self.height())
        img_w, img_h = float(self.image_width), float(self.image_height)

        # Calculate scale to fit image in window
        scale_x = win_w / img_w if img_w > 0 else 1.0
        scale_y = win_h / img_h if img_h > 0 else 1.0
        scale = min(scale_x, scale_y) * self._zoom

        # Build orthographic projection centered at origin
        l, r = -win_w / 2.0, win_w / 2.0
        b, t = -win_h / 2.0, win_h / 2.0
        n, f = -1.0, 1.0

        proj = np.array(
            [
                [2.0 / (r - l), 0, 0, -(r + l) / (r - l)],
                [0, 2.0 / (t - b), 0, -(t + b) / (t - b)],
                [0, 0, -2.0 / (f - n), -(f + n) / (f - n)],
                [0, 0, 0, 1.0],
            ],
            dtype=np.float32,
        )

        # Model-view: scale image for OpenGL coordinate system
        scaled_w = img_w * scale
        scaled_h = img_h * scale

        model_view = np.array(
            [
                [scaled_w, 0, 0, self._pan_x],
                [0, scaled_h, 0, self._pan_y],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )

        # Combined MVP (matrix multiplication: proj @ model_view)
        mvp = proj @ model_view

        # OpenGL expects column-major, so transpose
        return mvp.T.flatten()

    def jump_to_end(self) -> None:
        """Callback to jump to end frame."""
        self.set_frame(self._end_frame)

    def jump_to_start(self) -> None:
        """Callback to jump to start frame."""
        self.set_frame(self._start_frame)

    def set_frame(self, value: int) -> None:
        """Set current frame to display.

        Args:
            value: Frame number to display.

        """
        index = value - self._start_frame
        if index == self._current_index:
            return
        self._current_index = index
        self.setup_frame_for_rendering()

    def _start_timer(self, callback: Callable[[None], None]) -> None:
        self._timer.stop()
        self._timer = QtCore.QTimer()
        self._timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
        self._timer.timeout.connect(callback)
        self._timer.start(int(1000 / self._fps))

    def toggle_forward(self) -> None:
        self._play_state = PlayMode.Forward
        if not self._file_paths_to_load:
            return

        if self._play_forward:
            self._timer.stop()
        else:
            self._start_timer(self._next_frame_callback)
        self._play_forward = not self._play_forward

        self._play_backward = False
        self.playForward.emit(self._play_forward)
        self.update()

    def toggle_backward(self) -> None:
        self._play_state = PlayMode.Backward

        if not self._file_paths_to_load:
            return
        if self._play_backward:
            self._timer.stop()
        else:
            self._start_timer(self._previous_frame_callback)
        self._play_backward = not self._play_backward

        self._play_forward = False
        self.playBackward.emit(self._play_backward)
        self.update()

    def _next_frame_callback(self):
        self._current_index = (self._current_index + 1) % len(self._file_paths_to_load)
        self.setup_frame_for_rendering()

    def _previous_frame_callback(self):
        self._current_index = (self._current_index - 1) % len(self._file_paths_to_load)
        self.setup_frame_for_rendering()

    def set_fps(self, fps: float) -> None:
        self._fps = fps
        if self._timer.isActive():
            self._timer.stop()
            self._timer.start(int(1000 / self._fps))

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        if event.mimeData().urls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        for url in event.mimeData().urls():
            seq = ImageSequence(url.toLocalFile())
            seq.find_frames_on_disk()
            self.set_image_sequence(seq)
        event.accept()
        return


class ImageSequenceView(QtWidgets.QWidget):
    def __init__(self, fps: int = 24, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent=parent)

        # Create widgets
        self.viewport = GLImagePlane(fps=fps, parent=self)

        self.view_combobox = QtWidgets.QComboBox()
        self.channel_combo = QtWidgets.QComboBox()

        self.exposure_spin = QtWidgets.QDoubleSpinBox()
        self.exposure_spin.setRange(-100.0, 100.0)
        self.exposure_spin.setValue(0.0)
        self.exposure_spin.setSingleStep(0.1)

        self.gamma_spin = QtWidgets.QDoubleSpinBox()
        self.gamma_spin.setRange(0.01, 4.0)
        self.gamma_spin.setValue(1.0)
        self.gamma_spin.setSingleStep(0.1)

        # Layout
        controls_layout = QtWidgets.QHBoxLayout()
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(0)
        controls_layout.addWidget(QtWidgets.QLabel("View:"))
        controls_layout.addWidget(self.view_combobox)
        controls_layout.addWidget(QtWidgets.QLabel("Channel:"))
        controls_layout.addWidget(self.channel_combo)
        controls_layout.addStretch()
        controls_layout.addWidget(QtWidgets.QLabel("Exposure:"))
        controls_layout.addWidget(self.exposure_spin)
        controls_layout.addWidget(QtWidgets.QLabel("Gamma:"))
        controls_layout.addWidget(self.gamma_spin)

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addLayout(controls_layout)
        layout.addWidget(self.viewport, 1)
        self.setLayout(layout)

        # Populate views
        self._populate_views()

        # Connect signals
        self.view_combobox.currentTextChanged.connect(self._on_view_changed)
        self.exposure_spin.valueChanged.connect(self.viewport.update_exposure)
        self.gamma_spin.valueChanged.connect(self.viewport.update_gamma)
        self.viewport.channelsUpdated.connect(self._populate_channel_names_callback)
        self.playForward = self.viewport.playForward
        self.playBackward = self.viewport.playBackward

        self.channel_combo.activated.connect(self._channel_changed_callback)

        self.set_image_sequence = self.viewport.set_image_sequence
        self.set_frame = self.viewport.set_frame
        self.toggle_forward = self.viewport.toggle_forward
        self.toggle_backward = self.viewport.toggle_backward
        self.next_frame_callback = self.viewport._next_frame_callback
        self.previous_frame_callback = self.viewport._previous_frame_callback
        self.jumpToEnd = self.viewport.jump_to_end
        self.jumpToStart = self.viewport.jump_to_start
        self.set_fps = self.viewport.set_fps

        # signals
        self.frameChanged = self.viewport.frameChanged
        self.rangeChanged = self.viewport.rangeChanged
        self.cachedFrame = self.viewport.cachedFrame
        self.cachedChannelFrames = self.viewport.cachedChannelFrames

    def _channel_changed_callback(self, index: int) -> None:
        """Callback to execute when channel name is changes.

        Args:
            index: New index that is set.

        """
        if index < 0:
            return
        self.viewport.set_view_channel(self.channel_combo.itemText(index))

    def _populate_channel_names_callback(self, channels) -> None:
        self.channel_combo.clear()
        channels.sort(key=lambda x: x.lower())
        self.channel_combo.addItems(
            sorted(channels, key=lambda c: {DEFAULT_CHANNEL_NAME: 0}.get(c, 99))
        )
        self.channel_combo.setCurrentText(self.viewport.view_channel)

    def _populate_views(self) -> None:
        """Populate view dropdown."""
        display = config.getDefaultDisplay()
        self.view_combobox.addItems(list(config.getViews(display)))

    def _on_view_changed(self, view):
        """Handle view change."""
        self.viewport.ocio_view_color_space = view

        self.viewport.update_ocio_processor()

    def set_view_colorspace(self, name: str) -> None:
        self.view_combobox.setCurrentText(name)


def __test() -> None:
    """Test function."""
    fmt = QSurfaceFormat()
    fmt.setVersion(4, 1)
    fmt.setProfile(QSurfaceFormat.CoreProfile)
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QtWidgets.QApplication(sys.argv)
    viewer = ImageSequenceView()
    viewer.resize(1280, 720)
    viewer.show()

    src = os.path.join(os.path.dirname(__file__), "../../images/plate.1001.exr")
    seq = ImageSequence(src)
    seq.find_frames_on_disk()

    viewer.set_image_sequence(seq)

    sys.exit(app.exec())


if __name__ == "__main__":
    __test()
