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
import os

DEFAULT_START_FRAME = 1001
DEFAULT_END_FRAME = 1100
FPS_RATES = (
    "60",
    "50",
    "48",
    "30",
    "25",
    "24",
    "23.98",
    "12",
)


_IS_DEBUG = os.getenv("IMAGE_SEQUENCE_VIEW_DEBUG", "0") == "1"

logging.basicConfig(
    level=logging.DEBUG if _IS_DEBUG else logging.INFO,
    format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

LOG = logging.getLogger("image_sequence_view")
