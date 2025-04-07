# Copyright 2023 The Magenta Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Pulls in all magenta libraries that are in the public API.."""

import src.util.magenta.common.beam_search
import src.util.magenta.common.concurrency
import src.util.magenta.common.nade
import src.util.magenta.common.sequence_example_lib
import src.util.magenta.common.state_util
import src.util.magenta.common.testing_lib
import src.util.magenta.common.tf_utils
import src.util.magenta.pipelines.dag_pipeline
import src.util.magenta.pipelines.drum_pipelines
import src.util.magenta.pipelines.lead_sheet_pipelines
import src.util.magenta.pipelines.melody_pipelines
import src.util.magenta.pipelines.note_sequence_pipelines
import src.util.magenta.pipelines.pipeline
import src.util.magenta.pipelines.pipelines_common
import src.util.magenta.pipelines.statistics
import src.util.magenta.version
from src.util.magenta.version import __version__
