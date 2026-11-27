# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Process-role decisions that must survive a fake single-process run.

"Global rank 0 owns this side effect" is the usual way to express "exactly one
process does this": write configs, emit logs, build a dataset index, download a
shared file. A fake run keeps that decision *and* is a single process, so when
it stands for a non-zero logical rank (``HP_FAKE_RANK``) the rank test turns
false and nobody performs the duty - no config on disk, no log output, no index.

``is_main_process`` names the decision instead of the rank, so the fake case can
answer "yes" without every call site knowing about the fake backend.

This module lives at the package root on purpose. The question is asked at
import time by both ``hyper_parallel.data`` and ``hyper_parallel.trainer``, and
``hyper_parallel.distributed`` pulls in the whole sharding subsystem (whose
``encoder_parallel`` imports ``hyper_parallel.data`` back), so hosting it there
would create an import cycle. It depends on nothing but torch.
"""

__all__ = ["is_fake_backend_active", "is_main_process"]


def is_fake_backend_active() -> bool:
    """Report whether the default process group runs the fake backend.

    Returns:
        True when the default PG exists and is the fake backend.
    """
    try:
        import torch.distributed as dist
    except ImportError:
        return False

    if not dist.is_initialized():
        return False
    try:
        return str(dist.get_backend()).lower() == "fake"
    except (RuntimeError, ValueError):
        # Some torch builds raise when the default group is not a real PG.
        return False


def is_main_process() -> bool:
    """Report whether this process owns the global side effects.

    True when distributed is not initialized, when this is global rank 0, or
    when the fake backend is active. The fake case is unconditional on purpose:
    a fake run has exactly one process, so if it declined the duty no process
    would perform it.

    Returns:
        bool: True if this process should write configs, logs and shared files.
    """
    if is_fake_backend_active():
        return True

    try:
        import torch.distributed as dist
    except ImportError:
        return True

    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0
