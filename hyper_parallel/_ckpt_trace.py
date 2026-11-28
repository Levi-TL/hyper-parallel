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
"""Opt-in per-stage tracing for the checkpoint save path.

A save is a sequence of collectives: the optimizer broadcasts its state to the
replicate-group peers, then the checkpointer synchronizes on a barrier, then DCP
gathers plans and shards. When one of those collectives is issued by only some
ranks --- or with a different buffer on each rank --- every participant waits
forever, and the failure that surfaces is a device-sync timeout (``507048`` on
Ascend) reported by whichever rank happened to reach the *next* barrier. That
traceback names the victim, not the culprit: the rank that stalled sits in a
different collective and its stack is never printed by the first timeout.

``HP_CKPT_TRACE=1`` makes every rank log the stage it enters, so the last line
of each rank's log names the collective that never completed, and the ranks'
lines side by side show which stage only one of them reached.

The trace is off by default and costs one environment lookup per call site.
"""

__all__ = ["TRACE_ENV", "ckpt_trace", "ckpt_trace_enabled"]

import logging
import os
import time
from typing import Any

import torch.distributed as dist

#: Environment variable that switches the trace on.
TRACE_ENV = "HP_CKPT_TRACE"

logger = logging.getLogger(__name__)


def ckpt_trace_enabled() -> bool:
    """Report whether checkpoint stage tracing is switched on."""
    return bool(os.environ.get(TRACE_ENV))


def ckpt_trace(stage: str, **fields: Any) -> None:
    """Log ``stage`` for this rank, when tracing is enabled.

    Warning level on purpose: the interesting runs are the ones whose logger is
    configured to drop ``info``, and a hung save is exactly when that happens.

    Args:
        stage: Short dotted stage name, e.g. ``"save.pre_barrier"``.
        **fields: Extra ``key=value`` pairs worth correlating across ranks.
    """
    if not ckpt_trace_enabled():
        return

    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    detail = " ".join(f"{key}={value}" for key, value in fields.items())
    logger.warning(
        "[ckpt-trace] rank=%s stage=%s t=%.3f %s",
        rank,
        stage,
        time.monotonic(),
        detail,
    )
