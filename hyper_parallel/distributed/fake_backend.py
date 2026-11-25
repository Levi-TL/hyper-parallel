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
"""Single-process fake distributed backend support.

The fake backend keeps compute real (kernels, allocator, data, weights) while
replacing collectives with local no-ops. It lets one physical process stand
for one logical rank of an arbitrarily sized logical world:

* ``WORLD_SIZE`` is a logical value injected via ``HP_FAKE_WORLD_SIZE``;
* ``RANK`` / ``LOCAL_RANK`` are pinned to 0 (single process);
* collectives dispatched through the PyTorch fake process group return
  immediately-completed work and create no communicator.

The fake backend skips env rendezvous, so ``rank`` and ``world_size`` must be
passed explicitly to ``dist.init_process_group`` — the environment variables
alone are NOT enough. Omitting them yields a PG that reports the torch
argument defaults (rank=-1, world_size=-1) while ``dist.is_initialized()``
still returns True, which surfaces downstream as
"world_size must be a positive integer, got -1". Callers must follow
``dist.init_process_group(backend="fake", rank=0, world_size=<logical>)``
(the pattern used by Torchtitan's ``init_fake_mode`` and by PyTorch's own
fake-backend tests).

This module never mutates global backend registration at import time: the
NPU registration helper is only called on the ``backend == "fake"`` path so a
real HCCL run is unaffected (see PyTorch issue #159945 for the failure mode
of eagerly registering fake over a real device backend).
"""

import logging
import os

logger = logging.getLogger(__name__)

__all__ = [
    "HP_FAKE_WORLD_SIZE",
    "setup_fake_single_process_environment",
    "ensure_fake_backend_supports_npu",
]

HP_FAKE_WORLD_SIZE = "HP_FAKE_WORLD_SIZE"
"""Environment variable carrying the logical world size of a fake run."""

_NPU_PATCH_APPLIED = False


def setup_fake_single_process_environment() -> int:
    """Pin this process to logical rank 0 of a logical world.

    When ``HP_FAKE_WORLD_SIZE`` is set, ``WORLD_SIZE`` is overwritten with the
    logical value and ``RANK`` / ``LOCAL_RANK`` are pinned to 0: the fake
    backend is single-process by construction, so a stale launcher
    environment must not leak a physical rank into the logical world.

    Returns:
        The logical world size the fake process group will report.
    """
    raw = os.environ.get(HP_FAKE_WORLD_SIZE)
    if raw is None:
        # No explicit injection: honour whatever WORLD_SIZE the launcher set
        # (1 when the script runs bare), which keeps plain single-process use
        # and torchrun-launched fake debugging both working.
        return int(os.environ.get("WORLD_SIZE", "1"))

    logical_world_size = int(raw)
    if logical_world_size <= 0:
        raise ValueError(
            f"{HP_FAKE_WORLD_SIZE} must be a positive integer, got {raw!r}"
        )
    os.environ["WORLD_SIZE"] = str(logical_world_size)
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29513")
    return logical_world_size


def ensure_fake_backend_supports_npu() -> bool:
    """Make the fake process group usable with NPU tensors.

    PyTorch registers the fake backend for ``cpu/cuda/hpu/xpu`` only, so NPU
    tensors cannot dispatch collectives to a fake process group until ``npu``
    is added to the device-backend map. The patch is idempotent and scoped to
    fake runs; a real HCCL run never imports this path, and an existing
    ``npu`` mapping (e.g. registered by torch_npu) is never overwritten.

    Returns:
        True when ``npu`` resolves to the fake backend after the call.
    """
    global _NPU_PATCH_APPLIED
    if _NPU_PATCH_APPLIED:
        return True

    try:
        import torch.distributed as dist
    except ImportError:
        logger.warning(
            "torch is not installed; the NPU fake-backend registration patch "
            "is skipped."
        )
        return False

    backend_cls = getattr(dist, "Backend", None)
    device_map = getattr(backend_cls, "default_device_backend_map", None)
    if device_map is None:
        # Unknown layout: let the explicit init_process_group("fake") call
        # surface whatever error the installed torch raises.
        logger.warning(
            "This torch build exposes no Backend.default_device_backend_map; "
            "skipping the NPU fake-backend registration patch."
        )
        return False

    if "npu" in device_map:
        # torch_npu (or a prior patch) already mapped npu. Overwriting could
        # break real HCCL init order (pytorch/pytorch#159945), so only align
        # it when the existing mapping is already fake.
        _NPU_PATCH_APPLIED = device_map.get("npu") == "fake"
        return _NPU_PATCH_APPLIED

    # Preferred path: a full registration that carries the fake creator, so
    # dispatch (not just name resolution) recognises npu tensors.
    registered = _register_fake_for_npu(backend_cls)
    if not registered:
        # Fallback: amend the plain name-resolution map. Explicit
        # init_process_group("fake") and subgroups inheriting the default
        # backend resolve through this map when a device is involved.
        device_map["npu"] = "fake"
        registered = True
        logger.info("Mapped device 'npu' to the fake backend via device_backend_map.")

    _NPU_PATCH_APPLIED = registered
    return registered


def _register_fake_for_npu(backend_cls) -> bool:
    """Re-register the fake backend including ``npu`` in its device list.

    Registration details differ across torch versions, so this walks the
    known module/creator names and treats any mismatch as "not available"
    rather than an error.
    """
    import importlib

    for module_name in ("torch.distributed.fake", "torch.distributed.fake_pg"):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        creator = _find_fake_creator(module)
        if creator is None:
            continue
        try:
            backend_cls.register_backend(
                "fake",
                creator,
                devices=["cpu", "cuda", "hpu", "xpu", "npu"],
            )
            logger.info(
                "Registered the fake backend for npu via %s.", module_name
            )
            return True
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning(
                "register_backend('fake', ..., devices=[..., 'npu']) failed "
                "(torch=%s, module=%s): %s. Falling back to the device map.",
                getattr(importlib.import_module("torch"), "__version__", "?"),
                module_name,
                exc,
            )
            return False
    return False


def _find_fake_creator(module):
    """Return the fake backend creator callable from a candidate module."""
    for attr in (
        "_create_fake_process_group",
        "create_fake_process_group",
        "_create_process_group_fake",
        "_create_fake_backend",
    ):
        candidate = getattr(module, attr, None)
        if callable(candidate):
            return candidate
    return None
