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
* ``RANK`` is the logical rank this process stands for, injected via
  ``HP_FAKE_RANK`` (default 0). PyTorch's fake backend is built for exactly
  this ("run a single rank with a fake process group without needing multiple
  processes (simulates per-rank behavior)"): ``_create_fake_pg`` forwards the
  requested ``group_rank``/``group_size`` unchanged, so any rank is valid.
  A non-zero logical rank makes the shard ownership, mesh coordinate and
  checkpoint offsets of that rank real, which is what lets a single fake run
  save and reload one rank's shards;
* ``LOCAL_RANK`` names the device this process owns, never the logical rank:
  pinning ``npu:<logical rank>`` would fail on a machine with fewer cards.
  ``HP_FAKE_DEVICE_ID`` (default 0) selects that device, so a run can stand
  for logical rank 7 while occupying card 3. The index follows torch's device
  numbering, i.e. it indexes the *visible* list, so it means "card 3" only
  when all cards are visible. Binding a run to a physical card is still
  ``ASCEND_RT_VISIBLE_DEVICES``; the two knobs compose and a mismatch is
  rejected instead of silently landing on another card;
* collectives dispatched through the PyTorch fake process group return
  immediately-completed work and create no communicator.

Two properties of the PyTorch fake backend are not obvious and are handled
here:

1. ``rank`` and ``world_size`` must be passed explicitly. The fake backend
   skips env rendezvous, so ``dist.init_process_group(backend="fake")`` without
   them yields a PG that reports the torch argument defaults
   (rank=-1, world_size=-1) while ``dist.is_initialized()`` still returns True,
   which surfaces downstream as "world_size must be a positive integer, got -1".
   Callers must use ``init_process_group("fake", rank=0, world_size=<logical>)``
   (the pattern used by Torchtitan's ``init_fake_mode`` and by PyTorch's own
   fake-backend tests).

2. A process group resolves its backend per collective from the device type of
   the input tensors (``group._get_backend(tensor.device)``). The device set a
   PG binds is read from ``Backend.backend_capability["fake"]`` every time a PG
   is created, and torch registers ``cpu/cuda/hpu`` (plus ``xpu`` from 2.10) -
   never ``npu``. A fake PG therefore fails on the first NPU collective with
   "No backend type associated with device type npu" unless ``npu`` is added to
   that capability list BEFORE ``init_process_group``. ``new_group`` calls
   without an explicit backend re-read the same list, so one append covers the
   default group and every subgroup.

``Backend.default_device_backend_map`` (device -> default backend NAME, used
only when ``init_process_group`` is called without a backend) is deliberately
left untouched: overwriting its ``npu`` entry is what broke real HCCL runs in
pytorch/pytorch#159945.
"""

import logging
import os

from hyper_parallel.process_role import is_fake_backend_active  # noqa: F401  (re-exported)

logger = logging.getLogger(__name__)

__all__ = [
    "HP_FAKE_DEVICE_ID",
    "HP_FAKE_RANK",
    "HP_FAKE_WORLD_SIZE",
    "setup_fake_single_process_environment",
    "resolve_fake_device_id",
    "is_fake_backend_active",
    "ensure_fake_backend_supports_device",
    "verify_fake_backend_bound",
]

HP_FAKE_WORLD_SIZE = "HP_FAKE_WORLD_SIZE"
"""Environment variable carrying the logical world size of a fake run."""

HP_FAKE_RANK = "HP_FAKE_RANK"
"""Environment variable naming the logical rank a fake run stands for.

Defaults to 0. Must satisfy ``0 <= rank < HP_FAKE_WORLD_SIZE``.
"""

HP_FAKE_DEVICE_ID = "HP_FAKE_DEVICE_ID"
"""Environment variable naming the device a fake run occupies.

Defaults to 0. Independent of ``HP_FAKE_RANK``: the logical rank decides which
shard/mesh coordinate the process owns, the device id decides which card the
work actually runs on. The value follows torch's device numbering, i.e. it
indexes the visible device list (``ASCEND_RT_VISIBLE_DEVICES`` /
``CUDA_VISIBLE_DEVICES`` remap it), and is validated against that list.
"""

_PATCHED_DEVICES: set[str] = set()

# Modules that register the "fake" backend, one per torch layout. Released
# torch (<= 2.11) registers it on import of the testing module; newer torch
# registers it through torch.distributed._backends.fake.
_FAKE_BACKEND_MODULES = (
    "torch.testing._internal.distributed.fake_pg",
    "torch.distributed._backends.fake",
)

_NO_BACKEND_MESSAGE = "No backend type associated with device type"


def resolve_fake_device_id(device_count: int | None = None) -> int:
    """Resolve the device a fake run occupies from ``HP_FAKE_DEVICE_ID``.

    The device id is deliberately independent of the logical rank: a fake run
    may stand for rank 7 of an 8-rank world (shard ownership, mesh coordinate,
    checkpoint offsets) while occupying the only card it actually has.

    Args:
        device_count: Number of devices visible to this process. When given,
            an id outside ``[0, device_count)`` is rejected with the fix, since
            the process would otherwise bind a device that does not exist.

    Returns:
        The device index this process should bind.

    Raises:
        ValueError: The value is not a non-negative integer, or it is outside
            the visible device range.
    """
    raw = os.environ.get(HP_FAKE_DEVICE_ID, "0")
    try:
        device_id = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{HP_FAKE_DEVICE_ID} must be a non-negative integer, got {raw!r}"
        ) from exc
    if device_id < 0:
        raise ValueError(
            f"{HP_FAKE_DEVICE_ID}={raw!r} must be a non-negative integer."
        )
    if device_count is not None and device_id >= device_count:
        raise ValueError(
            f"{HP_FAKE_DEVICE_ID}={raw!r} is outside the {device_count} device(s) "
            "visible to this process. The id indexes the visible list, so a "
            "narrowed ASCEND_RT_VISIBLE_DEVICES/CUDA_VISIBLE_DEVICES remaps it: "
            "either point that variable at the card you want, or widen it and "
            "keep this id."
        )
    return device_id


def setup_fake_single_process_environment() -> int:
    """Pin this process to one logical rank of a logical world.

    When ``HP_FAKE_WORLD_SIZE`` is set, ``WORLD_SIZE`` is overwritten with the
    logical value and ``RANK`` is set to ``HP_FAKE_RANK`` (default 0), while
    ``LOCAL_RANK`` is set to the device this run owns (``HP_FAKE_DEVICE_ID``,
    default 0) instead of the logical rank. The fake backend is single-process
    by construction, so a stale launcher environment must not leak a physical
    rank into the logical world.

    Returns:
        The logical world size the fake process group will report.

    Raises:
        ValueError: The world size is not positive, the rank is out of range,
            a rank was requested without a matching world size, or the device
            id is not a non-negative integer.
    """
    raw = os.environ.get(HP_FAKE_WORLD_SIZE)
    if raw is None:
        # No explicit injection: honour whatever WORLD_SIZE the launcher set
        # (1 when the script runs bare), which keeps plain single-process use
        # and torchrun-launched fake debugging both working. A rank without a
        # world size is ambiguous - the rank range cannot be validated - so
        # reject it instead of silently pinning rank 0.
        requested_rank = os.environ.get(HP_FAKE_RANK)
        if requested_rank not in (None, "", "0"):
            raise ValueError(
                f"{HP_FAKE_RANK}={requested_rank!r} requires "
                f"{HP_FAKE_WORLD_SIZE} to be set as well."
            )
        return int(os.environ.get("WORLD_SIZE", "1"))

    logical_world_size = int(raw)
    if logical_world_size <= 0:
        raise ValueError(
            f"{HP_FAKE_WORLD_SIZE} must be a positive integer, got {raw!r}"
        )

    rank_raw = os.environ.get(HP_FAKE_RANK, "0")
    logical_rank = int(rank_raw)
    if not 0 <= logical_rank < logical_world_size:
        raise ValueError(
            f"{HP_FAKE_RANK}={rank_raw!r} is out of range for a logical world "
            f"of size {logical_world_size}."
        )

    os.environ["WORLD_SIZE"] = str(logical_world_size)
    os.environ["RANK"] = str(logical_rank)
    # LOCAL_RANK selects the device only; it must never follow the logical rank.
    device_id = resolve_fake_device_id()
    os.environ["LOCAL_RANK"] = str(device_id)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29513")
    logger.info(
        "Fake run stands for logical rank %s of %s on device %s.",
        logical_rank,
        logical_world_size,
        device_id,
    )
    return logical_world_size


def _fake_capability(dist) -> list[str] | None:
    """Return the mutable device list of the ``fake`` backend, if registered."""
    capability = getattr(dist.Backend, "backend_capability", None)
    if not isinstance(capability, dict):
        return None
    for name, devices in capability.items():
        if str(name).lower() == "fake" and isinstance(devices, list):
            return devices
    return None


def _import_fake_backend_module(dist) -> str | None:
    """Import whichever module registers ``fake`` in this torch build."""
    import importlib

    for module_name in _FAKE_BACKEND_MODULES:
        try:
            importlib.import_module(module_name)
        except Exception:  # pylint: disable=broad-exception-caught
            # Not this layout (or an incompatible one): try the next one.
            continue
        if _fake_capability(dist) is not None:
            return module_name
    return None


def ensure_fake_backend_supports_device(device_type: str) -> bool:
    """Bind the fake backend to ``device_type`` (e.g. ``npu``) in this process.

    Appends ``device_type`` to ``Backend.backend_capability["fake"]`` in place.
    Must run before ``dist.init_process_group(backend="fake", ...)``: that call
    (and every ``new_group`` without an explicit backend) snapshots the list to
    decide which device types the PG binds. Idempotent, and scoped to fake
    runs: real HCCL runs never call it, and the global default-backend map is
    never modified.

    Returns:
        True when the device type is in the fake capability list afterwards.
    """
    if device_type in _PATCHED_DEVICES:
        return True

    try:
        import torch.distributed as dist
    except ImportError:
        logger.warning(
            "torch is not installed; skipping the fake-backend device binding "
            "for '%s'.",
            device_type,
        )
        return False

    if _fake_capability(dist) is None and _import_fake_backend_module(dist) is None:
        logger.warning(
            "This torch build does not register a 'fake' distributed backend; "
            "cannot bind device type '%s' to it.",
            device_type,
        )
        return False

    devices = _fake_capability(dist)
    if device_type not in devices:
        devices.append(device_type)
    _PATCHED_DEVICES.add(device_type)
    logger.info(
        "Fake backend bound to device type '%s' (fake capability: %s).",
        device_type,
        devices,
    )
    return True


def verify_fake_backend_bound(device_type: str) -> bool:
    """Check that the default PG really has a backend for ``device_type``.

    Call right after ``init_process_group(backend="fake", ...)``. Turns the
    failure that would otherwise surface at the first collective on that device
    ("No backend type associated with device type ...") into an init-time error
    that names the cause and the fix.

    Returns:
        True when the backend resolves. False when this torch build does not
        expose the probe (the run is left to fail or succeed on its own).

    Raises:
        RuntimeError: The default PG has no backend bound to ``device_type``.
    """
    import torch
    import torch.distributed as dist

    default_group = dist.distributed_c10d._get_default_group()
    try:
        backend = default_group._get_backend(torch.device(device_type))
    except RuntimeError as exc:
        if _NO_BACKEND_MESSAGE in str(exc):
            raise RuntimeError(
                f"The fake process group has no backend bound to device type "
                f"'{device_type}' (torch {torch.__version__}, fake capability "
                f"{_fake_capability(dist)}). "
                "ensure_fake_backend_supports_device() must run before "
                "init_process_group(backend='fake', ...)."
            ) from exc
        logger.warning(
            "Could not verify the fake backend binding for '%s': %s",
            device_type,
            exc,
        )
        return False
    except AttributeError:
        logger.warning(
            "This torch build does not expose ProcessGroup._get_backend; "
            "skipping the fake backend binding check for '%s'.",
            device_type,
        )
        return False

    logger.info(
        "Fake process group: device type '%s' resolves to %s.",
        device_type,
        type(backend).__name__,
    )
    return True
