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
"""Unit tests for the single-process fake distributed backend.

Layer 1 (no torch required) covers the logical-world environment contract.
Layer 2 requires torch and exercises the fake process group end to end.
NPU-only cases self-skip on machines without an NPU runtime.
"""

import os
import unittest
from unittest import mock

from hyper_parallel.distributed.fake_backend import (
    HP_FAKE_WORLD_SIZE,
    ensure_fake_backend_supports_npu,
    setup_fake_single_process_environment,
)

try:
    import torch  # noqa: F401 pylint: disable=unused-import
    import torch.distributed as dist

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

_SAVED_ENV_KEYS = ("WORLD_SIZE", "RANK", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT")


class TestFakeEnvironment(unittest.TestCase):
    """HP_FAKE_WORLD_SIZE env contract (no torch needed)."""

    def setUp(self):
        self._saved = {key: os.environ.get(key) for key in _SAVED_ENV_KEYS}
        os.environ.pop(HP_FAKE_WORLD_SIZE, None)

    def tearDown(self):
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_injection_pins_single_process_and_logical_world(self):
        os.environ[HP_FAKE_WORLD_SIZE] = "8"
        world_size = setup_fake_single_process_environment()
        self.assertEqual(world_size, 8)
        self.assertEqual(os.environ["WORLD_SIZE"], "8")
        self.assertEqual(os.environ["RANK"], "0")
        self.assertEqual(os.environ["LOCAL_RANK"], "0")

    def test_injection_overrides_stale_launcher_env(self):
        # A stale torchrun environment must not leak into the logical world.
        os.environ[HP_FAKE_WORLD_SIZE] = "16"
        os.environ["WORLD_SIZE"] = "4"
        os.environ["RANK"] = "3"
        world_size = setup_fake_single_process_environment()
        self.assertEqual(world_size, 16)
        self.assertEqual(os.environ["WORLD_SIZE"], "16")
        self.assertEqual(os.environ["RANK"], "0")

    def test_without_injection_defaults_to_existing_world_size(self):
        os.environ.pop("WORLD_SIZE", None)
        self.assertEqual(setup_fake_single_process_environment(), 1)
        os.environ["WORLD_SIZE"] = "4"
        self.assertEqual(setup_fake_single_process_environment(), 4)

    def test_rejects_non_positive_world_size(self):
        for bad in ("0", "-2"):
            os.environ[HP_FAKE_WORLD_SIZE] = bad
            with self.assertRaises(ValueError):
                setup_fake_single_process_environment()

    def test_rejects_non_integer_world_size(self):
        os.environ[HP_FAKE_WORLD_SIZE] = "eight"
        with self.assertRaises(ValueError):
            setup_fake_single_process_environment()


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class TestFakeProcessGroup(unittest.TestCase):
    """Fake PG behaviour with an injected logical world size."""

    def setUp(self):
        self._saved = {key: os.environ.get(key) for key in _SAVED_ENV_KEYS}
        os.environ[HP_FAKE_WORLD_SIZE] = "8"
        setup_fake_single_process_environment()

    def tearDown(self):
        if dist.is_initialized():
            dist.destroy_process_group()
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_init_reports_logical_world_size(self):
        dist.init_process_group(backend="fake")
        self.assertTrue(dist.is_initialized())
        self.assertEqual(dist.get_world_size(), 8)
        self.assertEqual(dist.get_rank(), 0)

    def test_collectives_complete_without_communicator(self):
        dist.init_process_group(backend="fake")
        tensor = torch.ones(4)
        dist.all_reduce(tensor)
        self.assertTrue(torch.equal(tensor, torch.ones(4)))
        dist.barrier()

    def test_subgroup_creation_inherits_fake_backend(self):
        dist.init_process_group(backend="fake")
        group = dist.new_group(ranks=[0, 1])
        self.assertIsNotNone(group)
        self.assertEqual(dist.get_backend(group), "fake")


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class TestNpuRegistrationPatch(unittest.TestCase):
    """The NPU registration patch must be idempotent and non-destructive."""

    def setUp(self):
        import hyper_parallel.distributed.fake_backend as fake_backend
        self._fake_backend = fake_backend
        self._saved_patched = fake_backend._NPU_PATCH_APPLIED

    def tearDown(self):
        self._fake_backend._NPU_PATCH_APPLIED = self._saved_patched

    def test_does_not_overwrite_existing_npu_mapping(self):
        device_map = {"cpu": "gloo", "cuda": "nccl", "npu": "hccl"}
        with mock.patch.object(
            dist.Backend, "default_device_backend_map", device_map
        ):
            self._fake_backend._NPU_PATCH_APPLIED = False
            self.assertFalse(ensure_fake_backend_supports_npu())
            self.assertEqual(device_map["npu"], "hccl")
            # Repeated calls stay non-destructive.
            self.assertFalse(ensure_fake_backend_supports_npu())
            self.assertEqual(device_map["npu"], "hccl")

    def test_adds_npu_when_absent(self):
        device_map = {"cpu": "gloo", "cuda": "nccl"}
        with mock.patch.object(
            dist.Backend, "default_device_backend_map", device_map
        ):
            self._fake_backend._NPU_PATCH_APPLIED = False
            self.assertTrue(ensure_fake_backend_supports_npu())
            self.assertEqual(device_map.get("npu"), "fake")

    def test_idempotent_after_successful_patch(self):
        device_map = {"cpu": "gloo"}
        with mock.patch.object(
            dist.Backend, "default_device_backend_map", device_map
        ):
            self._fake_backend._NPU_PATCH_APPLIED = False
            self.assertTrue(ensure_fake_backend_supports_npu())
            # Second call short-circuits without touching the map again.
            self.assertTrue(ensure_fake_backend_supports_npu())
            self.assertEqual(device_map.get("npu"), "fake")


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class TestFakeParallelLayout(unittest.TestCase):
    """Logical world size flows through the pure topology resolver."""

    def test_dp_size_derived_from_logical_world(self):
        from hyper_parallel.distributed.mesh import (
            ParallelTopologySpec,
            resolve_parallel_layout,
        )

        spec = ParallelTopologySpec(tp_size=2, cp_size=1, pp_size=1)
        layout = resolve_parallel_layout(spec, world_size=8)
        self.assertEqual(layout.dp_size, 4)

    def test_rejects_indivisible_world(self):
        from hyper_parallel.distributed.mesh import (
            ParallelTopologySpec,
            resolve_parallel_layout,
        )

        spec = ParallelTopologySpec(tp_size=3, cp_size=1, pp_size=1)
        with self.assertRaises(ValueError):
            resolve_parallel_layout(spec, world_size=8)


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class TestFakeCheckpoint(unittest.TestCase):
    """DCP save and single-process restore under the fake backend."""

    def setUp(self):
        import tempfile
        self._tmp = tempfile.mkdtemp(prefix="hp_fake_ckpt_")
        self._saved = {key: os.environ.get(key) for key in _SAVED_ENV_KEYS}
        os.environ[HP_FAKE_WORLD_SIZE] = "1"
        setup_fake_single_process_environment()
        dist.init_process_group(backend="fake")

    def tearDown(self):
        if dist.is_initialized():
            dist.destroy_process_group()
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_dcp_save_and_load_roundtrip(self):
        import torch.distributed.checkpoint as dcp
        import torch.serialization

        original = {"weights": torch.arange(8, dtype=torch.float32)}
        dcp.save(original, checkpoint_id=self._tmp)

        restored = {"weights": torch.zeros(8, dtype=torch.float32)}
        dcp.load(
            restored,
            checkpoint_id=self._tmp,
            process_group=dist.group.WORLD,
        )
        self.assertTrue(torch.equal(restored["weights"], original["weights"]))


if __name__ == "__main__":
    unittest.main()
