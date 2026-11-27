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

from hyper_parallel.distributed.fake_backend import (
    HP_FAKE_DEVICE_ID,
    HP_FAKE_RANK,
    HP_FAKE_WORLD_SIZE,
    ensure_fake_backend_supports_device,
    is_fake_backend_active,
    resolve_fake_device_id,
    setup_fake_single_process_environment,
    verify_fake_backend_bound,
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
        os.environ.pop(HP_FAKE_RANK, None)
        os.environ.pop(HP_FAKE_DEVICE_ID, None)

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

    def test_rank_injection_selects_logical_rank(self):
        # The logical rank decides shard ownership and checkpoint offsets, so
        # a fake run must be able to stand for any rank of the logical world.
        os.environ[HP_FAKE_WORLD_SIZE] = "8"
        os.environ[HP_FAKE_RANK] = "5"
        os.environ["LOCAL_RANK"] = "3"
        self.assertEqual(setup_fake_single_process_environment(), 8)
        self.assertEqual(os.environ["RANK"], "5")
        # One physical device per process: LOCAL_RANK must not follow the
        # logical rank, or the process binds npu:5 on a smaller machine.
        self.assertEqual(os.environ["LOCAL_RANK"], "0")

    def test_rejects_rank_outside_logical_world(self):
        os.environ[HP_FAKE_WORLD_SIZE] = "4"
        for bad in ("4", "-1", "9"):
            os.environ[HP_FAKE_RANK] = bad
            with self.assertRaises(ValueError):
                setup_fake_single_process_environment()

    def test_rejects_rank_without_world_size(self):
        # A rank with no world size cannot be range-checked, so it is a
        # configuration error rather than a silent fallback to rank 0.
        os.environ.pop(HP_FAKE_WORLD_SIZE, None)
        os.environ[HP_FAKE_RANK] = "3"
        with self.assertRaises(ValueError):
            setup_fake_single_process_environment()

    def test_device_id_is_independent_of_logical_rank(self):
        # Standing for rank 7 must not force card 7: one process, one card.
        os.environ[HP_FAKE_WORLD_SIZE] = "8"
        os.environ[HP_FAKE_RANK] = "7"
        os.environ[HP_FAKE_DEVICE_ID] = "3"
        setup_fake_single_process_environment()
        self.assertEqual(os.environ["RANK"], "7")
        self.assertEqual(os.environ["LOCAL_RANK"], "3")

    def test_device_id_defaults_to_zero(self):
        os.environ[HP_FAKE_WORLD_SIZE] = "8"
        os.environ[HP_FAKE_RANK] = "5"
        setup_fake_single_process_environment()
        self.assertEqual(os.environ["LOCAL_RANK"], "0")

    def test_rejects_negative_or_non_integer_device_id(self):
        os.environ[HP_FAKE_WORLD_SIZE] = "8"
        for bad in ("-1", "three"):
            os.environ[HP_FAKE_DEVICE_ID] = bad
            with self.assertRaises(ValueError):
                setup_fake_single_process_environment()

    def test_rejects_device_id_outside_visible_devices(self):
        # The id indexes the visible list, so a narrowed
        # ASCEND_RT_VISIBLE_DEVICES makes card 3 unreachable.
        os.environ[HP_FAKE_DEVICE_ID] = "3"
        with self.assertRaises(ValueError):
            resolve_fake_device_id(device_count=1)
        self.assertEqual(resolve_fake_device_id(device_count=8), 3)


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
        # rank/world_size must be explicit: the fake backend skips env
        # rendezvous and would otherwise default to -1/-1.
        dist.init_process_group(backend="fake", rank=0, world_size=8)
        self.assertTrue(dist.is_initialized())
        self.assertEqual(dist.get_world_size(), 8)
        self.assertEqual(dist.get_rank(), 0)

    def test_init_accepts_non_zero_logical_rank(self):
        # PyTorch's _create_fake_pg forwards group_rank/group_size verbatim,
        # so a single process can stand for any rank of the logical world.
        dist.init_process_group(backend="fake", rank=5, world_size=8)
        self.assertEqual(dist.get_rank(), 5)
        self.assertEqual(dist.get_world_size(), 8)
        self.assertTrue(is_fake_backend_active())

    def test_collectives_complete_without_communicator(self):
        dist.init_process_group(backend="fake", rank=0, world_size=8)
        tensor = torch.ones(4)
        dist.all_reduce(tensor)
        self.assertTrue(torch.equal(tensor, torch.ones(4)))
        dist.barrier()

    def test_subgroup_creation_inherits_fake_backend(self):
        dist.init_process_group(backend="fake", rank=0, world_size=8)
        group = dist.new_group(ranks=[0, 1])
        self.assertIsNotNone(group)
        self.assertEqual(dist.get_backend(group), "fake")


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class TestDeviceBinding(unittest.TestCase):
    """Accelerator binding for the fake backend.

    ``meta`` stands in for ``npu``: binding a backend to a device type needs
    no hardware, so the NPU failure mode ("No backend type associated with
    device type npu") is reproduced exactly on a CPU-only machine.
    """

    STANDIN = "meta"

    def setUp(self):
        import hyper_parallel.distributed.fake_backend as fake_backend

        self._fake_backend = fake_backend
        capability = fake_backend._fake_capability(dist)
        if capability is None:
            self.skipTest("this torch build has no fake distributed backend")
        self._capability = capability
        self._snapshot = list(capability)
        fake_backend._PATCHED_DEVICES.discard(self.STANDIN)
        self._default_map = dict(dist.Backend.default_device_backend_map)
        self._saved = {key: os.environ.get(key) for key in _SAVED_ENV_KEYS}
        os.environ[HP_FAKE_WORLD_SIZE] = "8"
        setup_fake_single_process_environment()

    def tearDown(self):
        if dist.is_initialized():
            dist.destroy_process_group()
        # Restore in place: the PG factory reads this very list object.
        self._capability[:] = self._snapshot
        self._fake_backend._PATCHED_DEVICES.discard(self.STANDIN)
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_appends_device_to_fake_capability_only(self):
        self.assertNotIn(self.STANDIN, self._capability)
        self.assertTrue(ensure_fake_backend_supports_device(self.STANDIN))
        self.assertIn(self.STANDIN, self._capability)
        # The device -> default backend NAME map must stay untouched: it is
        # what a real HCCL init consults (pytorch/pytorch#159945).
        self.assertEqual(dist.Backend.default_device_backend_map, self._default_map)
        # Idempotent: a second call neither duplicates nor rewrites anything.
        self.assertTrue(ensure_fake_backend_supports_device(self.STANDIN))
        self.assertEqual(self._capability.count(self.STANDIN), 1)

    def test_default_group_and_subgroup_bind_the_device(self):
        ensure_fake_backend_supports_device(self.STANDIN)
        dist.init_process_group(backend="fake", rank=0, world_size=8)
        self.assertTrue(verify_fake_backend_bound(self.STANDIN))
        default_group = dist.distributed_c10d._get_default_group()
        backend = default_group._get_backend(torch.device(self.STANDIN))
        self.assertIn("Fake", type(backend).__name__)

        # new_group() without an explicit backend re-reads the same capability
        # list, so subgroups (mesh axes) bind the device as well.
        group = dist.new_group(ranks=[0, 1])
        self.assertEqual(dist.get_backend(group), "fake")
        subgroup_backend = group._get_backend(torch.device(self.STANDIN))
        self.assertIn("Fake", type(subgroup_backend).__name__)

    def test_unbound_device_fails_at_init_verification(self):
        # Without the binding, the PG is created fine but has no backend for
        # the device: exactly the state the internal NPU run crashed in.
        dist.init_process_group(backend="fake", rank=0, world_size=8)
        with self.assertRaises(RuntimeError) as ctx:
            verify_fake_backend_bound(self.STANDIN)
        self.assertIn("ensure_fake_backend_supports_device", str(ctx.exception))


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
class TestMirroredExpertTransport(unittest.TestCase):
    """Expert-parallel exchange on the fake backend: peers mirror the local rank.

    torch's FakeProcessGroup leaves all-to-all outputs untouched, so a ragged
    MoE dispatch cannot learn its receive counts (the internal NPU run failed
    with "all EP ranks must agree ..."). On a fake group the EP transport hands
    the sent rows straight back instead. Everything below runs on CPU.
    """

    NUM_EXPERTS = 8
    EP_SIZE = 2

    class _Mesh:
        ndim = 1

        def __init__(self, group):
            self._group = group

        def size(self):
            return TestMirroredExpertTransport.EP_SIZE

        def get_local_rank(self):
            return dist.get_rank(self._group)

        def get_group(self):
            return self._group

    def setUp(self):
        self._saved = {key: os.environ.get(key) for key in _SAVED_ENV_KEYS}
        os.environ[HP_FAKE_WORLD_SIZE] = "8"
        setup_fake_single_process_environment()
        dist.init_process_group(backend="fake", rank=0, world_size=8)
        self._group = dist.new_group(ranks=[0, 1])

    def tearDown(self):
        if dist.is_initialized():
            dist.destroy_process_group()
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def _slots(self):
        generator = torch.Generator().manual_seed(0)
        # Expert IDs ascending, as the slot protocol requires; experts 4-7 are
        # owned by the (absent) peer rank.
        ids = torch.tensor([0, 2, 3, 5, 6, 7], dtype=torch.int64)
        hidden = torch.randn(ids.numel(), 4, generator=generator, requires_grad=True)
        weights = torch.rand(ids.numel(), generator=generator, requires_grad=True)
        return hidden, ids, weights

    def test_counts_come_back_unchanged_with_flags(self):
        from hyper_parallel.distributed.expert_parallel.collectives import ep_exchange_counts

        metadata = torch.tensor([[3, 1], [3, 1]], dtype=torch.int64)
        received = ep_exchange_counts(metadata, self._group)
        self.assertTrue(torch.equal(received, metadata))
        self.assertIsNot(received, metadata)

    def test_exchange_is_identity_and_passes_gradients(self):
        from hyper_parallel.distributed.expert_parallel.collectives import (
            ep_all_to_all,
            ep_all_to_all_async,
        )

        for exchange in (ep_all_to_all, ep_all_to_all_async):
            with self.subTest(exchange=exchange.__name__):
                rows = torch.randn(6, 4, requires_grad=True)
                out = exchange(rows, [4, 2], [4, 2], self._group)
                self.assertEqual(out.shape, (6, 4))
                self.assertTrue(torch.equal(out.detach(), rows.detach()))
                out.sum().backward()
                self.assertTrue(torch.equal(rows.grad, torch.ones_like(rows)))

    def test_exchange_rejects_mismatched_receive_total(self):
        from hyper_parallel.distributed.expert_parallel.collectives import ep_all_to_all

        # A caller whose receive counts came from elsewhere (not the mirrored
        # count exchange) must fail loudly rather than hand back wrong shapes.
        with self.assertRaises(ValueError):
            ep_all_to_all(torch.randn(6, 4), [4, 2], [6, 2], self._group)

    def test_mirrored_ids_stay_inside_local_expert_range(self):
        from hyper_parallel.distributed.expert_parallel.collectives import mirror_expert_ids

        ids = torch.tensor([0, 2, 3, 5, 6, 7], dtype=torch.int64)
        for ep_rank in range(self.EP_SIZE):
            local = mirror_expert_ids(ids, 4, ep_rank)
            self.assertEqual(int(local.min()), ep_rank * 4)
            self.assertEqual(int(local.max()), ep_rank * 4 + 3)
            self.assertTrue(torch.equal(local % 4, ids % 4))

    def test_dispatch_and_combine_keep_every_slot(self):
        from hyper_parallel.distributed.expert_parallel.ep_context import EPContext

        context = EPContext(
            self._Mesh(self._group), num_experts=self.NUM_EXPERTS, overlap_shared_expert=False)
        hidden, ids, weights = self._slots()
        received, counts, _, send, receive, recv_inverse = context.dispatch(hidden, ids, weights)
        local_experts = self.NUM_EXPERTS // self.EP_SIZE

        # Row counts are those of a balanced EP run: every slot is computed.
        self.assertEqual(send, receive)
        self.assertEqual(tuple(counts.shape), (local_experts,))
        self.assertEqual(int(counts.sum()), ids.numel())
        self.assertEqual(received.shape, hidden.shape)
        self.assertEqual(int(recv_inverse.min()), 0)
        self.assertEqual(int(recv_inverse.max()), ids.numel() - 1)

        # Identity experts: combine returns each slot's own row, in slot order.
        returned = context.combine(received, send, receive, recv_inverse)
        self.assertTrue(torch.equal(returned.detach(), hidden.detach()))
        returned.sum().backward()
        self.assertTrue(torch.equal(hidden.grad, torch.ones_like(hidden)))

    def test_legacy_dispatch_folds_onto_local_experts(self):
        from hyper_parallel.distributed.expert_parallel import experts as ep_experts

        seen = {}

        class _Experts:
            local_expert_count = self.NUM_EXPERTS // self.EP_SIZE

            def __call__(self, states, indices):
                seen["indices"] = indices
                seen["rows"] = states.shape[0]
                return states

        class _Module:
            experts = _Experts()

        rows = torch.randn(6, 4)
        indices = torch.tensor([[0], [2], [3], [5], [6], [7]], dtype=torch.int64)
        out = ep_experts._run_ep_local_experts(  # pylint: disable=protected-access
            _Module(), rows, indices, [3, 3], [3, 3], self._group, expert_offset=0)
        self.assertEqual(out.shape, rows.shape)
        self.assertEqual(seen["rows"], rows.shape[0])
        self.assertGreaterEqual(int(seen["indices"].min()), 0)
        self.assertLess(int(seen["indices"].max()), _Experts.local_expert_count)


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class TestFakeCheckpoint(unittest.TestCase):
    """DCP save and single-process restore under the fake backend."""

    def setUp(self):
        import tempfile
        self._tmp = tempfile.mkdtemp(prefix="hp_fake_ckpt_")
        self._saved = {key: os.environ.get(key) for key in _SAVED_ENV_KEYS}
        os.environ[HP_FAKE_WORLD_SIZE] = "1"
        setup_fake_single_process_environment()
        dist.init_process_group(backend="fake", rank=0, world_size=1)

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
