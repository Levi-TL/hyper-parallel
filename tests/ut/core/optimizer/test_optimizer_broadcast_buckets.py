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

"""Unit tests for the fused checkpoint state broadcast buckets.

The HSDP state broadcast runs once per save, before any DCP work. Every rank
builds its own bucket list from its own local state, and the ranks of one
replicate group then issue one broadcast per bucket. A rank that owns a
parameter holds state in the parameter's dtype while a peer that does not own it
holds nothing, so bucketing by the local dtype made the two ranks disagree on
element size and packed length --- a broadcast that never completes.
"""

import torch

from hyper_parallel.core.optimizer.optimizer import (
    BaseDistributedOptimizer,
    _STATE_BROADCAST_DTYPE,
)


class _BroadcastProbe(BaseDistributedOptimizer):
    """Stand-in exposing only what ``_collect_broadcast_tensors`` reads.

    Building a real distributed optimizer needs a device mesh, a model and a
    process group; the bucketing decision is a pure function of the parameter map
    and the local state, so the probe carries just those two.
    """

    def __init__(self):
        self.state = {}
        self._param_to_broadcast_info = {}


def _probe(param, replicate_pgs, state=None):
    """Return a probe that owns ``param`` and broadcasts it from ``src_coord``."""
    probe = _BroadcastProbe()
    probe._param_to_broadcast_info = {param: ((0,), replicate_pgs)}
    if state is not None:
        probe.state[param] = dict(state)
    return probe


def test_state_bucket_is_identical_whether_or_not_the_rank_owns_the_state():
    """The owner keeps bf16 momentum; its peer has none. Both must bucket alike."""
    param = torch.nn.Parameter(torch.zeros(16, dtype=torch.bfloat16))
    replicate_pgs = (object(),)

    owner = _probe(param, replicate_pgs, {"momentum_buffer": torch.zeros_like(param)})
    peer = _probe(param, replicate_pgs)

    owner_buckets = owner._collect_broadcast_tensors("state", ["momentum_buffer"])
    peer_buckets = peer._collect_broadcast_tensors("state", ["momentum_buffer"])

    assert list(owner_buckets) == list(peer_buckets)
    assert sum(t.numel() for t in next(iter(owner_buckets.values()))) == sum(
        t.numel() for t in next(iter(peer_buckets.values()))
    )


def test_state_is_bucketed_by_the_exchange_dtype_not_by_the_local_dtype():
    """A bf16 state tensor must not put the owner in a different bucket."""
    param = torch.nn.Parameter(torch.zeros(16, dtype=torch.bfloat16))
    owner = _probe(param, (object(),), {"momentum_buffer": torch.zeros_like(param)})

    buckets = owner._collect_broadcast_tensors("state", ["momentum_buffer"])

    assert len(buckets) == 1
    (_, bucket_dtype, _), tensors = next(iter(buckets.items()))
    assert bucket_dtype is _STATE_BROADCAST_DTYPE
    assert tensors[0].dtype is torch.bfloat16, "the owner keeps its own state dtype"


def test_a_rank_without_ownership_receives_a_float32_placeholder():
    """The placeholder has to be the exchange dtype, so pack/unpack stays matched."""
    param = torch.nn.Parameter(torch.zeros(4, 8, dtype=torch.bfloat16))
    peer = _probe(param, (object(),))

    peer._collect_broadcast_tensors("state", ["exp_avg", "exp_avg_sq"])

    assert peer.state[param]["exp_avg"].dtype is _STATE_BROADCAST_DTYPE
    assert peer.state[param]["exp_avg"].shape == param.shape
    assert peer.state[param]["exp_avg_sq"].dtype is _STATE_BROADCAST_DTYPE


def test_parameter_buckets_keep_the_parameter_dtype():
    """Parameters already agree across ranks, so their bucketing is unchanged."""
    param = torch.nn.Parameter(torch.zeros(16, dtype=torch.bfloat16))
    probe = _probe(param, (object(),))

    buckets = probe._collect_broadcast_tensors("param")

    (_, bucket_dtype, _), _ = next(iter(buckets.items()))
    assert bucket_dtype is torch.bfloat16
