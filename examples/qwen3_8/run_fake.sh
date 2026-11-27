#!/usr/bin/env bash
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

# Single-process fake-backend run of the qwen3_8 (Qwen3.8-27B) online recipe.
#
# Compute (kernels, allocator, weights, data) is real; collectives are local
# no-ops. HP_FAKE_WORLD_SIZE keeps the recipe's parallel configuration
# (FSDP dp_shard=8 x cp=2 on 8 ranks) unchanged while one process stands for
# one logical rank of that world (HP_FAKE_RANK, default 0), so a single device
# exercises the full training step of the 8-rank topology.
#
# Usage:
#   ./run_fake.sh /path/to/Qwen3.8-27B /path/to/decif-30k.jsonl [overrides...]
#
# Environment:
#   HP_FAKE_WORLD_SIZE          logical world size (default 8, matching the
#                               recipe's FSDP+CP topology)
#   HP_FAKE_RANK                logical rank this process stands for (default
#                               0, must be < HP_FAKE_WORLD_SIZE). Rank-local
#                               shard ownership, mesh coordinate and
#                               checkpoint offsets follow it, so
#                               `HP_FAKE_RANK=3 ./run_fake.sh ...` exercises
#                               rank 3's shards on the device it occupies.
#   HP_FAKE_DEVICE_ID           card this process occupies (default 0), chosen
#                               independently of HP_FAKE_RANK. The id indexes
#                               the visible list, so it means "card 3" only
#                               when card 3 is visible.
#   ASCEND_RT_VISIBLE_DEVICES   passed through untouched: unset means all
#                               cards are visible and HP_FAKE_DEVICE_ID picks
#                               one; setting it to a single card (e.g. 3) makes
#                               that card device 0 and the only card this
#                               process can open
#
# Typical overrides (recipe defaults are train_iters=100, save_steps=10):
#   --training.train_iters=2 --checkpoint.save_steps=1   # short smoke run
#
# What a successful run proves: the recipe launches under a logical 8-rank
# world, rank-local memory is measured by the real NPU allocator, and the
# DCP checkpoint can be saved from a single process. It does NOT prove
# loss equivalence, throughput, communication overlap, or SPMD numerical
# correctness — escalate to a real 8-card run for those.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
OUTPUT_DIR="${PROJECT_ROOT}/outputs/qwen3_8/fake_run"

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 /path/to/Qwen3.8-27B /path/to/dataset.jsonl [overrides...]" >&2
    exit 1
fi
MODEL_PATH=$1
DATA_PATH=$2
shift 2

MODEL_PATH=$(cd "${MODEL_PATH}" 2>/dev/null && pwd) || {
    echo "Model directory does not exist: ${MODEL_PATH}" >&2
    exit 1
}
if [[ ! -s "${MODEL_PATH}/config.json" ]]; then
    echo "Qwen3.8-27B config.json is missing: ${MODEL_PATH}/config.json" >&2
    exit 1
fi
if [[ ! -s "${DATA_PATH}" ]]; then
    echo "Dataset file does not exist: ${DATA_PATH}" >&2
    exit 1
fi

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}"

# --- fake-backend switches ---------------------------------------------------
# 1. training.backend=fake         -> collectives become local no-ops; the
#    logical rank/world_size are passed explicitly to init_process_group (the
#    fake backend skips env rendezvous, so env vars alone would yield -1/-1).
# 2. HP_FAKE_WORLD_SIZE            -> logical world size (default 8).
# 3. HP_FAKE_RANK                  -> logical rank this process stands for
#    (default 0).
# 4. HP_FAKE_DEVICE_ID             -> card this process occupies (default 0),
#    independent of the logical rank. LOCAL_RANK is set from it, never from
#    HP_FAKE_RANK: one process owns one card. To keep the process from being
#    able to touch any other card at all, narrow the visible list instead:
#    ASCEND_RT_VISIBLE_DEVICES=3 makes card 3 device 0 of this process.
#    The visible list is left to the caller/scheduler here; a device id
#    outside it fails at startup with the cause.
# 5. memory.enable / mem_info      -> allocator snapshot (pickle) + peak stats
#    (per-step allocator accounting is real; NPU/HCCL resident memory and
#     communication buffers are NOT included - expect that gap vs a real
#     8-card run; see the README "fake backend" notes in
#     examples/training_demo/README.md).
# -----------------------------------------------------------------------------
export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-1800}
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-1800}

HP_FAKE_WORLD_SIZE="${HP_FAKE_WORLD_SIZE:-8}" \
HP_FAKE_RANK="${HP_FAKE_RANK:-0}" \
HP_FAKE_DEVICE_ID="${HP_FAKE_DEVICE_ID:-0}" \
python3 -m scripts.train_lm \
    "${PROJECT_ROOT}/hyper_parallel/models/qwen3_8/recipes/train_online.yaml" \
    --model.pretrained_model_name_or_path="${MODEL_PATH}" \
    --dataset.model_assets.tokenizer.pretrained_model_name_or_path="${MODEL_PATH}" \
    --dataset.model_assets.tokenizer.local_files_only=true \
    --dataset.data_path="${DATA_PATH}" \
    --training.backend=fake \
    --memory.enable=true \
    --memory.start_step=0 \
    --memory.end_step=1 \
    --memory.save_path="${OUTPUT_DIR}/memory_snapshot" \
    --memory.mem_info=true \
    "$@" \
    2>&1 | tee "${OUTPUT_DIR}/run_fake.log"
