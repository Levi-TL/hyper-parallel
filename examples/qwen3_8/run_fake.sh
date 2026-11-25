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
# logical rank 0 of the logical world, so a single device exercises the full
# training step of the 8-rank topology.
#
# Usage:
#   ./run_fake.sh /path/to/Qwen3.8-27B /path/to/decif-30k.jsonl [overrides...]
#
# Environment:
#   HP_FAKE_WORLD_SIZE          logical world size (default 8, matching the
#                               recipe's FSDP+CP topology)
#   ASCEND_RT_VISIBLE_DEVICES   NPU visible to this process (default 0)
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
# 1. training.backend=fake         -> collectives become local no-ops; rank 0
#    and world_size 8 are passed explicitly to init_process_group (the fake
#    backend skips env rendezvous, so env vars alone would yield -1/-1).
# 2. HP_FAKE_WORLD_SIZE            -> logical world size (default 8).
# 3. memory.enable / mem_info      -> allocator snapshot (pickle) + peak stats
#    (per-step allocator accounting is real; NPU/HCCL resident memory and
#     communication buffers are NOT included - expect that gap vs a real
#     8-card run; see the README "fake backend" notes in
#     examples/training_demo/README.md).
# -----------------------------------------------------------------------------
export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-1800}
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-1800}

HP_FAKE_WORLD_SIZE="${HP_FAKE_WORLD_SIZE:-8}" \
RANK=0 LOCAL_RANK=0 \
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
