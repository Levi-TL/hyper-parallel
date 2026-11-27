#!/bin/bash
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

# Single-process fake-backend run of the qwen3_moe demo.
#
# Compute (kernels, allocator, weights, data) is real; collectives are
# local no-ops. HP_FAKE_WORLD_SIZE keeps the parallel configuration of the
# YAML unchanged while one process stands for one logical rank of that world
# (HP_FAKE_RANK, default 0), so a single device (or even CPU for a smoke run)
# can exercise the full training step of an 8-rank topology.
#
# Usage:
#   ./run_fake.sh /path/to/Qwen3-30B-A3B [trainer overrides...]
#
# Environment:
#   HP_FAKE_WORLD_SIZE  logical world size (default 8, matching the YAML
#                       tp2 x cp2 x ep2 x dp2 topology)
#   HP_FAKE_RANK        logical rank this process stands for (default 0, must
#                       be < HP_FAKE_WORLD_SIZE). Shard ownership, mesh
#                       coordinate and checkpoint offsets follow it.
#   HP_FAKE_DEVICE_ID   card this process occupies (default 0), independent of
#                       HP_FAKE_RANK: `HP_FAKE_RANK=7 HP_FAKE_DEVICE_ID=3`
#                       simulates rank 7 while running on card 3 only. The id
#                       indexes the visible list, so a narrowed
#                       ASCEND_RT_VISIBLE_DEVICES remaps it.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
OUTPUT_DIR="${PROJECT_ROOT}/output/training_demo"
DATA_ROOT="${OUTPUT_DIR}/data"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 /path/to/Qwen3-30B-A3B [trainer overrides...]" >&2
    exit 1
fi
MODEL_PATH=$1
shift
MODEL_PATH=$(cd "${MODEL_PATH}" 2>/dev/null && pwd) || {
    echo "Model directory does not exist: ${MODEL_PATH}" >&2
    exit 1
}
if [[ ! -s "${MODEL_PATH}/config.json" ]]; then
    echo "Qwen3-30B-A3B config.json is missing: ${MODEL_PATH}/config.json" >&2
    exit 1
fi

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}" "${DATA_ROOT}"
if [[ ! -s "${DATA_ROOT}/parallel_online.jsonl" ]]; then
    python -m examples.training_demo.qwen3_moe.prepare_parallel_data \
        --output-dir "${DATA_ROOT}" \
        --num-samples 128 \
        --seq-length 128
fi

# --- fake-backend switches ---------------------------------------------------
# 1. training.backend=fake         -> collectives become local no-ops
# 2. HP_FAKE_WORLD_SIZE            -> logical world size (default 8)
# 3. HP_FAKE_RANK                  -> logical rank this process stands for
#    (default 0).
# 4. HP_FAKE_DEVICE_ID             -> card this process occupies (default 0).
#    Not derived from the logical rank: one process, one card.
# 5. memory.enable / mem_info      -> allocator snapshot (pickle) + peak stats
#    (per-step allocator accounting is real; NPU/HCCL resident memory is not
#     included - see examples/training_demo/README.md, "fake backend" section)
# -----------------------------------------------------------------------------

HP_FAKE_WORLD_SIZE="${HP_FAKE_WORLD_SIZE:-8}" \
HP_FAKE_RANK="${HP_FAKE_RANK:-0}" \
HP_FAKE_DEVICE_ID="${HP_FAKE_DEVICE_ID:-0}" \
python -m examples.training_demo.train_text \
    "${SCRIPT_DIR}/train_parallel_online.yaml" \
    --model.config_path="${MODEL_PATH}" \
    --dataset.model_assets.tokenizer.pretrained_model_name_or_path="${MODEL_PATH}" \
    --dataset.data_path="${DATA_ROOT}/parallel_online.jsonl" \
    --training.backend=fake \
    --memory.enable=true \
    --memory.start_step=0 \
    --memory.end_step=1 \
    --memory.save_path="${OUTPUT_DIR}/memory_snapshot" \
    --memory.mem_info=true \
    "$@" \
    2>&1 | tee "${OUTPUT_DIR}/run_fake.log"
