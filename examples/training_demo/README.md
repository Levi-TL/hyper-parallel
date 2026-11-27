# Training demos

The root keeps the shared `train_text.py` entry point. Model-owned builders,
recipes, data preparation, launchers, and validation manifests are grouped by
family:

- `qwen3_moe/`: cropped and full-model Qwen3-MoE text demos;
- `deepseek_v41/`: DeepSeek-V4.1 text/VLM crops and validation tooling.

## Qwen3-MoE cropped Hugging Face demos

These examples build a layer-cropped Qwen3-30B-A3B model with
`HyperAutoModelForCausalLM.from_config`. They read the complete Hugging Face
configuration, keep all original hidden, attention, vocabulary, and expert
dimensions, and change only `num_hidden_layers` (four by default). No model
checkpoint tensor is loaded, so model parameters are randomly initialized.

## Placement validation from YAML

YAML can directly control placement validation:

```yaml
model:
  _target_: examples.training_demo.qwen3_moe.cropped_qwen3_moe.build_cropped_qwen3_moe
  validate_placement: true
```

The value is retained by `Target`, passed through `BaseTrainer._build_model`,
forwarded by `build_cropped_qwen3_moe` to `HyperAutoModelForCausalLM.from_config`,
and finally used as the infrastructure `validate_mode` value. The typed CLI
override `--model.validate_placement=true` follows the same path. The launch
scripts do not override this field, so editing the YAML remains effective.

## Data modes

- Offline uses the Indexed Dataset format. Its targets are already shifted,
  so `labels_are_shifted: true`; the GQA CP member handles causal attention
  from the query position and visible KV prefix.
- Online tokenizes and packs JSONL text at runtime. Its transform emits
  pre-shifted labels, and packed document boundaries use a global block mask accepted by
  the `GroupedQueryAttention` CP member.

Both launchers automatically generate small deterministic local datasets when
the expected files are absent. This process does not access or download an
external dataset.

## Run

Pass a local Qwen3-30B-A3B Hugging Face directory containing `config.json` and
tokenizer assets as the first argument. The examples deliberately set
`local_files_only: true`: missing assets cause an explicit error instead of an
implicit model or weight download. Prepare the runtime according to the
project installation guide before launching the example.

```bash
bash examples/training_demo/qwen3_moe/run_parallel_offline.sh /path/to/Qwen3-30B-A3B
bash examples/training_demo/qwen3_moe/run_parallel_online.sh /path/to/Qwen3-30B-A3B
```

The default topology uses eight devices with TP=2, CP=2, EP=2, and FSDP. To
enable placement validation without editing YAML:

```bash
bash examples/training_demo/qwen3_moe/run_parallel_offline.sh \
    /path/to/Qwen3-30B-A3B \
    --model.validate_placement=true
```

Additional typed overrides are forwarded to the Trainer. For example, a
one-step smoke test is:

```bash
bash examples/training_demo/qwen3_moe/run_parallel_offline.sh \
    /path/to/Qwen3-30B-A3B \
    --training.train_iters=1
```

Logs and generated data are stored under `output/training_demo`.

## Fake backend: single-process logical-world runs

`training.backend: fake` runs the **same YAML topology in one physical
process**: kernels, allocator, weights, and data stay real, while collectives
become local no-ops (no communicator is created). `HP_FAKE_WORLD_SIZE` pins
the logical world size, and `HP_FAKE_RANK` (default 0) selects which logical
rank the process stands for, so `HP_FAKE_RANK=3` exercises rank 3's shards.

```bash
bash examples/training_demo/qwen3_moe/run_fake.sh /path/to/Qwen3-30B-A3B
```

`run_fake.sh` also enables the allocator memory snapshot
(`output/training_demo/memory_snapshot/*.pickle`, viewable with
[PyTorch memory_viz](https://pytorch.org/memory-viz)) and per-step peak
allocator statistics. Use it to answer, before cluster access, whether a
parallel configuration starts and how much memory logical rank 0 holds.

`HP_FAKE_RANK` selects a **logical** rank, never a physical card: `LOCAL_RANK`
stays 0, so the process runs on the first visible device whatever rank it
stands for. To pin the run to a specific card, set
`ASCEND_RT_VISIBLE_DEVICES=<card>` (it then is device 0 of that process).
Code that derives a device index from `dist.get_rank()` opens a context on
another card instead - `npu-smi` then shows one small, allocation-free entry
per such rank - which is why device resolution goes through the current
device of the process, not the rank.

What the fake backend proves and does not:

- Proves: configuration/mesh/shape bring-up, the full train step with real
  kernels, and the rank-local **allocator** memory account (parameters,
  optimizer state, activations, gradients, fragmentation).
- Does not prove: throughput or kernel timing, numerical equivalence or
  real collective behaviour, communication overlap, and pipeline-parallel
  scheduling (`pp_size > 1` fails fast with an explicit error).
- The account excludes device-runtime resident memory (HCCL communicator,
  ~GB-scale per card on real runs) and cross-rank timing interactions; add
  the environment-specific resident delta when comparing against a real run.
- Expert parallelism (`ep_size > 1`) runs with **mirrored** token exchange:
  torch's fake process group leaves all-to-all outputs empty, so on the fake
  path each peer is assumed to send this rank exactly what this rank sends it,
  and the local expert at the same offset stands in for a remote one
  (`hyper_parallel/distributed/expert_parallel/collectives.py`). Row counts,
  and therefore expert activations and exchange buffers, match a balanced EP
  run in expectation; routing imbalance between ranks and the loss value do
  not. This is the same assumption torch's fake `all_gather` makes.
- Data is read **by the fake process itself**, not broadcast: in a real run
  only TP rank 0 reads the batch and broadcasts it to its TP peers, but a fake
  collective moves no data, so a non-source rank would train on uninitialized
  buffers. The DataLoader context is therefore built with
  `tp_source_rank=None`, and every logical rank reads its own DP/CP shard
  (same coordinates, same shard, same order as the peer that would have
  broadcast to it).

Torch registers its fake backend for `cpu/cuda/hpu` (plus `xpu` from 2.10)
only, while a process group resolves its backend per collective from the
device type of the tensors involved. On the fake path, `npu` is therefore
appended to that device list before `init_process_group` and the binding is
verified right after init (`hyper_parallel/distributed/fake_backend.py`).
Real HCCL runs are unaffected: the default-backend map is never modified.

Not covered by the fake backend: `dataloader.load_balance` (collective loading
creates real Gloo/HCCL data-plane groups, so it fails fast), high
availability (creates real HCCL groups), and the Engram module with
`ep_size > 1` (its count exchange does not use the mirrored transport and
fails fast with a row-count error).

## Full pretrained model

Both full-model launchers load all 48 layers and the complete Hugging Face
checkpoint through `HyperAutoModelForCausalLM.from_pretrained`. Online uses the
packaged `hyper_parallel/models/qwen3_moe/recipes/train.yaml`;
Offline uses `examples/training_demo/qwen3_moe/train_parallel_full_offline.yaml` because
the two data paths instantiate different Dataset, DataLoader, and collate
targets. Ordinary values can be overridden on the command line, but the typed
configuration interface intentionally does not replace `_target_` values.

The Online launcher tokenizes and packs a deterministic local JSONL file at
runtime. It generates that file under `output/training_demo/data` when needed
and uses a 128-token smoke-test length by default; an appended typed override
can increase the sequence length:

```bash
bash examples/training_demo/qwen3_moe/run_parallel_full_online.sh \
    /path/to/Qwen3-30B-A3B
```

The Offline launcher requires an existing Indexed Dataset and never generates
or downloads one implicitly. Pass the dataset prefix without the `.bin` or
`.idx` suffix:

```bash
bash examples/training_demo/qwen3_moe/run_parallel_full_offline.sh \
    /path/to/Qwen3-30B-A3B \
    /path/to/offline_text_document
```

Both launchers validate the local model `config.json` before starting and force
model/tokenizer loading into `local_files_only` mode. The Offline launcher also
validates both Indexed Dataset files. Missing local assets therefore fail
explicitly rather than triggering a network download. Additional typed Trainer
overrides may be appended to either command.

## DeepSeek-V4.1 text pretraining

The [text configuration](deepseek_v41/train_deepseek_v41_online.yaml) retains all
40 decoder layers, divides text parameter widths by four, and uses 16 routed
experts with random initialization. It follows the VLM recipe's TP1/CP4/EP8,
FSDP8, FP32 main parameters and reduction, BF16 forward parameters, Host Engram,
fused CSA2 attention, MegaMoe/MegaMhc/MegaGate, and full recomputation.
Vision and encoder parallelism are disabled. Fixed expert destinations are a
benchmark setting; set `fix_router: false` for normal learned routing.

Prepare the environment with the [DeepSeek-V4.1 quick start](deepseek_v41/deepseek_v41_quick_start.md),
then run from the repository root:

```bash
bash examples/training_demo/deepseek_v41/run_deepseek_v41_online.sh \
    /path/to/DeepSeek-V4.1-Flash --training.train_iters=1 --checkpoint.restore_from=null

# Supply an existing text JSONL instead of generating the default test data.
bash examples/training_demo/deepseek_v41/run_deepseek_v41_online.sh \
    /path/to/DeepSeek-V4.1-Flash /path/to/text/train.jsonl
```

Records use a `text` field. The transform tokenizes documents into 4096-token
windows and drops partial tails. The launcher accepts a model path, an optional
dataset path, and dotted Trainer overrides; the old `tp1|tp2|cp2` selector is no
longer used. Fused CSA2 requires TP1. Generated Engram assets use parameter divisor 4.
Text logs use `run_text_tp1_cp4_ep8_fsdp8.log` by default, and checkpoints use
`output/training_demo/deepseek_v41/artifacts/text_checkpoints`.

## DeepSeek-V4.1 multimodal Online smoke

The [VLM configuration](deepseek_v41/train_deepseek_v41_vlm_online.yaml) also
retains 32 vision blocks with parameter divisor 4 and enables encoder CP4.
It uses `pretrain + image_caption` data and a 4096-token packed window.
See the [quick start](deepseek_v41/deepseek_v41_quick_start.md) for dependencies,
CC12M/fake data preparation, operator switches, HF weight-loading constraints,
and troubleshooting.

```bash
bash examples/training_demo/deepseek_v41/run_deepseek_v41_vlm_online.sh \
    /path/to/DeepSeek-V4.1-Flash /path/to/image_caption/train.jsonl
```

The default dataset is
`output/training_demo/deepseek_v41/mm_data/deepseek_v41_image_caption/train.jsonl`.
Both launchers clear a stale success marker and recreate it after every worker
exits successfully. Explicit Trainer overrides are passed last.

The [validation manifest](../../hyper_parallel/models/deepseek_v41/tools/deepseek_v41_validation.yaml)
defines the TP/CP/EP, recompute, precision and checkpoint validation matrix. The
[ND measurement guide](../../hyper_parallel/models/deepseek_v41/tools/nd_timing_round.md)
describes the separate layout measurement workflow.
