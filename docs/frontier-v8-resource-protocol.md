# V8 CPU preflight and fresh resource boundary

This phase prepares the [fixed scientific plan](../reports/frontier-v8-training-protocol-20261003/training-plan.json)
and CPU checks. No host is selected or staged, no current GPU ownership is
established, and no model is loaded. The new driver supports read-only
`--preflight-only`; `--execute` deliberately fails closed. Neither the plan,
a CPU receipt nor an opaque resource file can enable execution. A separately
reviewed fresh owned launcher and restoration guard remain required.

The original v6/v7 drivers, controllers, plans, attempts, checkpoints, receipts
and PID records retain their historical bytes. Do not substitute new argv into
an old controller, reuse an old session or nonce, or label old ownership proof
as current evidence.

## Scientific identity and CPU preflight

The source commit is bound by the separate `freeze-receipt.json` after its
source is committed and pushed. This avoids a circular commit identity in the
static plan. The external freeze records `schema_version: 1`,
`status: frontier_v8_protocol_frozen_to_committed_source`, `source_commit`,
`plan_sha256`, `implementation_sha256`, `data_manifest_sha256` and
`source_pushed_before_freeze: true`. The driver requires exact plan and freeze
hashes supplied through `--expected-plan-sha256` and
`--expected-freeze-sha256`, plus the exact `--expected-commit`. Its checkout
must be clean, have that HEAD, and match every declared committed source
blob. All inherited `GIT_*` overrides are removed from its Git probes.

The prepared dataset has 2,932 Train rows, 496 Calibration rows and the
unchanged primary and observed inventories. Preflight verifies all thirteen
JSONL files and the manifest as exact bytes. It parses only complete Train
and Calibration, before applying the trainer's fixed selection order. Train
uses seed 20261004 with shuffled sampling; **733 steps × accumulation 4**
declare one complete pass. Calibration uses all 496 rows in the trainer's
fixed balanced order. Validation, Test, OOD and old observed bodies are never
parsed by this preflight. Their byte hashes do not authorize evaluation.

The initializer remains exact published
`ZefanCai/Open-Jev-2B@0c7aa498b1627be8da4acf34c863ff0ee0a92785`, with
the published adapter/head/config/temperature inventory and directory hash.
The pinned base revision is
`Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc`.
The plan's eleven base-file hashes identify scientific content; their
historical inventory origin provides no installation or resource authority.
Every byte must be hashed again on a future host. Standard HF snapshot file
symlinks are allowed only when their regular destinations resolve inside the
declared cache; the fresh CPU receipt also binds their resolved target paths.
Other dataset/checkpoint/source inventories reject symlinks and special leaves.

The driver checks Linux x86_64, Python 3.11, the exact runtime-lock bytes and
all 76 first-match installed distribution versions. It records their actual
distribution roots and the import positions of torch, transformers, peft,
triton, safetensors and accelerate without importing those packages. The
fresh stage must match these positions; any declared overlay has an exact
regular-file inventory. This establishes package versions and positions,
not wheel provenance or a clean installation replay. Shared runtime files
remain read-only.

The declared numerical settings require CUDA 12.8, a BF16 backbone, FP32
head, highest FP32 matrix precision, CUDA matrix TF32 disabled and no
quantization. CPU preflight returns these requirements with
`numerical_settings_applied: false`; it does not initialize Torch/CUDA to
observe or apply them. A future owned launcher must apply and verify them
before any model allocation. The comparator's loaded-runtime records must
verify the same constants for both weights and both phases.

The future launcher must also verify actual loaded LoRA A/B and head tensors
against the publication, frozen BF16 base parameters, FP32 trainable LoRA/head
parameters, active unmerged adapters, finite nonzero first-step trainable
gradients and absent base gradients. File hashes and configured sampling do
not replace these observations. No such tensor or gradient check is claimed
by the CPU preflight.

## Fresh request and receipt formats

No example request supplies invented host paths. A future independently
reviewed CPU staging step must capture actual absolute paths and hashes on
the selected host. `scripts.run_frontier_training_v8` accepts `--request`,
`--plan`, `--freeze`, their expected hashes, `--expected-commit`, and exactly
one of `--preflight-only` or the currently unavailable `--execute`.

The request has `schema_version: 1`, the plan's `protocol_id`, a new
`attempt_id` of `frontier-v8-` followed by 16–64 lowercase hexadecimal digits,
an eligible `node_alias`, `source_commit`, six-package `runtime`, and
`runtime_receipt_sha256`. Required actual absolute paths are
`source_directory`, `dataset`, `released_checkpoint`, `task_directory`,
`training_run`, `comparison_output`, `completion_receipt`, `runtime_receipt`,
`resource_receipt`, `python`, `hf_cache` and `base_snapshot`.

The fresh task directory's name is the attempt ID and lies outside all
immutable inputs. Its request is `execution-request.json`; its fixed children
are `training`, `comparison`, `training-completion.json`,
`cpu-stage-receipt.json` and `resource-ready.json`. Input/source/cache paths
are outside that directory. Existing training, comparison or completion
outputs, old locks, snapshots and unexpected task-root artifacts are refused.
The preflight makes no writes; its output can later be saved independently as
`cpu-preflight.json`. A valid request passed to `--execute` consumes an
exclusive `attempt.lock.json` and produces an exclusive failure receipt,
including when preflight fails. The unavailable launch cannot retry itself
or overwrite a prior attempt. An invalid/ambiguous task path is rejected
before creating an artifact.

The fresh CPU stage records `schema_version: 1`,
`status: cpu_staged_no_cuda_initialization`, `cuda_initialized: false`,
`protocol_id`, `attempt_id`, `source_commit`, `plan_sha256`, `python`,
`python_sha256`, six-package `runtime`, `runtime_identity` (all locked
`versions` and `distribution_roots`, plus six `package_origins`), `hf_cache`,
`base_snapshot`, `base_snapshot_files_sha256`, and
`base_snapshot_symlink_targets`. An optional `overlay` is absolute and has
`overlay_files_sha256`. Reusing a historical stage under a new attempt is
rejected by its source/plan/attempt/path bindings.

The future training completion wrapper is separate from the input preflight.
Its required identities are `schema_version`, `protocol_id`, `status: complete`,
`source_commit`, `source_sha256`, `plan_sha256`, `freeze_receipt_sha256`,
`execution_request_path`/`execution_request_sha256`,
`training_preflight_receipt_sha256` (the comparison request supplies the path
as `training_preflight_receipt`), `runtime_receipt_path`/`runtime_receipt_sha256`,
`training_run`, exact six-package `runtime`, `gpu_uuid` and `visible_devices`.
It declares `completed_steps: 733`, `consumed_rows: 2932`,
`fixed_final_checkpoint: true`, `defer_heldout: true`, `resume_training: false`
and `automatic_promotion: false`. Its `artifacts_sha256` cover `run.json`,
`summary.json`, `training.jsonl`, `calibration.jsonl` and `reload_check.jsonl`;
`checkpoint` binds every final file and the directory hash; `calibration`
binds its journal hash, ordered-ID hash, count 496 and fitted temperature.
No such completion or runtime/resource evidence is generated in this phase.

The CPU completion helper checks actual trainer journal records, which lack
an `options` field, against immutable Calibration rows. It verifies all 733
contiguous finite optimizer steps and monotonic elapsed time, the published
initial identity, the fixed-final/deferred/no-resume boundary, Calibration-only
fitting, the recomputed Calibration reload error, and six finite phase records.
Allocated tensor and allocator-reserved peaks include resident weights and
allocator caching; they are not total device or minimum VRAM. Configured
row consumption follows the frozen loop; individual microbatch IDs are not
independently observed by the existing journal.

## Requirements for a later owned launcher

Only `ms-n1-1`, `ms-n1-3` and `ms-n4-1` through `ms-n4-4` are eligible.
N1-2 and all N8 nodes are excluded despite broader historical inventories.
Resolve the current alias/address/hostname, boot identity and ownership afresh.
Protect physical GPU0/1 and their R-KV work; keep unrelated processes outside
the control tree. Prefer a genuinely free H100 on the six allowed nodes.
Low utilization alone does not establish a free or reserved GPU.

Before a loan, capture current UID, PID birth ticks, boot ID, ancestry/nonce,
all original GPU UUIDs, configuration/optimizer/checkpoint content hashes and
queue progress. Verify a complete recoverable optimizer-boundary checkpoint.
The existing xiaofan pause scope covers only the freshly verified owned queue
tree and its sampling services. Reserve every original GPU for restoration
even if Jev uses one; never control a process through a stale numeric PID.

A new Linux pidfd controller and an independently running restoration guard
must be reviewed and bound before any pause or allocation. Require current
kernel process identities, live parent/guard relationships, a held exclusive
reservation and ownership evidence, the matching physical UUID and current
boot, plus a live no-compute/idle-baseline GPU probe. A `ready` boolean or an
old controller receipt cannot supply this authority. No such controller,
reservation or live probe is implemented by the CPU driver or comparator.

The future controller must bound the complete training-plus-comparison episode
to at most 4,500 seconds and provide up to 1,500 seconds for restoration. On
any error, timeout, interruption, owner loss or failed safety gate, terminate
only the new attempt's verified session/nonce tree and restore the original
checkpoint/configuration/services. Independently verify the four original
training ranks, the distributed coordinator, all three sampling endpoints,
original hashes and resumed queue progress. Preserve failure and restoration
proof; no automatic retry, resume, threshold change or promotion is permitted.

Training and comparison are separate phases. Training saves the fixed final
checkpoint and fits only Calibration. The comparator must finish all 496
Calibration predictions and fitting for **both** weights before any heldout
forward, then emit twenty complete Test/OOD journals with 1,896 predictions
per weight. CPU checks, a prepared command and synthetic safety gates grant
no model-run authority, prove no natural/official benchmark result and cannot
remove historical or initializer exposure.
