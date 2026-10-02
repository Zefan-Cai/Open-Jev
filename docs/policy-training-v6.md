# Prepared original-policy mixture v6

The fixed adaptation from the exact released Open-Jev-2B has now completed.
The [full report](../reports/policy-v6-training-20261002/README.md) records
692 steps, the same-runtime comparison, independent replay and unresolved
rejection/capacity failures. The candidate is not promoted. The preparation
below remains the original frozen protocol; the failed exclusion-boundary v5
checkpoint was not the initialization. Upstream author training/demo/test
rows contribute zero.

## Allocation and immutable preparation

The only training inputs are **2400 v4 Train + 128 v5 Train + 240 original v6
Train = 2768 distinct rows**. Source Calibration and Validation remain separate
from Train and from one another: 240 + 32 + 36 = 308 rows in each split. Their
per-source files are also retained under `heldout/`. Primary Test and OOD each
contain only the 36 original v6 rows. The previously observed v4 128+128 and
v5 32+32 selections stay under `observed-regression/`, with their original
selection order and locks; they are development regression controls.

After committing the preparation, training and generator changes, choose a
fresh output and supply the checkpoint/output paths on the future training host:

```bash
python -m scripts.prepare_policy_training_v6 \
  --v4 data/frontier-controls-v4-20261002 \
  --v5 data/temporal-windows-v5-20261002-r1 \
  --v6 runs/openjev-hf-data-20261002/original-policy-v6-candidate \
  --output data/policy-training-v6-20261002-r1 \
  --initial-checkpoint /path/on/training-host/released-2b/checkpoint \
  --training-output /path/on/training-host/policy-v6-adaptation
```

The standard-library preparation refuses an existing output or uncommitted
preparation/training/generator bytes. Its manifest pins the source commit,
implementation, all three source manifests, all five split byte hashes and
normalized row hashes. V6 must also match the fixed public candidate manifest
and its manifest/split hashes; rewriting a local manifest cannot replace it.
Original source rows, metadata and targets are retained.
Group/input/entity checks run across all source splits before output creation.
No Test, OOD, Calibration or Validation row is promoted into Train. A local
freeze binds a committed SHA; the operator must push that same SHA before a
future remote run. The preparation does not itself prove a push or execute one.

The completed CPU [freeze receipt](../reports/openjev-hf-data-20261002/prepared-training/freeze-receipt.json)
verifies pushed source `1ecbabb77d506d948dc7dd548f1cedb2f3f8f189`, all five
mixture files, ten retained heldout files and zero Train/heldout group overlap.
The public [manifest](../reports/openjev-hf-data-20261002/prepared-training/manifest.json)
and [comparison plan](../reports/openjev-hf-data-20261002/prepared-training/comparison-plan.json)
pin those exact bytes and settings. To recreate this exact manifest, run the
preparation at that source commit with fresh output; later source commits
intentionally produce a different source identity even when row bytes agree.
The recorded training-host paths are staging targets, not existing validated
model directories. No GPU training was launched by this freeze.

`comparison-plan.json` records exact settings and a structured training argv
for `python -m jev.train`: **692 steps × accumulation 4**, one full shuffled
pass through all 2768 rows, rank 8, length 4096, learning rates 2e-5/5e-5,
Brier weight 0.1 and seed 20261002. It pins the published released-package
identity. Preparation never loads model weights; stage and verify the exact
checkpoint files on the training host before invoking that argv. Main
Calibration supplies 308 separate rows; Validation is preserved and is not
consumed by the current runner.

## Released initialization and future comparison

`jev.train --initial-checkpoint` validates serving files, model/revision/rank/
length, method and the actual PEFT configuration before importing Torch. It
rejects explicit adapter base/revision conflicts and records the full config.
The released adapter leaves base name empty and revision null; those fields
are unspecified, while `model.json` supplies the pinned loading identity.
LoRA A/B and the head are enabled for gradients; the base remains frozen.
Training output must be separate from the source checkpoint and its ancestors.
Content hashes bind initialization in run metadata/training identity. Omitting
this flag keeps the existing fresh-base constructor behavior.

This minimal initialization mode **requires `--checkpoint-every 0` and cannot
combine with `--resume-training`**. It produces no resumable training snapshots.
An interruption requires a separately declared future attempt. The existing
fresh-base resume behavior remains separate.

The runner saves initialized/trained logits and performs its existing
calibration/reload checks. It does **not** automatically perform the plan's
full future comparison: recompute released/final weights in the same runtime,
apply both released and newly fitted calibration-only temperatures to both
logit states, and evaluate the preserved observed regressions. Report argmax,
Brier/NLL/ECE and Noul accuracy/coverage/accepted errors at inclusive 0.2/0.8.
Keep integer-cent boundaries, exact authority scopes, latest revocations,
untrusted-text pairs, temporal exclusions and state tracking visible. Neither
the fixed final checkpoint nor the thresholds may be retuned using Test/OOD.
Synthetic gains would not establish natural-request or official JevBench
progress, and no automated promotion is part of this preparation.

## Fixed comparison runner

`scripts.compare_policy_training_v6` implements the declared comparison after
training finishes. It requires the controlled driver's completion receipt,
which binds all final checkpoint files and the run, summary, step journal and
Calibration logits. It verifies the frozen mixture, selected rows, complete
692-step journal, initialization and Calibration-only temperature before
loading either model. The evaluation checkout must be a pushed commit whose
training and generator bytes still match the frozen training source.

```bash
python -m scripts.compare_policy_training_v6 \
  --dataset /path/to/policy-training-v6-20261002-r1 \
  --released-checkpoint /path/to/released-2b/checkpoint \
  --training-run /path/to/policy-v6-adaptation \
  --completion-receipt /path/to/completion-receipt.json \
  --output /path/to/fresh-comparison \
  --expected-commit EVALUATION_COMMIT
```

The runner loads released and adapted weights sequentially on the same CUDA
device and records 392 predictions for each: v6 Test/OOD 36 each, v4 128 each
and v5 32 each. Raw logits support the four combinations of model weights and
released/adapted temperatures. Reports retain family and boundary breakdowns,
paired decision changes and benign/injected outcome consistency. Input and
checkpoint hashes are checked again after inference. An incomplete prediction
or changed input prevents a complete summary. Per-row synchronized times are
recorded for diagnosis; ordered loads and compilation do not establish a
speedup. CPU checks of this runner do not establish a training result.
