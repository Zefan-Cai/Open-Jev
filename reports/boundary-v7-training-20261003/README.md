# Open-Jev-2B v7: training completed, comparison failed

Later update, 2026-10-03: the separately declared [S2 comparison](../boundary-v7-comparison-s2-20261003/README.md)
completed with independent replay; all four safety cells failed. This report
preserves the original attempt and its failure. Its outcome and original
evidence are unchanged; the candidate remains experimental and unpublished.

The frozen 948-step adaptation completed and saved an experimental candidate.
The required published-versus-candidate comparison failed at its runtime
identity guard before writing any comparison predictions. There is no valid
v7 comparison score or evaluated safety gate, and the candidate has not been
promoted or published. [outcome.json](outcome.json) records this boundary.

## Frozen training and observed execution

Training source was `87eb8b419685d272f059b3696e0f547e47ebdcc6`; the original
evaluation source was `37729b2340e8d00fb211784d2181dd167db4f96b`. The
[predeclared plan](../boundary-controls-v7-20261002/prepared-training/comparison-plan.json)
and [protocol](../../docs/boundary-v7-run-protocol.md) fixed 3,792 shuffled
Train rows, 948 steps with accumulation four, seed 20261003, rank eight,
maximum length 4,096 and 436 Calibration rows. No training retry or resume
occurred. The driver reports 3,792 consumed rows; it explicitly records that
direct per-row consumption was not observed.

The exact published `ZefanCai/Open-Jev-2B` initializer at
`0c7aa498b1627be8da4acf34c863ff0ee0a92785` was used. Its loaded LoRA and head
tensors matched the release, the BF16 backbone stayed frozen, and the FP32
LoRA A/B and head groups had finite first-step gradients with nonzero group
maxima. Base gradients were absent. Both initialization and gradient proofs
passed and are bound by the original completion receipt.

| Recorded measurement | Value | Scope |
|---|---:|---|
| Optimizer journal | 948 steps | Contiguous steps 1–948; finite loss and gradient norm |
| Optimizer-stage elapsed time | 729.106 s | Last training-loop journal timer |
| Complete frozen training pipeline | 850.086 s | Loading, internal baseline/final inference, Calibration, training, saving, cleanup and validation |
| Peak allocated tensors | 4.593 GiB | PyTorch journal measurement on an H100 80GB; excludes total/reserved GPU memory and does not establish a minimum VRAM requirement |

These measurements establish execution completion. They do not establish an
inference speedup. Exact values and measurement scopes are in `outcome.json`.
The candidate checkpoint directory SHA256 is
`0eef8f5fdbc3a6a0fdbeb3e020bd6bab5b5dd8a0d261831999e790c1e48087c6`;
the original training completion SHA256 is
`694486383d755c0a436c38b5f99ba7ff85bcac85058326f899f19659d4b7da05`.

## Comparison failure and score boundary

The original comparison returned code 1 with
`Published/final comparison runtime or recorded training packages differ`.
Its preserved output contains only `comparison.lock.json`: zero raw comparison
journals, zero comparison runtime records and no comparison summary. The
planned 904 rows per checkpoint, sixteen journals and four weight/temperature
cells were not completed. Quality scores, paired regressions and safety gates
are therefore unavailable.

The frozen training pipeline also produced its own internal prediction
artifacts. Their scores are not substituted for the required comparison here.
This report makes no model capability, official JevBench or natural-business
blind-test improvement claim.

An independent CPU reproduction found a sufficient cause: the legacy helper
records Torch's bare physical UUID, while the v7 guard compares it directly
with the allocation receipt's `GPU-`-prefixed representation. All other
recorded runtime terms passed in the reproducer. The failed v7 comparison
raised before persisting its runtime; the diagnosis used a historical
same-node runtime and the actual v7 completion, without reading a new v7 CUDA
device property. [The diagnosis summary](uuid-format-diagnosis.json) preserves
this evidence limit. The proposed correction normalizes only complete
physical UUIDs during comparison, preserves raw strings and continues to
reject different GPUs, MIG identifiers and other runtime changes.

## Preservation and restoration

All 29 files in the completed-training snapshot match its saved hash inventory,
including the six candidate checkpoint files, execution evidence, locks and
original failure log. Their original bytes remain preserved.

The borrowed resources were returned to the original workload. An independent
audit passed all 38 restoration checks, covering its optimizer/sampler tree,
configuration, four workers and one coordinator, real completions from three
sampling services, protected GPUs and exit of every owned controller, guard,
driver and descendant. The audit SHA256 is
`f7597cf2979365c99513ab9bd3dbd48bf97f81edd399fba95b9939c67f5a80cf`.
Unrelated workload process and environment details are omitted from this
public summary.

Public evidence includes [training completion](training-completion.json)
(device UUID and local request path redacted, original hash retained),
[original execution completion](experiment-completion.json) (unchanged bytes),
and the CPU diagnosis summary. A separate [comparison-only forensic protocol](../../docs/boundary-v7-comparison-supplement.md)
and [declaration](../boundary-v7-forensic-comparison-20261003/declaration.json)
are prepared and have not started as of this report. They require a new
declaration and output; training and the failed comparison will not be rerun
or overwritten under the old protocol. The candidate stays experimental.
