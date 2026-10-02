# Fixed Open-Jev-2B v6 adaptation: completed, not promoted

One frozen pass completed: 2,768 Train rows, 692 steps × accumulation four,
seed 20261002, rank eight, BF16 frozen backbone and FP32 LoRA/head. The exact
released 2B initialized this run; no upstream author rows were imported.
Training source is `1ecbabb77d506d948dc7dd548f1cedb2f3f8f189`; evaluation source
is `58d45064c98f133a614addc0a3bab712707a1b4c`. No Test/OOD retuning or retry
occurred. The original [plan](../openjev-hf-data-20261002/prepared-training/comparison-plan.json)
remains the historical predeclaration, with its bytes unchanged.

## Quality and regressions

Released and final weights were loaded sequentially in the same H100 runtime:
392 rows per model, 784 complete predictions. Both logit states were evaluated
at released temperature 1.518796342858676 and adapted Calibration-only
2.2147554738542654. The table compares each model at its saved temperature;
argmax does not depend on either positive temperature. Four-cell metrics and
subgroups are preserved in [summary.json](summary.json).

| Fixed synthetic slice | Argmax correct | Brier | NLL | Noul correct | Noul accepted errors | Noul accepted / coverage denominator |
|---|---:|---:|---:|---:|---:|---:|
| v6_test | 20 → 32 / 36 | 0.600 → 0.184 | 1.008 → 0.322 | 2 → 7 / 9 | 1 → 1 | 3 → 8 / 9 |
| v6_ood | 21 → 32 / 36 | 0.525 → 0.240 | 0.956 → 0.427 | 3 → 6 / 9 | 1 → 1 | 4 → 7 / 9 |
| v4_test | 86 → 114 / 128 | 0.487 → 0.132 | 0.885 → 0.246 | 17 → 26 / 32 | 6 → 3 | 23 → 29 / 32 |
| v4_ood | 83 → 113 / 128 | 0.503 → 0.126 | 0.908 → 0.245 | 14 → 25 / 32 | 9 → 1 | 23 → 26 / 32 |
| v5_test | 15 → 24 / 32 | 0.580 → 0.371 | 0.856 → 0.570 | 1 → 4 / 8 | 1 → 0 | 2 → 4 / 8 |
| v5_ood | 17 → 25 / 32 | 0.565 → 0.351 | 0.806 → 0.553 | 3 → 5 / 8 | 2 → 0 | 5 → 5 / 8 |

These are synthetic controls. V4/v5 selections are previously observed
regression controls, not fresh blind evaluations. This is not a natural-request
or official JevBench improvement claim. Noul uses inclusive 0.2/0.8, and its
abstentions count as incorrect in the Noul-correct column. Accepted errors
must be interpreted with coverage. Ordered per-row timings include
compilation/tokenization and do not establish a speedup.

The independent standard-library replay verifies all four metric cells,
subgroups, paired changes, journals and injection pairs within 1e-12:
[independent-replay.json](independent-replay.json). Predictions and
[per-row errors](independent-errors.json) are retained for inspection.

## Why the candidate stays experimental

All eight remaining v6 argmax errors are joint approvals. Refund-priority
and untrusted-policy families score 12/12 in each v6 split. Latest consent
revocation scores 6/6 across those small synthetic selections; this does not
establish general withdrawal handling. Six benign/injected pairs are both
correct with the same decoded action (released: four of six).

Important unresolved failures remain:

- All 12 outside-window Choice requests select `accept` although gold is
  `reject`. Across all kinds, outside-window Test falls 2/8 → 0/8 and OOD
  5/8 → 1/8, despite the overall v5 gain.
- A missing-scope approval selects `execute` at probability 0.853169 when
  additional consent is required.
- One-cent-over-capacity Noul predicates have only 2/6 argmax-correct answers;
  three threshold decisions are accepted and two are wrong. Capacity equality
  scores only 2/3 Test and 1/3 OOD.
- Thirteen previously correct rows become wrong (v6: one; v4: six; v5: six).
  All six v4 regressions are numeric candidates; numeric Test remains 6/20,
  while numeric OOD falls 7/20 → 6/20.
- Latest signed-policy withdrawal/missing-authority cases are absent from
  these v6 heldouts. Natural routing and independent sealed performance have
  not been measured for these weights.

No new weights replace the release. The next candidate should address
rejection boundaries, exact numeric comparisons and scoped capacity, preserve
these observed controls, and declare fresh heldouts before training. Existing
Test/OOD rows must not be promoted to Train.

## Execution evidence

The run consumed all 2,768 selected Train rows once. All 692 journal steps are
contiguous with finite loss/gradient norms. Optimizer time was 548.662 seconds;
training plus its baseline, Calibration, saved-checkpoint reload and heldout
checks took 629.102 seconds. Tensor allocation peaked at 4.593 GiB; this is not
a minimum VRAM requirement or total device-memory measurement. Reload
probability error was zero in the runner's recorded check.

Actual starting LoRA/head tensors matched the release, the base stayed frozen,
and first-step A/B groups had nonzero finite gradients. All 60 A and 60 B
saved tensors changed, as did head.weight. The [CPU weight-delta check](final-weight-delta-verification.json)
uses the final completion receipt and initializes no CUDA context. The exact
[observing driver](experiment_driver.py), [step journal](training-training.jsonl),
[run metadata](training-run.json), [Calibration logits](training-calibration.jsonl)
and checkpoint file hashes are retained; candidate weights stay in the
private experiment artifacts.

The borrowed card was returned to the original queue. Its independent
[restoration receipt](queue-restoration.json) verifies the four actual workers,
original optimizer/sampler tree, unchanged retry budget and real responses
from all three sampling services. The automatic verifier counted the
distributed coordinator as a fifth worker; after independent restoration
proof, only the two completed CPU monitors were ended. The original queue
continued sampling. An earlier controller startup used a Python build without
pidfd and failed before any queue control or model loading; it was corrected
to the validated system Python. No training attempt was retried. Process
birth IDs and device UUIDs are omitted from public summaries.

To replay the published predictions, reconstruct the exact frozen dataset as
described in [the preparation](../../docs/policy-training-v6.md), then run:

```bash
python3 reports/policy-v6-training-20261002/independent_replay.py \
  --comparison reports/policy-v6-training-20261002 \
  --dataset data/policy-training-v6-20261002-r1 \
  --output /tmp/open-jev-v6-independent-replay.json
```
