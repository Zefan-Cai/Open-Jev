# Evaluation versions and independent results

This ledger separates the historical public-subset experiment, independent
benchmark versions and new evaluations that have not run. The scores use
different metrics and task mixtures; do not interpret them as an accuracy trend.

[Machine-readable snapshot](../site/evaluation-ledger.json) ·
[Benchmark tables](benchmarks.md) · [Deployment](deployment.md)

## Historical 231-public protocol

Pinned JevBench commit: `f8ce71361165846101d02ebc83ad44e47ae44fc3`.
The evaluated tasks were 72 Original, 48 Easy and 111 Hard: 231 public tasks
from the then 534-task benchmark. The other 303 tasks were unavailable.

| Released model | Correct / 231 | Correct / 111 Hard |
|---|---:|---:|
| Open-Jev-2B | 150 | 46 |
| Open-Jev-9B | 179 | 66 |
| Open-Jev-27B-v1.1 | 197 | 80 |
| Jev 1.13.0 | 200 | 81 |

These are raw reference-match counts, not the current leaderboard composite.
The [original method and audits](jevbench-public.md) retain checkpoint identities,
normalization rules and candidate-order limitations. The 27B ran through
four-rank direct execution on shared H100 GPUs; its timing does not establish
single-GPU or HTTP latency. These historical results remain unchanged.

## Independent v1.4: historical report

The [evaluator's issue #3](https://github.com/Zefan-Cai/Open-Jev/issues/3)
reported 9B at #59 / composite 11.20 and 2B at #60 / 9.98. For 9B it reported
29.9% sealed accuracy versus 77.5% on the public slice. The discrepancy is
retained here, including the [maintainer follow-up](https://github.com/Zefan-Cai/Open-Jev/issues/3#issuecomment-5943119185).

The v1.4 sealed set used hard-style decisions and its public and sealed
difficulty mixtures differed. That limits a direct gap interpretation; it
does not establish that difficulty explains every error. Aggregate run
manifests are needed to distinguish task difficulty, checkpoint/configuration
and generalization. This issue is a historical record, not the current result.

## Independent v1.5.4 snapshot

Observed on 2026-10-02 UTC (2026-10-02 PDT), from the
[versioned board](https://benchmarkheaven.com/jev-models/v1.5.4) and its
[aggregate API](https://benchmarkheaven.com/api/jevbench/v1.5.4).
Each completed system has 1,624 decisions: 904 open and 720 sealed.
Only 601 of the 904 open decisions are published. The v1.5 method matches
open/sealed tier mixtures and uses per-item chance correction.

| Released model | Rank / composite | Open competence | Sealed competence |
|---|---:|---:|---:|
| Open-Jev-9B | #46 / 24.356 | 61.786 | 65.855 |
| Open-Jev-2B | #65 / 9.073 | 38.786 | 28.343 |

**Competence is chance-corrected, not raw accuracy.** Composite also includes
calibration, speed and cost under this version's rules. Rank changes across
versions do not isolate a model improvement or regression. In particular,
the historical v1.4 29.9% is not the current 9B sealed result.

For Noul, this protocol calls probabilities at or below 0.2 No, at or above
0.8 Yes, and the interval between them abstention. Abstentions count as wrong
for competence. The reported overall Noul abstention rates are 29.293% for
2B and 13.131% for 9B. Interpret primitive performance alongside raw accuracy,
coverage, thresholds and calibration; low competence alone does not diagnose
an answer-key or decision-head bug.

The local evaluator's raw measured P50/P95 are 0.563/1.641 seconds for 9B and
0.429/1.240 for 2B on its H100 host. The leaderboard also uses a synthetic
`x2 + 0.15 seconds` transport adjustment and estimated cost. Those adjusted
values are not measured latency/cost of our own hosted service. The JSON
snapshot records the original source-response hash and these distinctions.

## Fresh independent Open-Jev-27B-v1.1 evaluation: pending

No corresponding 27B v1.1 result was present in the observed v1.5.4 aggregate.
Preparing a handoff or rerunning the old public tasks does not fill this cell.
The independent evaluator controls its frozen suite and sealed task access.
Only aggregate results and a reproducible run manifest are requested; private
questions or labels are not needed.

Prepare an executable handoff from the published code checkout:

```bash
python3 scripts/prepare_independent_evaluation.py \
  --output runs/independent-27b-submission
```

The output pins the loader commit, model repository revision, checkpoint
checksum, base revision, saved temperature and explicit request settings.
It contains an install/serve script, an offline checksum/runtime verifier and
an optional reproduction of the old 231-public protocol. Follow its README
on an evaluator-owned NVIDIA host. The default 16,384-token evaluation limit
is an explicit override of the saved 4,096-token training limit; use
`--max-length 4096` for a separate submission at the saved limit.

New handoffs carry the observed Linux x86-64 / Python 3.11 / CUDA 12.8
dependency version lock and its checksum. Installation uses those exact
versions; a metadata-only preflight rejects lock, package-version or venv
location drift and runs `pip check` before model allocation. Preparation does
not install the environment or load weights, and the version lock is not an
artifact-hash lock or a completed clean-install replay. The older published
handoff and its original validation remain unchanged.

Record hardware, actual denominators, attempted/failed/missing requests,
per-primitive accuracy/calibration/coverage, warmup/retry policy, raw timings
and cost assumptions with the returned aggregate. The new sealed status
remains **pending independent evaluator** until those results exist.

## Controlled synthetic v6 adaptation: completed, candidate not promoted

The [fixed 2B run](../reports/policy-v6-training-20261002/README.md) completed
692 steps over 2,768 Train rows. A same-runtime comparison used 392 decisions
per checkpoint and retained both temperature ablations. Independent raw-logit
replay reproduced every metric cell and subgroup.

| Fixed slice | Released argmax correct | Adapted argmax correct |
|---|---:|---:|
| v6 Test | 20/36 | 32/36 |
| v6 OOD | 21/36 | 32/36 |
| observed v4 Test / OOD | 86/128 / 83/128 | 114/128 / 113/128 |
| observed v5 Test / OOD | 15/32 / 17/32 | 24/32 / 25/32 |

These synthetic controls do not fill an official or natural-request result.
All 12 outside-window Choice requests still select `accept`, joint capacity
and missing-scope failures remain, and 13 previously correct rows become
wrong. Noul thresholds, accepted errors, coverage, calibration and per-row
regressions are retained in the report. The final weights remain experimental;
released model results above stay unchanged.

## Controlled synthetic v7 adaptation and S2 comparison: completed, safety failed

The [v7 result report](../reports/boundary-v7-comparison-s2-20261003/README.md)
records the fixed 948-step pass over 3,792 Train rows and the separately
declared S2 comparison of its completed checkpoint. The original comparison
failed before writing predictions; its historical report and failure remain
preserved. S2 produced 904 predictions per weight on the same runtime and
physical H100. Independent CPU replay reproduced all 32 metric cells, four
safety cells and eight paired slices.

| Fixed slice | Released argmax correct | V7 argmax correct |
|---|---:|---:|
| New v7 Test | 142/256 (55.47%) | 206/256 (80.47%) |
| New v7 OOD | 142/256 (55.47%) | 182/256 (71.09%) |
| Observed v4 Test / OOD | 86/128 / 83/128 | 115/128 / 115/128 |
| Observed v5 Test / OOD | 15/32 / 17/32 | 25/32 / 24/32 |
| Observed v6 Test / OOD | 20/36 / 21/36 | 31/36 / 33/36 |

**All four weight/temperature cells fail the predeclared safety gates.** The
candidate selects a dangerous Choice action on 6/128 Test and 8/128 OOD rows.
All 12 old outside-window Choice requests still select `accept`; the release
rejected five. All six old latest-revocation controls are correct. At the
Calibration-only temperature and inclusive Noul .2/.8 thresholds, temporal
coverage reaches 33/64 with no accepted errors, but numeric, joint-capacity
and latest-authority families have 6, 12 and 6 accepted errors respectively.
There are 34 previously correct rows that become wrong across all 904 rows,
including six new Test and seventeen new OOD regressions.

These are synthetic predicate/Choice selections, with no real business
execution. Old slices are observed development regressions, not fresh
independent scores. The report retains calibration, coverage, every paired
regression and diagnostic timings with cold outliers; those timings establish
no deployment speedup. The candidate remains experimental and unpublished.
Natural business blind tests and the fresh independent 27B evaluation above
remain pending.

The [2026-10-03 training-input audit](../reports/frontier-v8-training-isolation-20261003/README.md)
subsequently found 151 legacy v4 Train rows matching 39 observed v4 input
fingerprints after renaming and reordering, with exact typed account/event
facts and rules retained. This adds a concrete exposure limitation to the v4
regression slices above; their recorded counts remain unchanged. A separate
future mixture excludes complete parents, including additional conservative
scaffold matches. This preparation does not erase historical exposure or
establish a new model result.
