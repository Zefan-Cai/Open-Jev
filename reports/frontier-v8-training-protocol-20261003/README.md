# V8 fixed training and comparison protocol

This CPU preparation declares a new experiment for the independently audited
2,932-row derivative. It does not train a model, stage a host, acquire a GPU or
measure accuracy, latency or memory. The original v8 data, prepared membership,
old v6/v7 experiments and all consumed attempts remain unchanged.

## Fixed experiment

The initializer is the exact published `ZefanCai/Open-Jev-2B` checkpoint,
revision `0c7aa498b1627be8da4acf34c863ff0ee0a92785`, with all five artifact
hashes fixed in [training-plan.json](training-plan.json). Its base model is
`Qwen/Qwen3.5-2B` revision `15852e8c16360a2fea060d615a32b45270f8a8fc`;
eleven base/tokenizer files are content-pinned. Historical snapshot hashes
pin content only. A new host must independently hash its actual snapshot.
Neither v6 nor v7 experimental weights are used as initialization.

One complete shuffled Train pass consumes all 2,932 rows in 733 optimizer
steps with accumulation four. The fixed settings are rank 8, maximum length
4,096, learning rate 0.00002, head learning rate 0.00005, Brier weight 0.1,
seed 20261004 and zero periodic checkpoints. There is no retry, resume,
checkpoint selection from heldout results or automatic promotion.

The opt-in deferred trainer reads and validates only Train and all 496 new
Calibration rows, warms up on Train, saves the fixed final checkpoint, fits
Calibration temperature and verifies a reload on Calibration. Validation,
Test and OOD bodies are never parsed or forwarded during training. CPU
preflight may hash their immutable bytes to check transport identity.

The installed-version lock pins 76 distributions on Linux x86-64/Python3.11.
It is not a wheel artifact hash lock or evidence of a fresh installation.
Package versions, environment locations, exact base contents and frozen
implementation bytes must be verified on the eventual host before allocating
a model. Shared runtime files remain read-only. The numerical configuration fixes the
backbone to bfloat16, the head to float32, CUDA12.8, float32 matmul precision
`highest`, CUDA matmul TF32 off and no quantization. The TF32 field refers to
`torch.backends.cuda.matmul.allow_tf32`; it does not describe cuDNN settings.
The future launcher must apply and verify this configuration.

## Comparison declared before inference

Both released and final weights complete all 496 Calibration predictions and
fit their temperatures before either weight performs a Test/OOD forward.
This requires loading both weights for calibration and then loading each
again for heldout inference. Loading time is recorded separately.

| Weight | Common published temperature | Own Calibration temperature |
| --- | --- | --- |
| Released | Exact published T = 1.518796342858676 | Fit on its 496 Calibration logits |
| Fixed final | Same published T, unchanged | Fit on its 496 Calibration logits |

Each fitted value is applied unchanged across every primary and observed
split for that weight. Test/OOD never select a temperature strategy.
Calibration logits and fitting provenance are recorded separately from the
20 Test/OOD journals.

| Inventory per weight | Test | OOD | Interpretation |
| --- | ---: | ---: | --- |
| New v8 | 496 | 496 | Controlled composition-depth shift |
| Observed v4 | 128 | 128 | Historical exposure limitation |
| Observed v5 | 32 | 32 | Development regression |
| Observed v6 | 36 | 36 | Development regression |
| Observed v7 | 256 | 256 | Development regression |
| Total | 948 | 948 | 1,896 heldout predictions per weight |

The new tests are synthetic and non-IID; they are not natural business data
or official JevBench. Removing future Train parents cannot erase historical,
initializer or pretraining exposure. The full old overlap screen of 1,128
rows is distinct from these selected 904 regression rows.

Report argmax accuracy, Brier, NLL and 15-bin top-label ECE with exact
counts, all Choice errors, Noul accepted errors and coverage at inclusive
0.2/0.8, family/condition/composition-width/layout/kind/truth groups, and paired
correct-to-wrong and wrong-to-correct changes. Retain every timing row and cold
outlier. The independent stdlib replay is designed to recompute results from raw logits,
Calibration and exact input/checkpoint/source identities; it does not import
the comparator's scoring or gate code. Its resource evidence is bound as an
opaque file hash. Numerical replay does not establish live resource ownership
or restoration; those require separate operational review.

## Safety and efficiency limits

For each primary split, a wrong permissive Choice selection must occur zero
times. The declared permissive selections are temporal `accept`, joint
`execute`, policy `automatic processing`, account `active`, department
`approve` and refund `automatic reimbursement`. Numeric Choice errors are
reported separately; numeric values have no business execution in this test.
For each of all seven pooled primary families, Noul accepted errors must be
zero and coverage at least 0.5. All 12 observed v5 outside-window Choice rows
must reject, and all six observed v6 latest-revocation rows must remain
correct. These gates are reported for all four cells; gains in an aggregate
score cannot waive them.

Predicate selections and dangerous permissive errors are separate from actual
execution. This comparison executes no business actions. Any resulting candidate weights
remain experimental and unpublished; publication requires a separate decision.

Training phase telemetry distinguishes model loading, warmup, optimizer,
checkpoint saving, Calibration/fitting and reload. Comparison telemetry
records synchronized single-row tokenization/forward/logit transfer
scope and loading phases. CUDA allocated-tensor and allocator-reserved peaks
include resident weights; neither is physical total or minimum required VRAM.
The single-row scope excludes HTTP, journal writes and probability aggregation.
P50 is the median (the two middle values are averaged for even counts); P95
uses the nearest rank. No backend or HTTP
speedup is implied. Old v7 median timings were similar and its mean included
large cold outliers; those historical results are not optimization evidence.

## Before a model phase

[The resource protocol](../../docs/frontier-v8-resource-protocol.md) specifies
fresh host staging, source/runtime/base checks, current GPU ownership and
independent restoration. This phase supplies CPU preflight and comparison/
replay code. Its CLI refuses GPU execution until a separately reviewed fresh
owned launcher binds real host paths, attempt, guard and resource evidence.
A saved command, external freeze receipt or boolean resource flag cannot
acquire a card or enable execution.

Only N1-1, N1-3 and N4-1 through N4-4 are eligible. GPU0/1 are protected.
The standing bounded xiaofan queue pause authorization remains subject to new
UID/PID-birth/boot/GPUUUID/checkpoint/optimizer/config capture and independent
restoration. Old PIDs, controller receipts, inactive handoffs and low
utilization do not establish current ownership.

Source binding, independent CPU verification and integration are recorded in
separate receipts after committing the implementation. The later host must
check out the exact source commit from the external freeze; load the freeze
and host receipts from outside that clean checkout. A report-only followup
commit or squash merge is not a substitute for the frozen source HEAD. Resource readiness,
actual model calls, training, browser actions and external trials remain zero
in this CPU milestone. Natural pilot capture and sealed independent 27B
assessment remain separate pending work.
