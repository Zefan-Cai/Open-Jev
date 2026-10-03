# V7 comparison-only forensic supplement

Execution status, 2026-10-03: S2 completed and its [independently replayed results](../reports/boundary-v7-comparison-s2-20261003/README.md)
fail all four safety cells. Its controller, driver, stage and attempt locks
are consumed and must not be restarted. The protocol below documents the
completed experiment; it is not authorization for another execution.

The original v7 training attempt completed all 948 finite optimizer steps.
Its required comparison exited with code 1 before writing any of the sixteen
prediction journals: a raw UUID equality check rejected the bare UUID form
returned by PyTorch against the NVIDIA `GPU-` form in the allocation receipt.
Historical same-device runtime evidence and an independent CPU reproduction
establish this guard defect; the failed v7 call did not persist its own runtime.
No comparison scores or safety-gate result exist for that attempt.

The first supplement, `boundary-v7-comparison-s1-20261003`, consumed two
CPU preflights without reaching inference. Its source bundle omitted the
separately frozen training commit; an objects-only continuation repaired that
input, then the source-overlap check rejected the required nested checkout.
Both failure logs and the original declaration are preserved in the
[CPU failure report](../reports/boundary-v7-preflight-layout-20261003/README.md).
S1 must not be executed again or have its inputs or logs overwritten.

The separately declared `boundary-v7-comparison-s2-20261003` runs inference
only, using the already completed checkpoint. It is an explicit supplemental
evaluation, with a new source commit, task directory, resource acquisition,
session, attempt lock and output directory. It does not retry training, resume
an optimizer, overwrite the failed comparison or replace any original receipt.

## Preserved origin and unchanged experiment

The committed [declaration](../reports/boundary-v7-forensic-comparison-20261003/declaration.json)
binds the original evaluation commit, request, completion, failure log,
lock-only comparison inventory, final controller receipt and independent full
restoration audit. The audit must confirm checkpoint/config/retry preservation,
four training ranks, three current sampling services, protected GPU0/1 and
all owned controller/guard/driver exits. Merely restarting a supervisor does
not satisfy this gate.

Training remains at source `87eb8b419685d272f059b3696e0f547e47ebdcc6` and
original evaluation `37729b2340e8d00fb211784d2181dd167db4f96b` remains intact.
Dataset, published initializer, completed checkpoint, base snapshot, overlay,
interpreter/package versions, BF16 base/FP32 head, physical GPU4, Calibration
temperature, 904 rows per weight and all predeclared metrics/safety gates are
unchanged. Only strict physical UUID representation, explicit supplemental
provenance handling and the exact source placement check change. Complete
UUIDs with an optional exact `GPU-` prefix and hexadecimal case variation denote the same device; different
UUIDs, MIG identifiers, ordinals and malformed identities are rejected.
Raw runtime records are preserved, and `visible_devices` remains exact.

## New request and CPU preflight

After the new source is pushed, latest CI passes and integration completes,
stage that exact source in a new task root outside the original task and
immutable data. Its source must be the actual canonical directory
`new-task/evaluation-source`, exactly alongside the request. A sibling, deeper
checkout, task root or symlink to a foreign checkout is rejected. Bidirectional
overlap with the original task, dataset or released checkpoint still rejects.

The new deployment bundle must include the integrated evaluation commit and
the separate frozen training commit `87eb8b419685d272f059b3696e0f547e47ebdcc6`.
Before upload, clone the actual bundle in a fresh local directory, check out
the declared evaluation commit and verify every current and frozen source
blob required by the comparator. Verify the same objects remotely before
preflight. An integrated squash history alone does not retain the frozen
training commit. Preserve all original input paths, including the original
`execution-request.json` and `completion-receipt.json`. Copy the immutable
terminal controller/audit bytes to `original-recovery/` in the new task.

Create `comparison-supplement-request.json` with schema version 1, kind
`v7_comparison_only_supplement`, supplement ID above, the new
`evaluation_commit`, committed `declaration_sha256`, frozen `plan_sha256`,
`original_task`, `original_controller_receipt`, `original_restoration_audit`,
and exact original `dataset`, `released_checkpoint`, `training_run`,
`completion_receipt`, six-package `runtime` and `gpu_uuid`. Its
`comparison_output` must be the new task's `comparison/`; `resource_receipt`
must be that task's `resource-ready.json`, and `controller_plan` must be
`controller-plan.json` there. The latter is created after CPU preflight and
binds the request, preflight, wrapper and independently reviewed controller.
It also fixes the canonical controller command (`/usr/bin/python3`, the
new task's `pause_v7_comparison_restore.py`, `--execute`) and working directory.
Use actual absolute paths, not
saved future-host commands. No training settings or temperature override is
accepted.

Run `scripts.run_boundary_comparison_supplement_v7` with `--request`,
`--expected-commit` and `--preflight-only`. This performs the full existing
comparison's CPU input/checkpoint/Calibration checks, verifies all origin
hashes and independently restored resource evidence, checks current package
versions and exclusively writes `comparison-supplement-preflight.json`.
It makes no model calls and cannot acquire or pause a GPU.

## Fresh bounded acquisition and one comparison

Read current MS ownership evidence again. GPU0/1 remain protected. A new
controller and live restoration guard must capture current checkpoint,
optimizer, config, UID/PID birth/boot/ancestry/nonce and GPU identity. Preserve
GPU2–7 for original-queue restoration; never reuse the consumed a2 controller
or its PIDs. Apply the same strict dual fresh empty GPU checks and bounded
restoration protocol from the original run. Review the new controller on CPU
and independently review its fresh capture before any pause.

Only the new controller may invoke the supplemental wrapper without
`--preflight-only`. The wrapper compares its CPU preflight binding, requires
its own session and creates `attempt.lock.json` exclusively with fsync before
inference. The lock binds the preflight, controller plan, resource bytes and
fresh controller identity. The comparator checks that exact preflight and all
six current package versions again, then the live controller parent, same
session/lock, fresh resource proof with mandatory live guard, and one empty
physical GPU before model loading. The
supplemental CLI cannot bypass these checks. Existing outputs or locks reject
a second attempt. On failure, preserve the new failure receipt and restore;
there is no automatic retry, new training or promotion.

Verify the parent's current command and working directory between two PID
birth checks. The forked restoration guard must have its own positive PID and
birth identity, matching UID/boot and controller parent; it cannot alias the
controller or inference process. The attempt lock records a timezone-aware
start time, and both launch and independent replay check that it follows the
resource check by at most 120 seconds. Launch additionally checks freshness
against current time and the live guard. Offline replay does not require an
already restored process to remain alive.

Released/final inference, the four temperature cells and the independent CPU
replay retain the original frozen protocol. Replay independently validates
the original-to-supplement provenance and new lock/resource evidence, without
importing the supplemental validator or scorer. Raw journals are hashed and
all scores, gates and paired regressions recomputed. Preserve failed gates
and retain the candidate as experimental. Predicate decisions and Choice
selections are not actual business execution; no natural-request or official
JevBench gain is established by this synthetic experiment.

Only after full independent restoration and complete replay may a result
report be integrated or announced. A completed training pipeline is a real
milestone; an unexecuted request, CPU fix or supplemental declaration is not
evidence of model improvement.
