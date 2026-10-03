# Frozen Train coverage diagnostics

The CPU diagnostic reads only the exact v7 mixture `train.jsonl`, pinned to SHA256
`83fa0a04e04c43fa21a86aa799b642dad5fcb1c3e69ffb7a8b92df42c759b6c0`.
It reports authored training support before a separately declared next candidate.
It makes no model calls and does not create examples, change training, fit
parameters or read Calibration, Validation, Test or OOD row bodies.

Run from the repository root with a fresh output file:

```sh
python -m scripts.audit_boundary_train_coverage --output /tmp/jev-train-coverage.json
python -m unittest tests.test_boundary_train_coverage -v
```

The output uses exclusive creation. The input must be named `train.jsonl`, including
after symlink resolution, match the exact frozen hash, and contain only Train
rows. The script imports existing pure CPU fact helpers from the independent
comparison replay; it invokes no replay campaign, evaluator, model or lifecycle
entry point. It hashes the helpers for reproducibility. It does not call the
all-split oracle audit, whose directory reader opens heldouts.

The report separates the state-derived disposition from the decision gold:
Choice gold is an action, amount or relation; Noul gold is `no` or `yes` for the
visible proposition. It retains source and scenario family, authored condition,
question kind and gold option position. Amount labels stay exact; a distinct
amount is not counted as a distinct business action. Missing Choice gold means
that a disposition occurs in Train states but is never a correct Choice answer.
It does not mean that the model cannot select that answer.

Signed capacity and limit margins come from valid current state records. They
use request amount minus the minimum complete valid granted capacity, or request
amount minus the current active policy limit. Missing, revoked or withdrawn
authority rows are reported separately. Active-policy margins include fraud
routes, where fraud precedence still determines the outcome. Temporal outside
margins are signed seconds from the nearest interval endpoint. Numeric comparison
margins use exact integer subtraction.

The three v7 structural cross-tabs cover scope field × affected role × kind/truth,
equivalent-outside side × kind/truth, and negative-comparison direction ×
kind/truth. They expose small counts and explicitly list unpopulated expected
cells. They describe Train only; they do not assert complete heldout coverage.

The public [static report](../reports/train-coverage-diagnostics-20261003/README.md)
documents narrow margins, fixed role vocabulary and older supervision gaps.
These findings support design questions. They do not establish why the v7 model
failed, validate a new candidate, or justify reusing consumed training attempts.
Any future data and training require a separate declaration and new heldouts;
observed evaluation rows cannot be copied or rewritten into Train.
