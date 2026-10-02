# Original v7 boundary data — audited, not trained

The [completed v6 report](../policy-v6-training-20261002/README.md) documents
32/36 correct on each new synthetic split alongside unresolved authorization,
numeric and severe outside-window rejection failures. This follow-up adds
**1,792 original CC0 rows in 224 whole eight-case groups**: 1,024 Train,
128 Calibration, 128 Validation and 256 each Test/controlled OOD. No upstream
training/demo row was imported and no model was run for this candidate.

The four families cover temporal rejection/equality/exception/UTC rendering,
integer-cent balances/comparisons, exact joint scope/latest capacity and
signed-policy withdrawal/missing authority. Question type and proposed truth
are balanced by condition. Independent state-derived checks caught and fixed
an initial structure/question-type shortcut before this data was frozen.
The small Calibration/Validation and scope-field cross-product limitations
remain recorded rather than claimed as full coverage.

- [Candidate lock](candidate-manifest.json) and [complete manifest](dataset-manifest.json).
- [Independent rule/condition replay](independent-audit.json): all 1,792 rows and 224 complete groups pass.
- [Original-source screen](original-source-overlap.json): all 4,000 rows across the original five v4/v5/v6 splits reserved against new copying, zero matches.
- [Visible-input screen](visible-overlap.json): public JevBench, natural routing and available author inputs, zero detected matches. Structured input fields exclude gold/predictions; source literals are unclassified. Partial lexical matching covers state strings, while full-input exact matching covers state/question/options.
- [32 original Train examples](sample-train.jsonl); raw upstream snapshots and full datasets remain ignored.
- [CPU verification](cpu-verification.json): 933 portable tests, 85 skipped; the final binding checks are separately rerun and GitHub CI is required.

The [protocol and commands](../../docs/boundary-controls-v7.md) fix a future
released-2B initialization and 3,792-row mixture, one shuffled pass of
**948 steps × accumulation 4**, with separate Calibration, primary and
observed regressions. The v7 comparison runner and current GPU handoff must
be verified before launch. Preparation is not a runtime enforcement mechanism
inside `jev.train`, and a saved argv does not establish that a campaign ran.
No synthetic gain, natural-user result, official JevBench score or weight
release is claimed here.
