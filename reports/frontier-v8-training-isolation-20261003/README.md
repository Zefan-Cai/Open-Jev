# V8 training-input isolation and deferred evaluation

The proposed 4,784-row concatenation was blocked before materialization.
A private audit found inherited equivalences between old Train and already
observed v4–v7 regression inputs. Original source data, old scores and the
frozen v8 dataset remain unchanged. This stage prepares a separately declared
whole-parent derivative and a Train/Calibration-only trainer; no model has
been trained or evaluated here.

## What the audit found

The old prepared Train contains 151 v4 state-tracking rows matching 39
normalized visible fingerprints in old v4 Test/OOD. A second typed account
comparison confirmed identical rule text, event sequence/status facts,
complete account-event relationships and the same reviewed query, including
the parsed Noul outcome, after identity renaming and record reordering.
Whole-parent closure for those confirmed matches is 47 groups /
188 Train rows. Matching does not consult targets or model predictions.

The 22 broader complete-parent scaffold signals repeat abstract menus:
12 v4 state-tracking, one v4 numeric, four v6 and five v7 scaffolds. Those
normalizations deliberately collapse magnitudes and origins. They are
conservative exclusions, not 22 additional independently established exact
clones. Retaining the same four blocking fingerprint modes, with shared
magnitude/time atoms disclosed separately, requires excluding a total of
335 old Train groups / 1,852 rows. This reduces coverage, including all old
v6 and v7 training parents; the next comparison must still retain their
observed regressions. Historical observed v4 results must not be interpreted
as fresh independent generalization evidence. Removing future Train parents
does not erase historical or initializer exposure.

| Membership | Rows | Groups | Choice | Noul |
| --- | ---: | ---: | ---: | ---: |
| Blocked raw proposal | 4,784 | 834 | 3,084 | 1,700 |
| Whole-parent exclusions | 1,852 | 335 | 1,133 | 719 |
| Declared derivative | 2,932 | 499 | 1,951 | 981 |

The derivative retains 1,812 v4, 128 v5 and all 992 new v8 Train rows.
Retained row bytes and relative order must be unchanged. The original
4,784-row proposal and its failure evidence remain available; there is no
silent gate relaxation or overwrite. Only opaque exclusion identities and
aggregate overlap evidence are passed to the materializer/data author.

## Split boundary

The [independent input screen](independent-inputs-r1.json) compares the complete retained Train with all
1,984 fresh v8 reserved rows and all 1,128 old observed Test/OOD rows.
The comparison inventory is separately bound: 904 selected observed rows
(v4 256, v5 64, v6 72, v7 512). Selected regression files are preserved as
bytes; they are not resampled during preparation. Cal/Val/Test/OOD remain
496 rows each at the frozen v8 composition depths. These are controlled
synthetic shifts, not IID, natural-business or official JevBench results.

All four blocking modes pass against both inventories. Eight shared old
magnitude/time atoms remain disclosed and nonblocking; no such collision is
present against the new reserved splits. See the [blocked raw proposal](blocked-full-proposal-r2.json),
[explicit exclusions](whole-parent-exclusion-declaration-r1.json) and
[preserved earlier audit](history/blocked-full-proposal-r1.json).

The materializer requires a passing input audit and an explicit exclusion
declaration, checks committed implementation bindings, preserves original
inputs and writes an exclusive provisional output. An independent audit
of that actual output is required before recording mixture preparation.
No executable training argv or initializer/step freeze is created here.

The [materializer review](independent-materializer-review-r1.json) and
[isolation review](independent-isolation-review-r1.json) bind their reviewed
source versions and hand-authored tests. The latter identifies four resolved
checker/proof-boundary findings; its draft-document hashes are historical.
The [focused repository check](root-focused-r1.json) passed 42 tests with no
skips. [Repository regression](root-full-r2.json) ran 1,124 tests:
1,039 passed, 85 skipped, no failures. Its first run correctly rejected the
new trainer in two old v7 staging fixtures. Only the temporary tests were
[repaired to use exact historical source](legacy-v7-fixture-receipt-r1.json);
the old driver, plans and guards remain unchanged. New negative checks prove
that edited/current source is rejected before data or checkpoint staging.
The [original failure and skip-count amendment](history/README.md) are preserved.
Actual output preparation is recorded separately below.

## Actual prepared output

Source `945f5af3cff3870b5752343f799e3d2c7ee02ef9` was committed, pushed and
confirmed on the remote branch before the materializer ran once. The actual
output is `data/frontier-training-v8-20261003-isolated-r1`. The independent
checker then ran once and verified all 13 JSONL files plus the manifest:
exact byte inventories, order, parent/row membership, provenance, source
commit, blocks and split counts. No original input or output was rewritten.

| Binding | SHA-256 |
| --- | --- |
| Prepared Train bytes | `67888a8f0cf88697902ab5fdac80ae41b91d8ee50e6dc949d156688404437702` |
| Prepared manifest | `ee8f8d689a748e2e7d71b43fa18e47183a5c7aca01a4bb39ffffe3f5dbd499a8` |
| Independent actual-output audit | `a7691eace88625783ff85b7f2cf98fb1115a0a7b5edb44f8f3ebe9a5f22e1f9c` |

The [prepared membership receipt](prepared-membership-receipt-r1.json),
[manifest](prepared-manifest-r1.json) and
[actual output audit](independent-output-audit-r1.json) fix the data identity.
The original provisional manifest flags remain intact. This external receipt
records completed CPU preparation; it freezes no initializer, training steps,
comparison or resource protocol and grants no model-run authority.

## Trainer boundary and limits

The opt-in `--defer-heldout` mode validates complete Train and Calibration
before sampling, hashes only those two files and warms up on Train. It
saves the fixed final checkpoint, fits temperature on Calibration and uses
Calibration for the reload check. It neither parses nor forwards Validation,
Test or OOD, and emits no initialized baselines or heldout metrics. The
legacy default retains its existing behavior. Deferred mode prohibits
resume and periodic snapshots. See [training documentation](../../docs/training.md).

Six synchronized phase timers record CUDA allocated-tensor and allocator-reserved
peaks. Allocated tensors include resident weights; reserved memory includes
allocator caching. Neither measures physical total or minimum VRAM. Timings include
phase file writes and optimizer logs. Guarded mock tests establish the file
and orchestration boundaries; actual numerical training, CUDA telemetry,
latency and memory performance remain unmeasured.

A new comparator/replay, exact published initializer, fixed finite training
budget, runtime lock and fresh resource/restoration protocol are still
required before any model phase. The old 1,196-step proposal is superseded
by the changed membership; no replacement budget is frozen in this stage.
Candidates remain experimental until separately declared safety gates pass.
Natural trials, new GPU actions and model calls in this stage are all zero.
