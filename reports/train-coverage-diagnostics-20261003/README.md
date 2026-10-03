# Static Train coverage, 2026-10-03

This report audits the exact 3,792-row frozen synthetic Train mixture only. It
reads no heldout row bodies and makes no model calls. The detailed
[coverage.json](coverage.json) binds Train SHA256
`83fa0a04e04c43fa21a86aa799b642dad5fcb1c3e69ffb7a8b92df42c759b6c0`
and the CPU implementation. Existing frozen datasets, predictions, results and
consumed attempts remain unchanged.

| Source | Train rows | Groups |
| --- | ---: | ---: |
| v4 frontier controls | 2,400 | 600 |
| v5 temporal windows | 128 | 32 |
| v6 original policy controls | 240 | 60 |
| v7 boundary controls | 1,024 | 128 |

The mixture contains 2,588 Choice and 1,204 Noul rows. Noul supervision contains
733 `no` and 471 `yes` labels; the older v4 component contributes 431 `no` and
169 `yes`. These are target counts, not model probabilities or accepted decisions.

Three non-money dispositions appear in Train states without ever being a correct
Choice answer in the whole mixture: `unknown` (100 rows), `security review`
(100), and `review reimbursement` (20). Source/family tables expose additional
older gaps which newer v7 examples cover under other contracts. In particular,
v4 timeline `review` and v6 scoped approval `request higher capacity` occur only
as Noul states in those source families. These are supervision representation
findings, not demonstrated causes of model errors.

Each v7 family has 256 rows and 32 eight-condition groups. Each authored
condition has 16 Choice, 8 Noul `no` and 8 Noul `yes` rows. All 18 scope field ×
affected required role × kind/truth cells are present, with only 1–4 rows per
cell. Equivalent-outside before/after and negative-comparison below/above also
cross all three kind/truth categories: 8 Choice, 4 Noul `no` and 4 Noul `yes`
per side. Train does not have an absent-cell problem in these specific crosses;
the cell counts are small.

State-derived numeric support is narrow:

| v7 quantity | Observed support |
| --- | --- |
| Joint amount minus minimum complete granted capacity | 0 cents: 64 rows; +1 cent: 64; no negative margin |
| Amount minus current active policy limit | 0 cents: 96; +1 cent: 96; no negative margin; includes fraud routes |
| Outside-window margin from nearest endpoint | −1 second: 80; +1 second: 80; no farther outside requests |
| Integer comparison left minus right | −1 cent: 48; 0: 32; +1 cent: 48 |
| Balance ledger length | 2 entries: 96; 4 entries: 32 |

Another 128 v7 joint rows and 64 v7 authority rows lack complete granted or active
authority and are excluded from execution margins. V7 uses only the required
role pair `inventory guardian` / `release approver` and policy authority
`settlement controls` in Train. Latest valid records already occur at varied
array positions. The data therefore supports array-order contrasts while
offering limited role naming and numeric magnitude variation. Ledger maximum
entry sizes range from 2,801 to 88,976 cents. Return durations are limited to
3,601, 86,403 and 172,801 seconds.

These counts suggest questions for a separately designed dataset: broaden safe
and unsafe numeric margins, vary authority names independently of the decision,
and cover every disposition with both Choice and balanced Noul supervision.
No examples or next-candidate plan are created by this diagnostic. Any new
candidate needs original data, declared group isolation and fresh heldouts;
observed v4–v7 Test/OOD rows must remain outside Train.

Seven focused checks pass: heldout filename and alias rejection before reads,
wrong-hash rejection before parsing, non-Train row rejection, hand-authored exact
margin arithmetic and blocked-authority exclusion, state-gold consistency and
input immutability, and an exact-Train read/count/cross-tab check. See the
[method and command](../../docs/train-coverage-diagnostics.md).
An [independent review](independent-validation.json) passed all 48 checks,
recomputing targets and coverage from Train alone. In a clean checkout, the
six fixture/gate tests always run; the exact-Train integration check skips
when the ignored frozen artifact is absent.

The [v7 comparison](../boundary-v7-comparison-s2-20261003/README.md) remains the
model evidence. Its failed safety gates are unchanged. This static report is
neither a capability improvement, an official JevBench result, a natural
application test nor a reason to publish candidate weights.
