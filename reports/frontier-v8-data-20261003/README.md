# V8 original controls: CPU data audit

This stage prepares original synthetic supervision for a later candidate.
The 2,976 CC0 rows were generated once, then independently checked without
changing the generator or data. No model inference or training occurred.
There is no model improvement, natural-user result or official JevBench score.
The frozen Train-only coverage report motivated the design; it does not
establish the cause of previous model failures.

| Split | Rows | Whole parents | Essential composition width |
| --- | ---: | ---: | --- |
| Train | 992 | 14 | 1, 2 |
| Calibration | 496 | 7 | 3 |
| Validation | 496 | 7 | 4 |
| Test | 496 | 7 | 5 |
| OOD | 496 | 7 | 6 |

Joint consent and policy tasks require width+1 roles/topics. All seven families
and their declared disposition schedules occur at each width. These splits
deliberately shift composition depth, including Test, Calibration and
Validation. They are not IID or natural business data.

The [visible contract](contract.json) specifies signed temporal corrections,
exact ledger arithmetic, joint consent/capacity, latest valid required
policies, aggregated account status, department routing and refund priority.
Unknown status, security review and review reimbursement receive positive
Choice supervision. Each authored index has two Choice presentations and
one true/one false Noul proposition: 1,488 Choice rows and 1,488 Noul rows
(744 yes, 744 no). Gold comes from visible rules and typed facts.

| Evidence | Actual result |
| --- | --- |
| Independent rule oracle | 2,976 targets, 264 essential-input interventions, 2,232 complete typed presentation comparisons passed |
| Scope support | 90 joint cells and 60 policy cells across split, first/last position, scope field and kind/truth |
| Numeric candidates | All five answer ranks covered across six authored balance cases per width; maximum count difference one |
| Isolation | No compiled/normalized row or complete-parent collision; no quarantine; all 42 essential baselines passed |
| Old observed-data screen | Fresh screen of all 1,128 reserved v4–v7 Test/OOD rows; original eight files unchanged |
| Source cross-review | 16/16 checks; additionally 2,232 identity/layout comparisons, 744 Choice candidate sets and 144 rank annotations passed |
| Focused tests | 44 passed: generator 7, independent oracle 23, isolation 14 |
| Repository regression | 1,081 tests: 996 passed, 85 skipped, no failures |

The independent oracle author did not read the generator source or old
heldout bodies. The isolation checker read reserved inputs privately and
reported aggregate counts/fingerprints to the author. The source cross-reviewer
authored the generator, so that review is not another independent gold source.
See [oracle evidence](independent-oracle-r4.json),
[isolation evidence](independent-isolation-r1.json),
[inventory binding](independent-isolation-source-inventory-r1.json),
[cross-review](source-cross-review-r1.json) and
[root verification](root-cpu-verification-r1.json).

Rows, cases and scaffolds are different counts: 744 authored indices reduce
to 742 exact projected causal facts and 353 symbolic scaffolds. Width-one
account endpoint probes include aliases; eight old atomic scaffolds recur.
Shared individual rules are disclosed, rather than claimed as new independent
tasks. Full cross-split abstract case/parent checks passed at distinct depths.

There are 54 order-indistinguishable cases. Each of the 744 indices has only
one Noul layout for both truths; its alternative layout is unobserved within
that parent. Middle roles/topics are essential, but only first/last mismatch
probes are fully crossed. Causal projection covers seven reviewed schemas;
the conservative neutral-subset screen is bounded to 16 operands. Neither
check proves arbitrary graph isomorphism or pretraining novelty.

Three failed CPU oracle attempts remain in [failure history](failure-history/).
R1 depended on policy-map insertion order; R2 depended on request-line order;
R3 conservatively rejected valid opaque-ID renaming. The final checker uses
a single injective identity map, dynamic topic keys, exact typed facts,
canonical UTC instants and declared record reversal. An added rank gate and
foreign-identity adversarial fixtures preserve audit integrity. All repairs
affected audit code/tests; original generator, contract, manifest and split
bytes stayed unchanged.

## Reproduction and freeze boundary

Source and dataset freeze are recorded separately. The original
[manifest](manifest.json) and contract retain their creation-time provisional
flags. They must not be silently rewritten to remove historical pending states.
The external freeze receipt, added after a real source commit, is authoritative
for reviewed data status. This commits only CPU preparation; initializer,
training mixture, steps and comparison protocol remain unfrozen and unrun.

To reproduce into a fresh directory, using the source commit named by the
freeze receipt:

```sh
python -m scripts.build_frontier_controls_v8 --output data/frontier-controls-v8-reproduction
python -m scripts.audit_frontier_controls_v8 --dataset data/frontier-controls-v8-reproduction --output runs/v8-reproduction-oracle.json
python -m unittest tests.test_frontier_controls_v8 tests.test_audit_frontier_controls_v8 tests.test_frontier_isolation_v8
```

The reproduction directory and receipt must not already exist; create `runs/`
first if necessary. Compare all five split hashes and the manifest with the
freeze receipt. Reserved old datasets are required to replay the isolation
screen; no opaque cache alone establishes that screen. Keep old observed
Test/OOD outside Train and preserve all completed v6/v7 attempts.
