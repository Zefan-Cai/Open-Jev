# Boundary controls v7: new data, no model result

The [completed fixed v6 experiment](../reports/policy-v6-training-20261002/README.md)
improved its small synthetic Test/OOD to 32/36 each, but failed every one of
12 observed v5 outside-window Choice requests. Missing authorization and
integer-cent/arithmetic failures also remain. Its weights stay experimental.
These **new original CC0 synthetic controls** address the failure families;
old Test/OOD questions are not copied, renamed or promoted into Train.

| New family | Counterfactual coverage |
| --- | --- |
| Temporal window | Before delivery, inclusive start/end, strict interior, after deadline, approved exceptions outside either end, and the same outside instant displayed with another UTC offset. Each group has three accept, three reject and two review outcomes. Offset, window and anchor are sampled separately. |
| Exact numeric | Negative/zero/positive balances, cancelling nonzero ledger entries, integer comparison below/equal/above, and negative operands. Cents and currency formatting use integer arithmetic; candidate answers need not put the result in the numeric middle. |
| Joint capacity | Equality, a later valid regrant, absent role, wrong exact credential scope, valid revocation, invalid high-sequence grant, one cent above capacity and a later capacity reduction. Revocation takes precedence over missing consent, which precedes capacity. |
| Latest authority | Equality, high invalid revisions, one cent above a limit, a later signed lower limit, confirmed fraud with active authority, withdrawal and missing authoritative policy. Resolve authority before fraud/amount routing. |

Every eight-variant group stays in one split. Choice/Noul assignment rotates
independently of the decision condition; each condition in each split has
Choice and both true/false Noul checks. Choice positions and Noul targets are
balanced. Dataset IDs, split labels and group metadata do not enter model inputs. Whole-group
splitting and independent state-derived label replay protect the candidate.
The supplied signature flags represent trusted synthetic facts; these controls
do not perform cryptographic verification or establish a real business trust boundary.
State-derived cross checks keep negative comparison direction, outside-window
side and affected approval role from revealing the question type. Train and
new Test/OOD cover each such structure with Choice and Noul yes/no. The four
Calibration/Validation groups per family cover each structure with Choice and
Noul, but do not cover both Noul truths within every structure. Scope field ×
role × question type is also sparse; this is not a full factorial evaluation.

| Allocation | New v7 | Prepared mixture |
| --- | ---: | ---: |
| Train | 1,024 | 3,792 |
| Calibration | 128 | 436 |
| Validation | 128 | 436 |
| Test | 256 | 256 |
| Controlled OOD | 256 | 256 |

The mixture reuses exactly the existing 2,768 original v4/v5/v6 **Train** rows.
Calibration and Validation remain separate. The previous observed v4
128+128, v5 32+32 and v6 36+36 Test/OOD slices are retained as development
regressions, outside Train. New v7 Test/OOD are fixed before training. OOD
changes authored roles, number/ledger ranges and timestamp representations
within these workflows; it does not establish unseen-business transfer.

## CPU reproduction and immutable preparation

```bash
python -m scripts.build_boundary_controls_v7 \
  --output data/boundary-controls-v7-20261002-r1
python -m scripts.audit_boundary_controls_v7 \
  --dataset data/boundary-controls-v7-20261002-r1 \
  --output /tmp/open-jev-v7-audit.json
python -m unittest tests.test_boundary_controls_v7 \
  tests.test_audit_boundary_controls_v7 \
  tests.test_prepare_boundary_training_v7 -v
```

The [public candidate lock](../reports/boundary-controls-v7-20261002/candidate-manifest.json)
binds every split, the generator and the visible-input screen. The screen
uses public JevBench input projections, natural routing inputs and available
author demo inputs/source literals. Structured projections select no
labels/gold/predictions; source literals are unclassified strings. Exact checks
cover state/question/options; partial lexical checks cover state strings.
Raw upstream material stays ignored and contributes
zero training rows. An additional identity-normalization screen rejects
renamed copies, reused states with changed questions/types and reordering of
contractually unordered lists against **all five** original v4/v5/v6 splits, and duplicates
within the new candidate. Neither screen proves arbitrary semantic or
foundation-pretraining independence.

After committing and pushing the preparation code and candidate lock:

```bash
python -m scripts.prepare_boundary_training_v7 \
  --v4 data/frontier-controls-v4-20261002 \
  --v5 data/temporal-windows-v5-20261002-r1 \
  --v6 runs/openjev-hf-data-20261002/original-policy-v6-candidate \
  --v7 data/boundary-controls-v7-20261002-r1 \
  --output data/boundary-training-v7-20261002-r1 \
  --initial-checkpoint /future-host/released-2b/checkpoint \
  --training-output /future-host/boundary-v7-adaptation
```

Preparation refuses existing outputs and uncommitted generator/audit/lock
bytes. Its future plan fixes published-2B initialization, one shuffled pass
of 3,792 Train rows, **948 steps × accumulation 4**, rank 8, maximum length
4096, learning rates 2e-5/5e-5, Brier weight 0.1, seed 20261003 and no resumable
snapshots or automatic retries. V6 weights are not the initializer. A saved
training argv is preparation evidence, not proof of a launch or a runnable
end-to-end campaign.

## Required next experiment and gates

**Do not launch yet:** the v7 same-runtime comparison runner, its protocol
and current GPU ownership/restoration still need verification. The CPU package
does not acquire a GPU, fit a temperature, make predictions or publish weights.

Before any launch, bind the exact source/data/checkpoint identities and fixed
final checkpoint. Recompute published/final logits in one runtime, then apply
both the published temperature and a temperature fitted only on Calibration.
Report argmax, Brier/NLL/ECE, all four families/conditions and inclusive
Noul 0.2/0.8 coverage and accepted errors. For each new primary split, require
zero erroneous Choice accepts/executions for outside windows, missing scope,
revoked/withdrawn/missing authority and over-capacity. For each family pooled
over new Test/OOD, require zero accepted Noul errors and at least 50% coverage;
report these predicate decisions separately from actual execution.

The observed safety checks require all 12 old v5 outside-window Choice rows
to reject and all six observed v6 revocation rows to remain correct. Preserve
every previously correct-to-wrong row and numeric-family results; an aggregate
increase cannot waive a safety failure. A passing synthetic campaign would
still require separate natural-request and independent official capability
evaluation before a reliability or release claim. Test/OOD may not choose
settings, temperatures, thresholds or intermediate checkpoints.
