# Original policy controls v6

This preparation produced **384 original CC0-1.0 synthetic rows from 96 four-variant groups**, with zero upstream examples imported. The subsequent fixed v6 training and independent comparison have [completed](../reports/policy-v6-training-20261002/README.md). Rejection and authorization failures keep its weights experimental. The original preparation protocol below records the data's origin; its builder does not run training.

Public error aggregates identified `temporal_numeric` and `long_policy` as weak development families; the [v4 analysis](frontier-controls-v4.md) describes those observations and their limits. External [OpenJev demos](https://github.com/abhishekgahlot2/openjev-server/tree/032a2c5791f3d8856cc26fdb6876c106fac8dbf8/demos) motivated the abstract refund, approval and guardrail uses. Original test text, labels and predictions were not imported, rewritten or used to generate cases. The [Hub/source audit](openjev-hf-data.md) explains why those public evaluation journals are reserved.

## Exact decision contracts

| Original family | Contract and four variants |
| --- | --- |
| Explicit refund priority | Verified recall overrides proof and reimbursement eligibility. Without recall, missing verified receipt rejects; otherwise confirmed defect or intact seal permits reimbursement. The amount limit applies only to permitted reimbursement. Variants cover equality, recall plus missing proof, missing proof without recall, and one cent above the limit. |
| Scoped joint approval | Both named roles need verified signatures for the exact resource, operation and currency specified by their credential scope. For each role the latest valid sequence wins. Any latest revocation rejects; otherwise missing consent precedes capacity review; equality permits execution. Variants cover both consents, a later revocation, a mismatched credential operation, and one cent above capacity. Unsigned and wrong-resource events are distractors. |
| Untrusted policy conflict | First resolve the largest signed revision by the named authority with exact department credential scope. Missing or withdrawn latest policy requires verification. With active policy, confirmed fraud precedes amount routing. Customer text and quotations have no policy authority. Variants are benign input, an injected instruction with identical verified facts, a real signed-limit change, and confirmed fraud. |

All money is integer cents; floating-point dollar arithmetic is absent. A supplied `verified_signature` field represents a trusted verification result for a synthetic exercise. The generator does not implement cryptographic verification or establish a real workflow's trust boundary. Conflicting signed sequence/revision IDs are rejected rather than silently resolved.

Each family has 20 train groups and three groups in each calibration, validation, test and controlled OOD split. This yields **240 train rows and 36 in each other split**, with 288 Choice and 96 Noul rows overall. Group membership is fixed before labels; all four related variants stay together. Choice answer positions rotate with a maximum count difference of one within each family/split. Noul labels alternate by group index. The whole dataset and group IDs are locked by the [candidate manifest](../reports/openjev-hf-data-20261002/original-candidate/candidate-manifest.json).

Controlled OOD changes authority role names, resource prefixes, currency, threshold ranges and question wording within the same workflows. It does not establish transfer to an unseen business domain. The benign/injected policy pair is an invariance check, not two independent customer requests.

## Preparation and verification

```bash
python -m scripts.build_policy_controls_v6 \
  --output runs/openjev-hf-data-20261002/original-policy-v6-candidate
python -m unittest tests.test_policy_controls_v6 -v
python -m jev.data validate \
  runs/openjev-hf-data-20261002/original-policy-v6-candidate
```

The builder refuses existing outputs. All labels are computed from facts without variant IDs, and the manifest pins the generator and each split. Five CPU tests separately replay every label through a decision table/event fold; they also check 48 refund truth-table/boundary cases, scope/currency mismatches, superseding revocations, integer-cent boundaries, policy withdrawal, injection invariance, complete groups and frozen outputs. [Test evidence](../reports/openjev-hf-data-20261002/original-candidate/cpu-tests.txt) records five passes.

The [independent replay and overlap receipt](../reports/openjev-hf-data-20261002/original-candidate/audit.json) checks all 384 rows against all **231 public JevBench input projections**, our 256 natural routing test inputs and 96 calibration inputs, available author journal input fields, and public demo source string literals. It records zero detected matches and zero quarantined groups using exact visible request fingerprints, normalized state leaves of at least eight words and 13-token state shingles. Upstream label/gold/score/probability fields are not selected for screening or generation. Private/judge benchmark inputs were not accessed.

The benchmark source JSON contains expected answers; the private projection selects only IDs, state and question type/instructions/criteria. Neither the generator nor the screen accesses expected-answer fields. Projection hashes are in the [receipt](../reports/openjev-hf-data-20261002/original-candidate/jevbench-input-projection.json). Raw benchmark inputs, upstream journals and complete generated data stay ignored; the public [12-row sample](../reports/openjev-hf-data-20261002/original-candidate/sample-train.jsonl) contains only newly written training controls.

Author refund/filter journals omit full input state, prompt-injection journals truncate 94 texts, and game/browser records expose only partial state. The screen covers what is publicly visible and cannot establish semantic or foundation-pretraining independence. Public JevBench now supplies development feedback; it is not an untouched independent evaluation.

## Fixed future allocation

The fixed candidate contributes **240 additional original training rows: 80 per family**, with 144 reserved calibration/validation/test/OOD rows. External author data contributes **zero**. The candidate manifest is not a training launch plan and does not modify an existing mixture. A later experiment must bind its baseline checkpoint, retained replay-source hashes, fixed steps and selected groups before inference, then report original versus adapted accuracy, probability calibration and injected/benign pair consistency. Real natural-test and JevBench performance require their own separately declared checks.

The separate [v6 mixture preparation](policy-training-v6.md) supplies that fixed allocation and released-checkpoint initialization contract. Its fixed training and independent comparison have now [completed](../reports/policy-v6-training-20261002/README.md); rejection/capacity failures keep the candidate experimental. [New v7 controls](boundary-controls-v7.md) address those failure families without promoting observed heldouts into Train.
