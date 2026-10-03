# Support routing with a review queue

The workflow takes human-written support text or a CSV, proposes a BANKING77
intent with real local Open-Jev probabilities, applies a frozen confidence
policy, lets a reviewer correct uncertain rows, and exports a final CSV.
Unreviewed uncertain rows remain unresolved; it never performs banking actions.

The English interface is available from the existing model server at
`http://127.0.0.1:8791/examples/support-routing/`. It contains 12 natural
official-training examples and a label index. It calls the server that served
the page; it has no simulated model or fallback oracle. The displayed model
identity and probability vector come from the response. Without an imported
calibration lock, every predicted row waits for human review.

## Prepare and freeze the evaluation

```bash
python -m scripts.evaluate_support_routing prepare \
  --output data/support-routing-20261002 \
  --calibration-rows 96 --test-rows 256 \
  --demo-dir examples/support-routing
```

The script downloads the two BANKING77 CSVs at fixed upstream revision
`57ec275d8078af65b7731c2a98be812d844a6d6b` and verifies their known SHA-256s.
The original dataset has 10,003 train and 3,080 test utterances across 77
human-labeled intents. Attribution: Casanueva et al. (2020), *Efficient Intent
Detection with Dual Sentence Encoders*, PolyAI BANKING77, [CC BY 4.0](https://github.com/PolyAI-LDN/task-specific-datasets/tree/57ec275d8078af65b7731c2a98be812d844a6d6b/banking_data).

Normalized official-test texts are reserved before deduplicating train.
Calibration takes 96 deterministic whole utterances from official train;
the remaining 9,851 training utterances build 77 intent documents for BM25.
The official test sample is independently shuffled with the saved seed and
never enters retrieval training or threshold fitting. Inputs and gold are
stored separately, and every input file is frozen in `manifest.json`.

Candidate selection uses the text alone: BM25 selects eight labels from the
77-label catalog, followed by an explicit human-review candidate. Gold never
inserts the correct candidate. This means routing failures can come from
retrieval or from the model; candidate recall and retrieval top-1 accuracy
make that distinction visible. This is a nine-candidate retrieval-and-routing
pipeline, not a direct 77-way classifier score.

## Measure a released checkpoint

Start the actual 2B, 9B or 27B server using the usual pinned local checkpoint
command. Then, from a second terminal:

```bash
python -m scripts.evaluate_support_routing evaluate \
  --dataset data/support-routing-20261002 \
  --endpoint http://127.0.0.1:8791/v1/systemone \
  --output runs/support-routing-2b-20261002
```

The runner measures calibration first, chooses the highest empirical coverage
that reaches 90% accepted accuracy with at least ten accepted calibration
rows, and writes `decisions.lock.json` **before collecting test predictions or
opening test gold**. If no threshold meets the policy, it sends every row for
review. It never retunes on test. Calibration and test must use the same model,
checkpoint hash, base revision, temperature, method, code revision and context
limit. Identity drift stops the run.

`summary.json` reports routing accuracy, candidate recall, accepted coverage,
error among accepted decisions, review rate, failed requests and HTTP wall
time. Constant-majority and retrieval-top-1 baselines use the same test rows.
All failures remain in the denominator; raw probability vectors and requests'
hashes are kept in the prediction journals. The threshold is an empirical
calibration selection, not a statistical guarantee. With 256 test rows,
accepted-error estimates may remain uncertain.

Import a measured `decisions.lock.json` into the browser only after its held-out
results meet the intended acceptance policy. A lock from another model is rejected. All measurements in a claim
must name the actual checkpoint size; offline fixture tests are not model
performance.

## Measured 2B pilot, October 2, 2026

The released Open-Jev-2B checkpoint was measured on one H200 using the Torch
backend, BF16 backbone and its released temperature 1.518796342858676. Code
commit `80ca8e81d08992cb2a4cbb6a0caa1355ad3e5aee`, checkpoint hash
`3076462e6356412082e79af909227b39b2863b90def79155ca0821aa506b7ded`, and
pinned base revision `15852e8c16360a2fea060d615a32b45270f8a8fc` are recorded
in the [raw summary](../reports/support-routing-20261002/summary.json).

| Measure | Calibration, 96 rows | Frozen test, 256 rows |
| --- | ---: | ---: |
| Model routing accuracy | 60/96, 62.5% | 166/256, 64.8% |
| BM25 retrieval top-1 accuracy | 79/96, 82.3% | 211/256, 82.4% |
| Gold intent in candidate set | 93/96, 96.9% | 249/256, 97.3% |
| Accepted coverage | 31/96, 32.3% | 88/256, 34.4% |
| Errors among accepted | 3/31, 9.7% | 14/88, 15.9% |
| Manual-review demand | 65/96, 67.7% | 168/256, 65.6% |
| Failed requests | 0 | 0 |

The calibration-only threshold was 0.7581042723393133 and remained unchanged
on test. Test HTTP wall time was 21.06 seconds in total. The model performed
worse than the lexical retrieval baseline, and the held-out accepted accuracy
of 84.1% missed the intended 90% target. Keep the browser's default review mode;
this checkpoint is not ready to automate this routing task under that policy.
The measured lock is retained as evidence, rather than adjusted using test.
These results apply to this 2B routing pilot, not the 9B or 27B checkpoints.

The frozen policy would accept 88 test rows and send 168 for manual review.
Fourteen of those 88 accepted predictions disagree with the benchmark labels.
Those are measured routing errors, not observed customer rework. No human
review was completed in this campaign: the saved reviewer and correction
fields are empty. Review demand does not measure completed work or saved
staff time, and this pilot has no production users or banking actions.

## Browser workflow verification

The [genuine browser capture](../reports/support-routing-20261002/browser-demo.png)
and [smoke receipt](../reports/support-routing-20261002/browser-smoke.json)
record 12 real released-2B model calls on the natural official-training demo
utterances, with no page errors. No confidence lock was imported; all 12
proposals entered the review queue. The
[exported final CSV](../reports/support-routing-20261002/demo-final-unresolved.csv)
keeps all 12 unresolved with no human label or reviewer. This demonstrates
model inference, queueing and export; it is not test accuracy or completed
human review.

The browser used a local page and an SSH bridge to the real model API, so UI
elapsed time includes transport overhead and is not inference latency. The
receipt pins the tested UI at source commit `80ca8e81d08992cb2a4cbb6a0caa1355ad3e5aee`.
The genuine screenshot retains the heading shown during that run. The subsequent
heading-only change to “Review each proposed support decision” and both UI hashes
are recorded. The later human no-route option described below was not part of
that original browser smoke.

## Finish human review

The browser provides a catalog selector for every row and requires a reviewer
name for human corrections. Export either the review queue or the final CSV.
For a request outside the catalog, choose **No matching banking intent**.
The CLI uses `reviewed_label=__review__` with a named reviewer for the same
final human disposition. The export records it as a human decision; an
unreviewed model `__review__` proposal remains unresolved.
The CLI supports the same last step: edit `reviewed_label` and `reviewer` in
the saved review queue, then run:

```bash
python -m scripts.evaluate_support_routing finalize \
  --queue runs/support-routing-2b-20261002/review-queue.csv \
  --index data/support-routing-20261002/index.json \
  --output runs/support-routing-2b-20261002/final.csv
```

The export records whether the final label came from the model, from a named
human, or remains unresolved. It does not infer a review label from gold.
The actual human-review completion count must come from saved reviews; merely
testing the correction mechanism with fixtures is not a completed human pilot.

The [real opt-in pilot protocol](natural-support-pilot.md) keeps request intake,
blind labels and human corrections separate and defines misroutes, abstention,
actual rework and completed trials. Its evidence-capture helper and real intake
are still pending; the protocol does not establish new natural-user results.

## Limits

Released checkpoints may already have trained on BANKING77 official train.
This is a supervised intent-routing evaluation, not an unseen-task zero-shot
claim. This campaign's calibration and retrieval subsets are separate from
each other, but they were not necessarily absent from older checkpoint
training. BANKING77 supplies no official out-of-scope requests, so review
rates are abstention and failure rates, not a measured OOS detection score.
The natural dataset represents a benchmark, not new production users.

```bash
python -m unittest tests.test_support_routing
```
