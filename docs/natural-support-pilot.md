# Real opt-in banking-support pilot: preparation protocol

**Status: awaiting real opt-in requests and an evidence-capture helper.** This
protocol has made zero model calls and completed zero external human trials.
It is not a completed blind test or an automated runnable pipeline. The
[BANKING77 pilot](support-routing.md) remains a benchmark result; its examples,
synthetic controls and browser smoke are not new customer requests.

The first trial uses the existing banking-intent catalog and local
`/examples/support-routing/` page. It proposes an intent, obtains human review
and exports a routing CSV. It performs no banking action. The generic hosted
workbench is a separate entry point: its eight-candidate limit and plain-text
request contract do not support this nine-candidate routing request unchanged.

## Intake and freeze before predictions

An opt-in participant supplies genuine support requests they are entitled to
share, with unnecessary account and contact details removed. Record consent
for this specific evaluation and retention scope. Benchmark copies, generated
requests, researcher-authored examples and internal smoke tests are excluded
from external trial counts. Source attestations are evidence of provenance,
not proof that every request is new to a model's pretraining.

Keep the private intake, consent records, labels and outputs outside the Git
checkout and its training `data/` tree, including resolved symlink targets.
Use opaque participant and conversation IDs. Keep all messages from one
conversation in the same cohort; do not split related messages across
calibration and blind evaluation. Reserve this entire cohort from training,
retrieval-index construction and later threshold fitting. Future development
must use a separate cohort and declaration.

The following names and values illustrate a schema only. They are
placeholders, not collected requests, participants or consent receipts:

```json
{
  "trial_id": "PLACEHOLDER_TRIAL",
  "participant_id": "PLACEHOLDER_PARTICIPANT",
  "external_participant": true,
  "source_kind": "real_opt_in_support_request",
  "consent_receipt": "PLACEHOLDER_PRIVATE_CONSENT_PATH",
  "requests": [{
    "id": "PLACEHOLDER_REQUEST",
    "conversation_group_id": "PLACEHOLDER_CONVERSATION",
    "source_created_at": "PLACEHOLDER_UTC_TIME"
  }]
}
```

Create `inputs.csv` with only `id,text`, matching the existing UI import and
runner input schema. An annotator who has not seen model outputs creates
`blind-gold.json` as an ID-to-label map, using the frozen banking catalog or
`__review__` for no matching intent/insufficient information. A second blinded
reviewer adjudicates disagreements before the gold file is sealed. Record
annotator roles and disagreements privately; never turn model proposals into
gold labels. Trial corrections later belong to a separate review artifact.

Before predictions, freeze SHA-256 commitments for `inputs.csv`,
`blind-gold.json`, intake/consent record, group assignments and the exact
`index.json`; freeze the review policy and expected model identity as well.
Store actual hashes, counts and a UTC freeze time in a private manifest. The
prediction operator receives inputs, index and the gold commitment, without
the gold file. A separate scoring operator opens gold only after prediction
evidence is complete and frozen. Hashes detect later changes; they do not by
themselves establish blinding or real provenance. Public reports should omit
request text, participant identifiers and consent contents.

## Use the existing loop, preserve the missing evidence

Use a released checkpoint and the [deployment guide](deployment.md). The
local page is `http://127.0.0.1:8791/examples/support-routing/`; inference uses
`POST /v1/systemone`. Candidate selection must remain text-only using the
frozen index and the existing eight-label shortlist plus `__review__`.

**Keep review-all mode: do not import the old confidence lock.** The released
2B BANKING77 lock accepted 88/256 test rows with 14 errors, missing its intended
90% accepted-accuracy target. Without an imported lock, the current UI sends
every successful prediction to review. A future acceptance policy requires
its own separate calibration cohort and frozen declaration; this blind
cohort cannot select its threshold.

The intended artifact flow is:

1. Freeze private intake, `inputs.csv`, index, blind-gold commitment and policy.
2. Import only `inputs.csv` into the local routing page and run once. Keep
   failed, pending and unsubmitted rows in the intended-input inventory.
3. Save a private per-row request/response journal before scoring. Preserve
   the exact candidate probabilities, selected label, request hash, HTTP
   status/failure and model metadata: model, checkpoint SHA-256, base revision,
   temperature, method, code commit and maximum length. Check normalized
   finite probabilities, argmax consistency and identical identity across
   the run. An identity change stops the trial; remaining rows stay pending.
4. Export the review queue. A named human reviews rows independently of the
   sealed gold. Save separate start/end timestamps for completed reviews,
   without estimating them from HTTP wall time or total browser duration.
5. Export the final CSV; freeze journal, review, timing and export hashes.
   Then let the scoring operator join completed predictions and gold by ID.
   Preserve the original freeze manifest and all failures.

These are protocol steps, not commands that already automate collection.
The UI review export currently contains
`id,text,proposed_label,top_probability,status,reviewed_label,reviewer,model`.
It omits raw probabilities, full identity, request hashes and review timing.
Those need separate private capture: probability/identity values are held in
page memory but not preserved in exports; human review timing is not recorded.
`scripts.evaluate_support_routing.measure` already journals probability and
identity evidence, but its public `evaluate` CLI prepares/fits the BANKING77
campaign. Do not run it unchanged on this natural cohort. A follow-up helper
must bind the private intake and frozen policy, capture the evidence and
validate the joins before a reported natural trial.

The existing finalization CLI is usable after genuine review fields are saved:

```bash
python -m scripts.evaluate_support_routing finalize \
  --queue /PRIVATE_PILOT_PATH/review-queue.csv \
  --index /PRIVATE_PILOT_PATH/index.json \
  --output /PRIVATE_PILOT_PATH/final.csv
```

Replace the placeholder path with the actual private directory. This writes
`id,text,final_label,decision_source,reviewer,top_probability`; it does not
create inference evidence, blind scores or timing receipts. The UI final
export also retains the model name. Neither route supplies missing reviews
from gold. The human selector now offers **No matching banking intent**;
setting `reviewed_label=__review__` with a reviewer also works in the CLI.
Both export `final_label=__review__` and `decision_source=human` for this
completed human no-route disposition. A model's `__review__` proposal alone
is abstention and remains unresolved until human review. A named no-route
decision closes a routing row; it does not establish a completed external
trial or a solved business request.

## Metrics and denominators

Let `N` be all provenance-valid requests frozen for the trial, including
failed, pending and unresolved rows. Report counts alongside rates:

| Measure | Definition |
| --- | --- |
| Routing accuracy | Successful model proposals matching blind gold / `N`; failures and pending rows contribute no correct prediction. Also report completed-prediction accuracy separately. |
| Misroutes | Successful proposals differing from blind gold / `N`, plus the same count / completed predictions. A `__review__` proposal matching `__review__` gold is not a misroute. |
| Acceptance and accepted errors | Policy-accepted non-review proposals / `N`; gold disagreements among those proposals / accepted proposals. In review-all mode acceptance is zero and accepted-error rate is undefined, not a measured zero-error guarantee. |
| Abstention and review demand | Successful `__review__` selections / `N` and policy review demand / `N` separately. HTTP failures and pending rows have their own counts; they are not semantic abstentions. |
| Retrieval recall | Requests whose gold banking label appears in the saved candidate set / requests with a banking gold label; show `__review__`-gold cases separately. |
| Human review completion | Rows with a valid named human decision / `N`; report final unresolved rows / `N`. Review demand alone is not completed work. |
| Human rework | Completed human decisions differing from their successful model proposal / completed human reviews of successful proposals; also show the count / `N`. This measures actual corrections, not gold disagreement or predicted staffing demand. |
| Review time | Sum and distribution of recorded start-to-end durations, with timed-review count / completed human reviews. Missing timing stays missing. This does not establish time saved without a separate human-only baseline. |
| Final routing accuracy | Final resolved decisions matching blind gold / `N`; unresolved rows contribute no correct result. Keep model accuracy separate. |

No change to banking accounts is executed, so these metrics measure routing
decisions and reviewed outputs, not business-action success.

## What counts as a completed external trial

A trial completes only when a real external opt-in participant supplied at
least one provenance-valid request, real inference finished, every frozen row
has a named human final disposition, the final output was exported and the
participant confirmed it was usable for the declared routing task. Record a
private receipt such as:

```json
{
  "trial_id": "PLACEHOLDER_TRIAL",
  "participant_id": "PLACEHOLDER_PARTICIPANT",
  "consent_receipt_sha256": "PLACEHOLDER_SHA256",
  "final_export_sha256": "PLACEHOLDER_SHA256",
  "usable_result_confirmed_at": "PLACEHOLDER_UTC_TIME"
}
```

Count unique validated trial IDs and unique participants separately. Report
completed trials / started consented external trials, with incomplete trials
and their reasons retained. Page views, model calls, download clicks, fixtures,
internal demonstrations and benchmark runs do not count as completed external
trials. The next required work is the capture/validation helper, followed by
actual opt-in intake and a frozen run; preparation alone establishes no
natural-user gain.
