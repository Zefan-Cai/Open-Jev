# Graded retrieval controls and typed reranking

`ir-control-v1` contains **11,600 original typed records**, representing
**8,800 requests, 400 queries and 200 system families**. The implementation
supports pointwise Noul/Score, pairwise, setwise and listwise Choice/Score
reranking. Its independent audit derives every relevance label from the final
query and passage text. These are finite synthetic controls, not TREC data.

A separate frozen pilot has real Jev responses for **6 held-out queries from
3 families**, with 8 candidates per query. It supplies a small execution check;
it does not establish general retrieval quality. No IR training or external
TREC evaluation is claimed here. The generation manifests describe the state
at generation time; subsequent provider execution is recorded separately below.

## Community source and reproduction boundary

[Shengyao Zhuang's post](https://x.com/ShengyaoZhuang/status/2101212268440723895)
reports Jev reranking on DL19/DL20. The author's public implementation is in
[`ielab/llm-rankers`](https://github.com/ielab/llm-rankers/tree/ac843ed63a302900d76722b83be926fdb84a3936/jev),
with an initial Jev commit 35 seconds before the post. The
[source review](../reports/ir-control-v1/source-code-research.json) pins the
code, interfaces, algorithm settings and timing boundaries. Later prompt
studies were committed after the post and are not attributed to that experiment.

Our builders preserve the typed interface shapes and question IDs. The query,
passage, rubric and instruction wording are independently authored. Neither
`original-control-v1` nor the separate `general-ir-v1` request profile is an
exact replay of the author's prompts. No third-party examples or TREC passages
were imported into this corpus. Original generated data are CC0-1.0.

The [tweet table transcription](../reports/ir-control-v1/tweet-table-provenance.json)
is explicitly **author-reported**. The original experiment used the Vercel
`typesafe-ai/jev` gateway, BM25 top 100 (`k1=0.9`, `b=0.4`), approximate
`cl100k_base` limits of 32 query/128 passage tokens, 4 request workers,
12 query workers and a 20 requests/second limit. Its API latency is mean
successful HTTP request time, excluding throttle and retry waits. It is not
the time to rerank one query. Cost is a successful-input-token estimate at
$0.042 per million, not an invoice. Paper baselines were copied by the source
author, not rerun there. Unpinned downloads, mutable provider aliases and absent
raw run files prevent an exact independent reconstruction of those results.

## What the original labels mean

Each query asks for the current retry ceiling and backoff delay for a fictional
system and operating profile. Each family contains eight passages shared by
two profile-counterfactual queries. The eight grades per query are
`[0, 0, 1, 1, 1, 1, 2, 3]` before deterministic shuffling.

| Grade | Visible evidence |
| ---: | --- |
| 0 | Passage concerns another system, even if a quoted search example names the requested system |
| 1 | Correct system, but wrong profile, withdrawn guidance or neither requested value |
| 2 | Current guidance for the exact system/profile supplies one requested value |
| 3 | Current guidance for the exact system/profile supplies both requested values |

Score targets are one-hot four-level grades. Noul is positive only for grade 3,
so it supervises complete-answer detection, not all graded distinctions.
Choice targets distribute mass uniformly over tied best passages. These
targets supervise best-passage selection; they are neither measured uncertainty
nor full ranking supervision. In particular, assigning all probability to the
best passage leaves the ordering of the other passages unspecified.

The model receives only state, question, kind and options. Relevance judgments,
query/group IDs, split labels, source and generation metadata are offline
annotations. The independent body parser is an auditor for this finite grammar,
not a production retrieval model or a source of model evaluation predictions.

## Request formats and runtime

| Method | State | Questions per request | Frozen cases per query |
| --- | --- | --- | ---: |
| Pointwise Noul | `query`, `passage` | `relevant`: Noul | 8 |
| Pointwise Score | `query`, `passage` | `relevance`: Score | 8 |
| Pairwise | `query`, `passage_A`, `passage_B` | `more_relevant`: Choice A/B | 2, opposite orders of one pair |
| Setwise | `query`, `passages` with P1…P4 | `most_relevant`: Choice, null descriptions | 2 disjoint subsets |
| Listwise Choice | `query`, `passages` with P1…P8 | `most_relevant`: Choice | 1 |
| Listwise Score | `query`, `passages` with P1…P8 | One `relevance_Pn`: Score per passage | 1, containing 8 typed questions |

This totals **22 request cases and 29 typed records per query**. The frozen
pairwise/setwise cases are local decision supervision; they do not contain all
adaptive comparisons in a complete heap execution.

[`jev.ir_eval.rerank`](../jev/ir_eval.py) executes the complete algorithms with
an actual `predict(request)` callback and never receives qrels:

- Pointwise sorts returned scalar Noul or Score values.
- Pairwise binary heapsort asks both passage orders. Its comparison is
  `(P(A forward) + 1 - P(A reversed)) / 2`. Near-exact probability ties retain
  original input order, an explicit tie-handling difference from the source.
- Setwise heapsort uses 10 children plus their parent, at most 11 passages per
  comparison. Pairwise/setwise rerank the top `k=10` by default; the remaining
  documents retain their input order.
- Listwise sorts Choice probabilities or per-passage Scores in bottom-to-top
  windows. With 100 candidates, window/step 20/10 makes 9 calls; 100/100 makes
  1 call. Pointwise makes 100 calls. Heap counts depend on actual comparisons.

For an already running local Open-Jev server, this example makes real calls:

```python
import json
from urllib.request import Request, urlopen
from jev.ir_eval import rerank

def predict(request):
    body = json.dumps(request).encode()
    http = Request("http://127.0.0.1:8791/v1/inference", data=body,
                   headers={"Content-Type": "application/json"})
    with urlopen(http, timeout=60) as response:
        return json.load(response)

query = "Which setting controls retry count?"
documents = [
    {"id": "a", "text": "The retry ceiling limits the number of attempts."},
    {"id": "b", "text": "The backoff interval controls waiting time."},
]
result = rerank(query, documents, "listwise_score", predict,
                request_profile="general-ir-v1")
print(result["ranking"])
```

Requests, raw callback responses and callback durations remain in the trace.
A malformed answer raises `RerankFailure` with that trace; it does not become a
zero relevance label. Probability mass must equal 1 within `1e-6`. Score and
its probability expectation must also agree within `1e-6` by default.
For a provider that independently rounds both to two decimals, explicitly set
`score_rounding_digits=2`: the four-level maximum combined rounding difference
is `(1 + 0 + 1 + 2 + 3) × 0.005 = 0.035`. This does not relax probability mass
validation or modify returned values. The runner does not truncate text or
implement a provider-specific retry, concurrency or throttling policy.

## Stable splits and audit

| Split | Families | Queries | Requests | Typed records |
| --- | ---: | ---: | ---: | ---: |
| Train | 127 | 254 | 5,588 | 7,366 |
| Calibration | 8 | 16 | 352 | 464 |
| Validation | 8 | 16 | 352 | 464 |
| Test | 17 | 34 | 748 | 986 |
| OOD | 40 | 80 | 1,760 | 2,320 |
| Total | 200 | 400 | 8,800 | 11,600 |

Every system family, both queries and all passage/method counterfactuals stay
in one split. Family index modulo 10 values 8–9 reserve exactly 20% for OOD;
the remaining families use stable group hashing. OOD uses different brief
wording and profile names. This is a controlled wording/profile shift, not a
claim of broad domain generalization. The generator requires multiples of ten
families so expansion never reassigns an existing family.

The frozen 10-family pilot contains 20 queries, 440 cases and 580 typed rows.
Its seven JSONL files remained byte-identical after stabilizing expansion.
All 10 family assignments were preserved, and none of its 3 test/OOD families
entered the full corpus's train/calibration/validation splits. See the
[split verification](../reports/ir-control-v1/stable-split-verification.json).
The original pilot generator is preserved as a source snapshot; the audit
requires its actual hash, not an unbound allowlist entry.

Use a new directory to regenerate; the generator rejects nonempty outputs:

```bash
python3 -m jev.ir_data --output-dir data/ir-control-v1-rebuilt \
  --groups 200 --ood-groups 40 --seed 42
python3 reports/ir-control-v1/verify.py --data data/ir-control-v1-rebuilt \
  --output reports/ir-control-v1/rebuilt-audit.json
python3 -m unittest tests.test_ir_control -v
```

The [full manifest](../reports/ir-control-v1/manifest.json),
[independent audit](../reports/ir-control-v1/independent-audit.json) and
[test execution record](../reports/ir-control-v1/test-verification.json)
bind counts, file/source hashes and verification results. The corpus is published
as the separate [`ir-control-v1` HF config](https://huggingface.co/datasets/ZefanCai/Open-Jev/tree/b0aad4004b8d74f4a6ca66c7a9175fe17478d688/data/ir-control-v1).
All five splits and 11,600 rows passed anonymous Parquet/raw byte-for-byte
round-trip verification. The additive upload preserved 202 existing files and
all earlier config definitions; only the root dataset card changed. The
[release evidence](../reports/ir-data-release-20260920/README.md) records the
pinned revision and checksums. Training mixtures remain unchanged.

## Real Jev pilot observations

On September 20, Jev 1.13.0 returned HTTP 200 for all 132 held-out requests,
containing 174 typed answers. This is **6 queries**, not 174 queries. Four
complete ranking modes can be reconstructed from those recorded requests:

| Mode | Strictly valid full queries | nDCG@10, all 6 queries |
| --- | ---: | ---: |
| Pointwise Noul | 6/6 | 0.974823 |
| Pointwise Score | 5/6 | 0.833333 |
| Listwise Choice | 6/6 | 0.940929 |
| Listwise Score | 6/6 | 1.000000 |

These scores use linear gains, with only 8 candidates per query. One pointwise
Score vector sums to 0.99; its entire query is absent from the strict run and
contributes zero while remaining in the six-query denominator. A separately
labelled observation sorting raw scalar Scores obtains 1.0 for both Score
modes; it bypasses probability validation and is not strict protocol success.
Score/expectation rounding uses the explicit two-decimal policy above.

The [ranking report](../reports/ir-control-v1/jev-pilot-ranking.json) binds the
frozen input and raw response hashes, per-query rankings, failures and the
shuffled-input baseline. It can be recomputed with:

```bash
python3 reports/ir-control-v1/evaluate_pilot.py
```

This replay makes no API calls and never substitutes oracle responses. It does
not report its CPU replay time as provider latency. The pairwise/setwise local
probe results cannot reconstruct complete heap rankings and have no full-query
nDCG claim here. All pilot results remain separate from the existing frozen
189-request provider comparison.

## Open-Jev TREC inference remains pending

The real held-out inputs are now prepared: DL19 has **43 judged queries with
4,300 candidate occurrences and 9,260 qrels**; DL20 has **54 queries with 5,400
candidate occurrences and 11,386 qrels**. The
[preparation evidence](../reports/ir-control-v1/trec-holdout/README.md) records
immutable source revisions, checksums, official-query agreement and complete
judged-query coverage. Raw texts stay in the isolated local holdout directory;
they were not uploaded as Open-Jev training data. **Open-Jev TREC inference has
not yet been run on this holdout.** The separate hosted evaluation is summarized
in the [IR-control report](../reports/ir-control-v1/README.md).

The downloaded BM25 rankings give linear nDCG@10 of 0.505831 on DL19 and
0.479637 on DL20. These are arithmetic checks of supplied rankings, not a new
retrieval run or a reproduction of the author's Jev results. The author's
original downloads were not pinned, and runtime query/passage truncation has
not been baked into the stored full texts.

The earlier [external evaluation contract](../reports/ir-control-v1/external-evaluation-contract.json)
records the required provenance and its original manifest template.
`load_external_holdout` reads checksummed `candidates.jsonl` and `qrels.txt`
outside the training-data tree, including resolved symlinks. It requires an
`evaluation_only` manifest, TREC-DL19/DL20 identity and 100 unique ordered BM25
candidates per query. Passing these checks verifies input structure and the
declared settings; it does not prove independent BM25 reproduction.

`ndcg_at_k` uses the qrel grade directly, as
[`trec_eval ndcg_cut`](https://github.com/usnistgov/trec_eval/blob/dc0c991c80bae2087de774ed76278e11d9d9f4c6/m_ndcg_cut.c)
does, rather than `2**grade - 1`. Unjudged or absent documents have zero gain;
duplicate run documents are rejected; every qrel query remains in the average,
including missing/failed queries at zero. The
[pinned NIST source evidence](../reports/ir-control-v1/trec-ndcg-source.json)
records this metric choice. Loader unit-test fixtures are synthetic schema
checks and supply no TREC benchmark result.
