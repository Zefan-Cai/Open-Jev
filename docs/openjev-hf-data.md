# Audit of the openjev/openjev Hub release

The inspected Hub release supplies model weights and serving code, but no released training corpus was found. **Zero upstream training rows were imported.** Its public demo journals contain useful comparison cases and model predictions; they must not be relabelled as training data. A separate [original policy candidate](policy-controls-v6.md) supplies 384 newly written synthetic rows for a separately frozen experiment. The predeclared v5 training plan and its weights remain unchanged.

The model is pinned at [ac97900](https://huggingface.co/openjev/openjev/tree/ac97900fd034fdd7e7e536f3d4c21b836cae0750). Its organization profile links to [abhishekgahlot2/openjev-server](https://github.com/abhishekgahlot2/openjev-server/tree/032a2c5791f3d8856cc26fdb6876c106fac8dbf8); that server is pinned at `032a2c5791f3d8856cc26fdb6876c106fac8dbf8`. The linked static Space is pinned at `0a2e8cfd84eb07b6e70cacbb94e1087ddbeeab3e`. These are external projects with a different option-letter readout from ZefanCai's candidate-scoring Open-Jev.

## Availability and source licenses

| Material | Observed availability | Training disposition |
| --- | --- | --- |
| Hub model | 12 BF16 weight shards; weights CC-BY-NC-4.0; `helper/` and `serve/` Apache-2.0 | Model/code licenses do not establish a license or release of training examples. No weights downloaded. |
| Hub dataset listing | Organization's public dataset API returned `[]`; same-name dataset API returned 401 | No accessible training corpus identified. A 401 does not prove that a private dataset exists or does not exist. |
| Author server | Complete GitHub tree, 152 entries, `truncated:false`; Apache-2.0 code; public demos and journals | Serving and synthetic-demo code are available. No native training corpus or training pipeline identified in this tree. |
| BANKING77 demo | 300 messages explicitly loaded from `mteb/banking77` **test** | Reserve for evaluation. 20 visible texts match our frozen 256-row natural test; zero match its 96-row calibration input. |
| Prompt-injection demo | All 662 public messages, 546 original train plus 116 test, already used in its published evaluation | Reserve these published evaluation cases; 94 journal texts are truncated to 200 characters. |
| Policy/guardrail/filter journals | 400 records from 200 refund cases, 200 guardrail records, 60 table rows with eight predicates each | These are published synthetic evaluations. Refund/filter journals omit input state; predictions are not independent gold. |
| Game journals | Chess 31, poker 76/127/367, snake 600 decisions | Model choices and game outcomes are not per-action expert training labels. State is only partially logged. |

The BANKING77 mirror's [pinned card](https://huggingface.co/datasets/mteb/banking77/blob/18072d2685ea682290f7b8924d94c62acc19c0b2/README.md) lists MIT and 9,993/3,076 train/test rows. Our existing [routing converter](community-routing-v3.md) uses the attributed original PolyAI release, CC-BY-4.0, with 10,003/3,080 rows. These are different source artifacts; the mirror's metadata must not replace the existing source attribution or split pins.

The [deepset card](https://huggingface.co/datasets/deepset/prompt-injections/blob/4f61ecb038e9c3fb77e21034b22511b523772cdd/README.md) itself contains conflicting licenses: top-level Apache-2.0 and nested `dataset_info.license: cc-by-4.0`. The author server's Apache code license cannot resolve that data-license discrepancy. No dataset from either mirror was copied into the new candidate.

The refund demo also needs a specification review before its oracle could be adapted. Its written rule 9 says the earliest matching rule wins, while `truth()` checks defective items before unopened returns. An unopened defective item returned after ten days, with a verified order and a request for replacement, therefore receives a replacement in code although the earliest written matching rule promises a refund. Our original candidate defines explicit, consistent precedence and uses different policy, entities, text and labels.

## Evidence and limitations

The [source manifest](../reports/openjev-hf-data-20261002/source-manifest.json) records 48 fetched snapshots with URLs, revisions where available, response status, byte count and SHA-256. Raw upstream prose, logs and API results stay under ignored `runs/openjev-hf-data-20261002/raw/`; the public report contains counts, hashes and links. The [source audit](../reports/openjev-hf-data-20261002/source-audit.json) distinguishes observed files from author performance claims. The [visible-text overlap receipt](../reports/openjev-hf-data-20261002/visible-overlap.json) reads only our natural-test/calibration input CSVs, without their gold files.

The model card describes 10,000 text evaluation questions from 34 public sources, including 3,078 development-used questions. Neither those descriptions nor published accuracy values supply an importable training corpus, and the underlying training-data IDs/schema are not released in the inspected artifacts. Those author scores were not independently rerun here.

The pinned upstream `MANIFEST.json` has an outdated README size/hash: 12,158 declared bytes versus 12,769 fetched bytes. This is a metadata inconsistency; weights were not fetched or checked, and no weight-corruption claim follows from it.

Future external-source allocation remains **zero** until there is a pinned, licensed corpus with original IDs, label provenance and train/evaluation split separation. Existing prepared BANKING77/CLINC150 data retain their own source policy. Author journals may inform comparisons and requirements, while the new original controls remain separately attributed and independently checked.

The separate [v6 training preparation](policy-training-v6.md) combines 240 original policy Train rows with 2400 v4 and 128 v5 Train rows. Its 2768-row mixture and fixed full-pass plan preserve calibration, validation and regression splits. The [completed fixed run](../reports/policy-v6-training-20261002/README.md) starts from the released 2B checkpoint and improves these synthetic aggregates, while retaining serious rejection and capacity failures. No new weights replace the release; zero upstream rows were imported.
