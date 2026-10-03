# V7 supplemental CPU preflight failures and source layout repair

Later update, 2026-10-03: [S2 completed](../boundary-v7-comparison-s2-20261003/README.md)
and its scores passed independent replay, while all four safety cells failed.
The account below preserves the earlier S1 failures and S2 preparation;
neither consumed attempt nor its evidence was rerun or overwritten.

V7 training remains complete at 948 finite optimizer steps. There are still
no valid v7 comparison scores. The first supplemental evaluation, S1, reached
two CPU preflight failures and never acquired resources or ran supplemental
inference. Its original request, source, bundle and both failure logs remain
preserved. S2 is separately declared and has not run.

## What failed

The first deployment bundle included evaluation commit
`d32aad0a0d2f13174823748c4f00ddb37622317c` and its integration ancestry, but
omitted frozen training commit `87eb8b419685d272f059b3696e0f547e47ebdcc6`.
Squash integration did not make that separate commit reachable. Preflight
failed when reading a frozen source blob. The earlier CPU review missed this
bundle completeness check; that review is retained alongside the failure.

A separately reviewed CPU continuation added only two remote Git pack files. It
preserved all 24 original remote Git files, the source checkout, original inputs,
request and first failed log. Subsequent read-only remote checks confirm
[Git counts and preservation](remote-git-preservation-check.json) and
[all original non-Git file hashes](remote-preserved-files-check.json). The separate local clone fixture had
26 original files and three new pack files. Source validation then failed on
`Supplement task overlaps original evidence or source`.

The protocol requires the checkout inside `new-task/evaluation-source`.
The old validator prohibited overlap between the task and that very checkout,
so it rejected the prescribed layout. The independent replay had the same
defect. Tests previously used a sibling source fixture and missed this actual
deployment layout.

The [failure evidence](failure-evidence.json) binds both
[initial](initial-cpu-preflight-failure.log) and
[continuation](source-continuation-cpu-preflight-failure.log) logs by SHA-256,
the consumed request, source bundles and
[original S1 declaration](consumed-s1-declaration.json). No successful
instrumentation receipt was produced; absence of launch artifacts is not a
successful measurement of all instrumentation counters.

## Repair and verification

Both public validators now require the source exactly at the request's
`evaluation-source` sibling directory. The wrapper requires a canonical
directory; a foreign checkout symlink is rejected. Only the source is removed
from the old overlap collection. Both directions of overlap with the preserved
original task, immutable dataset and released checkpoint still reject.

Tests now use the actual nested layout, reject wrong locations and foreign
symlinks, exercise both directions of all three preserved-input overlaps,
verify original bytes remain unchanged and reject the consumed S1 identity.
Independent replay review reproduced the old failure on the same valid
fixture and confirmed scoring, model, temperature and gate functions remain
unchanged. Repository and packaging results are recorded in
[validation](validation.json).

The [S2 protocol](../../docs/boundary-v7-comparison-supplement.md) requires a
fresh task, request, source, output and resource acquisition. The deployment
bundle must contain both the new integrated evaluation source and frozen
training commit, with actual clone/blob verification before upload and again
remotely. S1 cannot be rerun or rewritten.

CPU validation and a declaration establish readiness only. The same-runtime
comparison still needs committed-source remote preflight, fresh ownership
evidence, an independently reviewed restoration controller, one bounded
comparison, independent score replay and full resource restoration. The
completed candidate remains experimental; this repair establishes no natural
business or official JevBench improvement.
