# Original compositional controls: v8 CPU data protocol

This is a separately designed synthetic candidate, based on the frozen
Train-only coverage audit. It contains no imported upstream examples or
old v4–v7 Test/OOD rewrites. Dataset preparation establishes no model gain,
safety pass, natural-user result or official JevBench score. Training and
its comparison protocol are separate future work.

The [visible contract](../reports/frontier-v8-data-20261003/contract.json)
defines seven tasks. The generator and independent oracle both apply the
explicit rules and recorded facts; condition names never determine gold.

| Task | Required decisions and essential composition |
| --- | --- |
| Temporal window | Inclusive accept/reject/review after signed delivery corrections; exception takes precedence |
| Exact integer arithmetic | Ledger balance and comparison of ledger sums, with exact USD-cent labels |
| Joint authority | Every required role needs valid consent; revoke precedes missing, then minimum capacity |
| Latest policy | Every required topic resolves its latest valid policy before fraud or capacity |
| Account status | Aggregate active/paused/closed/unknown over required accounts |
| Department route | Aggregate approve/manager/security; fraud precedes policy availability |
| Refund priority | Aggregate automatic/recall/reject/review; recall precedes document and eligibility |

Each authored case index has two Choice questions and one true and one false
Noul proposition. All counterfactual cases and equivalent presentations
belong to their original whole group. Option positions and false proposals
are checked alongside action, kind and truth counts.

The contract uses essential composition widths 1/2 for Train,
3 for Calibration, 4 for Validation, 5 for Test and 6 for OOD. Joint authority
and latest policy need one more required role/topic than the stated width.
The actual dataset has 2,976 rows in 42 parents: Train992 and each other split496.
These are controlled shifts in composition depth. Test is not an IID sample;
Calibration and Validation also use unseen depths. These controls do not
establish unseen-business or natural generalization.

Depth is supported by applicable operands, corrections, required components
or request lines. Additional invalid records, obsolete history, identities,
wording or numeric scales do not establish parent independence. The independent
checker blocks cross-split compiled inputs and wording/order/identity clones,
then abstracts numeric magnitudes and time origins on complete parent menus.
Shared atomic rule motifs are disclosed separately. Every essential component
must be able to change an authored case result; decorative padding cannot
satisfy this requirement.

The [CPU audit report](../reports/frontier-v8-data-20261003/README.md)
records 744 authored case indices, 742 exact projected causal fact cases
and 353 symbolic scaffolds. Those denominators must remain separate.
There are 54 cases whose record-order reversal is indistinguishable.
For each of the 744 indices, both Noul truths use the same layout; the
alternative Noul layout is absent within that parent. This avoids a
truth/order shortcut but does not provide full condition/truth/layout crossing.

Mismatch probes cover first and last required roles/topics and their declared
scope fields with both kinds and Noul truths. Middle roles/topics remain
necessary for execution, but their mismatch combinations are not exhaustive.
At width one, first/last account probes may share an atomic case; duplicate
and shared-atom counts must remain visible.

Generation is deterministic and uses exclusive provisional outputs. CPU
label, schema, balance, margin, essentiality and isolation audits must pass
before a source/data freeze is declared. Old observed-input overlap screening
is performed by a separate checker; it provides aggregate counts and opaque
fingerprints to the author, never old row bodies. Preserve failed provisional
outputs and receipts; a repaired candidate gets a fresh output directory.

No existing v6/v7 training, comparison, controller, driver, lock or exclusive
replay output is reused. Any later training needs a new frozen mixture,
initializer and controlled evaluation plan. Released weights and historical
failed gates remain unchanged.
