# V7 resource acquisition: first failure, verified restoration

The first resource acquisition ended before a resource-ready receipt,
experiment driver, model load or optimizer step. It is not a model result.
The frozen v7 training experiment has not started.

## Preserved first acquisition

The acquisition began on N1-1 at 2026-10-03 00:08:36 UTC. After the authorized
original queue pause, an inventory query timed out after 12 seconds during
idle verification. The controller restored the original queue and completed
at 00:16:24 UTC. Independent read-only verification completed at 00:17:39 UTC.

The audit verified the preserved optimizer/checkpoint/adapter/sampler-tree
file inventory and full hashes, unchanged configuration and logical retry
budget, four actual training ranks plus their coordinator, three current
sampling services and fresh HTTP completions. Protected GPU0/1 were unchanged;
the Jev controller, guard and all owned descendants had exited. The original
queue again occupies GPU2–7. These cards were returned to that queue, not
allocated to R-KV.

The same inventory query subsequently completed three read-only samples in
203, 216 and 271 ms. This does not establish the cause of the earlier timeout.
The executed controller and its failed acquisition receipts are preserved
unchanged. It must never be restarted.

Hashes and the sanitized audit summary are in [acquisition-a1.json](acquisition-a1.json).
Full operational receipts remain local and are bound by their SHA256 hashes.

## Separate next acquisition

The proposed identity is `boundary-v7-acquisition-a2-20261002`, in a separate
task directory. It uses a separately reviewed controller and new execution,
staging, preflight, ownership and restoration receipts. It does not reuse the
previous acquisition's PIDs, ownership nonce, empty observations or locks.

Only an idle-read `subprocess.TimeoutExpired` becomes a nonconfirming sample:
discard partial data, record the timeout, reset the idle count, and require
two fresh consecutive successful empty samples. Both queries, process-identity
reads and sleeps share the absolute idle deadline; observations completing
at or after it are refused. Process spawning and cleanup can add wall time;
the policy bounds acceptance of observations, not all operating-system overhead.
UUID, ownership, parse and other subprocess errors remain hard failures.

An exclusive, fsynced `acquisition.lock.json` is created before guard creation
or pause. It survives failure, including failure before model loading, so
this acquisition cannot silently run again. Existing model-attempt safeguards,
the 4500-second driver allowance and 1500-second restoration allowance remain.

Launch requires a separate recorded root declaration, pushed public record
and passing CI, CPU staging/preflight, fresh live checkpoint/optimizer/config/
process ownership evidence, and a live restoration guard. Prefer an actually
empty H100; the latest independent six-node snapshot at 00:22:20–00:22:33 UTC
found all 48 cards occupied. The already authorized original-queue borrowing
may be used only after these new checks. GPU0/1 remain protected.

The [frozen training protocol](../../docs/boundary-v7-run-protocol.md) and
[comparison plan](../boundary-controls-v7-20261002/prepared-training/comparison-plan.json)
remain unchanged: exact released 2B initializer, 3792 Train rows, 948 steps ×
accumulation four, one configured shuffled pass, seed 20261003, no resume,
model retry, Test/OOD tuning or automatic promotion. A failed resource
acquisition is not authorization to rerun a started model attempt.

No natural business, official JevBench or model-capability improvement is
claimed by this resource record.
