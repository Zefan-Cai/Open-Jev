# Preserved CPU audit and regression history

The r1/r2 source snapshots bind the corresponding raw-proposal audit receipts.
They are historical checker bytes, not instructions to run a completed phase.
The raw proposal remains blocked; its inputs and all original data are unchanged.

The first full repository test run failed two legacy v7 staging fixtures because
they compared the current trainer with the old frozen implementation. The old
driver correctly refused that source mismatch. Its guard and frozen plan remain
unchanged; only temporary test fixtures are repaired. The original wrapper
misparsed the FAILED summary's skip count as zero. The separate interpretation
amendment records the actual 85 skips without rewriting the original receipt.
