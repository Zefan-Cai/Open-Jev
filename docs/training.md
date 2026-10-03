# Resumable training

`python -m jev.train` retains the original pilot defaults. `--checkpoint-every 0` disables periodic snapshots. For longer jobs, add a positive optimizer-step interval:

```bash
python -m jev.train \
  --model Qwen/Qwen3.5-2B \
  --revision 15852e8c16360a2fea060d615a32b45270f8a8fc \
  --data data/your-validated-mixture --output runs/2b-long-run \
  --train-rows 0 --steps 27000 --accumulation 4 \
  --max-length 4096 --checkpoint-every 500
```

This example's step count is illustrative. Choose the actual number of optimizer steps from the intended consumed-row budget. One optimizer step consumes `accumulation` rows; sampling advances deterministically through the source/kind-balanced training order and cycles if the budget exceeds the available rows. A final partial pass repeats at most `accumulation - 1` initial rows when rounding up to a whole optimizer step.

A snapshot is written every N completed optimizer steps and at the final step before calibration. It is first assembled in an `.incomplete-*` directory, then atomically renamed to `training-checkpoints/step-00000500`. Unpublished incomplete directories are never resume candidates. Existing named snapshots are not overwritten or pruned automatically.

Each snapshot contains:

- Exact trainable parameters, including LoRA adapters and the decision head.
- AdamW optimizer state and the number of completed optimizer steps.
- Python, Torch CPU, and visible CUDA RNG states.
- The original baseline test/OOD/calibration outputs and training log through that step.
- A checksum manifest and run identity binding model/revision, all training and evaluation parameters, dataset bytes, selected row IDs/order, optimizer settings, relevant implementation-file hashes and runtime package versions.

Frozen backbone weights are reconstructed from the original pinned model revision. Resumable training therefore requires a full 40-character model revision. The snapshot itself is **training state only**, marked `inference_ready: false`; it contains no calibrated `temperature.json` and cannot be passed to the inference server. The final serving checkpoint remains the separate `checkpoint/` directory produced after the complete training loop and calibration.

## Resume after interruption

Pass the snapshot directory and repeat the original training settings, including the original total `--steps`:

```bash
python -m jev.train \
  --model Qwen/Qwen3.5-2B \
  --revision 15852e8c16360a2fea060d615a32b45270f8a8fc \
  --data data/your-validated-mixture --output runs/2b-long-run \
  --train-rows 0 --steps 27000 --accumulation 4 \
  --max-length 4096 --checkpoint-every 500 \
  --resume-training runs/2b-long-run/training-checkpoints/step-00000500
```

Resume reconstructs a fresh model with trainable adapters, restores the saved parameters and optimizer, restores RNG after initialization, and starts at the next optimizer step. It does not repeat the original baseline with already trained weights: saved baseline rows must match the original held-out IDs, order and targets exactly. The original calibration split stays fixed.

Changing the dataset bytes, training order, model/revision, optimizer/training/evaluation settings, relevant code files, or Torch/Transformers/PEFT versions is rejected. Moving identical dataset files or choosing a new output directory is allowed. The checkpoint interval can change. Unrelated Git commits do not invalidate a resume when the relevant implementation-file bytes are unchanged. The visible CUDA device count must match the saved RNG state. Extending a finished run or changing its schedule is a new experiment, not supported by this strict resume path.

The output directory must not already contain a completed `summary.json`; completed experiment outputs are protected. When resuming into an interrupted run directory, existing run metadata and post-snapshot training logs are copied under `resume-history/` before the log is restored to the checkpoint cursor. Existing baseline files must be byte-identical. This restoration never touches final calibration artifacts. If choosing an older snapshot while later snapshot directories already exist, use a fresh output directory to avoid colliding with their immutable step names.

Saving and restoring all RNG states preserves the stochastic training trajectory, but does not promise bitwise equality for nondeterministic GPU kernels or different hardware. The CPU test below verifies exact parameter/loss continuity with AdamW, dropout and both Python/Torch RNG. A bounded real-model interrupted/resumed comparison is recommended before a long production experiment.

```bash
python -m unittest tests.test_training_resume -v
```

The tests also cover incompatible run identities, reordered held-out references, corrupted snapshots, mismatched log cursors, preservation of interrupted logs, and separation from final calibration. The numerical test requires the optional Torch runtime; the identity/artifact tests use only the Python standard library.

For a full-data pass, set `--training-sampling shuffled`. This retains each selected record once and avoids the long tail of the largest source after smaller round-robin buckets are exhausted. Short source-balanced pilots retain the original `source_kind_round_robin` option. The sampling mode is part of the resume identity.

## Defer heldout evaluation

Use the explicit `--defer-heldout` flag for a separately declared fixed-final
training run. This mode reads, schema-validates, samples and hashes only
`train.jsonl` and `calibration.jsonl`. Validation, Test and OOD are neither
parsed nor forwarded. Warmup uses Train; the final temperature and checkpoint
reload check use Calibration. The initialized baseline and all heldout metrics
remain deferred to a separate comparison. A Calibration fit is not a capability
or safety result.

This mode requires `--checkpoint-every 0` and prohibits `--resume-training`.
The default pilot/resume behavior above is preserved. The flag is part of the
run identity. A fresh attempt, exact initializer, data membership, step budget
and comparison protocol still need their own declarations before launch.

Deferred summaries report synchronized timing, peak CUDA allocated tensors
and peak allocator-reserved memory
for model loading, warmup, optimizer steps, checkpoint saving, Calibration and
temperature fitting, and checkpoint reload. Peaks include resident weights;
reserved memory also includes allocator caching. Neither figure measures
physical total or minimum VRAM. Phase timings include their file
writes, and optimizer timing includes training logs. These are instrumentation
boundaries, not measured performance results until an actual run completes.

### N1-1 real-model resume check

On September 20, 2026 UTC, Qwen3.5-2B on N1-1 physical GPU 3 ran four
optimizer steps, then resumed twice from its step-2 snapshot. The first resumed
loss was identical across all three runs (`0.31363993883132935`). The initial
strict maximum-parameter-error threshold of `1e-5` **failed**: uninterrupted versus
resumed was `9.93046e-5`. Repeating the same resume also differed by `9.68892e-5`;
this supports nondeterministic GPU backward numerics rather than a resume-only
failure. It does not establish exact GPU trajectory equivalence. Qwen's fast
path was active. The original failed check and the repeat comparison are kept
in `reports/runtime-resume/`; no threshold was silently relaxed.
