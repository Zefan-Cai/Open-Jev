# Open-Jev on Apple Silicon (MPS)

Open-Jev runs on the Apple GPU through PyTorch's MPS backend. Everything below
is opt-in: with no flag or variable set, CUDA and CPU behave exactly as before.

**Status.** The released 2B checkpoint runs entirely on Metal with no CPU fallback
(M4 MacBook Air, 10-core GPU, 24 GB, macOS 26.6, torch 2.14). On a reduced six-workload
matrix, MPS at the released bfloat16 is 1.3–3.5× faster than CPU float32 uncached and
2.6–14× faster cached; CPU bfloat16 is 4–12× slower than CPU float32. MPS and CPU
make the same decisions on every request, with probabilities differing by at most
4.5e-3. None of the knobs in step 5 below is faster than the defaults on MPS beyond
about 8% run-to-run noise. The Air has no fan and throttles within minutes: these are
heat-soaked numbers, and a cool machine is up to 1.7× faster. No comparison with the
H100 rows is claimed. Raw reports, full tables and the MPS profile are on the
[`mps-apple-silicon-support` branch of StGerman/Open-Jev](https://github.com/StGerman/Open-Jev/blob/a30fe54/docs/apple-silicon.md#released-checkpoint-results).
The random-fixture checks (M4 Pro) are under [Fixture results](#fixture-results).

Docker cannot use the Apple GPU: containers on macOS run in a Linux VM without
Metal. Run natively.

## Setup

Apple's requirements for [PyTorch on Metal](https://developer.apple.com/metal/pytorch/):
Apple silicon, macOS 14.0 or later, Python 3.10 or later, Xcode command-line tools.

```bash
uv venv --python 3.12 .venv            # or python3.12 -m venv .venv
uv pip install --python .venv/bin/python -e '.[train]'
.venv/bin/python -c "import torch; print(torch.ones(1, device='mps'))"   # tensor([1.], device='mps:0')
```

Fetch the pinned checkpoint package and its base with the same verified fetcher the
Docker image uses; it checks every package file against the published manifest and
refuses a base revision other than the one `checkpoint/model.json` names:

```bash
export JEV_MODEL_ROOT="$HOME/models/open-jev"
JEV_PACKAGE_REPO=ZefanCai/Open-Jev-2B JEV_PACKAGE_REVISION=0c7aa498b1627be8da4acf34c863ff0ee0a92785 \
JEV_BASE_MODEL=Qwen/Qwen3.5-2B JEV_BASE_REVISION=15852e8c16360a2fea060d615a32b45270f8a8fc \
  .venv/bin/python docker/fetch_models.py
export HF_HUB_CACHE="$JEV_MODEL_ROOT/hub" HF_HUB_OFFLINE=1
export CKPT="$JEV_MODEL_ROOT/Open-Jev-2B/package/checkpoint"
```

## Running

```bash
.venv/bin/python -m jev.server --checkpoint "$CKPT" --device mps
.venv/bin/python -m jev.predict --checkpoint "$CKPT" --device mps --request configs/example-request.json
```

| Setting | Effect on MPS |
| --- | --- |
| `--device mps` | Required; the default stays `cuda:0` |
| `JEV_TORCH_DTYPE=bfloat16\|float16\|float32` | Backbone dtype; unset keeps the released `bfloat16`. The head stays float32. Any other dtype changes numerics against the saved calibration, so its quality is not comparable to the released numbers. Responses record the loaded dtypes in `metadata.backbone_parameter_dtypes` |
| `PYTORCH_ENABLE_MPS_FALLBACK` | Leave unset. Set to `1`, it silently runs unsupported ops on the CPU; unset, they raise |
| `JEV_LOAD_8BIT`, `JEV_LOAD_4BIT` | Refused on MPS: bitsandbytes has no Apple backend |
| `JEV_PROFILE=1` | Per-phase cached-path timings, now synchronizing Metal instead of calling CUDA |

## Proving the GPU is used

A working request does not show that the GPU did the work. This check exits non-zero
unless every parameter is resident on MPS, Metal holds allocated memory, no
CPU-fallback warning is raised and MPSProfiler logs no CPU fallback. It also reports
Apple GPU `Device Utilization %` (from `ioreg`, no sudo) idle and while scoring:

```bash
env -u PYTORCH_ENABLE_MPS_FALLBACK .venv/bin/python -m scripts.check_mps_engagement \
  --checkpoint "$CKPT" --profile-log reports/mps-profile.log --output reports/mps-engagement.json
```

`--profile-log` re-runs the scoring in a child process with MPSProfiler statistics
(`PYTORCH_MPS_LOG_PROFILE_INFO`: operation, copy and CPU-fallback tables with GPU
time), following the [PyTorch MPS Backend wiki](https://github.com/pytorch/pytorch/wiki/MPS-Backend#pytorch-performance-profiling-using-mps-profiler).
For a timeline, `--signposts` wraps the scoring in `torch.mps.profiler.profile(mode="interval")`;
record it with Instruments (`Logging` template for the OS Signposts, `Metal System Trace`
for the GPU):

```bash
xcrun xctrace record --template Logging --output mps.trace --launch -- \
  .venv/bin/python -m scripts.check_mps_engagement --checkpoint "$CKPT" --signposts
```

Profiling slows execution; never profile inside a timed benchmark run.

## Measuring MPS against CPU on the same Mac

A same-machine comparison is a stronger claim than one against the published H100
rows. `scripts/benchmark_inference_latency.py` synchronizes Metal around every timed
call; without that, asynchronous dispatch would make every MPS time an understatement.
MPS has no peak-memory counter, so samples record point-in-time allocator readings
(`mps_allocated_bytes_at_start/_at_end`, `mps_driver_allocated_bytes_at_end`), never a peak.

1. **Prepare identical inputs once.** Every later run replays this file; the harness
   refuses a changed request by checksum.

   ```bash
   B=reports/inference-latency; R="$B/2b-m4pro-prepare/requests.json"
   .venv/bin/python -m scripts.benchmark_inference_latency --checkpoint "$CKPT" --prepare-only \
     --request examples/community/drone.json --request examples/workflows/customer_service.json \
     --contexts 128 512 --candidates 2 8 --output "$B/2b-m4pro-prepare"
   ```

   Six workloads instead of the published eleven: CPU would exceed the time budget on the
   1,024-token, 32-candidate points. Run those separately (`--contexts 1024 --candidates 32`,
   fewer repetitions) if needed, and state the reduced matrix with any result.

2. **Pilot each device** (`--warmup 1 --repetitions 1`) and size the real runs from its
   per-call times; `--max-seconds` is capped at 3600.
3. **Measure**, plugged in, High Power energy mode, under `caffeinate -dimsu`, with
   `pmset -g therm` recorded before and after. Alternate the order (CPU then MPS, later
   MPS then CPU) with a cooldown between runs, and run each configuration at least twice:

   ```bash
   caffeinate -dimsu .venv/bin/python -m scripts.benchmark_inference_latency --checkpoint "$CKPT" \
     --requests "$R" --device mps --warmup 3 --repetitions 20 --output "$B/2b-m4pro-mps-run1"
   caffeinate -dimsu .venv/bin/python -m scripts.benchmark_inference_latency --checkpoint "$CKPT" \
     --requests "$R" --device cpu --warmup 3 --repetitions 20 --output "$B/2b-m4pro-cpu-run1"
   ```

   Measure CPU at its default `bfloat16` and again with `JEV_TORCH_DTYPE=float32`, and
   compare MPS against the faster of the two, so a slow CPU dtype path does not inflate
   the GPU speedup. One uncached CPU `bfloat16` request can take longer than the harness's
   default 120 s loopback timeout; pass `--http-timeout 1800` for CPU runs. On a fanless
   Mac, a 5-minute cooldown does not undo throttling: run knob comparisons back to back
   at the same thermal state, with baseline runs in between.
4. **Compare answers across devices.** Drift between backends is expected; changed
   decisions are what matter, and are listed per request:

   ```bash
   .venv/bin/python -m scripts.compare_device_reports "$B/2b-m4pro-cpu-run1" "$B/2b-m4pro-mps-run1" \
     --output "$B/2b-m4pro-cpu-vs-mps.json"
   ```

5. **Try one knob at a time on MPS**, reporting speed and parity together: `JEV_RAGGED_SUFFIX=1`,
   `JEV_PREFILL_CHUNK=128` or `256`, `JEV_NO_QPREFILL=1`, `--batch-size 8` or `64`,
   `JEV_TORCH_DTYPE=float16`. The first three change only the cached path, so the uncached
   rows of the same run are a built-in control. Do not sweep `JEV_PREFIX_CACHE`: every run
   already measures cached and uncached paths. A knob that changes a decision is not a free
   win; on CUDA, `JEV_NO_QPREFILL=1` was 11% faster but cost 2 of 420 answers.

Report with the numbers: the power mode and thermal state, the reduced workload matrix,
the dtype on each device, and that no hardware-matched claim against the H100 rows is
made (different kernels: the fused linear-attention kernels are CUDA-only, so Metal runs
the pure-PyTorch reference path).

## Fixture results

Random hybrid-Qwen fixture from `tests/test_prefix_cache.py`, all three model profiles
with and without LoRA, `PYTORCH_ENABLE_MPS_FALLBACK` unset; M4 Pro (16-core GPU, 24 GB),
macOS 26.5.1, Python 3.12.12, torch 2.14.0, transformers 5.10.2, peft 0.19.1. These test
execution and numerics, not model quality.

| Backbone dtype | MPS vs CPU, max logit error | Decisions differing (6 configs) | Error vs float32 reference, MPS / CPU |
| --- | ---: | ---: | --- |
| float32 | 4.8e-7 – 1.3e-6 | 0 | same as CPU |
| bfloat16 | 6.7e-3 – 2.3e-2 | 2 | 4.8e-3 – 2.3e-2 / 3.6e-3 – 2.3e-2 |
| float16 | 8.7e-4 – 3.7e-3 | 0 | 8.7e-4 – 7.9e-3 / 8.7e-4 – 5.7e-3 |

bfloat16 is as noisy on CPU as on Metal; the two differing decisions are near-tied
candidates of a random model disagreeing between two equally noisy computations. With
the backbone in bfloat16, MPSProfiler recorded 97 graphs and 80 kernels for uncached,
cached and ragged scoring and logged no CPU fallback. Most dispatches were small copy,
elementwise and reduction kernels from the linear-attention reference path; matrix
multiplies were a small share. Re-profile on the released checkpoint before tuning.

`tests/test_mps_support.py` holds these checks and runs wherever MPS is available:
`env -u PYTORCH_ENABLE_MPS_FALLBACK .venv/bin/python -m unittest tests.test_mps_support -v`.
