"""Prove a checkpoint runs on the Apple GPU (torch MPS) with no silent CPU fallback.

Hard checks, any of which exits non-zero: an MPS backend is available;
PYTORCH_ENABLE_MPS_FALLBACK is not 1 (it would hide unsupported ops); every
model parameter is resident on MPS; Metal holds allocated memory after load;
no CPU-fallback warning is raised while scoring. Reported evidence: Apple GPU
Device Utilization % from ioreg, idle versus while scoring (no sudo needed).

--profile-log re-runs the scoring in a child process under MPSProfiler
statistics (operation, copy and CPU-fallback tables, see the PyTorch MPS
Backend wiki) and records whether it logged any CPU fallback. --signposts wraps
the scoring in torch.mps.profiler for viewing as OS Signposts in Instruments.
"""
import argparse
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import threading
import time
import warnings

from scripts.benchmark_inference_latency import apple_gpu_statistics, sysctl

# MPSProfiler LogOptions (aten/src/ATen/mps/MPSProfiler.h):
# OPERATION_STATS | COPY_STATS | CPU_FALLBACK_STATS | INCLUDE_GPU_TIME.
PROFILE_LOG_OPTIONS = (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7)


def profiled_child_environment():
    # MPSProfiler is not safe against transformers' threaded weight loading:
    # the child segfaulted mid-load on the released 2B checkpoint.
    return {**os.environ, "PYTORCH_MPS_LOG_PROFILE_INFO": str(PROFILE_LOG_OPTIONS),
            "HF_DEACTIVATE_ASYNC_LOAD": "1"}


def utilization():
    match = re.search(r'"Device Utilization %"=(\d+)', apple_gpu_statistics() or "")
    return int(match.group(1)) if match else None


def sample_utilization(stop, samples, interval):
    while not stop.is_set():
        if (value := utilization()) is not None:
            samples.append(value)
        stop.wait(interval)


def describe(samples):
    return {"samples": len(samples), "median": statistics.median(samples) if samples else None,
            "max": max(samples, default=None)}


def score_loop(predictor, request, repetitions):
    import torch
    times = []
    for _ in range(repetitions):
        torch.mps.synchronize()
        begin = time.perf_counter()
        predictor.predict(request)
        torch.mps.synchronize()
        times.append((time.perf_counter() - begin) * 1000)
    return times


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--request", type=Path, default=Path("configs/example-request.json"))
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sample-interval", type=float, default=0.1)
    parser.add_argument("--profile-log", type=Path, help="Write an MPSProfiler statistics log from a child run here")
    parser.add_argument("--signposts", action="store_true", help="Emit torch.mps.profiler OS Signposts while scoring")
    parser.add_argument("--profiled-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    import torch
    from jev.serving import load_predictor

    if not torch.backends.mps.is_available():
        raise SystemExit("torch reports no available MPS backend on this machine")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise SystemExit("unset PYTORCH_ENABLE_MPS_FALLBACK: it silently runs unsupported ops on the CPU")
    request = json.loads(args.request.read_text())
    if args.profiled_child:  # Statistics print at exit, under PYTORCH_MPS_LOG_PROFILE_INFO.
        score_loop(load_predictor(checkpoint=args.checkpoint, device="mps", batch_size=args.batch_size),
                   request, args.repetitions)
        return 0

    idle = []
    for _ in range(20):
        if (value := utilization()) is not None:
            idle.append(value)
        time.sleep(args.sample_interval)
    loading = time.perf_counter()
    predictor = load_predictor(checkpoint=args.checkpoint, device="mps", batch_size=args.batch_size)
    torch.mps.synchronize()
    load_seconds = time.perf_counter() - loading
    model = predictor.scorer.model
    off_device = [name for name, value in model.named_parameters() if value.device.type != "mps"]
    allocated, driver = torch.mps.current_allocated_memory(), torch.mps.driver_allocated_memory()
    score_loop(predictor, request, 2)  # Warm Metal pipelines before sampling utilization.

    busy, stop = [], threading.Event()
    sampler = threading.Thread(target=sample_utilization, args=(stop, busy, args.sample_interval), daemon=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sampler.start()
        try:
            if args.signposts:
                with torch.mps.profiler.profile(mode="interval", wait_until_completed=False):
                    times = score_loop(predictor, request, args.repetitions)
            else:
                times = score_loop(predictor, request, args.repetitions)
        finally:
            stop.set()
            sampler.join()
    fallbacks = sorted({str(w.message) for w in caught if "fall back" in str(w.message)})

    report = {
        "device": "mps", "chip": sysctl("machdep.cpu.brand_string"), "torch": torch.__version__,
        "checkpoint": str(args.checkpoint), "request": str(args.request),
        "backbone_parameter_dtypes": predictor.provenance.get("backbone_parameter_dtypes"),
        "model_load_seconds": load_seconds, "parameters": sum(1 for _ in model.parameters()),
        "parameters_off_mps": off_device, "mps_allocated_bytes_after_load": allocated,
        "mps_driver_allocated_bytes_after_load": driver, "cpu_fallback_warnings": fallbacks,
        "gpu_device_utilization_pct": {"idle": describe(idle), "while_scoring": describe(busy)},
        "request_ms": {"repetitions": len(times), "median": statistics.median(times), "max": max(times)},
        "signposts": args.signposts,
    }
    if args.profile_log:
        child = subprocess.run(
            [sys.executable, "-m", "scripts.check_mps_engagement", "--profiled-child",
             "--checkpoint", str(args.checkpoint), "--request", str(args.request),
             "--repetitions", str(min(args.repetitions, 5)), "--batch-size", str(args.batch_size)],
            env=profiled_child_environment(), capture_output=True, text=True, check=False)
        args.profile_log.write_text(child.stdout + child.stderr)
        log = child.stdout + child.stderr
        report["profile"] = {"log": str(args.profile_log), "exit_code": child.returncode,
                             "no_cpu_fallback_logged": "There are no CPU Fallbacks logged" in log,
                             "cpu_fallback_table_present": bool(re.search(r"CPU Fallback", log)
                                                                and "There are no CPU Fallbacks" not in log)}
    idle_median, busy_median = report["gpu_device_utilization_pct"]["idle"]["median"], \
        report["gpu_device_utilization_pct"]["while_scoring"]["median"]
    report["checks"] = {
        "all_parameters_on_mps": not off_device,
        "metal_memory_allocated": allocated > 0,
        "no_cpu_fallback_warnings": not fallbacks,
        "profiler_logged_no_cpu_fallback": report.get("profile", {}).get("no_cpu_fallback_logged"),
        "gpu_utilization_rose_while_scoring": (None if idle_median is None or busy_median is None
                                               else busy_median > idle_median),
    }
    passed = (report["checks"]["all_parameters_on_mps"] and report["checks"]["metal_memory_allocated"]
              and report["checks"]["no_cpu_fallback_warnings"]
              and report["checks"]["profiler_logged_no_cpu_fallback"] is not False)
    report["status"] = "passed" if passed else "failed"
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n")
    print(rendered)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
