"""Measure a released checkpoint: warmed Predictor and loopback HTTP, cache off/on.

The output directory is new, contains exact requests and every warmup/timed
attempt, and is never overwritten. This tool does not allocate cluster resources.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import http.client
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import threading
import time

from jev.api import candidate_prompts, compile_request
from jev.server import make_server, strict_json
from scripts.benchmark_prefix_cache import compare_answers


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def percentile(values, quantile):
    """Linear interpolation, equivalent to NumPy's default percentile method."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def sysctl(name):
    """macOS hardware identity, e.g. machdep.cpu.brand_string -> "Apple M4 Pro"; None elsewhere."""
    try:
        return subprocess.check_output(["sysctl", "-n", name], text=True, timeout=5).strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def apple_gpu_statistics():
    """Apple GPU PerformanceStatistics from ioreg, including Device Utilization %; no sudo needed."""
    try:
        text = subprocess.check_output(["ioreg", "-r", "-d", "1", "-w", "0", "-c", "AGXAccelerator"],
                                       text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r'"PerformanceStatistics" = (\{.*?\})', text)
    return match.group(1) if match else None


def summarize(samples):
    groups = {}
    for sample in samples:
        if sample["phase"] == "measured":
            groups.setdefault((sample["request_id"], sample["transport"], sample["mode"]), []).append(sample)
    result = []
    for (request_id, transport, mode), rows in groups.items():
        successes = [row["wall_ms"] for row in rows if row["success"]]
        all_times = [row["wall_ms"] for row in rows]
        result.append({"request_id": request_id, "transport": transport, "mode": mode,
                       "attempts": len(rows), "successes": len(successes),
                       "errors": len(rows) - len(successes),
                       "p50_ms": percentile(successes, .5), "p95_ms": percentile(successes, .95),
                       "min_ms": min(successes) if successes else None,
                       "max_ms": max(successes) if successes else None,
                       "mean_ms": statistics.mean(successes) if successes else None,
                       "all_attempt_p50_ms": percentile(all_times, .5),
                       "all_attempt_p95_ms": percentile(all_times, .95)})
    return result


def request_metadata(tokenizer, request):
    records = compile_request(request["state"], request["questions"])
    prompts = [tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
        for record in records for prompt in candidate_prompts(record)]
    encoded = tokenizer(prompts, padding=False, truncation=False)["input_ids"]
    state = request["state"]
    rendered_state = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, sort_keys=True)
    state_tokens = len(tokenizer([rendered_state], padding=False, truncation=False)["input_ids"][0])
    return {"question_count": len(records), "candidate_sequences": len(encoded),
            "state_tokens": state_tokens, "candidate_input_tokens": list(map(len, encoded)),
            "logical_input_tokens": sum(map(len, encoded)), "max_candidate_input_tokens": max(map(len, encoded)),
            "request_sha256": digest(request)}


def make_workloads(tokenizer, sources=(), contexts=(128, 512, 1024), candidates=(2, 8, 32)):
    workloads = []
    for path in sources:
        request = strict_json(Path(path).read_bytes())
        # Keep the same state/questions for every service; model aliases are transport details.
        request = {"state": request["state"], "questions": request["questions"]}
        workloads.append({"id": Path(path).stem, "source": str(path), "kind": "demo_request",
                          "request": request, **request_metadata(tokenizer, request)})
    sentence = "The warehouse robot is stationary. Its route is clear and the safety check passed. "
    for target in contexts:
        text = "Route 00 is permitted. Choose route_00. " + sentence * (target // 8 + 1)
        tokens = tokenizer([text], padding=False, truncation=False)["input_ids"][0]
        state = tokenizer.decode(tokens[:target])
        for count in candidates:
            request = {"state": state, "questions": {"route": {
                "type": "choice", "instructions": "Which route is permitted by the observation?",
                "criteria": {f"route_{i:02}": f"Take route number {i:02}." for i in range(count)}}}}
            workloads.append({"id": f"context-{target}-choice-{count}", "kind": "synthetic_scaling",
                              "target_state_tokens": target, "request": request,
                              **request_metadata(tokenizer, request)})
    if len({item["id"] for item in workloads}) != len(workloads):
        raise ValueError("workload IDs must be unique")
    return workloads


def validate_response(request, response):
    # The serving formatter validates distributions; this additionally checks wire coverage.
    expected = compile_request(request["state"], request["questions"])
    if set(response.get("answers", {})) != {record["id"] for record in expected}:
        raise ValueError("response question coverage differs")
    for record in expected:
        answer = response["answers"][record["id"]]
        if answer["type"] != record["kind"]:
            raise ValueError("response question type differs")
        values = [1 - answer["noul"], answer["noul"]] if record["kind"] == "noul" else list(answer["probabilities"].values())
        if record["kind"] != "noul" and list(answer["probabilities"]) != record["answer_keys"]:
            raise ValueError("response candidate coverage/order differs")
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values) or not math.isclose(sum(values), 1, abs_tol=1e-6):
            raise ValueError("invalid response probability distribution")


def benchmark(predictor, workloads, *, output, warmup=3, repetitions=20,
              probability_tolerance=1e-4, max_seconds=900, synchronize=None, http_timeout=120):
    """One loaded model, concurrency one, four paths alternated within each repeat."""
    import torch
    device = torch.device(predictor.scorer.model.device_name)
    synchronize = synchronize or (lambda: torch.cuda.synchronize(device) if device.type == "cuda" else
                                  torch.mps.synchronize() if device.type == "mps" else None)
    original_mode = predictor.prefix_cache

    class SynchronizedPredictor:
        model_name, method = predictor.model_name, predictor.method

        def predict(self, request):
            synchronize()
            response = predictor.predict(request)
            synchronize()
            return response

    server = make_server(SynchronizedPredictor(), "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    samples, parity = [], []
    started = time.monotonic()
    exhausted = False
    fatal_error = False
    output = Path(output)

    def run(workload, phase, repetition, order, transport, cached):
        predictor.prefix_cache = cached
        sample = {"request_id": workload["id"], "request_sha256": workload["request_sha256"],
                  "phase": phase, "repetition": repetition, "order": order,
                  "transport": transport, "mode": "cached" if cached else "uncached",
                  "started_at": datetime.now(timezone.utc).isoformat(), "success": False}
        connection = None
        begin = time.perf_counter()
        try:
            synchronize()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            elif device.type == "mps":
                # MPS has no peak counter: point-in-time allocator readings, never a peak.
                sample["mps_allocated_bytes_at_start"] = torch.mps.current_allocated_memory()
            begin = time.perf_counter()
            if transport == "predictor":
                response = predictor.predict(workload["request"])
                synchronize()
            else:
                payload = json.dumps(workload["request"], ensure_ascii=False, allow_nan=False).encode()
                connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=http_timeout)
                connection.request("POST", "/v1/inference", payload, {"Content-Type": "application/json"})
                received = connection.getresponse()
                raw = received.read()
                sample["http_status"] = received.status
                sample["http_response_text"] = raw.decode("utf-8", errors="replace")
                response = strict_json(raw)
                sample["response"] = response
                if received.status != 200:
                    sample["error_response"] = response
                    raise RuntimeError(f"HTTP {received.status}")
            sample["response"] = response
            validate_response(workload["request"], response)
            sample["success"] = True
            sample["response"] = response
        except Exception as error:
            sample["error_type"], sample["error"] = type(error).__name__, str(error)
            if transport == "http_loopback" and isinstance(error, (OSError, http.client.HTTPException)):
                sample["fatal"] = True  # A timed-out handler may still be computing; never overlap another request.
        finally:
            try:
                synchronize()
                if device.type == "cuda":
                    sample["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
                elif device.type == "mps":
                    sample["mps_allocated_bytes_at_end"] = torch.mps.current_allocated_memory()
                    sample["mps_driver_allocated_bytes_at_end"] = torch.mps.driver_allocated_memory()
            except Exception as error:
                sample.update(success=False, fatal=True, synchronization_error=f"{type(error).__name__}: {error}")
            sample["wall_ms"] = (time.perf_counter() - begin) * 1000
            if connection:
                connection.close()
        return sample

    try:
        with (output / "samples.jsonl").open("x") as raw_file:
            for workload in workloads:
                for iteration in range(warmup + repetitions):
                    if time.monotonic() - started > max_seconds:
                        exhausted = True
                        break
                    phase = "warmup" if iteration < warmup else "measured"
                    repetition = iteration if phase == "warmup" else iteration - warmup
                    paths = [("predictor", False), ("predictor", True), ("http_loopback", False), ("http_loopback", True)]
                    offset = iteration % len(paths)
                    paths = paths[offset:] + paths[:offset]
                    pair = {}
                    for order, (transport, cached) in enumerate(paths):
                        sample = run(workload, phase, repetition, order, transport, cached)
                        samples.append(sample)
                        raw_file.write(json.dumps(sample, ensure_ascii=False, allow_nan=False) + "\n")
                        raw_file.flush()
                        pair[(transport, cached)] = sample
                        if sample.get("fatal"):
                            fatal_error = True
                            break
                    if fatal_error:
                        break
                    if phase == "measured":
                        for transport in ("predictor", "http_loopback"):
                            left, right = pair[(transport, False)], pair[(transport, True)]
                            check = (compare_answers(left["response"], right["response"], probability_tolerance)
                                     if left["success"] and right["success"] else {"passed": False, "reason": "request_error"})
                            parity.append({"request_id": workload["id"], "transport": transport,
                                           "repetition": repetition, **check})
                if exhausted or fatal_error:
                    break
                print(json.dumps({"workload_complete": workload["id"], "elapsed_seconds": time.monotonic() - started}), flush=True)
    finally:
        predictor.prefix_cache = original_mode
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    summaries = summarize(samples)
    comparisons = []
    for workload in workloads:
        for transport in ("predictor", "http_loopback"):
            rows = {row["mode"]: row for row in summaries if row["request_id"] == workload["id"] and row["transport"] == transport}
            checks = [row for row in parity if row["request_id"] == workload["id"] and row["transport"] == transport]
            passed = len(checks) == repetitions and all(row["passed"] for row in checks)
            comparisons.append({"request_id": workload["id"], "transport": transport,
                                "parity_passed": passed,
                                "compared_pairs": sum("max_probability_error" in row for row in checks),
                                "max_probability_error": max((row["max_probability_error"] for row in checks if "max_probability_error" in row), default=None),
                                "uncached_over_cached_p50": (rows["uncached"]["p50_ms"] / rows["cached"]["p50_ms"]
                                                             if passed else None)})
    status = ("runtime_error" if fatal_error else "budget_exhausted" if exhausted else "request_errors" if any(not row["success"] for row in samples)
              else "parity_failed" if not all(row["passed"] for row in parity) else "passed")
    return {"status": status, "summaries": summaries, "cache_comparisons": comparisons, "parity": parity,
            "warmup_attempts": sum(row["phase"] == "warmup" for row in samples),
            "measured_attempts": sum(row["phase"] == "measured" for row in samples),
            "errors_including_warmup": sum(not row["success"] for row in samples),
            "elapsed_seconds": time.monotonic() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--request", type=Path, action="append", default=[])
    parser.add_argument("--requests", type=Path, help="Previously prepared requests.json; no regeneration")
    parser.add_argument("--contexts", type=int, nargs="+", default=[128, 512, 1024])
    parser.add_argument("--candidates", type=int, nargs="+", default=[2, 8, 32])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--max-seconds", type=float, default=900)
    parser.add_argument("--max-probability-error", type=float, default=1e-4)
    parser.add_argument("--http-timeout", type=float, default=120,
                        help="Loopback HTTP timeout in seconds; raise for slow CPU runs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true", help="CPU tokenizer only; no model/GPU allocation")
    args = parser.parse_args()
    if (not 1 <= args.repetitions <= 200 or not 1 <= args.warmup <= 10 or not 1 <= args.batch_size <= 256
            or not 1 <= args.max_seconds <= 3600 or not 0 <= args.max_probability_error <= 1
            or not 1 <= args.http_timeout <= 3600
            or any(not 32 <= n <= 3072 for n in args.contexts) or any(not 2 <= n <= 255 for n in args.candidates)):
        parser.error("invalid bounded benchmark arguments")
    if args.output.resolve().is_relative_to(args.checkpoint.resolve()):
        parser.error("output must be outside checkpoint")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(), "status": "initializing"}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    try:
        config = strict_json((args.checkpoint / "model.json").read_bytes())
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(config["model_id"], revision=config["revision"])
        if args.requests:
            workloads = strict_json(args.requests.read_bytes())["workloads"]
            for workload in workloads:
                if workload["request_sha256"] != digest(workload["request"]):
                    raise ValueError("prepared request checksum differs")
                workload.update(request_metadata(tokenizer, workload["request"]))
        else:
            workloads = make_workloads(tokenizer, args.request, args.contexts, args.candidates)
        if not 1 <= len(workloads) <= 32 or any(row["max_candidate_input_tokens"] > config["max_length"] for row in workloads):
            raise ValueError("requires 1-32 workloads within checkpoint context limit; no truncation")
        (args.output / "requests.json").write_text(json.dumps({"schema_version": 1, "workloads": workloads}, indent=2, ensure_ascii=False) + "\n")
        report = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
                  "checkpoint": str(args.checkpoint.resolve()), "checkpoint_model_config": config,
                  "configuration": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items() if key != "request"},
                  "scope": {"predictor": "Warm model, request compilation + tokenization + synchronized inference + typed formatting and validation; model loading excluded.",
                            "http_loopback": "Same warm model over real loopback HTTP, fresh TCP connection each request; includes JSON encoding, server, synchronized inference, response decoding and validation. No external Internet.",
                            "concurrency": 1, "cache": "request-local only; no cross-request result or prefix reuse",
                            "percentile_method": "linear interpolation at (n-1)*q, successful measured attempts; failures and warmups retained separately",
                            "comparability": "A hosted API includes network, routing and service scheduling; no direct model-only speedup may be inferred from that difference."},
                  "workload_manifest_sha256": digest(workloads)}
        if args.prepare_only:
            report["status"] = "prepared_no_model_inference"
        else:
            import torch
            from jev.serving import load_predictor
            loading = time.perf_counter()
            predictor = load_predictor(checkpoint=args.checkpoint, device=args.device, batch_size=args.batch_size)
            if torch.device(args.device).type == "cuda":
                torch.cuda.synchronize(torch.device(args.device))
            elif torch.device(args.device).type == "mps":
                torch.mps.synchronize()
            report["model_load_seconds"] = time.perf_counter() - loading
            report["provenance"] = predictor.provenance
            report["temperature"] = predictor.temperature
            report["runtime"] = {"python": platform.python_version(), "platform": platform.platform(),
                                 "hostname": platform.node(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                                 "torch_cuda": torch.version.cuda,
                                 "gpu": (torch.cuda.get_device_name(torch.device(args.device)) if torch.device(args.device).type == "cuda"
                                         else sysctl("machdep.cpu.brand_string") if torch.device(args.device).type == "mps" else None),
                                 "backbone_parameter_dtypes": sorted({str(parameter.dtype) for parameter in predictor.scorer.model.backbone.parameters()}),
                                 "head_parameter_dtypes": sorted({str(parameter.dtype) for parameter in predictor.scorer.model.head.parameters()}),
                                 "attention_implementation": getattr(predictor.scorer.model.backbone.config, "_attn_implementation", None),
                                 **{name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft")}}
            if torch.device(args.device).type != "cuda":
                # Additive for CPU and Apple MPS runs only, so CUDA reports keep their exact shape.
                # CPU thread count bounds CPU speed; the chip identity is the SoC for both devices.
                report["runtime"].update(cpu=sysctl("machdep.cpu.brand_string"), torch_num_threads=torch.get_num_threads(),
                                         pytorch_enable_mps_fallback=os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"))
            if torch.device(args.device).type == "mps":
                report["runtime"]["mps_recommended_max_memory_bytes"] = torch.mps.recommended_max_memory()
                report["gpu_snapshot_before"] = apple_gpu_statistics()
            else:
                try:
                    report["gpu_snapshot_before"] = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,utilization.gpu", "--format=csv,noheader"], text=True)
                except (OSError, subprocess.CalledProcessError):
                    report["gpu_snapshot_before"] = None
            report.update(benchmark(predictor, workloads, output=args.output, warmup=args.warmup,
                                    repetitions=args.repetitions, probability_tolerance=args.max_probability_error,
                                    max_seconds=args.max_seconds, http_timeout=args.http_timeout))
    except BaseException as error:
        report.update(status="runtime_error", error_type=type(error).__name__, error=str(error))
    report["source_sha256"] = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
                               ("scripts/benchmark_inference_latency.py", "scripts/benchmark_prefix_cache.py", "jev/model.py", "jev/serving.py", "jev/server.py", "jev/prefix_cache.py")}
    (args.output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "output": str(args.output.resolve())}))
    return 0 if report["status"] in ("passed", "prepared_no_model_inference") else 2


if __name__ == "__main__":
    raise SystemExit(main())
