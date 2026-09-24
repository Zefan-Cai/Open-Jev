"""Compare the answers of two latency-benchmark runs, e.g. CPU against Apple MPS.

Both runs must have replayed identical requests (checked by request SHA-256).
For every request, the first successful measured Predictor response in the
chosen cache mode is compared with compare_answers: question, type and
candidate coverage, selected decisions, usage, and maximum probability error.
Drift between backends is expected and reported as a number; exceeding the
tolerance does not fail the run, but a changed decision is listed per request.
"""
import argparse
import json
from pathlib import Path

from scripts.benchmark_prefix_cache import compare_answers


def responses(run, mode):
    rows = {}
    for line in (Path(run) / "samples.jsonl").read_text().splitlines():
        sample = json.loads(line)
        if (sample["phase"] == "measured" and sample["transport"] == "predictor" and sample["mode"] == mode
                and sample["success"] and sample["request_id"] not in rows):
            rows[sample["request_id"]] = sample
    return rows


def compare_runs(left, right, *, mode="uncached", tolerance=1e-4):
    a, b = responses(left, mode), responses(right, mode)
    if not a or set(a) != set(b):
        raise ValueError("runs do not cover the same successful requests")
    if any(a[key]["request_sha256"] != b[key]["request_sha256"] for key in a):
        raise ValueError("runs replayed different request contents")
    results = {key: compare_answers(a[key]["response"], b[key]["response"], tolerance) for key in sorted(a)}
    return {"left": str(left), "right": str(right), "mode": mode, "probability_tolerance": tolerance,
            "requests": len(results), "all_decisions_equal": all(row["argmax_equal"] for row in results.values()),
            "all_within_tolerance": all(row["passed"] for row in results.values()),
            "max_probability_error": max(row["max_probability_error"] for row in results.values()),
            "changed_decisions": {key: row["changed_decisions"] for key, row in results.items() if row["changed_decisions"]},
            "per_request": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("left", type=Path, help="benchmark output directory, e.g. the CPU run")
    parser.add_argument("right", type=Path, help="benchmark output directory, e.g. the MPS run")
    parser.add_argument("--mode", choices=("uncached", "cached"), default="uncached")
    parser.add_argument("--max-probability-error", type=float, default=1e-4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare_runs(args.left, args.right, mode=args.mode, tolerance=args.max_probability_error)
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
