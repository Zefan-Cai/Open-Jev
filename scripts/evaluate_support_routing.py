"""A local BANKING77 routing and human-review loop, with frozen calibration.

Retrieval uses official training text; calibration uses reserved training
utterances; the official test split never chooses candidates using its label
and never chooses a confidence threshold. Standard library only.
"""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import random
import re
import time
from urllib.request import Request, urlopen

from jev.community_routing_v3 import SOURCES, normalize, option_text


UNKNOWN = "__review__"
VERSION = "support-routing-v1"
QUESTION = "Which banking support intent matches this customer's request? Select review if none of the available intents matches. Classify only; do not perform the request."


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def make_index(rows):
    docs = {}
    for row in rows:
        docs.setdefault(row["category"], Counter()).update(tokens(row["text"]))
    frequency = Counter(t for terms in docs.values() for t in terms)
    return {"schema_version": 1, "method": "BM25 over official-train intent documents; no gold-guided shortlist",
            "labels": {label: option_text(label) for label in sorted(docs)},
            "counts": {label: dict(sorted(terms.items())) for label, terms in sorted(docs.items())},
            "lengths": {label: sum(terms.values()) for label, terms in docs.items()},
            "document_frequency": dict(frequency), "average_length": sum(sum(x.values()) for x in docs.values()) / len(docs)}


def shortlist(text, index, count=8):
    n = len(index["labels"])
    if not 1 <= count <= n:
        raise ValueError("shortlist size is outside the label catalog")
    query = Counter(tokens(text))
    scored = []
    for label, terms in index["counts"].items():
        score = 0.0
        for token, qcount in query.items():
            tf = terms.get(token, 0)
            df = index["document_frequency"].get(token, 0)
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            denominator = tf + 1.2 * (0.25 + 0.75 * index["lengths"][label] / index["average_length"])
            score += qcount * idf * tf * 2.2 / denominator
        scored.append((score, label))
    return [label for _, label in sorted(scored, key=lambda x: (-x[0], x[1]))[:count]]


def request_for(text, index, count=8):
    labels = shortlist(text, index, count)
    return {"model": "open-jev", "state": {"customer_request": text}, "questions": {"route": {
        "type": "choice", "instructions": QUESTION, "criteria": {**{x: index["labels"][x] for x in labels},
                                                                  UNKNOWN: "Human review: no matching intent is available or the request is insufficiently specific."}}}}


def prepare(output, *, seed=20261002, calibration_rows=96, test_rows=256, demo_dir=None):
    if calibration_rows < 1 or not 1 <= test_rows <= 3080:
        raise ValueError("Invalid split sample sizes")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    source = SOURCES["banking77"]
    datasets, receipts = {}, {}
    for name, info in source["files"].items():
        url = f"https://raw.githubusercontent.com/{source['repository']}/{source['revision']}/{info['path']}"
        with urlopen(url, timeout=30) as response:
            raw = response.read()
        if sha(raw) != info["sha256"]:
            raise ValueError("Pinned BANKING77 source checksum changed")
        datasets[info["split"]] = list(csv.DictReader(io.StringIO(raw.decode())))
        receipts[name] = {"sha256": sha(raw), "source_url": url, "rows": len(datasets[info["split"]])}
    # Reserve normalized official-test text before train deduplication/sampling.
    test_text = {normalize(r["text"]) for r in datasets["test"]}
    seen, train = set(), []
    for i, row in enumerate(datasets["train"]):
        norm = normalize(row["text"])
        if norm in test_text or norm in seen:
            continue
        seen.add(norm)
        train.append({**row, "id": f"banking77-train-{i}"})
    train.sort(key=lambda r: sha(f"{seed}:cal:{normalize(r['text'])}".encode()))
    if calibration_rows >= len(train):
        raise ValueError("Calibration consumes all retrieval training rows")
    cal, retrieval = train[:calibration_rows], train[calibration_rows:]
    index = make_index(retrieval)
    test = [{**r, "id": f"banking77-test-{i}"} for i, r in enumerate(datasets["test"])]
    random.Random(seed).shuffle(test)
    test = test[:test_rows]
    write_json(output / "index.json", index)
    for split, rows in (("calibration", cal), ("test", test)):
        write_csv(output / f"{split}-inputs.csv", rows, ["id", "text"])
        write_json(output / f"{split}-gold.json", {r["id"]: r["category"] for r in rows})
    majority = Counter(r["category"] for r in retrieval).most_common(1)[0][0]
    manifest = {"schema_version": 1, "version": VERSION, "seed": seed, "sources": receipts,
                "license": source["license"], "attribution": source["attribution"],
                "counts": {"retrieval_train": len(retrieval), "calibration_official_train": len(cal), "official_test_sample": len(test)},
                "majority_baseline_label": majority,
                "threshold_policy": "Fit on reserved official-train calibration only, then lock before official-test predictions; no test threshold fitting.",
                "training_exposure": "Released checkpoints may have trained on BANKING77 official train. This is supervised intent transfer, not guaranteed unseen-task zero-shot.",
                "unknown_scope": "BANKING77 has no official out-of-scope examples. Review is abstention, not an evaluated OOS class.",
                "files_sha256": {p.name: sha(p.read_bytes()) for p in sorted(output.iterdir()) if p.is_file()}}
    write_json(output / "manifest.json", manifest)
    if demo_dir:
        demo = Path(demo_dir)
        demo.mkdir(parents=True, exist_ok=True)
        write_json(demo / "index.json", index)
        # Calibration examples can demonstrate the workflow without disclosing test inputs.
        write_csv(demo / "sample.csv", cal[:12], ["id", "text"])
        write_json(demo / "source.json", {"license": source["license"], "attribution": source["attribution"], "sources": receipts,
                                         "sample": "12 natural official-training utterances; no synthetic users and no test labels"})
    return manifest


def infer(request, endpoint, timeout=120):
    started = time.perf_counter()
    req = Request(endpoint, json.dumps(request, allow_nan=False).encode(), {"Content-Type": "application/json"})
    with urlopen(req, timeout=timeout) as response:
        result = json.load(response)
    answer = result["answers"]["route"]
    probs = answer["probabilities"]
    if set(probs) != set(request["questions"]["route"]["criteria"]) or any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1 for p in probs.values()) or not math.isclose(sum(probs.values()), 1, abs_tol=1e-5):
        raise ValueError("Invalid probability vector")
    chosen = max(probs, key=probs.__getitem__)
    if answer["choice"] != chosen:
        raise ValueError("Choice is not probability argmax")
    return {"prediction": chosen, "top_probability": probs[chosen], "probabilities": probs,
            "model": result.get("model"), "metadata": result.get("metadata", {}), "usage": result.get("usage", {}),
            "wall_ms": (time.perf_counter() - started) * 1000}


def measure(rows, index, endpoint, output, count=8):
    results, identity = [], None
    with Path(output).open("x") as handle:
        for row in rows:
            started = time.perf_counter()
            request = request_for(row["text"], index, count)
            result = {"id": row["id"], "text": row["text"], "candidate_labels": list(request["questions"]["route"]["criteria"]),
                      "retrieval_prediction": next(iter(request["questions"]["route"]["criteria"])), "request_sha256": sha(json.dumps(request, sort_keys=True).encode())}
            try:
                result.update(infer(request, endpoint))
                current = {"model": result["model"], **{k: result["metadata"].get(k) for k in ["checkpoint_sha256", "base_revision", "temperature", "method", "code_commit", "max_length"]}}
                if any(current.get(k) is None for k in current) or not current["model"]:
                    raise ValueError("Missing model/checkpoint identity metadata")
                if identity is not None and current != identity:
                    raise ValueError("Model identity changed during the frozen run")
                identity = current
                result["success"] = True
            except Exception as error:
                result.update(success=False, prediction=UNKNOWN, top_probability=0.0, error=f"{type(error).__name__}: {error}")
            result["wall_ms"] = (time.perf_counter() - started) * 1000
            results.append(result)
            handle.write(json.dumps(result, allow_nan=False) + "\n")
            handle.flush()
            if "identity changed" in result.get("error", ""):
                raise ValueError(result["error"])
    return results, identity


def metrics(results, gold, threshold, majority):
    accepted = [r for r in results if r["success"] and r["prediction"] != UNKNOWN and r["top_probability"] >= threshold]
    n, correct = len(results), sum(r["success"] and r["prediction"] == gold[r["id"]] for r in results)
    kept_correct = sum(r["prediction"] == gold[r["id"]] for r in accepted)
    return {"n": n, "correct": correct, "routing_accuracy": correct / n if n else None,
            "candidate_recall": sum(gold[r["id"]] in r["candidate_labels"] for r in results) / n if n else None,
            "accepted": len(accepted), "accepted_correct": kept_correct, "coverage": len(accepted) / n if n else None,
            "error_among_accepted": (len(accepted) - kept_correct) / len(accepted) if accepted else None,
            "review_rate": 1 - len(accepted) / n if n else None,
            "failures": sum(not r["success"] for r in results),
            "majority_baseline_accuracy": sum(gold[r["id"]] == majority for r in results) / n if n else None,
            "retrieval_top1_accuracy": sum(r["retrieval_prediction"] == gold[r["id"]] for r in results) / n if n else None,
            "wall_seconds": sum(r.get("wall_ms", 0) for r in results) / 1000}


def fit_threshold(results, gold, target=0.90, minimum_accepted=10):
    if not 0 < target <= 1 or minimum_accepted < 1:
        raise ValueError("Invalid calibration policy")
    thresholds = sorted({0.0, 1.0, *[r["top_probability"] for r in results if r["success"]]})
    eligible = []
    for threshold in thresholds:
        accepted = [r for r in results if r["success"] and r["prediction"] != UNKNOWN and r["top_probability"] >= threshold]
        correct = sum(r["prediction"] == gold[r["id"]] for r in accepted)
        if len(accepted) >= minimum_accepted and correct / len(accepted) >= target:
            eligible.append((len(accepted), -threshold, threshold))
    return max(eligible)[2] if eligible else 1.000001


def evaluate(dataset, endpoint, output, *, candidates=8, target=0.90, minimum_accepted=10):
    dataset, output = Path(dataset), Path(output)
    manifest = json.loads((dataset / "manifest.json").read_text())
    for name, expected in manifest["files_sha256"].items():
        if sha((dataset / name).read_bytes()) != expected:
            raise ValueError("Frozen input changed: " + name)
    output.mkdir(parents=True, exist_ok=False)
    index = json.loads((dataset / "index.json").read_text())
    cal_gold = json.loads((dataset / "calibration-gold.json").read_text())
    cal, identity = measure(read_csv(dataset / "calibration-inputs.csv"), index, endpoint, output / "calibration-predictions.jsonl", candidates)
    threshold = fit_threshold(cal, cal_gold, target, minimum_accepted)
    lock = {"schema_version": 1, "threshold": threshold, "confidence_measure": "top_label_probability", "target_calibration_accuracy": target,
            "minimum_accepted": minimum_accepted, "identity": identity, "dataset_manifest_sha256": sha((dataset / "manifest.json").read_bytes()),
            "calibration_predictions_sha256": sha((output / "calibration-predictions.jsonl").read_bytes()), "candidates": candidates,
            "selection": "maximum calibration coverage meeting empirical target; test never chooses threshold; no statistical guarantee"}
    write_json(output / "decisions.lock.json", lock)  # Persist before opening test gold or collecting test predictions.
    test, test_identity = measure(read_csv(dataset / "test-inputs.csv"), index, endpoint, output / "test-predictions.jsonl", candidates)
    if test_identity != identity:
        raise ValueError("Calibration and test model identities differ")
    test_gold = json.loads((dataset / "test-gold.json").read_text())
    summary = {"schema_version": 1, "status": "complete" if all(r["success"] for r in cal + test) else "complete_with_failures",
               "scope": "Natural BANKING77 official test sample; not an OOS benchmark or untouched-task zero-shot claim",
               "identity": identity, "threshold": threshold, "endpoint": endpoint, "manifest": manifest,
               "calibration": metrics(cal, cal_gold, threshold, manifest["majority_baseline_label"]),
               "test": metrics(test, test_gold, threshold, manifest["majority_baseline_label"])}
    write_json(output / "summary.json", summary)
    queue_rows = [{"id": r["id"], "text": r["text"], "proposed_label": r["prediction"], "top_probability": r["top_probability"],
                   "status": "accepted" if r["success"] and r["prediction"] != UNKNOWN and r["top_probability"] >= threshold else "review",
                   "reviewed_label": "", "reviewer": ""} for r in test]
    write_csv(output / "review-queue.csv", queue_rows, ["id", "text", "proposed_label", "top_probability", "status", "reviewed_label", "reviewer"])
    return summary


def finalize(queue, index, output):
    rows = read_csv(queue)
    labels = set(index["labels"])
    for row in rows:
        reviewed = row.get("reviewed_label", "").strip()
        if reviewed and (reviewed not in labels | {UNKNOWN} or not row.get("reviewer", "").strip()):
            raise ValueError("Reviewed rows require a valid catalog/no-route label and reviewer")
        row["final_label"] = reviewed or (row["proposed_label"] if row["status"] == "accepted" else "")
        row["decision_source"] = "human" if reviewed else "model" if row["final_label"] else "unresolved"
    write_csv(output, rows, ["id", "text", "final_label", "decision_source", "reviewer", "top_probability"])
    return {"n": len(rows), "human_reviewed": sum(r["decision_source"] == "human" for r in rows),
            "unresolved": sum(r["decision_source"] == "unresolved" for r in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--output", required=True)
    prep.add_argument("--calibration-rows", type=int, default=96)
    prep.add_argument("--test-rows", type=int, default=256)
    prep.add_argument("--seed", type=int, default=20261002)
    prep.add_argument("--demo-dir")
    run = sub.add_parser("evaluate")
    run.add_argument("--dataset", required=True)
    run.add_argument("--endpoint", default="http://127.0.0.1:8791/v1/systemone")
    run.add_argument("--output", required=True)
    run.add_argument("--candidates", type=int, default=8)
    run.add_argument("--target", type=float, default=.90)
    run.add_argument("--minimum-accepted", type=int, default=10)
    final = sub.add_parser("finalize")
    final.add_argument("--queue", required=True)
    final.add_argument("--index", required=True)
    final.add_argument("--output", required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "prepare":
        result = prepare(**args)
    elif command == "evaluate":
        result = evaluate(**args)
    else:
        args["index"] = json.loads(Path(args["index"]).read_text())
        result = finalize(**args)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
