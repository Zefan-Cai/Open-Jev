"""Four-rank training with explicit weights-only initialization or strict resume.

Launch with torchrun --standalone --nproc-per-node=4 -m jev.train_distributed.
Single-process snapshots are deliberately not a supported resume format.
"""
import argparse
from datetime import timedelta
import gc
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import socket
import subprocess
import tempfile
import time

from .train import (BASELINE_FILES, SNAPSHOT_FILES, OPTIMIZER_SETTINGS,
                    _file_sha256, _json_sha256, evaluate as evaluate_cuda, read_rows,
                    restore_training_artifacts, source_checkout_commit, training_identity,
                    validate_baseline_identity, validate_resume_identity)


WORLD_SIZE = 4


def checked_reload_error(reference, actual):
    if (not reference or len(reference) != len(actual) or
            any(not math.isfinite(value) for value in [*reference, *actual])):
        raise ValueError("Checkpoint reload produced nonfinite or mismatched logits")
    error = max(abs(x - y) for x, y in zip(reference, actual))
    if not math.isfinite(error) or error > 0.05:
        raise ValueError(f"Checkpoint reload mismatch: {error}")
    return error


def validate_allocation(policy, hostname, visible, inventory):
    if (hostname != policy.get("expected_hostname") or
            policy.get("distributed_training_gpu_indices") != [0, 1, 2, 3] or
            policy.get("allowed_gpu_indices") != [0, 1, 2, 3]):
        raise ValueError("NCCL host/GPU allocation differs from resource policy")
    devices = {}
    for line in inventory.splitlines():
        index, uuid = [part.strip() for part in line.split(",")]
        if int(index) in devices:
            raise ValueError("Duplicate physical GPU in inventory")
        devices[int(index)] = uuid
    expected = [devices.get(i) for i in range(WORLD_SIZE)]
    if any(not uuid or not uuid.startswith("GPU-") for uuid in expected) or visible.split(",") != expected:
        raise ValueError("CUDA_VISIBLE_DEVICES must contain the physical GPU 0–3 UUIDs in order")
    return {"hostname": hostname, "physical_gpu_indices": [0, 1, 2, 3], "visible_uuids": expected}


def evaluate(model, rows, path):
    """Use the unchanged evaluator on CUDA; retain its record form on CPU."""
    import torch
    if next(model.parameters()).device.type == "cuda":
        return evaluate_cuda(model, rows, path)
    results = []
    model.eval()
    with torch.inference_mode(), Path(path).open("w") as handle:
        for row in rows:
            start = time.perf_counter()
            logits = model([row])[0].float()
            value = {key: row[key] for key in ("id", "kind", "source", "group_id", "target")}
            value.update(question_id=row["metadata"].get("question_id", row["kind"]),
                         target_basis=row["metadata"].get("target_basis", "hard_label"),
                         logits=logits.tolist(), probabilities=logits.softmax(-1).tolist(),
                         latency_seconds=time.perf_counter() - start)
            handle.write(json.dumps(value) + "\n")
            handle.flush()
            results.append(value)
    return results


def rank_row_index(completed_steps, rank, row_count, world_size=WORLD_SIZE):
    if world_size != WORLD_SIZE or not 0 <= rank < world_size or row_count < 1 or completed_steps < 0:
        raise ValueError("Exactly four ranks, a nonempty dataset and a valid cursor are required")
    return (completed_steps * WORLD_SIZE + rank) % row_count


def distribution_identity(backend):
    if backend not in ("nccl", "gloo"):
        raise ValueError("Unsupported distributed backend")
    return {"world_size": WORLD_SIZE, "local_rows_per_step": 1, "global_batch_size": WORLD_SIZE,
            "backend": backend, "device_type": "cuda" if backend == "nccl" else "cpu",
            "gradient_reduction": "mean; no additional division of local loss",
            "row_order": "train[(completed_steps * 4 + rank) % len(train)]",
            "rng_policy": "Common initialization seed; independent SHA256(seed,rank) training streams; restore all four streams on DDP resume",
            "bitwise_single_process_equivalence": False,
            "implementation_sha256": _file_sha256(__file__)}


def on_rank_zero(function):
    """All ranks participate; propagate rank-zero I/O/evaluation failures."""
    import torch.distributed as dist
    result = [None]
    if dist.get_rank() == 0:
        try:
            result[0] = {"value": function(), "error": None}
        except Exception as error:
            result[0] = {"value": None, "error": f"{type(error).__name__}: {error}"}
    dist.broadcast_object_list(result, src=0)
    if result[0]["error"] is not None:
        raise RuntimeError("Rank-zero operation failed: " + result[0]["error"])
    return result[0]["value"]


def seed_training_rank(seed, rank, device):
    import torch
    value = int(_json_sha256(["open-jev-ddp-rng-v1", seed, rank])[:16], 16) % (2**63 - 1)
    random.seed(value)
    torch.random.default_generator.manual_seed(value)
    if device.type == "cuda":
        torch.cuda.manual_seed(value)
    return value


def capture_rng(device):
    import torch
    return {"python": random.getstate(), "torch_cpu": torch.get_rng_state(),
            "device_type": device.type,
            "torch_device": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}


def restore_rng(rng, device):
    import torch
    if rng["device_type"] != device.type or ((rng["torch_device"] is None) != (device.type == "cpu")):
        raise ValueError("Checkpoint RNG device type differs")
    random.setstate(rng["python"])
    torch.set_rng_state(rng["torch_cpu"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(rng["torch_device"], device)


def parameter_groups(model, optimizer):
    names = {id(p): name for name, p in model.named_parameters() if p.requires_grad}
    groups = [[names[id(p)] for p in group["params"]] for group in optimizer.param_groups]
    flattened = [name for group in groups for name in group]
    if len(flattened) != len(set(flattened)) or set(flattened) != set(names.values()):
        raise ValueError("Optimizer must contain every trainable parameter exactly once")
    return groups


def read_distributed_checkpoint(path, identity, distribution):
    path = Path(path)
    info = json.loads((path / "resume.json").read_text())
    if (info.get("schema_version") != 2 or info.get("kind") != "distributed_training_resume_only"
            or info.get("complete") is not True or info.get("inference_ready") is not False):
        raise ValueError("Not a complete DDP snapshot; single-process migration is unsupported")
    validate_resume_identity(info.get("identity", {}), identity)
    validate_resume_identity(info.get("distribution", {}), distribution)
    step = info.get("completed_step")
    if type(step) is not int or not 0 < step <= identity["arguments"]["steps"]:
        raise ValueError("Invalid completed optimizer step")
    if set(info.get("files_sha256", {})) != set(SNAPSHOT_FILES):
        raise ValueError("Incomplete DDP snapshot file manifest")
    for name, digest in info["files_sha256"].items():
        if _file_sha256(path / name) != digest:
            raise ValueError("DDP snapshot checksum mismatch: " + name)
    logs = [json.loads(line) for line in (path / "training.jsonl").read_text().splitlines() if line.strip()]
    if [row.get("step") for row in logs] != list(range(1, step + 1)):
        raise ValueError("DDP snapshot log cursor differs from completed step")
    return info


def validate_initialization_provenance(value):
    if isinstance(value, dict) and value.get("kind") == "inference_package_weights_only":
        from .inference_initialization import validate_inference_initialization_provenance
        return validate_inference_initialization_provenance(value)
    hashes = {"source_manifest_sha256", "source_training_state_sha256", "source_training_identity_sha256",
              "source_distribution_sha256"}
    fixed = {"kind": "ddp_snapshot_weights_only", "optimizer": "new_adamw_state",
             "rng": "new_run_seed_and_rank_streams", "step_cursor": 0,
             "baseline_initialization": "warm_start_checkpoint"}
    if (not isinstance(value, dict) or set(value) != hashes | set(fixed) | {"source_completed_step"}
            or any(value.get(key) != expected or type(value.get(key)) is not type(expected) for key, expected in fixed.items())
            or type(value.get("source_completed_step")) is not int or value["source_completed_step"] < 1
            or any(not isinstance(value.get(key), str) or re.fullmatch(r"[0-9a-f]{64}", value[key]) is None for key in hashes)):
        raise ValueError("Invalid weights-only initialization provenance")
    return value


def read_initialization_checkpoint(path, model, revision, lora_rank):
    """Validate a prior complete snapshot against its own immutable run identity."""
    path = Path(path)
    hint = json.loads((path / "resume.json").read_text())
    metadata = read_distributed_checkpoint(path, hint.get("identity", {}), hint.get("distribution", {}))
    original = metadata["identity"]["arguments"]
    expected = {"model": model, "revision": revision, "lora_rank": lora_rank}
    if any(original.get(key) != value or type(original.get(key)) is not type(value) for key, value in expected.items()):
        raise ValueError("Initialization snapshot base model, revision or LoRA rank differs")
    if lora_rank < 1 or metadata["distribution"].get("world_size") != WORLD_SIZE:
        raise ValueError("Initialization requires a trained four-rank LoRA snapshot")
    provenance = {"kind": "ddp_snapshot_weights_only", "source_manifest_sha256": _file_sha256(path / "resume.json"),
                  "source_training_state_sha256": metadata["files_sha256"]["training_state.pt"],
                  "source_training_identity_sha256": _json_sha256(metadata["identity"]),
                  "source_distribution_sha256": _json_sha256(metadata["distribution"]),
                  "source_completed_step": metadata["completed_step"], "optimizer": "new_adamw_state",
                  "rng": "new_run_seed_and_rank_streams", "step_cursor": 0,
                  "baseline_initialization": "warm_start_checkpoint"}
    return metadata, validate_initialization_provenance(provenance)


def bind_initialization_identity(identity, *, initialization=None, resume_metadata=None):
    """A resumed new-stage run inherits its original, tensor-bound seed identity."""
    if initialization is not None and resume_metadata is not None:
        raise ValueError("Weights-only initialization and exact resume are mutually exclusive")
    if resume_metadata is not None:
        initialization = resume_metadata.get("identity", {}).get("initialization")
    if initialization is None:
        return identity
    return {**identity, "initialization": dict(validate_initialization_provenance(initialization))}


def initialize_training_weights(model, path, metadata, provenance):
    """Copy only LoRA/head tensors; never restore optimizer, RNG, cursor or logs."""
    import torch
    path = Path(path)
    validate_initialization_provenance(provenance)
    if (provenance["source_training_identity_sha256"] != _json_sha256(metadata["identity"])
            or provenance["source_distribution_sha256"] != _json_sha256(metadata["distribution"])
            or provenance["source_completed_step"] != metadata["completed_step"]):
        raise ValueError("Initialization provenance differs from snapshot metadata")
    if (_file_sha256(path / "resume.json") != provenance["source_manifest_sha256"]
            or _file_sha256(path / "training_state.pt") != provenance["source_training_state_sha256"]):
        raise ValueError("Initialization snapshot changed after preflight")
    state = torch.load(path / "training_state.pt", map_location="cpu", weights_only=True)
    expected = _json_sha256({"training": metadata["identity"], "distribution": metadata["distribution"]})
    if (state.get("identity_sha256") != expected or type(state.get("completed_step")) is not int
            or state["completed_step"] != metadata["completed_step"]):
        raise ValueError("Initialization tensor identity/cursor mismatch")
    parameters = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    saved = state.get("trainable_parameters")
    if not isinstance(saved, dict) or set(saved) != set(parameters) or not parameters:
        raise ValueError("Initialization trainable parameter names differ")
    for name, parameter in parameters.items():
        value = saved[name]
        if (not isinstance(value, torch.Tensor) or value.shape != parameter.shape or value.dtype != parameter.dtype
                or not torch.isfinite(value).all()):
            raise ValueError("Initialization parameter shape/dtype/value differs: " + name)
    # Validate every tensor before copying any, so a rejected source cannot
    # leave a partially initialized model behind.
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(saved[name])
    return provenance


def save_distributed_checkpoint(model, optimizer, completed_step, output, identity, distribution, run_metadata, device):
    """Gather four RNG streams; rank zero atomically publishes immutable bytes."""
    import torch
    import torch.distributed as dist
    rngs = [None] * WORLD_SIZE
    dist.all_gather_object(rngs, capture_rng(device))

    def publish():
        if type(completed_step) is not int or not 0 < completed_step <= identity["arguments"]["steps"]:
            raise ValueError("Invalid snapshot optimizer step")
        directory = Path(output) / "training-checkpoints"
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"step-{completed_step:08d}"
        if target.exists():
            raise ValueError("DDP snapshot already exists: " + str(target))
        temporary = Path(tempfile.mkdtemp(prefix=".incomplete-", dir=directory))
        try:
            state = {"completed_step": completed_step,
                     "identity_sha256": _json_sha256({"training": identity, "distribution": distribution}),
                     "trainable_parameters": {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad},
                     "parameter_groups": parameter_groups(model, optimizer),
                     "optimizer": optimizer.state_dict(), "rng_by_rank": rngs}
            torch.save(state, temporary / "training_state.pt")
            for name in ("training.jsonl", *BASELINE_FILES):
                shutil.copyfile(Path(output) / name, temporary / name)
            info = {"schema_version": 2, "kind": "distributed_training_resume_only", "complete": True,
                    "inference_ready": False, "completed_step": completed_step, "identity": identity,
                    "distribution": distribution, "run_metadata": run_metadata,
                    "files_sha256": {name: _file_sha256(temporary / name) for name in SNAPSHOT_FILES}}
            (temporary / "resume.json").write_text(json.dumps(info, indent=2) + "\n")
            os.replace(temporary, target)
            return str(target)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    return on_rank_zero(publish)


def restore_distributed_state(model, optimizer, path, metadata, rank, device):
    import torch
    state = torch.load(Path(path) / "training_state.pt", map_location="cpu", weights_only=True)
    expected = _json_sha256({"training": metadata["identity"], "distribution": metadata["distribution"]})
    if state.get("identity_sha256") != expected or state.get("completed_step") != metadata["completed_step"]:
        raise ValueError("DDP tensor identity/cursor mismatch")
    parameters = {name: p for name, p in model.named_parameters() if p.requires_grad}
    saved = state["trainable_parameters"]
    if set(saved) != set(parameters) or state["parameter_groups"] != parameter_groups(model, optimizer):
        raise ValueError("DDP trainable parameter names or optimizer group order differ")
    for name, p in parameters.items():
        if saved[name].shape != p.shape or saved[name].dtype != p.dtype or not torch.isfinite(saved[name]).all():
            raise ValueError("DDP parameter shape/dtype/value differs: " + name)
    current_groups = optimizer.state_dict()["param_groups"]
    saved_groups = state["optimizer"]["param_groups"]
    if len(current_groups) != len(saved_groups):
        raise ValueError("Optimizer group count differs")
    saved_ids = []
    for current, previous, names in zip(current_groups, saved_groups, state["parameter_groups"]):
        if ({k: v for k, v in current.items() if k != "params"} !=
                {k: v for k, v in previous.items() if k != "params"}) or len(previous["params"]) != len(names):
            raise ValueError("Optimizer hyperparameters/group lengths differ")
        for identifier, name in zip(previous["params"], names):
            saved_ids.append(identifier)
            entry = state["optimizer"]["state"].get(identifier, {})
            if set(entry) != {"step", "exp_avg", "exp_avg_sq"} or entry["step"].item() != metadata["completed_step"]:
                raise ValueError("Missing/inconsistent AdamW optimizer state: " + name)
            for key in ("exp_avg", "exp_avg_sq"):
                value = entry[key]
                if value.shape != parameters[name].shape or value.dtype != parameters[name].dtype or not torch.isfinite(value).all():
                    raise ValueError("Optimizer moment shape/dtype/value differs: " + name)
    if len(saved_ids) != len(set(saved_ids)) or set(saved_ids) != set(state["optimizer"]["state"]):
        raise ValueError("Optimizer state parameter IDs differ")
    if len(state["rng_by_rank"]) != WORLD_SIZE or not 0 <= rank < WORLD_SIZE:
        raise ValueError("Exactly four saved rank RNG streams are required")
    with torch.no_grad():
        for name, p in parameters.items():
            p.copy_(saved[name])
    optimizer.load_state_dict(state["optimizer"])
    restore_rng(state["rng_by_rank"][rank], device)
    return metadata["completed_step"]


def distributed_step(ddp, optimizer, row, brier_weight):
    """One row per rank, DDP-averaged global gradient, then clipping and AdamW."""
    import torch
    import torch.distributed as dist
    optimizer.zero_grad(set_to_none=True)
    logits = ddp([row])[0].float()
    target = torch.tensor(row["target"], device=logits.device)
    loss = -(target * logits.log_softmax(-1)).sum() + brier_weight * ((logits.softmax(-1) - target) ** 2).sum()
    finite = torch.isfinite(loss).to(dtype=torch.int32)
    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    if not finite.item():
        raise FloatingPointError("Nonfinite loss on a rank; no optimizer step performed")
    # DDP averages four local gradients. Dividing by four here would be wrong.
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_([p for p in ddp.parameters() if p.requires_grad], OPTIMIZER_SETTINGS["gradient_clip_norm"])
    finite = torch.isfinite(norm).to(dtype=torch.int32)
    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    if not finite.item():
        raise FloatingPointError("Nonfinite synchronized gradient; no optimizer step performed")
    optimizer.step()
    mean_loss = loss.detach()
    dist.all_reduce(mean_loss)
    return mean_loss.item() / WORLD_SIZE, norm.item()


def final_evaluation(model, args, out, rows, baseline, meta):
    """Unwrapped rank-zero model; preserve this run's measured initialization."""
    from .metrics import evaluate_probabilities, fit_temperature, softmax
    model.save(out / "checkpoint")
    calibration = evaluate(model, rows["calibration"], out / "calibration.jsonl")
    temperature = fit_temperature([r["logits"] for r in calibration], [r["target"] for r in calibration])
    baseline_temperature = fit_temperature([r["logits"] for r in baseline["calibration"]], [r["target"] for r in baseline["calibration"]])
    import hashlib
    (out / "checkpoint/temperature.json").write_text(json.dumps({
        "temperature": temperature, "split": "calibration", "n": len(calibration),
        "ids_sha256": hashlib.sha256(json.dumps(meta["calibration_ids"]).encode()).hexdigest()}, indent=2) + "\n")
    trained = {s: evaluate(model, rows[s], out / f"trained_{s}.jsonl") for s in ("test", "ood")}
    metrics = {}
    for split in ("test", "ood"):
        for prefix, values, temp in (("baseline", baseline[split], 1), ("baseline_calibrated", baseline[split], baseline_temperature),
                                     ("trained", trained[split], 1), ("calibrated", trained[split], temperature)):
            name = prefix + "_" + split
            probs = [softmax([x / temp for x in row["logits"]]) for row in values]
            metric = evaluate_probabilities([r["target"] for r in values], probs)
            metric["mean_latency_seconds"] = sum(r["latency_seconds"] for r in values) / len(values)
            for field in ("kind", "source", "question_id"):
                groups = sorted({(r["source"] + "/" + r[field]) if field == "question_id" else r[field] for r in values})
                metric["by_question" if field == "question_id" else "by_" + field] = {}
                for key in groups:
                    pairs = [(r, p) for r, p in zip(values, probs) if ((r["source"] + "/" + r[field]) if field == "question_id" else r[field]) == key]
                    group_metric = evaluate_probabilities([r["target"] for r, _ in pairs], [p for _, p in pairs])
                    if field == "kind" and key == "score":
                        group_metric["ordinal_mae"] = sum(abs(sum(i * v for i, v in enumerate(p)) - sum(i * v for i, v in enumerate(r["target"]))) for r, p in pairs) / len(pairs)
                    if field == "source" and key.startswith("wikispeedia"):
                        group_metric.update(optimal_action_hit=sum(r["target"][max(range(len(p)), key=p.__getitem__)] > 0 for r, p in pairs) / len(pairs),
                                            optimal_set_probability_mass=sum(sum(x for x, y in zip(p, r["target"]) if y > 0) for r, p in pairs) / len(pairs))
                    metric["by_question" if field == "question_id" else "by_" + field][key] = group_metric
            metrics[name] = metric
            if prefix == "calibrated":
                with (out / f"{name}.jsonl").open("w") as handle:
                    for row, prob in zip(values, probs):
                        handle.write(json.dumps({**row, "probabilities": prob, "temperature": temp}) + "\n")
    return {"baseline_temperature": baseline_temperature, "temperature": temperature, "metrics": metrics}, trained["test"][0]["logits"]


def run(args):
    # Validate the source checkout before reading any argument, as jev/train.py does.
    source_commit = source_checkout_commit(__file__)
    initialize_from = getattr(args, "initialize_training_weights", None)
    initialize_package = getattr(args, "initialize_inference_weights", None)
    package_manifest = getattr(args, "initialization_manifest_sha256", None)
    if sum(bool(value) for value in (initialize_from, initialize_package, args.resume_training)) > 1:
        raise ValueError("Weights-only initialization and exact resume are mutually exclusive")
    if bool(initialize_package) != bool(package_manifest):
        raise ValueError("Inference-package initialization requires its pinned manifest SHA256")
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from importlib.metadata import version
    from .data import read_split_directory, validate_records
    from .model import DecisionModel
    rank, local_rank, size = (int(os.environ.get(k, -1)) for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
    if size != WORLD_SIZE or local_rank != rank or int(os.environ.get("LOCAL_WORLD_SIZE", -1)) != WORLD_SIZE:
        raise ValueError("Use one node with torchrun --nproc-per-node=4")
    if args.accumulation != WORLD_SIZE or args.steps < 1 or args.checkpoint_every < 1:
        raise ValueError("Global accumulation must be 4; steps and checkpoint frequency must be positive")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", args.revision):
        raise ValueError("A pinned 40-character upstream revision is required")
    allocation = None
    if args.backend == "nccl":
        policy = json.loads(Path(args.resource_policy).read_text())
        inventory = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True)
        allocation = validate_allocation(policy, socket.gethostname(), os.environ.get("CUDA_VISIBLE_DEVICES", ""), inventory)
        allocation["resource_policy_sha256"] = _file_sha256(args.resource_policy)
    device = torch.device("cuda", local_rank) if args.backend == "nccl" else torch.device("cpu")
    if device.type == "cuda":
        if torch.cuda.device_count() != WORLD_SIZE:
            raise ValueError("Exactly four CUDA devices must be visible")
        torch.cuda.set_device(device)
    dist.init_process_group(args.backend, timeout=timedelta(seconds=args.timeout_seconds))
    try:
        torch.random.default_generator.manual_seed(args.seed)
        random.seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(args.seed)
        validate_records(read_split_directory(args.data))
        rows = {s: read_rows(Path(args.data) / f"{s}.jsonl",
                            args.train_rows if s == "train" else args.calibration_rows if s == "calibration" else args.eval_rows,
                            args.seed, balanced=args.training_sampling == "source_kind_round_robin" if s == "train" else True)
                for s in ("train", "calibration", "test", "ood")}
        if not all(rows.values()):
            raise ValueError("Every selected split must be nonempty")
        hashes = {s: _file_sha256(Path(args.data) / f"{s}.jsonl") for s in ("train", "calibration", "validation", "test", "ood")}
        identity = training_identity(args, hashes, rows, {"torch": str(torch.__version__), "transformers": version("transformers"), "peft": version("peft")})
        initialization_metadata, initialization = (read_initialization_checkpoint(
            initialize_from, args.model, args.revision, args.lora_rank) if initialize_from else (None, None))
        if initialize_package:
            from .inference_initialization import read_inference_initialization_package
            initialization_metadata, initialization = read_inference_initialization_package(
                initialize_package, package_manifest, args.model, args.revision, args.lora_rank)
        resume_hint = json.loads((Path(args.resume_training) / "resume.json").read_text()) if args.resume_training else None
        identity = bind_initialization_identity(identity, initialization=initialization, resume_metadata=resume_hint)
        initialization = identity.get("initialization")
        if initialization and initialization["kind"] == "inference_package_weights_only":
            identity["implementation_sha256"]["inference_initialization.py"] = _file_sha256(
                Path(__file__).with_name("inference_initialization.py"))
        distribution = distribution_identity(args.backend)
        identities = [None] * WORLD_SIZE
        dist.all_gather_object(identities, _json_sha256({"training": identity, "distribution": distribution}))
        if len(set(identities)) != 1:
            raise ValueError("Ranks disagree on training identity")
        resume = read_distributed_checkpoint(args.resume_training, identity, distribution) if args.resume_training else None
        out = Path(args.output)
        def prepare_output():
            if resume:
                if out.resolve() == Path(args.resume_training).resolve() or (out / "summary.json").exists():
                    raise ValueError("Cannot overwrite an immutable snapshot or completed run")
                out.mkdir(parents=True, exist_ok=True)
                restore_training_artifacts(args.resume_training, out)
            else:
                out.mkdir(parents=True, exist_ok=False)
        on_rank_zero(prepare_output)
        meta = {**(resume["run_metadata"] if resume else {}), **vars(args), "commit": source_commit,
                "started_at": resume["run_metadata"]["started_at"] if resume else time.time(),
                "initialization": ("strict_same_run_ddp_resume" if resume else
                                   initialization["kind"] if initialization else "fresh_pinned_upstream"),
                "baseline_initialization": "warm_start_checkpoint" if initialization else "pretrained",
                "distribution": distribution, "identity_sha256": identities[0], "data_sha256": hashes,
                "allocation": allocation,
                "evaluation_ids": [r["id"] for r in rows["test"]], "ood_ids": [r["id"] for r in rows["ood"]],
                "calibration_ids": [r["id"] for r in rows["calibration"]],
                "training_rows_consumed": args.steps * WORLD_SIZE,
                "resume_step": resume["completed_step"] if resume else 0,
                "rank_training_seeds": [int(_json_sha256(["open-jev-ddp-rng-v1", args.seed, r])[:16], 16) % (2**63 - 1) for r in range(WORLD_SIZE)]}
        if initialization:
            meta["initialization_provenance"] = initialization
        if initialize_from:
            meta["initialization_source_snapshot"] = str(Path(initialize_from).resolve())
        if initialize_package:
            meta["initialization_source_package"] = str(Path(initialize_package).resolve())
        on_rank_zero(lambda: (out / "run.json").write_text(json.dumps(meta, indent=2) + "\n"))
        model = DecisionModel(args.model, args.revision, device=str(device), lora_rank=args.lora_rank, max_length=args.max_length)
        if initialize_from:
            initialize_training_weights(model, initialize_from, initialization_metadata, initialization)
        if initialize_package:
            from .inference_initialization import initialize_inference_weights
            initialize_inference_weights(model, initialize_package, initialization_metadata, initialization)
        meta["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        optimizer = torch.optim.AdamW([
            {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": args.lr},
            {"params": model.head.parameters(), "lr": args.head_lr}], weight_decay=OPTIMIZER_SETTINGS["weight_decay"],
            betas=tuple(OPTIMIZER_SETTINGS["betas"]), eps=OPTIMIZER_SETTINGS["eps"])
        ddp = DistributedDataParallel(model, device_ids=[local_rank] if device.type == "cuda" else None, broadcast_buffers=False)
        def original_baselines():
            if not resume:
                for split, name in zip(("test", "ood", "calibration"), BASELINE_FILES):
                    evaluate(model, rows[split], out / name)
                (out / "training.jsonl").write_text("")
            for split, name in zip(("test", "ood", "calibration"), BASELINE_FILES):
                values = [json.loads(line) for line in (out / name).read_text().splitlines() if line.strip()]
                validate_baseline_identity(values, rows[split], name)
                if any(len(value["logits"]) != len(value["target"]) or
                       any(not math.isfinite(x) for x in value["logits"]) for value in values):
                    raise ValueError("Original baseline contains invalid logits: " + name)
            hashes = {name: _file_sha256(out / name) for name in BASELINE_FILES}
            if resume and hashes != resume["run_metadata"].get("original_baseline_sha256"):
                raise ValueError("Restored baseline differs from the original fresh DDP run")
            return hashes
        meta["original_baseline_sha256"] = on_rank_zero(original_baselines)
        start_step = restore_distributed_state(model, optimizer, args.resume_training, resume, rank, device) if resume else 0
        if not resume:
            seed_training_rank(args.seed, rank, device)
        previous_elapsed = json.loads((out / "training.jsonl").read_text().splitlines()[-1])["elapsed_seconds"] if resume else 0
        on_rank_zero(lambda: (out / "run.json").write_text(json.dumps(meta, indent=2) + "\n"))
        model.train()
        started = time.perf_counter()
        for completed in range(start_step, args.steps):
            row = rows["train"][rank_row_index(completed, rank, len(rows["train"]))]
            loss, norm = distributed_step(ddp, optimizer, row, args.brier_weight)
            peak = torch.tensor(torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0, device=device)
            dist.all_reduce(peak, op=dist.ReduceOp.MAX)
            if rank == 0:
                record = {"step": completed + 1, "loss": loss, "gradient_norm": norm,
                          "elapsed_seconds": previous_elapsed + time.perf_counter() - started,
                          "peak_memory_gib": peak.item(), "world_size": WORLD_SIZE, "global_batch_size": WORLD_SIZE}
                with (out / "training.jsonl").open("a") as handle:
                    handle.write(json.dumps(record) + "\n")
                print(json.dumps(record), flush=True)
            if (completed + 1) % args.checkpoint_every == 0 or completed + 1 == args.steps:
                save_distributed_checkpoint(model, optimizer, completed + 1, out, identity, distribution, meta, device)
        del ddp, optimizer
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        result = None
        reference = None
        def evaluate_final():
            nonlocal result, reference
            baseline = {}
            for split, name in zip(("test", "ood", "calibration"), BASELINE_FILES):
                if _file_sha256(out / name) != meta["original_baseline_sha256"][name]:
                    raise ValueError("Original baseline changed during training")
                baseline[split] = [json.loads(line) for line in (out / name).read_text().splitlines() if line.strip()]
                validate_baseline_identity(baseline[split], rows[split], name)
            result, reference = final_evaluation(model, args, out, rows, baseline, meta)
        on_rank_zero(evaluate_final)
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        def reload_and_finish():
            restored = DecisionModel.load(out / "checkpoint", device=str(device))
            actual = evaluate(restored, rows["test"][:1], out / "reload_check.jsonl")[0]["logits"]
            error = checked_reload_error(reference, actual)
            summary = {"status": "complete", "model": args.model, "steps": args.steps,
                       "trained_rows_consumed": args.steps * WORLD_SIZE, "distribution": distribution,
                       "checkpoint_reload_max_error": error, **result,
                       "limitations": ["Fresh four-rank run, not continuation of a single-process pilot", "DDP reduction/RNG streams do not promise bitwise single-process equivalence", "Synthetic held-out metrics do not establish real-world task competence"]}
            if initialization:
                summary.update(initialization_provenance=initialization, baseline_initialization="warm_start_checkpoint")
                source = "released inference package" if initialization["kind"] == "inference_package_weights_only" else "prior DDP snapshot"
                summary["limitations"][0] = f"New four-rank training stage initialized from a {source}; optimizer/RNG/cursor reset; baseline measures that initialization on the new held-out data"
            (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        on_rank_zero(reload_and_finish)
    finally:
        dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "revision", "data", "output"):
        p.add_argument("--" + name, required=True)
    for name, default in (("steps", 27581), ("train-rows", 0), ("calibration-rows", 512), ("eval-rows", 512),
                          ("max-length", 4096), ("lora-rank", 8), ("accumulation", 4), ("seed", 20260920),
                          ("checkpoint-every", 500), ("timeout-seconds", 3600)):
        p.add_argument("--" + name, type=int, default=default)
    for name, default in (("lr", 5e-5), ("head-lr", 1e-4), ("brier-weight", 0.1)):
        p.add_argument("--" + name, type=float, default=default)
    p.add_argument("--training-sampling", choices=("shuffled", "source_kind_round_robin"), default="shuffled")
    p.add_argument("--backend", choices=("nccl", "gloo"), default="nccl")
    p.add_argument("--resource-policy", default=str(Path(__file__).resolve().parents[1] / "state/auto_research/resource_policy.json"))
    initialization = p.add_mutually_exclusive_group()
    initialization.add_argument("--resume-training", help="Only a matching schema-2 DDP optimizer-boundary snapshot is accepted")
    initialization.add_argument("--initialize-training-weights", help="Start a new run with only LoRA/head weights from a complete schema-2 snapshot; reset optimizer/RNG/cursor")
    initialization.add_argument("--initialize-inference-weights", help="Start a new run from a complete released inference package directory; reset optimizer/RNG/cursor and refit calibration")
    p.add_argument("--initialization-manifest-sha256", help="Required pinned manifest SHA256 for --initialize-inference-weights")
    run(p.parse_args())


if __name__ == "__main__":
    main()
