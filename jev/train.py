"""Bounded LoRA pilot with immutable held-out rows and separate calibration."""
import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import tempfile
import subprocess
import time
from pathlib import Path


def source_checkout_commit(source_file):
    """Read the source checkout's HEAD independently of the caller's directory."""
    root = Path(source_file).resolve().parent.parent
    message = (f"Training requires an Open-Jev source checkout; cannot resolve its commit at {root}. "
               "Install with pip install -e '.[train]' from that checkout.")
    if not (root / ".git").exists():
        raise RuntimeError(message)
    env = os.environ.copy()
    for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR"):
        env.pop(name, None)
    try:
        top = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            text=True, stderr=subprocess.DEVNULL, env=env).strip()
        if Path(top).resolve() != root:
            raise RuntimeError(message)
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "--verify", "HEAD"],
            text=True, stderr=subprocess.DEVNULL, env=env).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(message) from error


def read_rows(path, limit, seed, balanced=False):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    expected_split = Path(path).stem
    if any(row["split"] != expected_split for row in rows):
        raise ValueError(f"Rows in {path} must have split={expected_split}")
    random.Random(seed).shuffle(rows)
    if balanced:
        buckets = {}
        for row in rows:
            buckets.setdefault((row["source"], row["kind"]), []).append(row)
        rows = []
        while any(buckets.values()):
            for bucket in buckets.values():
                if bucket:
                    rows.append(bucket.pop())
    return rows[:limit] if limit else rows


def evaluate(model, rows, path):
    import torch
    model.eval()
    results = []
    with torch.inference_mode(), open(path, "w") as handle:
        for row in rows:
            torch.cuda.synchronize()
            start = time.perf_counter()
            logits = model([row])[0].float()
            torch.cuda.synchronize()
            result = {"id": row["id"], "kind": row["kind"], "source": row["source"],
                      "group_id": row["group_id"],
                      "question_id": row["metadata"].get("question_id", row["kind"]),
                      "target_basis": row["metadata"].get("target_basis", "hard_label"),
                      "target": row["target"], "logits": logits.cpu().tolist(),
                      "probabilities": logits.softmax(-1).cpu().tolist(),
                      "latency_seconds": time.perf_counter() - start}
            handle.write(json.dumps(result) + "\n")
            handle.flush()
            results.append(result)
    return results



IDENTITY_ARGUMENTS = ("model", "revision", "steps", "train_rows", "calibration_rows", "eval_rows",
                      "max_length", "lora_rank", "accumulation", "lr", "head_lr", "brier_weight", "seed")
OPTIMIZER_SETTINGS = {"name": "AdamW", "weight_decay": 0.01, "betas": [0.9, 0.999], "eps": 1e-8,
                      "gradient_clip_norm": 1.0}
BASELINE_FILES = ("baseline_test.jsonl", "baseline_ood.jsonl", "baseline_calibration.jsonl")
SNAPSHOT_FILES = ("training_state.pt", "training.jsonl", *BASELINE_FILES)


def _json_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _file_sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def training_identity(args, data_sha256, rows_by_split, runtime, initial_checkpoint=None):
    """Paths/checkpoint frequency are transport details; all training facts bind."""
    return {"arguments": {**{key: getattr(args, key) for key in IDENTITY_ARGUMENTS},
                           "training_sampling": getattr(args, "training_sampling", "source_kind_round_robin"),
                           **({"defer_heldout": True} if getattr(args, "defer_heldout", False) else {})},
            "optimizer": OPTIMIZER_SETTINGS, "data_sha256": data_sha256,
            "selected_ids": {split: [row["id"] for row in rows] for split, rows in rows_by_split.items()},
            "runtime": runtime, **({"initial_checkpoint": initial_checkpoint} if initial_checkpoint else {}),
            "implementation_sha256": {name: _file_sha256(Path(__file__).with_name(name))
                                       for name in ("train.py", "model.py", "api.py", "data.py", "metrics.py")}}


def validate_resume_identity(saved, current):
    if saved != current:
        changed = sorted(key for key in saved.keys() | current.keys() if saved.get(key) != current.get(key))
        raise ValueError("Training resume identity differs: " + ", ".join(changed))


def validate_baseline_identity(values, rows, name):
    if len(values) != len(rows) or any(
            any(result.get(key) != row[key] for key in ("id", "group_id", "source", "kind", "target"))
            for result, row in zip(values, rows)):
        raise ValueError(f"Saved {name} baseline does not match selected held-out rows")


def save_training_checkpoint(model, optimizer, completed_step, output, identity, run_metadata):
    """Publish only complete optimizer-boundary snapshots, without calibration."""
    import torch
    output = Path(output)
    if type(completed_step) is not int or not 0 < completed_step <= identity["arguments"]["steps"]:
        raise ValueError("checkpoint step is outside the planned training run")
    directory = output / "training-checkpoints"
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"step-{completed_step:08d}"
    if destination.exists():
        raise ValueError(f"Training snapshot already exists: {destination}")
    temporary = Path(tempfile.mkdtemp(prefix=".incomplete-", dir=directory))
    try:
        parameters = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters() if parameter.requires_grad}
        state = {"completed_step": completed_step, "identity_sha256": _json_sha256(identity),
                 "trainable_parameters": parameters, "optimizer": optimizer.state_dict(),
                 "rng": {"python": random.getstate(), "torch_cpu": torch.get_rng_state(),
                         "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}}
        torch.save(state, temporary / "training_state.pt")
        for name in ("training.jsonl", *BASELINE_FILES):
            shutil.copyfile(output / name, temporary / name)
        metadata = {"schema_version": 1, "kind": "training_resume_only", "inference_ready": False,
                    "complete": True, "completed_step": completed_step, "identity": identity,
                    "run_metadata": run_metadata,
                    "files_sha256": {name: _file_sha256(temporary / name) for name in SNAPSHOT_FILES}}
        (temporary / "resume.json").write_text(json.dumps(metadata, indent=2) + "\n")
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def read_training_checkpoint(path, current_identity):
    """Validate all bytes/identity before allocating tensors or restoring state."""
    path = Path(path)
    info = json.loads((path / "resume.json").read_text())
    if (info.get("schema_version") != 1 or info.get("kind") != "training_resume_only"
            or info.get("complete") is not True or info.get("inference_ready") is not False):
        raise ValueError("Not a complete training-resume checkpoint")
    validate_resume_identity(info.get("identity", {}), current_identity)
    step = info.get("completed_step")
    if type(step) is not int or not 0 < step <= current_identity["arguments"]["steps"]:
        raise ValueError("invalid completed checkpoint step")
    hashes = info.get("files_sha256", {})
    if set(hashes) != set(SNAPSHOT_FILES):
        raise ValueError("training checkpoint file manifest is incomplete")
    for name, expected in hashes.items():
        if _file_sha256(path / name) != expected:
            raise ValueError(f"training checkpoint file checksum mismatch: {name}")
    log = [json.loads(line) for line in (path / "training.jsonl").read_text().splitlines() if line.strip()]
    if [row.get("step") for row in log] != list(range(1, step + 1)):
        raise ValueError("training checkpoint log cursor does not match completed step")
    return info


def restore_training_state(model, optimizer, path, metadata):
    """Fresh trainable LoRA construction is required; inference adapters freeze."""
    import torch
    state = torch.load(Path(path) / "training_state.pt", map_location="cpu", weights_only=True)
    if state.get("identity_sha256") != _json_sha256(metadata["identity"]) or state.get("completed_step") != metadata["completed_step"]:
        raise ValueError("training tensor payload identity/cursor mismatch")
    parameters = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    saved = state["trainable_parameters"]
    if set(parameters) != set(saved):
        raise ValueError("trainable parameter names differ from checkpoint")
    for name, parameter in parameters.items():
        if saved[name].shape != parameter.shape or saved[name].dtype != parameter.dtype:
            raise ValueError(f"trainable parameter shape/dtype mismatch: {name}")
    cuda_rng = state["rng"]["torch_cuda"]
    visible_cuda = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if len(cuda_rng) != visible_cuda:
        raise ValueError("visible CUDA device count differs from RNG checkpoint")
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(saved[name])
    optimizer.load_state_dict(state["optimizer"])
    random.setstate(state["rng"]["python"])
    torch.set_rng_state(state["rng"]["torch_cpu"])
    if cuda_rng:
        torch.cuda.set_rng_state_all(cuda_rng)
    return metadata["completed_step"]


def restore_training_artifacts(checkpoint, output):
    """Keep interrupted post-snapshot logs before restarting at the saved cursor."""
    checkpoint, output = Path(checkpoint), Path(output)
    archive = output / "resume-history" / str(time.time_ns())
    for name in ("run.json", "training.jsonl"):
        if (output / name).exists():
            archive.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(output / name, archive / name)
    for name in BASELINE_FILES:
        if (output / name).exists() and _file_sha256(output / name) != _file_sha256(checkpoint / name):
            raise ValueError(f"Refusing to replace a different baseline file: {name}")
    for name in ("training.jsonl", *BASELINE_FILES):
        shutil.copyfile(checkpoint / name, output / name)


def initial_checkpoint_identity(args):
    path = getattr(args, "initial_checkpoint", None)
    if not path:
        return None
    if getattr(args, "resume_training", None):
        raise ValueError("initial-checkpoint and resume-training cannot be combined")
    if getattr(args, "checkpoint_every", 0) > 0:
        raise ValueError("initial-checkpoint adaptation does not support resumable snapshots; checkpoint-every must be zero")
    path = Path(path)
    source, output = path.resolve(), Path(args.output).resolve()
    if source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError("Training output and initial checkpoint must not overlap or contain one another")
    weights = [p.relative_to(path).as_posix() for p in (path / "adapter").glob("adapter_model.*")
               if p.suffix in (".bin", ".safetensors")]
    names = ["model.json", "head.pt", "temperature.json", "adapter/adapter_config.json", *weights]
    if not weights or any(not (path / name).is_file() for name in names):
        raise ValueError("Initial checkpoint requires config, LoRA weights, head and temperature")
    hashes = {name: _file_sha256(path / name) for name in sorted(names)}
    config = json.loads((path / "model.json").read_text())
    adapter = json.loads((path / "adapter/adapter_config.json").read_text())
    if (config.get("method") != "independent_candidate_lora_nll_brier" or adapter.get("peft_type") != "LORA"
            or type(adapter.get("r")) is not int or adapter["r"] != config["lora_rank"] or adapter.get("rank_pattern")
            or adapter.get("base_model_name_or_path") not in (None, "", config["model_id"])
            or adapter.get("revision") not in (None, config["revision"])):
        raise ValueError("Initial adapter method/type/rank/base/revision differs from model configuration")
    temperature = float(json.loads((path / "temperature.json").read_text())["temperature"])
    if (config["model_id"], config["revision"], config["lora_rank"], config["max_length"]) != (
            args.model, args.revision, args.lora_rank, args.max_length) or not re.fullmatch(r"[0-9a-fA-F]{40}", config["revision"]):
        raise ValueError("Initial checkpoint model/revision/rank/max_length differs from requested profile")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Initial checkpoint temperature must be finite and positive")
    return {"files_sha256": hashes, "sha256": _json_sha256(hashes), "config": config,
            "adapter_config": adapter, "temperature": temperature}


def initialize_model(args, model_class, initial_checkpoint):
    if initial_checkpoint is None:
        return model_class(args.model, args.revision, lora_rank=args.lora_rank, max_length=args.max_length)
    if initial_checkpoint_identity(args) != initial_checkpoint:
        raise ValueError("Initial checkpoint changed after preflight")
    model = model_class.load(args.initial_checkpoint)
    model.requires_grad_(False)
    for name, parameter in model.named_parameters():
        if name.startswith("head.") or ".lora_A." in name or ".lora_B." in name:
            parameter.requires_grad_(True)
    model.backbone.gradient_checkpointing_enable()
    model.backbone.enable_input_require_grads()
    return model


def run(args):
    source_commit = source_checkout_commit(__file__)
    defer_heldout = getattr(args, "defer_heldout", False)
    if defer_heldout and (getattr(args, "checkpoint_every", 0) or getattr(args, "resume_training", None)):
        raise ValueError("defer-heldout requires checkpoint-every=0 and no resume-training")
    initial_checkpoint = initial_checkpoint_identity(args)
    import torch
    from .model import DecisionModel
    from importlib.metadata import version
    from .metrics import evaluate_probabilities, fit_temperature, softmax
    from .data import read_split_directory, validate_records
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    args.checkpoint_every = getattr(args, "checkpoint_every", 0)
    args.resume_training = getattr(args, "resume_training", None)
    args.training_sampling = getattr(args, "training_sampling", "source_kind_round_robin")
    if args.checkpoint_every < 0 or args.steps < 1 or args.accumulation < 1:
        raise ValueError("checkpoint frequency must be nonnegative; steps/accumulation must be positive")
    if (args.checkpoint_every or args.resume_training) and not re.fullmatch(r"[0-9a-fA-F]{40}", args.revision):
        raise ValueError("Resumable training requires a pinned 40-character model revision")
    if args.resume_training and out.resolve() == Path(args.resume_training).resolve():
        raise ValueError("Training output must not be the immutable snapshot directory")
    if (out / "summary.json").exists():
        raise ValueError("Output already contains a completed run summary; choose a fresh output directory")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if defer_heldout:
        train = read_rows(Path(args.data) / "train.jsonl", 0, args.seed,
                          balanced=args.training_sampling == "source_kind_round_robin")
        calibration = read_rows(Path(args.data) / "calibration.jsonl", 0, args.seed, balanced=True)
        validate_records([*train, *calibration])
        train = train[:args.train_rows] if args.train_rows else train
        calibration = calibration[:args.calibration_rows] if args.calibration_rows else calibration
        test, ood = [], []
        hashed_splits = ("train", "calibration")
    else:
        validate_records(read_split_directory(args.data))
        train = read_rows(Path(args.data) / "train.jsonl", args.train_rows, args.seed,
                          balanced=args.training_sampling == "source_kind_round_robin")
        calibration = read_rows(Path(args.data) / "calibration.jsonl", args.calibration_rows, args.seed, balanced=True)
        test = read_rows(Path(args.data) / "test.jsonl", args.eval_rows, args.seed, balanced=True)
        ood = read_rows(Path(args.data) / "ood.jsonl", args.eval_rows, args.seed, balanced=True)
        hashed_splits = ("train", "calibration", "validation", "test", "ood")
    if not all([train, calibration] if defer_heldout else [train, calibration, test, ood]):
        raise ValueError("Every required split must be nonempty")
    data_sha256 = {split: _file_sha256(Path(args.data) / f"{split}.jsonl")
                   for split in hashed_splits if (Path(args.data) / f"{split}.jsonl").exists()}
    runtime = {"torch": str(torch.__version__), "transformers": version("transformers"), "peft": version("peft")}
    rows_by_split = {"train": train, "calibration": calibration}
    if not defer_heldout:
        rows_by_split.update(test=test, ood=ood)
    identity = training_identity(args, data_sha256, rows_by_split, runtime, initial_checkpoint)
    resume = read_training_checkpoint(args.resume_training, identity) if args.resume_training else None
    if resume:
        restore_training_artifacts(args.resume_training, out)
    meta = dict(resume["run_metadata"]) if resume else {}
    meta.update(vars(args))
    meta.update({"commit": source_commit,
                 "initial_checkpoint_identity": initial_checkpoint,
                 "baseline_initialization": "inference_checkpoint" if initial_checkpoint else "base_model",
                 "torch": str(torch.__version__), "gpu": torch.cuda.get_device_name(),
                 "data_sha256": data_sha256, "run_identity_sha256": _json_sha256(identity),
                 "evaluation_ids": [r["id"] for r in test], "ood_ids": [r["id"] for r in ood],
                 "calibration_ids": [r["id"] for r in calibration],
                 "started_at": resume["run_metadata"]["started_at"] if resume else time.time(), "phase": "load"})
    phase_metrics = {}
    if defer_heldout:
        meta.update(heldout_evaluation={"status": "deferred", "splits": ["validation", "test", "ood"],
                                       "model_calls": 0}, data_validation_splits=["train", "calibration"],
                    checkpoint_selection="fixed_final_step")

    def begin_phase():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        return time.perf_counter()

    def end_phase(name, started):
        torch.cuda.synchronize()
        phase_metrics[name] = {"elapsed_seconds": time.perf_counter() - started,
                               "peak_allocated_tensor_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
                               "peak_reserved_allocator_memory_gib": torch.cuda.max_memory_reserved() / 2**30}

    if resume:
        meta.update(resumed_from=str(args.resume_training), resume_step=resume["completed_step"],
                    resumed_at=time.time(), output=str(out), checkpoint_every=args.checkpoint_every)
    (out / "run.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({"event": "load", "model": args.model, "revision": args.revision}), flush=True)
    phase_start = begin_phase() if defer_heldout else None
    model = initialize_model(args, DecisionModel, initial_checkpoint)
    if defer_heldout:
        end_phase("model_loading", phase_start)
    from transformers import __version__ as transformers_version
    from transformers.models.qwen3_5.modeling_qwen3_5 import is_fast_path_available
    meta.update(transformers=transformers_version, fast_path_available=bool(is_fast_path_available),
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                evaluation_sampling="source_kind_round_robin", training_sampling=args.training_sampling,
                training_rows_consumed=args.steps * args.accumulation)
    (out / "run.json").write_text(json.dumps(meta, indent=2) + "\n")
    phase_start = begin_phase() if defer_heldout else None
    model.eval()
    with torch.inference_mode():
        model([train[0] if defer_heldout else test[0]])
    if defer_heldout:
        end_phase("warmup", phase_start)
    # A resumed model must retain the ORIGINAL base scores, never call its
    # already-trained weights a baseline. Snapshot hashes bind these artifacts.
    if defer_heldout:
        baseline_temperature = None
        print(json.dumps({"event": "heldout_evaluation_deferred", "splits": ["validation", "test", "ood"]}), flush=True)
    elif resume:
        baseline, baseline_ood, baseline_cal = [[json.loads(line) for line in (out / name).read_text().splitlines() if line.strip()]
                                               for name in BASELINE_FILES]
        for name, values, rows in (("test", baseline, test), ("ood", baseline_ood, ood), ("calibration", baseline_cal, calibration)):
            validate_baseline_identity(values, rows, name)
    else:
        # Measure the actual initialized weights before any optimizer update.
        baseline = evaluate(model, test, out / "baseline_test.jsonl")
        baseline_ood = evaluate(model, ood, out / "baseline_ood.jsonl")
        baseline_cal = evaluate(model, calibration, out / "baseline_calibration.jsonl")
    if not defer_heldout:
        baseline_temperature = fit_temperature([r["logits"] for r in baseline_cal], [r["target"] for r in baseline_cal])
        print(json.dumps({"event": "baseline_complete", "rows": len(baseline)}), flush=True)
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": args.lr},
        {"params": model.head.parameters(), "lr": args.head_lr},
    ], weight_decay=OPTIMIZER_SETTINGS["weight_decay"], betas=tuple(OPTIMIZER_SETTINGS["betas"]), eps=OPTIMIZER_SETTINGS["eps"])
    start_step = restore_training_state(model, optimizer, args.resume_training, resume) if resume else 0
    previous_elapsed = json.loads((out / "training.jsonl").read_text().splitlines()[-1]).get("elapsed_seconds", 0.0) if resume else 0.0
    if resume:
        print(json.dumps({"event": "training_resumed", "completed_step": start_step, "checkpoint": str(args.resume_training)}), flush=True)
    start = begin_phase() if defer_heldout else time.perf_counter()
    model.train()
    with open(out / "training.jsonl", "a" if resume else "w") as log:
        for step in range(start_step, args.steps):
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            for micro in range(args.accumulation):
                row = train[(step * args.accumulation + micro) % len(train)]
                logits = model([row])[0].float()
                target = torch.tensor(row["target"], device=logits.device)
                loss = -(target * logits.log_softmax(-1)).sum()
                loss = loss + args.brier_weight * ((logits.softmax(-1) - target) ** 2).sum()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite loss; run invalid")
                (loss / args.accumulation).backward()
                loss_sum += loss.item() / args.accumulation
                del loss, logits, target
            norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            if not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite gradient; run invalid")
            optimizer.step()
            record = {"step": step + 1, "loss": loss_sum, "gradient_norm": norm.item(),
                      "elapsed_seconds": previous_elapsed + time.perf_counter() - start,
                      "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30}
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(json.dumps(record), flush=True)
            if args.checkpoint_every and ((step + 1) % args.checkpoint_every == 0 or step + 1 == args.steps):
                checkpoint_path = save_training_checkpoint(model, optimizer, step + 1, out, identity, meta)
                print(json.dumps({"event": "training_checkpoint", "step": step + 1, "path": str(checkpoint_path), "inference_ready": False}), flush=True)
    if defer_heldout:
        end_phase("optimizer", start)
    phase_start = begin_phase() if defer_heldout else None
    model.save(out / "checkpoint")
    if defer_heldout:
        end_phase("checkpoint_save", phase_start)
    phase_start = begin_phase() if defer_heldout else None
    cal = evaluate(model, calibration, out / "calibration.jsonl")
    temperature = fit_temperature([r["logits"] for r in cal], [r["target"] for r in cal])
    (out / "checkpoint/temperature.json").write_text(json.dumps({
        "temperature": temperature, "split": "calibration", "n": len(cal),
        "ids_sha256": hashlib.sha256(json.dumps(meta["calibration_ids"]).encode()).hexdigest(),
    }, indent=2) + "\n")
    if defer_heldout:
        end_phase("calibration_and_temperature", phase_start)
        reference = cal[0]["logits"]
        del optimizer, model
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        phase_start = begin_phase()
        model = DecisionModel.load(out / "checkpoint")
        check = evaluate(model, calibration[:1], out / "reload_check.jsonl")[0]["logits"]
        if not reference or len(reference) != len(check):
            raise ValueError("Checkpoint reload logits must have equal nonempty lengths")
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in [*reference, *check]):
            raise ValueError("Checkpoint reload logits must be finite numbers")
        reload_error = max(abs(float(a) - float(b)) for a, b in zip(reference, check))
        if reload_error > 0.05:
            raise ValueError(f"Checkpoint reload mismatch: {reload_error}")
        end_phase("checkpoint_reload", phase_start)
        meta.update(phase="complete", phase_metrics=phase_metrics)
        (out / "run.json").write_text(json.dumps(meta, indent=2) + "\n")
        summary = {"status": "complete", "model": args.model, "steps": args.steps,
                   "trained_rows_consumed": args.steps * args.accumulation,
                   "baseline_temperature": None, "temperature": temperature, "metrics": {},
                   "heldout_evaluation": meta["heldout_evaluation"], "checkpoint_selection": "fixed_final_step",
                   "checkpoint_reload_split": "calibration", "checkpoint_reload_max_error": reload_error,
                   "phase_metrics": phase_metrics, "elapsed_seconds": time.time() - meta["started_at"],
                   "limitations": ["Validation/Test/OOD were not loaded or evaluated; comparison remains deferred",
                                   "Calibration temperature is not a held-out capability or safety result",
                                   "Phase peaks measure CUDA allocated tensors and reserved allocator memory, including resident weights, not total device or minimum VRAM",
                                   "Phase timings include their file writes; optimizer timing includes training logs"]}
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary), flush=True)
        return
    trained = evaluate(model, test, out / "trained_test.jsonl")
    trained_ood = evaluate(model, ood, out / "trained_ood.jsonl")
    metrics = {}
    for name, values, temp in [("baseline_test", baseline, 1), ("baseline_ood", baseline_ood, 1),
                               ("baseline_calibrated_test", baseline, baseline_temperature),
                               ("baseline_calibrated_ood", baseline_ood, baseline_temperature),
                               ("trained_test", trained, 1), ("trained_ood", trained_ood, 1),
                               ("calibrated_test", trained, temperature), ("calibrated_ood", trained_ood, temperature)]:
        probs = [softmax([x / temp for x in r["logits"]]) for r in values]
        metrics[name] = evaluate_probabilities([r["target"] for r in values], probs)
        metrics[name]["mean_latency_seconds"] = sum(r["latency_seconds"] for r in values) / len(values)
        metrics[name]["by_kind"] = {}
        for kind in sorted({r["kind"] for r in values}):
            pairs = [(r, p) for r, p in zip(values, probs) if r["kind"] == kind]
            metrics[name]["by_kind"][kind] = evaluate_probabilities(
                [r["target"] for r, p in pairs], [p for r, p in pairs])
            if kind == "score":
                metrics[name]["by_kind"][kind]["ordinal_mae"] = sum(
                    abs(sum(i * x for i, x in enumerate(p)) -
                        sum(i * x for i, x in enumerate(r["target"]))) for r, p in pairs) / len(pairs)
        metrics[name]["by_source"] = {}
        for source in sorted({r["source"] for r in values}):
            pairs = [(r, p) for r, p in zip(values, probs) if r["source"] == source]
            metrics[name]["by_source"][source] = evaluate_probabilities(
                [r["target"] for r, p in pairs], [p for r, p in pairs])
            if source.startswith("wikispeedia"):
                metrics[name]["by_source"][source].update(
                    optimal_action_hit=sum(r["target"][max(range(len(p)), key=p.__getitem__)] > 0 for r, p in pairs) / len(pairs),
                    optimal_set_probability_mass=sum(sum(x for x, y in zip(p, r["target"]) if y > 0) for r, p in pairs) / len(pairs))
        metrics[name]["by_question"] = {}
        for source, question in sorted({(r["source"], r["question_id"]) for r in values}):
            pairs = [(r, p) for r, p in zip(values, probs) if (r["source"], r["question_id"]) == (source, question)]
            metrics[name]["by_question"][source + "/" + question] = evaluate_probabilities(
                [r["target"] for r, p in pairs], [p for r, p in pairs])
        if name.startswith("calibrated"):
            with open(out / f"{name}.jsonl", "w") as handle:
                for row, prob in zip(values, probs):
                    handle.write(json.dumps({**row, "probabilities": prob, "temperature": temp}) + "\n")
    # Verify persisted weights in a fresh model after releasing training memory.
    reference = trained[0]["logits"]
    del optimizer, model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    model = DecisionModel.load(out / "checkpoint")
    check = evaluate(model, test[:1], out / "reload_check.jsonl")[0]["logits"]
    reload_error = max(abs(a - b) for a, b in zip(reference, check))
    if reload_error > 0.05:
        raise ValueError(f"Checkpoint reload mismatch: {reload_error}")
    summary = {"status": "complete", "model": args.model, "steps": args.steps,
               "trained_rows_consumed": args.steps * args.accumulation,
               "baseline_temperature": baseline_temperature,
               "temperature": temperature, "metrics": metrics,
               "checkpoint_reload_max_error": reload_error,
               "elapsed_seconds": time.time() - meta["started_at"],
               "limitations": ["One small pilot seed; not a Jev performance reproduction",
                               "No autoregressive latency comparator; scorer latency includes tokenization",
                               "Synthetic OOD measures template/entity transfer, not general world knowledge"]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--train-rows", type=int, default=4000)
    p.add_argument("--calibration-rows", type=int, default=128)
    p.add_argument("--eval-rows", type=int, default=128)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--accumulation", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--head-lr", type=float, default=1e-4)
    p.add_argument("--brier-weight", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--training-sampling", choices=("source_kind_round_robin", "shuffled"),
                   default="source_kind_round_robin", help="Use shuffled for a full pass without a source-sorted tail")
    p.add_argument("--checkpoint-every", type=int, default=0, help="Save resumable state every N completed optimizer steps; zero disables")
    p.add_argument("--resume-training", help="Resume from a training-checkpoints/step-* directory with matching run identity")
    p.add_argument("--initial-checkpoint", help="Adapt matching inference LoRA/head weights; excludes resume and requires checkpoint-every=0")
    p.add_argument("--defer-heldout", action="store_true", help="Load only Train/Calibration and defer Validation/Test/OOD; requires no resume or snapshots")
    run(p.parse_args())


if __name__ == "__main__":
    main()
