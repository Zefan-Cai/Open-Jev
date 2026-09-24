"""Bounded candidate batching shared by HTTP and Python inference clients."""
import math
import time
import hashlib
import subprocess
from pathlib import Path
import json

from .api import compile_request, format_response
from .metrics import softmax


def candidate_batches(records, limit):
    """Yield (record index, partial record) groups with at most limit sequences.

    A large Choice is split before scoring and normalized only after its logits
    are reassembled. Normalizing each chunk would change the decision.
    """
    if not isinstance(limit, int) or limit < 1:
        raise ValueError("candidate batch size must be positive")
    batch, size = [], 0
    for index, record in enumerate(records):
        pieces = [record] if record["kind"] == "noul" else [
            {**record, "options": record["options"][start:start + limit]}
            for start in range(0, len(record["options"]), limit)
        ]
        for piece in pieces:
            count = 1 if piece["kind"] == "noul" else len(piece["options"])
            if batch and size + count > limit:
                yield batch
                batch, size = [], 0
            batch.append((index, piece))
            size += count
    if batch:
        yield batch


class Predictor:
    """scorer.score(records) returns raw logits and actual encoded token count."""

    def __init__(self, scorer, *, model_name, temperature=1.0, batch_size=32,
                 max_questions=4096, max_candidates=65536, method="checkpoint", provenance=None,
                 prefix_cache=False):
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if min(batch_size, max_questions, max_candidates) < 1:
            raise ValueError("inference limits must be positive")
        self.scorer, self.model_name, self.method = scorer, model_name, method
        self.temperature, self.batch_size = temperature, batch_size
        self.max_questions, self.max_candidates = max_questions, max_candidates
        self.provenance = dict(provenance or {})
        self.prefix_cache = prefix_cache

    def predict(self, request):
        if not isinstance(request, dict) or not {"state", "questions"} <= request.keys():
            raise ValueError("request requires state and questions")
        # jev-latest is an explicit wire-compatibility alias, not a TypeSafe model.
        if request.get("model") not in (None, "open-jev", "jev-latest", self.model_name):
            raise ValueError("requested model is not loaded; see /v1/models")
        if not isinstance(request["questions"], dict) or len(request["questions"]) > self.max_questions:
            raise ValueError("questions must be an object within the server question limit")
        records = compile_request(request["state"], request["questions"])
        count = sum(1 if r["kind"] == "noul" else len(r["options"]) for r in records)
        if count > self.max_candidates:
            raise ValueError("request exceeds the candidate limit")
        start, input_tokens = time.perf_counter(), 0
        cache_stats = {"enabled": False, "mode": "independent_candidates"}
        if self.prefix_cache:
            score_request = getattr(self.scorer, "score_request", None)
            if not callable(score_request):
                raise ValueError("Selected backend does not support request-local prefix caching")
            logits, cache_stats = score_request(records, batch_size=self.batch_size)
            if len(logits) != len(records):
                raise RuntimeError("backend returned the wrong number of rows")
            for record, row in zip(records, logits):
                expected = 2 if record["kind"] == "noul" else len(record["options"])
                if len(row) != expected or any(not math.isfinite(v) for v in row):
                    raise RuntimeError("backend returned invalid logits")
            input_tokens = cache_stats["logical_input_tokens"]
        else:
            logits = [[] for _ in records]
            for batch in candidate_batches(records, self.batch_size):
                rows, token_count = self.scorer.score([piece for _, piece in batch])
                if len(rows) != len(batch):
                    raise RuntimeError("backend returned the wrong number of rows")
                input_tokens += token_count
                for (index, piece), row in zip(batch, rows):
                    expected = 2 if piece["kind"] == "noul" else len(piece["options"])
                    if len(row) != expected or any(not math.isfinite(v) for v in row):
                        raise RuntimeError("backend returned invalid logits")
                    logits[index].extend(row)
        result = format_response(records, [softmax(row, self.temperature) for row in logits])
        result.update(model=self.model_name, usage={"input_tokens": input_tokens, "output_tokens": 0},
                      metadata={"method": self.method, "temperature": self.temperature,
                                "candidate_sequences": count,
                                "inference_seconds": time.perf_counter() - start,
                                "prefix_cache": cache_stats, **self.provenance})
        return result


class TorchScorer:
    def __init__(self, model):
        self.model = model

    def score(self, records):
        import torch
        with torch.inference_mode():
            rows = self.model(records)
        return [row.float().cpu().tolist() for row in rows], self.model.last_input_tokens

    def score_request(self, records, *, batch_size):
        rows, stats = self.model.score_cached(records, batch_size=batch_size)
        return [row.float().cpu().tolist() for row in rows], stats


def load_predictor(*, checkpoint=None, model_id=None, revision=None, device="cuda:0",
                   max_length=None, batch_size=32, prefix_cache=False):
    from .model import DecisionModel
    if bool(checkpoint) == bool(model_id):
        raise ValueError("choose exactly one checkpoint or base model")
    if checkpoint:
        path = Path(checkpoint)
        model = DecisionModel.load(path, device=device)
        temperature = json.loads((path / "temperature.json").read_text())["temperature"]
        method = "lora_decision_head"
        digest = hashlib.sha256()
        for file in sorted(path.rglob("*")):
            if file.is_file():
                digest.update(str(file.relative_to(path)).encode() + b"\0")
                with file.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
        provenance = {"checkpoint_sha256": digest.hexdigest()}
    else:
        if not revision:
            raise ValueError("base model requires a pinned revision")
        model = DecisionModel(model_id, revision, device=device, lora_rank=0,
                              max_length=max_length or 4096).eval()
        temperature, method = 1.0, "pretrained_yes_minus_no_no_training"
        provenance = {}
    if max_length is not None:
        if max_length < 1:
            raise ValueError("max length must be positive")
        model.max_length = max_length
    # The dtypes actually loaded, not the requested one: a quantized or
    # JEV_TORCH_DTYPE run is then visible in every response's metadata.
    provenance.update(base_revision=model.revision, max_length=model.max_length,
                      backbone_parameter_dtypes=sorted({str(p.dtype) for p in model.backbone.parameters()}))
    try:
        provenance["code_commit"] = subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        provenance["code_commit"] = None
    return Predictor(TorchScorer(model), model_name=model.model_id, temperature=temperature,
                     batch_size=batch_size, method=method, provenance=provenance, prefix_cache=prefix_cache)
