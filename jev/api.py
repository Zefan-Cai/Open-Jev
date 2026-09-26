"""Lightweight System One request compilation and typed response formatting."""

import json
import math
from collections.abc import Mapping, Sequence

from .metrics import choice_confidence, score_confidence


def _render(value) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _description(value, *, optional=False):
    if value is None and optional:
        return None
    if not isinstance(value, (str, dict, list)):
        raise ValueError("instructions and descriptions must be text, an object, or an array")
    # Check nested values too; JSON does not support NaN or arbitrary objects.
    json.dumps(value, allow_nan=False)
    return _render(value)


def compile_request(state, questions: Mapping) -> list[dict]:
    """Compile a shared state and typed questions into isolated model records.

    Records contain no target. `answer_keys` and `legend` are software-only
    metadata; `candidate_prompts` never passes them or question IDs to a model.
    Choice candidate names and descriptions are both visible. Score candidate
    positions are not visible; code maps positions back to numeric levels.
    """
    if not isinstance(state, (str, dict, list)):
        raise ValueError("state must be text, a JSON object, or an array")
    state_copy = json.loads(json.dumps(state, ensure_ascii=False, allow_nan=False))
    if not isinstance(questions, Mapping) or not questions:
        raise ValueError("questions must be a nonempty mapping")
    records = []
    for question_id, definition in questions.items():
        if not isinstance(question_id, str) or not isinstance(definition, Mapping):
            raise ValueError("question IDs must be strings and definitions must be mappings")
        kind = definition.get("type")
        if kind not in ("choice", "score", "noul"):
            raise ValueError("question type must be choice, score, or noul")
        question = _description(definition.get("instructions"))
        criteria = definition.get("criteria")
        record = {"id": question_id, "state": state_copy, "kind": kind, "question": question}
        if kind == "choice":
            if not isinstance(criteria, Mapping) or not 1 <= len(criteria) <= 255:
                raise ValueError("Choice requires between 1 and 255 candidates")
            if any(not isinstance(key, str) for key in criteria):
                raise ValueError("Choice candidate names must be strings")
            descriptions = [_description(value, optional=True) for value in criteria.values()]
            record["answer_keys"] = list(criteria)
            record["options"] = [
                name if description is None else f"{name}: {description}"
                for name, description in zip(criteria, descriptions)
            ]
        elif kind == "score":
            if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
                raise ValueError("Score requires an array of 2 to 10 descriptive levels")
            record["options"] = [_description(level) for level in criteria]
            record["answer_keys"] = [str(index) for index in range(len(criteria))]
            record["legend"] = dict(zip(record["answer_keys"], json.loads(json.dumps(criteria))))
        else:
            if criteria is not None:
                if not isinstance(criteria, Mapping) or not set(criteria).issubset({"true", "false"}):
                    raise ValueError("Noul criteria must contain only true and false descriptions")
                for key, label in (("true", "Yes"), ("false", "No")):
                    if key in criteria:
                        description = _description(criteria[key])
                        record["question"] += f"\n{label} means: {description}"
            record["options"] = ["no", "yes"]
            record["answer_keys"] = ["false", "true"]
        records.append(record)
    return records


def candidate_prompts(record: dict) -> list[str]:
    """Render independent candidates using only declared model input fields."""
    prefix = f"Context:\n{_render(record['state'])}\n\nQuestion: {_render(record['question'])}\n"
    if record["kind"] == "noul":
        return [prefix + "Is the answer to this question yes? Answer Yes or No."]
    # Candidate order, question IDs, adjacent score levels and targets are absent.
    return [prefix + f"Proposed answer: {_render(option)}\nIs this proposed answer correct? Answer Yes or No."
            for option in record["options"]]


def format_response(records: Sequence[dict], probabilities: Sequence[Sequence[float]]) -> dict:
    """Return typed answers; invalid model probabilities fail validation.

    Callers apply any calibration temperature before this function. Probabilities
    must already be normalized (within 1e-6 numerical tolerance). This function
    never generates or parses model-produced text.
    """
    if len(records) != len(probabilities) or not records:
        raise ValueError("records and probability rows must have equal nonzero length")
    answers = {}
    for record, values in zip(records, probabilities):
        question_id, kind = record["id"], record["kind"]
        if question_id in answers:
            raise ValueError("duplicate question ID in response records")
        keys = record["answer_keys"]
        if len(values) != len(keys):
            raise ValueError("probability count must match the declared answer space")
        probs = [float(value) for value in values]
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in probs):
            raise ValueError("probabilities must be finite and in [0, 1]")
        total = sum(probs)
        if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("probabilities must sum to one")
        probs = [value / total for value in probs]
        if kind == "noul":
            if keys != ["false", "true"]:
                raise ValueError("Noul probabilities must be ordered false, true")
            answer = {"type": kind, "noul": probs[1]}
        elif kind == "choice":
            selected = max(range(len(probs)), key=probs.__getitem__)
            answer = {"type": kind, "choice": keys[selected], "probabilities": dict(zip(keys, probs)),
                      "confidence": choice_confidence(probs)}
        elif kind == "score":
            answer = {"type": kind, "score": sum(index * value for index, value in enumerate(probs)),
                      "probabilities": dict(zip(keys, probs)), "confidence": score_confidence(probs),
                      "legend": record["legend"]}
        else:
            raise ValueError("unknown record kind")
        answers[question_id] = answer
    return {"answers": answers}
