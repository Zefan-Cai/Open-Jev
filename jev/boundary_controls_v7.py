"""Original exact boundary controls; no old evaluation rows or model calls."""
from collections import Counter, defaultdict
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import random

from .data import SPLITS, _write_dataset, input_fingerprint, validate_records


VERSION = "boundary-controls-v7"
GROUP_COUNTS = {"train": 32, "calibration": 4, "validation": 4, "test": 8, "ood": 8}
FAMILIES = ("temporal_window", "exact_numeric", "joint_capacity", "latest_authority")
CONDITIONS = {
    "temporal_window": ("before", "at_start", "inside", "at_deadline", "after", "before_exception", "after_exception", "equivalent_outside"),
    "exact_numeric": ("balance_negative", "balance_zero", "balance_positive", "balance_cancellation", "compare_below", "compare_equal", "compare_above", "compare_negative"),
    "joint_capacity": ("equality", "latest_regrant", "role_absent", "scope_mismatch", "latest_revoke", "revoke_ignores_invalid_grant", "one_cent_above", "latest_reduced_capacity"),
    "latest_authority": ("equality", "ignore_invalid_high_revision", "one_cent_above", "latest_lower_limit", "fraud_with_active", "fraud_over_capacity", "latest_withdrawn", "missing_authority"),
}
OPTIONS = {
    "temporal_window": ["accept", "reject", "review"],
    "joint_capacity": ["execute", "request missing consent", "reject revoked consent", "request higher capacity"],
    "latest_authority": ["automatic processing", "capacity review", "fraud review", "verify policy"],
}
TEMPORAL_POLICY = (
    "An approved exception routes the case to review, including a request outside the window. "
    "Without an approved exception, accept exactly when the request instant is from delivery "
    "through delivery plus return_window_seconds, including both endpoints; otherwise reject. "
    "Use the stated fixed UTC offsets to compare absolute instants, rather than displayed clocks."
)
NUMERIC_POLICY = (
    "For task balance, add all credit amount_cents and subtract all debit amount_cents. "
    "Each amount is a nonnegative integer count of USD cents; 100 cents is one USD. "
    "The order of entries does not affect the balance. For task integer_compare, compare "
    "the signed integer left_cents to right_cents: below, equal, or above."
)
JOINT_POLICY = (
    "Each required role needs a verified signed event whose credential_scope exactly matches "
    "the requested resource, operation and currency. For each role use its greatest valid "
    "sequence, regardless of list order; unverified, wrong-role or wrong-scope events have no authority. "
    "Any latest valid revoke means reject revoked consent, even if another role is absent. "
    "Otherwise an absent role means request missing consent. With all latest events grant, "
    "request higher capacity if amount_cents exceeds either capacity_cents; equality means execute."
)
AUTHORITY_POLICY = (
    "Resolve authority before checking fraud or amount. A valid policy needs a verified signature "
    "from policy_authority and exact request department in both department and credential_scope. "
    "Use its greatest valid revision, ignoring array order and all invalid records. No valid policy "
    "or a latest withdrawn policy means verify policy; never fall back to an older active policy. "
    "With the latest policy active, confirmed_fraud means fraud review regardless of amount. "
    "Otherwise amount_cents at or below its automatic_limit_cents means automatic processing, "
    "and an amount above that limit means capacity review."
)
DISCLOSURE = (
    "Original CC0 synthetic boundary controls motivated by previously observed aggregate failures. "
    "The generator reads or copies no old Test/OOD input, state or label. All eight case and representation "
    "variants stay in one split. OOD changes controlled magnitudes, offsets and irrelevant-list "
    "lengths; it does not establish unseen-business or natural-request generalization."
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def integer(value, *, nonnegative=False):
    if type(value) is not int or (nonnegative and value < 0):
        raise ValueError("An exact integer is required")
    return value


def money(cents):
    cents = integer(cents)
    absolute = abs(cents)
    return f"USD {'-' if cents < 0 else ''}{absolute // 100}.{absolute % 100:02d}"


def instant(value):
    result = datetime.fromisoformat(value)
    if result.utcoffset() is None or result.microsecond:
        raise ValueError("A whole-second instant with an explicit fixed UTC offset is required")
    return result.astimezone(timezone.utc)


def balance(state):
    total = 0
    for entry in state["ledger"]:
        amount = integer(entry["amount_cents"], nonnegative=True)
        if entry["direction"] not in ("credit", "debit"):
            raise ValueError("Unknown ledger direction")
        total += amount if entry["direction"] == "credit" else -amount
    return total


def oracle(family, state):
    """Use only supplied facts; condition/variant metadata never selects gold."""
    if family == "temporal_window":
        start, request = instant(state["delivered_at"]), instant(state["request_received_at"])
        duration = integer(state["return_window_seconds"], nonnegative=True)
        if type(state["exception_approved"]) is not bool:
            raise ValueError("Exception approval must be boolean")
        if state["exception_approved"]:
            return "review"
        return "accept" if start <= request <= start + timedelta(seconds=duration) else "reject"
    if family == "exact_numeric":
        if state["task"] == "balance":
            return money(balance(state))
        if state["task"] != "integer_compare":
            raise ValueError("Unknown numeric task")
        left, right = integer(state["left_cents"]), integer(state["right_cents"])
        return "below" if left < right else "above" if left > right else "equal"
    if family == "joint_capacity":
        request = state["request"]
        scope = {key: request[key] for key in ("resource", "operation", "currency")}
        latest = {}
        for role in state["trusted_policy"]["required_roles"]:
            events = [e for e in state["signed_events"] if e["issuer_role"] == role
                      and e["verified_signature"] is True and e["credential_scope"] == scope]
            if events:
                sequences = [integer(e["sequence"], nonnegative=True) for e in events]
                if len(set(sequences)) != len(sequences):
                    raise ValueError("Conflicting valid sequence for one role")
                latest[role] = max(events, key=lambda e: e["sequence"])
        if any(e["status"] == "revoke" for e in latest.values()):
            return "reject revoked consent"
        if len(latest) != len(state["trusted_policy"]["required_roles"]):
            return "request missing consent"
        if any(e["status"] != "grant" for e in latest.values()):
            raise ValueError("Unknown valid consent status")
        amount = integer(request["amount_cents"], nonnegative=True)
        return "request higher capacity" if any(amount > integer(e["capacity_cents"], nonnegative=True) for e in latest.values()) else "execute"
    if family == "latest_authority":
        request = state["request"]
        policies = [p for p in state["signed_policy_registry"] if p["verified_signature"] is True
                    and p["issuer_role"] == state["policy_authority"] and p["department"] == request["department"]
                    and p["credential_scope"] == request["department"]]
        if not policies:
            return "verify policy"
        revisions = [integer(p["revision"], nonnegative=True) for p in policies]
        if len(set(revisions)) != len(revisions):
            raise ValueError("Conflicting valid policy revision")
        current = max(policies, key=lambda p: p["revision"])
        if current["status"] == "withdrawn":
            return "verify policy"
        if current["status"] != "active" or type(request["confirmed_fraud"]) is not bool:
            raise ValueError("Unknown policy status or fraud fact")
        if request["confirmed_fraud"]:
            return "fraud review"
        return "automatic processing" if integer(request["amount_cents"], nonnegative=True) <= integer(current["automatic_limit_cents"], nonnegative=True) else "capacity review"
    raise ValueError("Unknown boundary family")


def temporal_cases(rng, entity, ood, index):
    offsets = (-345, 525, 765) if ood else (-420, 0, 330)
    years = (2028, 2032, 2036, 2040) if ood else (2026, 2027, 2029, 2030)
    month, day = rng.choice(((2, 28), (2, 29), (12, 31))) if ood else (rng.randint(1, 12), rng.randint(20, 27))
    start = datetime(rng.choice(years), month, day, rng.randrange(24), rng.randrange(60), rng.randrange(60), tzinfo=timezone.utc)
    duration = rng.choice((7507, 86407, 2678437) if ood else (3601, 86403, 172801))
    delivery_offset, request_offset = rng.choice(offsets), rng.choice(offsets)
    inside = rng.randint(1, duration - 1)
    # Rotate structural sides separately from kind and the Noul truth schedule.
    side = (index // 2 + index // 4) % 2
    deltas = (-1, 0, inside, duration, duration + 1, -1, duration + 1, -1 if side == 0 else duration + 1)
    cases = []
    for case, delta in enumerate(deltas):
        offset = rng.choice([x for x in offsets if x != request_offset]) if case == 7 else request_offset
        cases.append({"case_ref": entity, "delivered_at": start.astimezone(timezone(timedelta(minutes=delivery_offset))).isoformat(),
                      "request_received_at": (start + timedelta(seconds=delta)).astimezone(timezone(timedelta(minutes=offset))).isoformat(),
                      "return_window_seconds": duration, "exception_approved": case in (5, 6), "policy": TEMPORAL_POLICY})
    return cases


def numeric_cases(rng, entity, ood, index):
    a = rng.randint(20000001, 90000000) if ood else rng.randint(1001, 90000)
    difference = rng.randint(101, 799)
    cancellation = rng.randint(1000001, 9000000) if ood else rng.randint(1, 999)
    cases = []
    for case, debit in enumerate((a + difference, a, a - difference, a - difference)):
        ledger = [{"entry_ref": entity + "/credit", "direction": "credit", "amount_cents": a},
                  {"entry_ref": entity + "/debit", "direction": "debit", "amount_cents": debit}]
        if case == 3:
            ledger += [{"entry_ref": entity + "/cancel-" + direction, "direction": direction, "amount_cents": cancellation} for direction in ("credit", "debit")]
        if ood:
            ledger += [{"entry_ref": entity + "/reserve-" + direction, "direction": direction, "amount_cents": cancellation + 1} for direction in ("credit", "debit")]
        rng.shuffle(ledger)
        cases.append({"case_ref": entity, "task": "balance", "currency": "USD", "ledger": ledger, "policy": NUMERIC_POLICY})
    side = (index // 2 + index // 4) % 2
    pairs = ((a - 1, a), (a, a), (a + 1, a), (-a, -a - 1) if side == 0 else (-a - 1, -a))
    cases += [{"case_ref": entity, "task": "integer_compare", "left_cents": left, "right_cents": right, "policy": NUMERIC_POLICY} for left, right in pairs]
    return cases


def joint_cases(rng, entity, ood, index):
    limit = rng.randint(2000001, 9000000) if ood else rng.randint(1001, 90000)
    roles = ["custody delegate", "transfer reviewer"] if ood else ["inventory guardian", "release approver"]
    request = {"resource": ("/cross-vault/" if ood else "/dispatch/") + entity + "/unit", "operation": "move" if ood else "release", "currency": "EUR" if ood else "USD", "amount_cents": limit}
    scope = {key: request[key] for key in ("resource", "operation", "currency")}
    sequence = rng.randint(21, 71)
    events = [{"issuer_role": role, "verified_signature": True, "credential_scope": copy.deepcopy(scope),
               "sequence": sequence, "status": "grant", "capacity_cents": limit + offset} for role, offset in zip(roles, (0, rng.randint(2, 101)))]
    base = {"request": request, "signed_events": events, "trusted_policy": {"required_roles": roles, "rules": JOINT_POLICY}}
    selected = roles[(index // 2 + index // 4) % 2]
    cases = []
    for case in range(8):
        state = copy.deepcopy(base)
        event = next(e for e in state["signed_events"] if e["issuer_role"] == selected)
        if case == 1:
            state["signed_events"] += [{**copy.deepcopy(event), "sequence": sequence + 1, "status": "revoke"}, {**copy.deepcopy(event), "sequence": sequence + 2, "status": "grant"}]
        elif case == 2:
            state["signed_events"].remove(event)
        elif case == 3:
            field = ("resource", "operation", "currency")[index % 3]
            event["credential_scope"][field] += "-different"
        elif case in (4, 5):
            state["signed_events"].append({**copy.deepcopy(event), "sequence": sequence + 1, "status": "revoke"})
            if case == 5:
                state["signed_events"].append({**copy.deepcopy(event), "sequence": sequence + 100, "verified_signature": False, "status": "grant", "capacity_cents": limit * 10})
        elif case == 6:
            state["request"]["amount_cents"] += 1
        elif case == 7:
            state["signed_events"].append({**copy.deepcopy(event), "sequence": sequence + 1, "capacity_cents": limit - 1})
        # Extra invalid records cannot repair a missing or revoked valid consent.
        for role in roles:
            state["signed_events"].append({"issuer_role": role, "verified_signature": False, "credential_scope": copy.deepcopy(scope), "sequence": sequence + 200, "status": "grant", "capacity_cents": limit * 10})
            if ood:
                state["signed_events"].append({"issuer_role": role, "verified_signature": True, "credential_scope": {**scope, "currency": "other currency"}, "sequence": sequence + 300, "status": "grant", "capacity_cents": limit * 20})
        rng.shuffle(state["signed_events"])
        cases.append(state)
    return cases


def authority_cases(rng, entity, ood, index):
    limit = rng.randint(2000001, 9000000) if ood else rng.randint(1001, 90000)
    revision = rng.randint(31, 91)
    department = ("inter-depot " if ood else "fulfilment ") + entity
    authority = "transfer controls" if ood else "settlement controls"
    current = {"department": department, "issuer_role": authority, "credential_scope": department,
               "verified_signature": True, "revision": revision, "status": "active", "automatic_limit_cents": limit}
    base = {"request": {"department": department, "amount_cents": limit, "confirmed_fraud": False}, "case_ref": entity,
            "policy_authority": authority, "signed_policy_registry": [{**current, "revision": revision - 1, "automatic_limit_cents": limit * 2}, current], "trust_contract": AUTHORITY_POLICY}
    cases = []
    for case in range(8):
        state = copy.deepcopy(base)
        if case == 1:
            state["signed_policy_registry"] += [{**current, "revision": revision + 100, "verified_signature": False, "automatic_limit_cents": 0}, {**current, "revision": revision + 200, "credential_scope": department + "-other", "automatic_limit_cents": 0}]
        elif case == 2:
            state["request"]["amount_cents"] += 1
        elif case == 3:
            state["signed_policy_registry"].append({**current, "revision": revision + 1, "automatic_limit_cents": limit - 1})
        elif case in (4, 5):
            state["request"]["confirmed_fraud"] = True
            if case == 5:
                state["request"]["amount_cents"] += 1
        elif case == 6:
            state["signed_policy_registry"].append({**current, "revision": revision + 1, "status": "withdrawn"})
        elif case == 7:
            state["request"]["confirmed_fraud"] = True
            state["signed_policy_registry"] = [{**current, "verified_signature": False}, {**current, "issuer_role": authority + "-unauthorized", "revision": revision + 1}]
        if ood:
            state["signed_policy_registry"] += [{**current, "department": department + f"-elsewhere-{n}", "credential_scope": department + f"-elsewhere-{n}", "revision": revision + 300 + n} for n in range(3)]
        rng.shuffle(state["signed_policy_registry"])
        cases.append(state)
    return cases


def records(seed=20261003):
    builders = dict(zip(FAMILIES, (temporal_cases, numeric_cases, joint_cases, authority_cases)))
    for split in SPLITS:
        for family in FAMILIES:
            cursors = Counter()
            for index in range(GROUP_COUNTS[split]):
                group = f"{VERSION}/{seed}/{split}/{family}/{index}"
                rng = random.Random(digest([seed, group]))
                entity = "boundary-" + digest([group, "entity"])[:16]
                cases = builders[family](rng, entity, split == "ood", index)
                order = list(range(8)); rng.shuffle(order)
                for variant, case in enumerate(order):
                    state = cases[case]
                    answer = oracle(family, state)
                    task = state.get("task", family)
                    if family == "exact_numeric":
                        if task == "balance":
                            rank = (index * 3 + case) % 5
                            total = balance(state)
                            choices = [money(total + offset) for offset in range(-rank, 5 - rank)]
                        else:
                            choices = ["below", "equal", "above"]
                    else:
                        choices = list(OPTIONS[family])
                    kind = "noul" if (index + case) % 2 else "choice"
                    proposed = None
                    if kind == "noul":
                        truth = (index // 2 + case // 2) % 2 == 0
                        proposed = answer if truth else rng.choice([choice for choice in choices if choice != answer])
                        question = f"Does the supplied rule establish '{proposed}' for this case?"
                        options = ["no", "yes"]
                        target = [float(not truth), float(truth)]
                    else:
                        question = "Apply the supplied exact rule to the recorded facts. Which stated outcome follows?"
                        choices.remove(answer); rng.shuffle(choices)
                        position = cursors[task] % (len(choices) + 1)
                        cursors[task] += 1
                        choices.insert(position, answer)
                        options, target = choices, [float(choice == answer) for choice in choices]
                    yield {"id": group + f"/v{variant}", "group_id": group, "split": split, "source": VERSION,
                           "state": state, "question": question, "kind": kind, "options": options, "target": target,
                           "metadata": {"family": "policy", "scenario_family": family, "condition": CONDITIONS[family][case],
                                        "entity_ids": [entity], "template_id": VERSION + "/" + family + ("/controlled-ood" if split == "ood" else "/id"),
                                        "proposed_outcome": proposed, "target_basis": "exact_fact_oracle",
                                        "provenance": {"type": "synthetic", "license": "CC0-1.0", "generator_version": VERSION,
                                                       "seed": seed, "group_index": index, "variant": variant,
                                                       "split_policy": "whole_eight_case_groups_with_controlled_ood", "upstream_rows_imported": 0}}}


def audit(rows):
    summary = validate_records(rows)
    groups, inputs = defaultdict(list), set()
    coverage = defaultdict(Counter)
    for row in rows:
        fingerprint = input_fingerprint(row)
        if fingerprint in inputs:
            raise ValueError("Duplicate input is forbidden, including within one split")
        inputs.add(fingerprint)
        family = row["metadata"]["scenario_family"]
        answer = oracle(family, row["state"])
        if row["kind"] == "noul":
            proposal = row["metadata"]["proposed_outcome"]
            if row["question"] != f"Does the supplied rule establish '{proposal}' for this case?":
                raise ValueError("Visible Noul proposal differs from metadata")
            answer = "yes" if answer == proposal else "no"
        if row["options"][row["target"].index(1.0)] != answer:
            raise ValueError("Fact oracle disagrees with target")
        groups[row["group_id"]].append(row)
        coverage[(row["split"], family, row["metadata"]["condition"])][row["kind"] + ("/" + answer if row["kind"] == "noul" else "")] += 1
    for values in groups.values():
        family = values[0]["metadata"]["scenario_family"]
        if len(values) != 8 or {v["metadata"]["provenance"]["variant"] for v in values} != set(range(8)) or {v["metadata"]["condition"] for v in values} != set(CONDITIONS[family]):
            raise ValueError("Every group must retain all eight distinct conditions and variants")
    for split in SPLITS:
        for family in FAMILIES:
            for condition in CONDITIONS[family]:
                if not all(coverage[(split, family, condition)][key] > 0 for key in ("choice", "noul/no", "noul/yes")):
                    raise ValueError("Condition must cover Choice and both Noul truths in every split")
    return {**summary, "oracle_checked": len(rows), "complete_eight_case_groups": len(groups),
            "condition_kind_truth_coverage": {"/".join(key): dict(value) for key, value in sorted(coverage.items())},
            "upstream_rows_imported": 0, "scope": DISCLOSURE}


def build(output, seed=20261003):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Refusing to overwrite a frozen v7 candidate")
    rows = list(records(seed))
    verified = audit(rows)
    manifest = _write_dataset(rows, output, {"type": "synthetic", "status": "candidate_not_trained", "generator_version": VERSION,
                                            "seed": seed, "groups_per_family_split": GROUP_COUNTS, "variants_per_group": 8,
                                            "license": "CC0-1.0", "scope": DISCLOSURE})
    manifest.update(generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), oracle_audit=verified,
                    group_ids_by_split={split: sorted({r["group_id"] for r in rows if r["split"] == split}) for split in SPLITS})
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
