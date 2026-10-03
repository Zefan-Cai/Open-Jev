"""Independent visible-input and whole-parent isolation checks; no model calls.

Group names and ancestry declarations identify records but never prove isolation.
Numeric/time abstraction is checked on whole-parent menus, not individual atoms.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
from itertools import combinations
import json
from pathlib import Path
import re
import unicodedata

from jev.api import candidate_prompts
from jev.data import SPLITS


PROSE_KEYS = {"policy", "rules", "trust_contract", "instructions", "description"}
PRESENTATION_KEYS = {"case_ref", "case_id", "event_id", "record_id", "request_id",
                     "entry_ref", "item_ref", "note", "notes", "layout", "display_name"}
IDENTITY_KEYS = {"account", "account_id", "resource", "operation", "currency",
                 "department", "issuer_role", "required_roles", "policy_authority",
                 "subject", "object", "credential_scope", "role", "authority"}
UNORDERED_KEYS = {"ledger", "entries", "signed_events", "signed_policy_registry",
                  "account_events", "events", "facts", "required_roles"}
ISO_TIME = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:Z|[+-]\d{2}:\d{2})")
V8_FIELDS = {
    "temporal_window": {"case_ref", "recorded_delivery_at", "correction_authority", "delivery_corrections",
                        "request_received_at", "return_window_seconds", "exception_approved", "policy"},
    "exact_numeric": {"case_ref", "task", "currency", "policy", "ledger", "left_ledger", "right_ledger"},
    "joint_capacity": {"case_ref", "request", "trusted_policy", "signed_events"},
    "latest_authority": {"case_ref", "request", "required_topics", "policy_authorities", "signed_policy_registry", "trust_contract"},
    "account_status": {"case_ref", "required_accounts", "events", "rule"},
    "department_route": {"case_ref", "request_lines", "policy_authority", "signed_policies", "policy"},
    "refund_priority": {"case_ref", "purchase_document_verified", "items", "trusted_policy"},
}
SPLIT_WIDTHS = {"train": [1, 2], "calibration": [3], "validation": [4], "test": [5], "ood": [6]}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def latest(records, field):
    values = [record[field] for record in records]
    require(all(type(v) is int and v >= 0 for v in values) and len(values) == len(set(values)),
            "Actual valid chronology must be unique nonnegative integers")
    return max(records, key=lambda r: r[field]) if records else None


def v8_family(state):
    if "delivery_corrections" in state: return "temporal_window"
    if "left_ledger" in state or state.get("task") in ("balance", "integer_compare"): return "exact_numeric"
    if "signed_events" in state and "trusted_policy" in state: return "joint_capacity"
    if "required_topics" in state: return "latest_authority"
    if "required_accounts" in state: return "account_status"
    if "request_lines" in state: return "department_route"
    if "purchase_document_verified" in state and "items" in state: return "refund_priority"
    raise ValueError("Unsupported v8 causal schema")


def causal_projection(state, *, abstract=False):
    """Project the seven reviewed contracts, retaining essential fan-in only.

    Dynamic mapping keys are resolved through actual membership before aliases
    disappear. Invalid/obsolete records and arbitrary decoration cannot supply
    width. No authored width, group ID, lineage hash or condition is consulted.
    """
    family = v8_family(state)
    require(set(state) - PRESENTATION_KEYS <= V8_FIELDS[family], "Unreviewed state field cannot prove causal independence")
    number = (lambda value: "<number>") if abstract else (lambda value: value)
    core, width = {"family": family}, 0

    def operands(entries):
        require(isinstance(entries, list) and bool(entries), "Nonempty exact operand ledger required")
        normalized = []
        for entry in entries:
            require(set(entry) <= {"entry_ref", "direction", "amount_cents"}, "Unreviewed operand field")
            amount, direction = entry["amount_cents"], entry["direction"]
            require(type(amount) is int and amount > 0 and direction in ("credit", "debit"),
                    "Essential operands must be strictly positive with explicit direction")
            normalized.append({"direction": direction, "amount_cents": amount})
        credit = [e["amount_cents"] for e in normalized if e["direction"] == "credit"]
        debit = [e["amount_cents"] for e in normalized if e["direction"] == "debit"]
        require(len(credit) == len(debit), "Balanced credit/debit operand counts required")
        if len(credit) > 1:
            require(not set(credit) & set(debit), "Exact cancellation padding cannot prove operand topology")
        return len(credit), sorted([(e["direction"], number(e["amount_cents"])) for e in normalized], key=canonical), sum(credit) - sum(debit)

    if family == "temporal_window":
        corrections = [record for record in state["delivery_corrections"]
            if record["verified_signature"] is True and record["issuer_role"] == state["correction_authority"]
            and record["case_ref"] == state["case_ref"]]
        require(len({r["sequence"] for r in corrections}) == len(corrections), "Correction sequences must be unique")
        require(all(type(r["seconds"]) is int and r["seconds"] > 0 and r["direction"] in ("delay", "advance")
                    for r in corrections), "Essential corrections must be signed nonzero operands")
        signed = [r["seconds"] * (1 if r["direction"] == "delay" else -1) for r in corrections]
        require(not ({r["seconds"] for r in corrections if r["direction"] == "delay"} &
                     {r["seconds"] for r in corrections if r["direction"] == "advance"}),
                "Neutral correction pair cannot prove composition")
        width = len(corrections)
        anchor = instant(state["recorded_delivery_at"]) + sum(signed)
        elapsed, duration = instant(state["request_received_at"]) - anchor, state["return_window_seconds"]
        require(type(duration) is int and duration > 0 and type(state["exception_approved"]) is bool,
                "Typed interval and exception required")
        position = "before" if elapsed < 0 else "start" if elapsed == 0 else "after" if elapsed > duration else "deadline" if elapsed == duration else "inside"
        core.update(corrections=sorted([(r["direction"], number(r["seconds"])) for r in corrections], key=canonical),
            position=position, exception=state["exception_approved"], anchor=number(anchor),
            request_instant=number(instant(state["request_received_at"])), duration=number(duration))
    elif family == "exact_numeric":
        require(state["currency"] == "USD", "Reviewed numeric contract uses USD")
        core["task"] = state["task"]
        if state["task"] == "balance":
            width, terms, total = operands(state["ledger"])
            core.update(terms=terms, total_sign=relation(total, 0))
        else:
            width, left, left_sum = operands(state["left_ledger"])
            right_width, right, right_sum = operands(state["right_ledger"])
            require(right_width == width, "Both compared operand ledgers must have same essential width")
            core.update(left=left, right=right, left_sign=relation(left_sum, 0), right_sign=relation(right_sum, 0),
                        comparison=relation(left_sum, right_sum))
    elif family == "joint_capacity":
        request, roles = state["request"], state["trusted_policy"]["required_roles"]
        require(len(roles) == len(set(roles)) and len(roles) >= 2, "Distinct required roles are essential")
        scope = {key: request[key] for key in ("resource", "operation", "currency")}
        members = []
        for role in roles:
            record = latest([r for r in state["signed_events"] if r["issuer_role"] == role
                and r["verified_signature"] is True and r["credential_scope"] == scope], "sequence")
            members.append(None if record is None else {"status": record["status"],
                "sequence": number(record["sequence"]), "capacity": number(record["capacity_cents"]),
                "amount_relation": relation(request["amount_cents"], record["capacity_cents"])})
        width = len(roles) - 1
        core.update(members=sorted(members, key=canonical), amount=number(request["amount_cents"]))
    elif family == "latest_authority":
        request, topics, authorities = state["request"], state["required_topics"], state["policy_authorities"]
        require(len(topics) == len(set(topics)) and len(topics) >= 2 and set(authorities) == set(topics),
                "Required topic membership and authority mapping must align exactly")
        members = []
        for topic in topics:
            record = latest([r for r in state["signed_policy_registry"] if r["topic"] == topic
                and r["issuer_role"] == authorities[topic] and r["verified_signature"] is True
                and r["department"] == request["department"]
                and r["credential_scope"] == {"department": request["department"], "topic": topic}], "revision")
            members.append(None if record is None else {"status": record["status"],
                "revision": number(record["revision"]), "limit": number(record["automatic_limit_cents"]),
                "amount_relation": relation(request["amount_cents"], record["automatic_limit_cents"])})
        width = len(topics) - 1
        core.update(members=sorted(members, key=canonical), amount=number(request["amount_cents"]), fraud=request["confirmed_fraud"])
    elif family == "account_status":
        accounts = state["required_accounts"]
        require(len(accounts) == len(set(accounts)), "Distinct required account membership is essential")
        members = []
        for account in accounts:
            record = latest([r for r in state["events"] if r["account"] == account], "sequence")
            members.append(None if record is None else {"status": record["status"], "sequence": number(record["sequence"])})
        width = len(accounts)
        core["members"] = sorted(members, key=canonical)
    elif family == "department_route":
        lines, authority = state["request_lines"], state["policy_authority"]
        require(len(lines) == len({line["department"] for line in lines}), "Request departments must be distinct essential members")
        members = []
        for line in lines:
            record = latest([r for r in state["signed_policies"] if r["verified_signature"] is True
                and r["issuer_role"] == authority and r["department"] == line["department"]
                and r["credential_scope"] == line["department"]], "revision")
            members.append({"amount": number(line["amount_cents"]), "fraud": line["confirmed_fraud"],
                "policy": None if record is None else {"status": record["status"], "revision": number(record["revision"]),
                "limit": number(record["approval_limit_cents"]), "amount_relation": relation(line["amount_cents"], record["approval_limit_cents"])}})
        width = len(lines)
        core["members"] = sorted(members, key=canonical)
    else:
        items = state["items"]
        require(len(items) == len({item["item_ref"] for item in items}), "Refund items must be distinct essential members")
        require(all(type(item["amount_cents"]) is int and item["amount_cents"] > 0 for item in items),
                "Zero-value refund padding cannot prove topology")
        width = len(items)
        members = [{"amount": number(item["amount_cents"]), "defect": item["defect_confirmed"],
                    "seal": item["seal_intact"], "recall": item["verified_recall"]} for item in items]
        total = sum(item["amount_cents"] for item in items)
        limit = state["trusted_policy"]["automatic_limit_cents"]
        core.update(members=sorted(members, key=canonical), document=state["purchase_document_verified"],
                    limit=number(limit), amount_relation=relation(total, limit))
    require(1 <= width <= 6, "Essential composition width outside reviewed range")
    return core, width


def neutral_profiles(terms, *, include_whole=False):
    """Exact zero-sum subset shapes, without mistaking a whole zero ledger for padding."""
    require(len(terms) <= 16, "Neutral-subgraph screen supports at most sixteen operands")
    found = set()
    for size in range(1, len(terms) + int(include_whole)):
        for subset in combinations(terms, size):
            if sum(subset) == 0:
                found.add((sum(value > 0 for value in subset), sum(value < 0 for value in subset)))
    return found


def parent_neutral_padding(group):
    """Block a neutral operand shape present throughout a parent's case menu.

    This is a conservative exact-arithmetic screen, not a complete symbolic
    isomorphism solver. Whole zero-balance conditions remain valid; only proper
    subsets recurring in every case are suspicious. Corrections may not be an
    entirely neutral decoration either.
    """
    family = v8_family(group[0]["state"])
    profiles = []
    seen = set()
    for row in group:
        state = row["state"]
        if family == "exact_numeric":
            ledgers = [state["ledger"]] if state["task"] == "balance" else [state["left_ledger"], state["right_ledger"]]
            for entries in ledgers:
                terms = sorted(e["amount_cents"] * (1 if e["direction"] == "credit" else -1) for e in entries)
                if tuple(terms) not in seen:
                    seen.add(tuple(terms))
                    profiles.append(neutral_profiles(terms))
        elif family == "temporal_window":
            terms = sorted(e["seconds"] * (1 if e["direction"] == "delay" else -1)
                for e in state["delivery_corrections"] if e["verified_signature"] is True
                and e["issuer_role"] == state["correction_authority"] and e["case_ref"] == state["case_ref"])
            if tuple(terms) not in seen:
                seen.add(tuple(terms))
                profiles.append(neutral_profiles(terms, include_whole=True))
    return sorted(set.intersection(*profiles)) if profiles else []


def core_outcome(core):
    family = core["family"]
    if family == "temporal_window":
        return "review" if core["exception"] else "reject" if core["position"] in ("before", "after") else "accept"
    if family == "exact_numeric":
        if core["task"] == "integer_compare": return core["comparison"]
        total = sum(amount * (1 if direction == "credit" else -1) for direction, amount in core["terms"])
        return f"USD {'-' if total < 0 else ''}{abs(total)//100}.{abs(total)%100:02d}"
    members = core.get("members", [])
    if family == "joint_capacity":
        if any(m is not None and m["status"] == "revoke" for m in members): return "reject revoked consent"
        if any(m is None for m in members): return "request missing consent"
        return "request higher capacity" if any(m["amount_relation"] == "above" for m in members) else "execute"
    if family == "latest_authority":
        if any(m is None or m["status"] == "withdrawn" for m in members): return "verify policy"
        if core["fraud"]: return "fraud review"
        return "capacity review" if any(m["amount_relation"] == "above" for m in members) else "automatic processing"
    if family == "account_status":
        if any(m is not None and m["status"] == "closed" for m in members): return "closed"
        if any(m is not None and m["status"] == "paused" for m in members): return "paused"
        return "unknown" if any(m is None for m in members) else "active"
    if family == "department_route":
        if any(m["fraud"] for m in members): return "security review"
        return "manager review" if any(m["policy"] is None or m["policy"]["status"] == "withdrawn"
            or m["policy"]["amount_relation"] == "above" for m in members) else "approve"
    if any(m["recall"] for m in members): return "recall remediation"
    if not core["document"] or any(not m["defect"] and not m["seal"] for m in members): return "reject request"
    return "review reimbursement" if core["amount_relation"] == "above" else "automatic reimbursement"


def essential_witness(group):
    """Require an observed valid case in which each claimed component can matter."""
    cores = [causal_projection(r["state"])[0] for r in group]
    family = cores[0]["family"]
    if family == "temporal_window":
        # Every signed nonzero correction affects one inclusive endpoint case.
        return {c["position"] for c in cores if not c["exception"]} >= {"start", "deadline"}
    if family == "exact_numeric":
        return any(c["task"] == "balance" for c in cores) and any(c.get("comparison") == "equal" for c in cores)
    expected = {"joint_capacity": "execute", "latest_authority": "automatic processing",
                "account_status": "active", "department_route": "approve",
                "refund_priority": "automatic reimbursement"}[family]
    # In these reviewed conjunction/aggregate contracts, each selected member of
    # a complete baseline can be removed/invalidated to change this outcome.
    return any(core_outcome(c) == expected and bool(c["members"]) for c in cores)


def causal_scope_cells(row):
    """Derive causal mismatched scope, never use declared condition as the source."""
    state = row["state"]
    family = v8_family(state)
    cells = set()
    if family == "joint_capacity":
        roles, request = state["trusted_policy"]["required_roles"], state["request"]
        scope = {key: request[key] for key in ("resource", "operation", "currency")}
        for index, role in enumerate(roles):
            if any(e["issuer_role"] == role and e["verified_signature"] is True and e["credential_scope"] == scope
                   for e in state["signed_events"]): continue
            for e in state["signed_events"]:
                if e["issuer_role"] != role or e["verified_signature"] is not True: continue
                mismatched = [field for field in scope if e["credential_scope"].get(field) != scope[field]]
                if len(mismatched) == 1 and set(e["credential_scope"]) == set(scope):
                    cells.add((index, mismatched[0]))
    elif family == "latest_authority":
        topics, request = state["required_topics"], state["request"]
        for index, topic in enumerate(topics):
            authority = state["policy_authorities"][topic]
            expected = {"department": request["department"], "topic": topic}
            if any(e["topic"] == topic and e["department"] == request["department"] and e["credential_scope"] == expected
                   and e["verified_signature"] is True and e["issuer_role"] == authority for e in state["signed_policy_registry"]): continue
            for e in state["signed_policy_registry"]:
                if e["verified_signature"] is not True or e["issuer_role"] != authority: continue
                bad_department = e["department"] != expected["department"] or e["credential_scope"].get("department") != expected["department"]
                bad_topic = e["topic"] != topic or e["credential_scope"].get("topic") != topic
                if bad_department != bad_topic:
                    cells.add((index, "department" if bad_department else "topic"))
    return cells


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def norm(value):
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def instant(value):
    parsed = datetime.fromisoformat(value)
    if parsed.utcoffset() is None or parsed.microsecond:
        raise ValueError("Whole-second offset timestamp required")
    return int(parsed.timestamp())


def relation(left, right):
    return "below" if left < right else "above" if left > right else "equal"


def reduce_ledger(entries):
    """Remove exact zero/cancellation padding, retaining the remaining sum graph."""
    positive, negative = defaultdict(list), defaultdict(list)
    for entry in entries:
        if set(entry) != {"direction", "amount_cents"}:
            return entries  # A different ledger schema needs independent review.
        amount = entry["amount_cents"]
        if type(amount) is not int or amount < 0 or entry["direction"] not in ("credit", "debit"):
            raise ValueError("Invalid exact ledger record")
        if amount:
            (positive if entry["direction"] == "credit" else negative)[amount].append(entry)
    result = []
    for amount in sorted(set(positive) | set(negative)):
        remove = min(len(positive[amount]), len(negative[amount]))
        result.extend(positive[amount][remove:])
        result.extend(negative[amount][remove:])
    # A complete zero-balance expression is a legitimate numerical condition,
    # not an empty padding graph. Only a proper neutral tail may be removed.
    return result if result else entries


def numeric_axes(state):
    """Only visible exact relations, never authored conditions or target labels."""
    axes = {}
    if {"left_cents", "right_cents"} <= state.keys():
        axes["integer_comparison"] = relation(state["left_cents"], state["right_cents"])
    if "ledger" in state and isinstance(state["ledger"], list):
        ledger = reduce_ledger(state["ledger"])
        if all(set(e) == {"direction", "amount_cents"} for e in ledger):
            total = sum(e["amount_cents"] * (1 if e["direction"] == "credit" else -1) for e in ledger)
            axes["ledger_total_sign"] = relation(total, 0)
    if {"delivered_at", "request_received_at", "return_window_seconds"} <= state.keys():
        elapsed = instant(state["request_received_at"]) - instant(state["delivered_at"])
        duration = state["return_window_seconds"]
        axes["interval_position"] = ("before" if elapsed < 0 else "start" if elapsed == 0
            else "after" if elapsed > duration else "deadline" if elapsed == duration else "inside")
    request = state.get("request", {})
    if isinstance(request, dict) and type(request.get("amount_cents")) is int:
        amount = request["amount_cents"]
        for key, quantity in (("signed_events", "capacity_cents"),
                              ("signed_policy_registry", "automatic_limit_cents")):
            if key in state:
                axes[key + "_amount_relations"] = sorted(relation(amount, e[quantity])
                    for e in state[key] if type(e.get(quantity)) is int)
    return axes


def normalize_state(state, *, abstract=False):
    """Normalize visible references and explicitly unordered facts.

    Abstract mode removes magnitudes/origins but retains visible relations and
    chronology ranks. A schema-specific causal review remains required: arbitrary
    extra state keys cannot be accepted as evidence of causal split separation.
    """
    if not isinstance(state, dict):
        raise ValueError("The independent structural audit requires typed object states")
    identities = defaultdict(list)

    def shape(value, key="", path=(), collect=True):
        if isinstance(value, dict):
            result = {}
            for name, item in sorted(value.items()):
                if name in PRESENTATION_KEYS or name in PROSE_KEYS and isinstance(item, str):
                    continue
                result[name] = shape(item, name, path + (name,), collect)
            return result
        if isinstance(value, list):
            if key == "ledger":
                cleaned = [{name: item for name, item in entry.items() if name not in PRESENTATION_KEYS}
                           if isinstance(entry, dict) else entry for entry in value]
                items = reduce_ledger(cleaned)
            else:
                items = value
            prepared = []
            order_values = sorted({item[field] for item in items if isinstance(item, dict)
                for field in ("sequence", "revision") if type(item.get(field)) is int})
            for item in items:
                if abstract and isinstance(item, dict):
                    item = dict(item)
                    for field in ("sequence", "revision"):
                        if type(item.get(field)) is int:
                            item[field] = {"chronology_rank": order_values.index(item[field])}
                prepared.append(shape(item, key, path + ("[]",), collect))
            return sorted(prepared, key=canonical) if key in UNORDERED_KEYS else prepared
        if isinstance(value, str):
            if ISO_TIME.fullmatch(value):
                return "<instant>" if abstract else instant(value)
            if key in IDENTITY_KEYS or key.endswith("_id"):
                if collect:
                    identities[norm(value)].append(path)
                return "<identity>"
            return norm(value)
        if type(value) in (int, float):
            return "<number>" if abstract and key != "chronology_rank" else value
        if value is None or type(value) is bool:
            return value
        raise ValueError("Non-JSON state value")

    result = shape(state)
    # Equality classes are based on actual value occurrences. Renaming IDs cannot
    # change them; distinct symmetric identities remain distinct graph vertices.
    references = sorted([sorted([list(p) for p in paths]) for paths in identities.values()], key=canonical)
    return {"facts": result, "reference_partition": references,
            "numeric_relations": numeric_axes(state) if abstract else None}


def row_fingerprints(row, *, strict_v8=False):
    exact = digest(sorted(norm(prompt) for prompt in candidate_prompts(row)))
    proposition = None
    if row["kind"] == "noul":
        quoted = re.findall(r"['\"‘“]([^'\"’”]+)['\"’”]", row["question"])
        if len(quoted) != 1:
            raise ValueError("Noul semantic normalization requires one visible quoted proposition")
        proposition = norm(quoted[0])
    if strict_v8:
        visible, _ = causal_projection(row["state"])
        abstract, _ = causal_projection(row["state"], abstract=True)
    else:
        visible = normalize_state(row["state"])
        abstract = normalize_state(row["state"], abstract=True)
        # Supported older causal contracts can share projected atoms. Unsupported
        # legacy schemas retain the mechanical fingerprint; no content is exposed.
        is_legacy_numeric = "left_cents" in row["state"] or ("ledger" in row["state"]
            and not all("entry_ref" in e for e in row["state"]["ledger"]))
        if not is_legacy_numeric:
            try:
                visible, _ = causal_projection(row["state"])
                abstract, _ = causal_projection(row["state"], abstract=True)
            except (KeyError, ValueError):
                pass
    semantic = digest({"state": visible, "kind": row["kind"],
        "proposition": proposition,
        "options": sorted(norm(option) for option in row["options"])})
    scaffold = digest(abstract)
    body = digest({"state": normalize_state(row["state"]), "kind": row["kind"],
                   "proposition": proposition, "options": sorted(norm(x) for x in row["options"])})
    return {"compiled_visible": exact, "wording_order_identity": semantic,
            "normalized_visible_body": body, "magnitude_time_scaffold": scaffold}


def collision_index(rows, *, strict_v8=False):
    groups, indices = defaultdict(set), defaultdict(lambda: defaultdict(set))
    groups_by_split = {}
    for row in rows:
        group, split = row["group_id"], row["split"]
        if group in groups_by_split and groups_by_split[group] != split:
            raise ValueError("Actual parent group crosses splits")
        groups_by_split[group] = split
        fps = row_fingerprints(row, strict_v8=strict_v8)
        groups[group].add(fps["magnitude_time_scaffold"])
        for mode, fp in fps.items():
            indices[mode][fp].add((split, group))
    whole = defaultdict(set)
    for group, atoms in groups.items():
        whole[digest(sorted(atoms))].add((groups_by_split[group], group))
    indices["whole_parent_scaffold"] = whole
    return indices


def audit_rows(rows, *, observed_rows=(), strict_v8=False, observed_fingerprints=None):
    """Return opaque collisions; never export an old reserved row body or label."""
    indices = collision_index(rows, strict_v8=strict_v8)
    observed = observed_fingerprints if observed_fingerprints is not None else collision_index(observed_rows)
    collisions, shared = [], {}
    for mode, index in indices.items():
        cross = [{"fingerprint": fp, "new_groups": sorted(group for _, group in refs),
                  "splits": sorted({split for split, _ in refs})}
                 for fp, refs in index.items() if len({split for split, _ in refs}) > 1]
        if mode == "magnitude_time_scaffold":
            shared[mode] = {"cross_split_atomic_scaffolds": len(cross),
                           "policy": "Disclosed shared atoms; whole-parent reuse remains a blocker."}
        else:
            collisions.extend(dict(item, reason=mode) for item in cross)
        matches = [{"fingerprint": fp, "new_groups": sorted({group for _, group in refs}),
                    "reason": "observed_" + mode} for fp, refs in index.items() if fp in observed[mode]]
        if mode == "magnitude_time_scaffold":
            shared["observed_atomic_scaffolds"] = len(matches)
        else:
            collisions.extend(matches)
    return {"status": "isolation_collisions_detected" if collisions else "mechanical_checks_passed_pending_causal_contract_review",
        "new_rows": len(rows), "observed_rows_screened": len(observed_rows),
        "collisions": collisions, "quarantined_new_groups": sorted({g for c in collisions for g in c["new_groups"]}),
        "shared_atomic_scaffolds": shared, "model_calls": 0,
        "limits": ["Mechanical equality checks do not prove semantic independence or foundation-pretraining novelty.",
                   "Schema-specific causal projection and reviewed contract remain required before freeze.",
                   "Group IDs, condition metadata and ancestry hashes are not used as isolation proof."]}


def audit_v8(rows, contract, *, observed_rows=(), observed_fingerprints=None):
    require(contract.get("version") == "frontier-controls-v8" and contract.get("split_widths") == SPLIT_WIDTHS,
            "Unsupported independently reviewed contract or split-width declaration")
    require(set(contract["family_specs"]) == set(V8_FIELDS), "All seven reviewed families must be declared")
    groups = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    errors, group_reports, crosses, outcome_counts = [], {}, {}, Counter()
    seen_family_width = set()
    for group, bundle in groups.items():
        try:
            family, width = None, None
            case_presentations = defaultdict(set)
            case_facts, symbolic_cases, cells = defaultdict(set), defaultdict(set), Counter()
            for row in bundle:
                core, actual_width = causal_projection(row["state"])
                actual_family = core["family"]
                if family is None: family, width = actual_family, actual_width
                require((actual_family, actual_width) == (family, width), "Parent includes mixed actual causal families or widths")
                require(actual_width in SPLIT_WIDTHS[row["split"]], "Actual essential width belongs to another split")
                metadata = row["metadata"]
                require(metadata.get("scenario_family") == family and metadata.get("composition_width") == width,
                        "Declared family/width does not match actual causal graph")
                policy = row["state"]["trusted_policy"]["rules"] if family == "joint_capacity" else (
                    row["state"]["trusted_policy"]["ordered_rule"] if family == "refund_priority" else
                    row["state"]["trust_contract"] if family == "latest_authority" else
                    row["state"]["rule"] if family == "account_status" else row["state"]["policy"])
                require(policy == contract["family_specs"][family]["policy"], "Visible policy differs from reviewed typed contract")
                case, presentation = metadata.get("case_index"), metadata.get("presentation_index")
                require(type(case) is int and type(presentation) is int and 0 <= presentation <= 3,
                        "Semantic case and presentation assertions required")
                require(presentation not in case_presentations[case], "Duplicate case/presentation identity")
                case_presentations[case].add(presentation)
                case_facts[digest(core)].add(case)
                symbolic_cases[digest(causal_projection(row["state"], abstract=True)[0])].add(case)
                require(row["kind"] == ("choice" if presentation < 2 else "noul"), "Presentation kind differs from four-form bundle")
                outcome = core_outcome(core)
                outcome_counts[(row["split"], family, "balance" if family == "exact_numeric" and core["task"] == "balance" else outcome)] += 1
                truth = "choice"
                if row["kind"] == "noul":
                    matches = re.findall(r"'([^']+)'", row["question"])
                    require(len(matches) == 1, "One visible Noul proposition required")
                    truth = "noul_yes" if matches[0] == outcome else "noul_no"
                    require(truth == ("noul_yes" if presentation == 2 else "noul_no"), "Actual Noul truth differs from presentation assertion")
                for index, field in causal_scope_cells(row):
                    members = row["state"]["trusted_policy"]["required_roles"] if family == "joint_capacity" else row["state"]["required_topics"]
                    position = "first" if index == 0 else "last" if index == len(members) - 1 else "middle"
                    cells[(position, field, truth)] += 1
                    assertion = "affected_role_index" if family == "joint_capacity" else "affected_topic_index"
                    require(metadata.get(assertion) == index and metadata.get("scope_field") == field,
                            "Scope/affected membership declaration differs from actual missing valid authority")
            require(family is not None and (family, width) not in seen_family_width, "Multiple parents use the same family/essential-width scaffold")
            seen_family_width.add((family, width))
            case_count = contract["family_specs"][family]["case_count"]
            require(set(case_presentations) == set(range(case_count)) and all(forms == {0, 1, 2, 3} for forms in case_presentations.values()),
                    "All causal case descendants must remain in one complete parent bundle")
            require(not parent_neutral_padding(bundle), "A recurring zero-sum padding subgraph cannot establish parent independence")
            require(essential_witness(bundle), "No complete valid case witnesses essential component influence")
            if family in ("joint_capacity", "latest_authority"):
                fields = ("resource", "operation", "currency") if family == "joint_capacity" else ("department", "topic")
                mandatory = {(position, field, truth) for position in ("first", "last")
                             for field in fields for truth in ("choice", "noul_no", "noul_yes")}
                require(all(cells[key] > 0 for key in mandatory), "Actual first/last scope-kind-truth crosses are incomplete")
            group_reports[group] = {"family": family, "essential_width": width, "rows": len(bundle),
                "declared_case_count": case_count, "unique_causal_fact_cases": len(case_facts),
                "collapsed_case_classes": sum(len(ids) > 1 for ids in case_facts.values()),
                "case_aliases": [sorted(ids) for ids in case_facts.values() if len(ids) > 1],
                "distinct_symbolic_case_scaffolds": len(symbolic_cases),
                "symbolic_case_aliases": [sorted(ids) for ids in symbolic_cases.values() if len(ids) > 1],
                "essential_baseline_witness": True, "recurring_neutral_padding": False}
            crosses[group] = {"/".join(key): count for key, count in sorted(cells.items())}
        except (KeyError, ValueError, TypeError) as error:
            errors.append({"new_group": group, "reason": str(error), "error_type": type(error).__name__})
    require(len(rows) == 2976 and len(groups) == 42, "Actual dataset size differs from separately declared 2976-row/42-parent candidate")
    require(dict(Counter(r["split"] for r in rows)) == {"train": 992, "calibration": 496, "validation": 496, "test": 496, "ood": 496},
            "Actual split quotas differ from declaration")
    if errors:
        return {"status": "causal_contract_or_isolation_failed", "new_rows": len(rows), "model_calls": 0,
            "errors": errors, "quarantined_new_groups": sorted({e["new_group"] for e in errors}),
            "parent_reports": group_reports, "actual_scope_kind_truth": crosses}
    result = audit_rows(rows, observed_rows=observed_rows, strict_v8=True, observed_fingerprints=observed_fingerprints)
    if observed_fingerprints is not None:
        require(bool(observed_rows), "Opaque hashes alone cannot prove observed-data screening")
        recomputed = collision_index(observed_rows)
        require({mode: set(index) for mode, index in recomputed.items()} ==
                {mode: set(index) for mode, index in observed_fingerprints.items()},
                "Opaque observed index differs from fresh visible-input replay")
    overlap_done = bool(observed_rows)
    if overlap_done:
        require(len(observed_rows) == 1128 and all(r["split"] in ("test", "ood") for r in observed_rows)
                and dict(Counter(r["source"] for r in observed_rows)) == {
                    "frontier-controls-v4": 480, "temporal-windows-v5": 64,
                    "original-policy-controls-v6-candidate": 72, "boundary-controls-v7": 512},
                "Complete observed v4-v7 Test/OOD inventory required")
    if not result["collisions"]:
        result["status"] = "independent_v8_isolation_and_overlap_passed_pending_root_freeze" if overlap_done else "v8_causal_isolation_passed_observed_overlap_pending"
    result.update(parent_reports=group_reports, actual_scope_kind_truth=crosses,
        reviewed_contract_version=contract["version"], actual_widths_by_split=SPLIT_WIDTHS,
        shared_atomic_rule_motifs={"/".join(key): count for key, count in sorted(outcome_counts.items())},
        middle_role_topic_mismatch_coverage="Not exhaustive; only declared first/last scope crossings are mandatory.")
    result["limits"] += ["Essential widths are controlled composition shifts; Test is not IID or natural business data.",
        "Causal projection handles seven reviewed schemas, not exhaustive mathematical graph isomorphism.",
        "Recurring exact zero-sum operand-shape screen is conservative and bounded to sixteen operands; independent source/oracle review remains required."]
    return result


def load_rows(root, *, splits=SPLITS):
    root = Path(root)
    rows, bindings = [], {}
    manifest = json.loads((root / "manifest.json").read_text())
    for split in splits:
        path = root / (split + ".jsonl")
        raw = path.read_bytes()
        checksum = hashlib.sha256(raw).hexdigest()
        if manifest["files_sha256"].get(path.name) != checksum:
            raise ValueError("Dataset split hash changed")
        bindings[path.name] = checksum
        for line in raw.decode().splitlines():
            row = json.loads(line)
            if row["split"] != split:
                raise ValueError("Row split differs from physical split file")
            rows.append(row)
    return rows, bindings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--contract", default=str(Path(__file__).resolve().parents[1] / "reports/frontier-v8-data-20261003/contract.json"))
    parser.add_argument("--observed-datasets", nargs="*", default=[])
    parser.add_argument("--observed-fingerprints")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    contract = Path(args.contract).read_bytes()
    document = json.loads(contract)
    rows, bindings = load_rows(args.dataset)
    old_rows, old_bindings = [], {}
    for root in args.observed_datasets:
        old, checksums = load_rows(root, splits=("test", "ood"))
        old_rows.extend(old)
        old_bindings[str(root)] = checksums
    opaque = None
    if args.observed_fingerprints:
        cache = json.loads(Path(args.observed_fingerprints).read_text())
        require(cache["audit_source_sha256"] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "Observed fingerprint cache belongs to another normalizer version")
        opaque = {key: set(values) for key, values in cache["fingerprints"].items()}
    result = audit_v8(rows, document, observed_rows=old_rows, observed_fingerprints=opaque)
    result.update(dataset_split_sha256=bindings, observed_split_sha256=old_bindings,
        contract_sha256=hashlib.sha256(contract).hexdigest(),
        audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with Path(args.output).open("x") as output:
        output.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "new_rows", "quarantined_new_groups")}, indent=2))
    raise SystemExit(result["status"] != "independent_v8_isolation_and_overlap_passed_pending_root_freeze")


if __name__ == "__main__":
    main()
