"""Independent arithmetic/time/authority replay for the original v7 controls."""
from collections import Counter, defaultdict
import copy
from datetime import date
import hashlib
import itertools
import json
from pathlib import Path
import re
import tempfile
import unittest

from jev.api import candidate_prompts
from jev.boundary_controls_v7 import CONDITIONS, FAMILIES, GROUP_COUNTS, audit, build, oracle, records
from jev.data import SPLITS, input_fingerprint
from scripts.audit_boundary_controls_v7 import CONTRACTS, audit_rows


def epoch_seconds(text):
    match = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})([+-])(\d{2}):(\d{2})", text)
    if not match:
        raise ValueError("Explicit whole-second fixed-offset timestamp required")
    year, month, day, hour, minute, second, sign, oh, om = match.groups()
    year, month, day, hour, minute, second, oh, om = map(int, (year, month, day, hour, minute, second, oh, om))
    if hour >= 24 or minute >= 60 or second >= 60 or oh >= 24 or om >= 60:
        raise ValueError("Invalid displayed time or offset")
    days = date(year, month, day).toordinal() - date(1970, 1, 1).toordinal()
    offset = (oh * 60 + om) * (1 if sign == "+" else -1)
    return days * 86400 + hour * 3600 + minute * 60 + second - offset * 60


def exact_int(value, nonnegative=False):
    if type(value) is not int or (nonnegative and value < 0):
        raise ValueError("Integer cents/facts required")
    return value


def integer_balance(state):
    for entry in state["ledger"]:
        exact_int(entry["amount_cents"], True)
        if entry["direction"] not in ("credit", "debit"):
            raise ValueError("Unexpected direction")
    credits = sum(e["amount_cents"] for e in state["ledger"] if e["direction"] == "credit")
    debits = sum(e["amount_cents"] for e in state["ledger"] if e["direction"] == "debit")
    return credits - debits


def amount_label(cents):
    amount = str(abs(cents) // 100) + "." + str(abs(cents) % 100).zfill(2)
    return "USD " + ("-" if cents < 0 else "") + amount


def independent_answer(family, state):
    if family == "temporal_window":
        elapsed = epoch_seconds(state["request_received_at"]) - epoch_seconds(state["delivered_at"])
        duration = exact_int(state["return_window_seconds"], True)
        if type(state["exception_approved"]) is not bool:
            raise ValueError("Approval must be boolean")
        return "review" if state["exception_approved"] else ("accept" if elapsed in range(duration + 1) else "reject")
    if family == "exact_numeric":
        if state["task"] == "balance":
            return amount_label(integer_balance(state))
        if state["task"] != "integer_compare":
            raise ValueError("Unknown numeric task")
        left, right = exact_int(state["left_cents"]), exact_int(state["right_cents"])
        return ("below", "equal", "above")[(left > right) - (left < right) + 1]
    if family == "joint_capacity":
        q = state["request"]
        expected_scope = {"resource": q["resource"], "operation": q["operation"], "currency": q["currency"]}
        current = []
        for role in state["trusted_policy"]["required_roles"]:
            values = sorted((e for e in state["signed_events"] if e["issuer_role"] == role
                             and e["verified_signature"] is True and e["credential_scope"] == expected_scope), key=lambda e: e["sequence"], reverse=True)
            for event in values:
                exact_int(event["sequence"], True)
                exact_int(event["capacity_cents"], True)
                if event["status"] not in ("grant", "revoke"):
                    raise ValueError("Unexpected consent status")
            if len({e["sequence"] for e in values}) != len(values):
                raise ValueError("Conflicting role sequence")
            current.append(values[0] if values else None)
        if any(e is not None and e["status"] == "revoke" for e in current):
            return "reject revoked consent"
        if None in current:
            return "request missing consent"
        capacity = min(exact_int(e["capacity_cents"], True) for e in current)
        return "execute" if exact_int(q["amount_cents"], True) <= capacity else "request higher capacity"
    if family == "latest_authority":
        q = state["request"]
        valid = sorted((p for p in state["signed_policy_registry"] if p["verified_signature"] is True
                        and (p["issuer_role"], p["department"], p["credential_scope"]) == (state["policy_authority"], q["department"], q["department"])), key=lambda p: p["revision"], reverse=True)
        for policy in valid:
            exact_int(policy["revision"], True)
            exact_int(policy["automatic_limit_cents"], True)
            if policy["status"] not in ("active", "withdrawn"):
                raise ValueError("Unexpected policy status")
        exact_int(q["amount_cents"], True)
        if type(q["confirmed_fraud"]) is not bool:
            raise ValueError("Fraud must be boolean")
        if len({p["revision"] for p in valid}) != len(valid):
            raise ValueError("Conflicting revision")
        if not valid or valid[0]["status"] == "withdrawn":
            return "verify policy"
        if q["confirmed_fraud"]:
            return "fraud review"
        return "capacity review" if q["amount_cents"] > valid[0]["automatic_limit_cents"] else "automatic processing"
    raise ValueError("Unknown family")


def replay(row):
    family, state = row["metadata"]["scenario_family"], row["state"]
    contract = state["trusted_policy"]["rules"] if family == "joint_capacity" else state["trust_contract"] if family == "latest_authority" else state["policy"]
    if contract != CONTRACTS[family]:
        raise ValueError("Visible rule differs from independently reviewed contract")
    answer = independent_answer(family, state)
    if row["kind"] == "noul":
        match = re.fullmatch(r"Does the supplied rule establish '([^']+)' for this case\?", row["question"])
        if not match or match[1] != row["metadata"]["proposed_outcome"]:
            raise ValueError("Visible proposal does not match metadata")
        answer = "yes" if match[1] == answer else "no"
    if row["options"][row["target"].index(1.0)] != answer:
        raise ValueError("Independent replay disagrees with target")
    return answer


class BoundaryControlsV7Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = list(records())

    def test_complete_schema_split_counts_independent_replay_and_determinism(self):
        self.assertEqual(self.rows, list(records()))
        result = audit(self.rows)
        self.assertEqual(result["records"], 1792)
        self.assertEqual(result["groups"], 224)
        self.assertEqual(result["splits"], dict(train=1024, calibration=128, validation=128, test=256, ood=256))
        groups = defaultdict(list)
        for row in self.rows:
            replay(row)
            self.assertTrue(all(value in (0., 1.) for value in row["target"]))
            groups[row["group_id"]].append(row)
        for values in groups.values():
            self.assertEqual(len(values), 8)
            self.assertEqual({v["split"] for v in values}, {values[0]["split"]})
            self.assertEqual({v["metadata"]["provenance"]["variant"] for v in values}, set(range(8)))
            self.assertEqual({v["metadata"]["condition"] for v in values}, set(CONDITIONS[values[0]["metadata"]["scenario_family"]]))

    def test_condition_kind_truth_correct_position_and_action_coverage(self):
        coverage, positions, actions = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
        for row in self.rows:
            split, family, condition = row["split"], row["metadata"]["scenario_family"], row["metadata"]["condition"]
            answer = replay(row)
            coverage[(split, family, condition)][row["kind"] + ("/" + answer if row["kind"] == "noul" else "")] += 1
            actions[(split, family)][independent_answer(family, row["state"])] += 1
            if row["kind"] == "choice":
                positions[(split, family, row["state"].get("task", family), len(row["options"]))][row["target"].index(1.)] += 1
        for split, family in itertools.product(SPLITS, FAMILIES):
            for condition in CONDITIONS[family]:
                counts = coverage[(split, family, condition)]
                self.assertEqual(counts, {"choice": GROUP_COUNTS[split] // 2, "noul/yes": GROUP_COUNTS[split] // 4, "noul/no": GROUP_COUNTS[split] // 4})
            if family == "temporal_window":
                self.assertEqual(actions[(split, family)], dict(accept=3 * GROUP_COUNTS[split], reject=3 * GROUP_COUNTS[split], review=2 * GROUP_COUNTS[split]))
            elif family in ("joint_capacity", "latest_authority"):
                self.assertEqual(set(actions[(split, family)].values()), {2 * GROUP_COUNTS[split]})
        for key, counts in positions.items():
            self.assertEqual(set(counts), set(range(key[-1])))
            self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

    def test_structural_alternatives_cross_kind_and_noul_truth(self):
        coverage = defaultdict(Counter)
        for row in self.rows:
            family, condition = row["metadata"]["scenario_family"], row["metadata"]["condition"]
            state = row["state"]
            if family == "temporal_window" and condition == "equivalent_outside":
                side = int(epoch_seconds(state["request_received_at"]) > epoch_seconds(state["delivered_at"]))
            elif family == "exact_numeric" and condition == "compare_negative":
                side = int(state["left_cents"] > state["right_cents"])
            elif family == "joint_capacity" and condition not in ("equality", "one_cent_above"):
                roles = state["trusted_policy"]["required_roles"]
                scope = {key: state["request"][key] for key in ("resource", "operation", "currency")}
                valid = [e for e in state["signed_events"] if e["verified_signature"] is True and e["credential_scope"] == scope]
                counts = Counter(e["issuer_role"] for e in valid)
                selected = [role for role in roles if counts[role] != 1]
                self.assertEqual(len(selected), 1)
                side = roles.index(selected[0])
            else:
                continue
            key = (row["split"], family, condition, side)
            label = row["kind"] + ("/" + replay(row) if row["kind"] == "noul" else "")
            coverage[key][label] += 1
        expected_conditions = [("temporal_window", "equivalent_outside"), ("exact_numeric", "compare_negative")]
        expected_conditions += [("joint_capacity", condition) for condition in CONDITIONS["joint_capacity"] if condition not in ("equality", "one_cent_above")]
        for split, (family, condition), side in itertools.product(SPLITS, expected_conditions, (0, 1)):
            counts = coverage[(split, family, condition, side)]
            n = GROUP_COUNTS[split]
            self.assertEqual(counts["choice"], n // 4)
            self.assertEqual(counts["noul/yes"] + counts["noul/no"], n // 4)
            if n >= 8:
                self.assertEqual(counts["noul/yes"], n // 8)
                self.assertEqual(counts["noul/no"], n // 8)
            # Four-group Calibration/Validation have one Noul row per side;
            # their two truth labels are balanced across the condition only.

    def test_temporal_inclusive_offsets_exception_and_equivalent_negative(self):
        groups = defaultdict(dict)
        for row in self.rows:
            if row["metadata"]["scenario_family"] == "temporal_window":
                groups[row["group_id"]][row["metadata"]["condition"]] = row["state"]
        for cases in groups.values():
            for name, state in cases.items():
                delta = epoch_seconds(state["request_received_at"]) - epoch_seconds(state["delivered_at"])
                duration = state["return_window_seconds"]
                if name in ("before", "before_exception"):
                    self.assertEqual(delta, -1)
                elif name == "at_start": self.assertEqual(delta, 0)
                elif name == "inside": self.assertTrue(0 < delta < duration)
                elif name == "at_deadline": self.assertEqual(delta, duration)
                elif name in ("after", "after_exception"): self.assertEqual(delta, duration + 1)
            equivalent = cases["equivalent_outside"]
            other = cases["before"] if epoch_seconds(equivalent["request_received_at"]) < epoch_seconds(equivalent["delivered_at"]) else cases["after"]
            self.assertNotEqual(equivalent["request_received_at"], other["request_received_at"])
            self.assertEqual(epoch_seconds(equivalent["request_received_at"]), epoch_seconds(other["request_received_at"]))
        base = {"delivered_at": "2028-02-29T05:30:00+05:30", "return_window_seconds": 3600, "exception_approved": False}
        for timestamp, expected in (("2028-02-28T19:59:59-04:00", "reject"), ("2028-02-28T20:00:00-04:00", "accept"), ("2028-02-28T21:00:00-04:00", "accept"), ("2028-02-28T21:00:01-04:00", "reject")):
            state = {**base, "request_received_at": timestamp}
            self.assertEqual(independent_answer("temporal_window", state), expected)
            self.assertEqual(oracle("temporal_window", state), expected)
            self.assertEqual(oracle("temporal_window", {**state, "exception_approved": True}), "review")

    def test_numeric_signed_zero_cancelled_integer_money_and_nonmedian_gold(self):
        ranks = set()
        for row in self.rows:
            if row["metadata"]["scenario_family"] != "exact_numeric": continue
            state, condition = row["state"], row["metadata"]["condition"]
            if state["task"] == "balance":
                total = integer_balance(state)
                if condition == "balance_negative": self.assertLess(total, 0)
                elif condition == "balance_zero": self.assertEqual(total, 0)
                else: self.assertGreater(total, 0)
                if condition == "balance_cancellation":
                    self.assertGreater(len(state["ledger"]), 3)
                    cancelled = [e for e in state["ledger"] if "/cancel-" in e["entry_ref"]]
                    self.assertEqual({e["direction"] for e in cancelled}, {"credit", "debit"})
                    self.assertEqual(len({e["amount_cents"] for e in cancelled}), 1)
                    self.assertGreater(cancelled[0]["amount_cents"], 0)
                if row["kind"] == "choice":
                    cents = []
                    for option in row["options"]:
                        match = re.fullmatch(r"USD (-?)(\d+)\.(\d{2})", option)
                        self.assertIsNotNone(match)
                        cents.append((int(match[2]) * 100 + int(match[3])) * (-1 if match[1] else 1))
                    self.assertEqual(len(set(cents)), 5)
                    ranks.add(sorted(cents).index(total))
                shuffled = copy.deepcopy(state); shuffled["ledger"].reverse()
                self.assertEqual(oracle("exact_numeric", shuffled), independent_answer("exact_numeric", state))
            elif condition == "compare_negative":
                self.assertLess(state["left_cents"], 0); self.assertLess(state["right_cents"], 0)
                self.assertIn(independent_answer("exact_numeric", state), ("below", "above"))
        self.assertEqual(ranks, set(range(5)))
        state = {"task": "balance", "ledger": [{"direction": "credit", "amount_cents": 999}, {"direction": "debit", "amount_cents": 1000}]}
        self.assertEqual(oracle("exact_numeric", state), "USD -0.01")
        for bad in (True, .1, -1):
            broken = copy.deepcopy(state); broken["ledger"][0]["amount_cents"] = bad
            with self.assertRaises(ValueError): oracle("exact_numeric", broken)

    def test_joint_scope_sequence_revocation_missing_and_one_cent_mutations(self):
        row = next(r for r in self.rows if r["metadata"]["scenario_family"] == "joint_capacity" and r["metadata"]["condition"] == "equality")
        base = row["state"]
        for delta in (-1, 0, 1):
            state = copy.deepcopy(base); state["request"]["amount_cents"] += delta
            self.assertEqual(oracle("joint_capacity", state), independent_answer("joint_capacity", state))
            self.assertEqual(oracle("joint_capacity", state), "execute" if delta <= 0 else "request higher capacity")
        for field in ("resource", "operation", "currency"):
            state = copy.deepcopy(base)
            event = next(e for e in state["signed_events"] if e["verified_signature"])
            event["credential_scope"][field] += "-mismatch"
            self.assertEqual(oracle("joint_capacity", state), "request missing consent")
        state = copy.deepcopy(base)
        valid = [e for e in state["signed_events"] if e["verified_signature"]]
        state["signed_events"].remove(valid[0])
        state["signed_events"].append({**valid[1], "sequence": valid[1]["sequence"] + 1, "status": "revoke"})
        self.assertEqual(independent_answer("joint_capacity", state), "reject revoked consent")
        self.assertEqual(oracle("joint_capacity", state), "reject revoked consent")
        state["signed_events"].append({**valid[1], "sequence": valid[1]["sequence"] + 1, "status": "grant"})
        with self.assertRaisesRegex(ValueError, "Conflicting"): oracle("joint_capacity", state)

    def test_latest_authority_missing_withdrawal_and_fraud_priority_mutations(self):
        row = next(r for r in self.rows if r["metadata"]["scenario_family"] == "latest_authority" and r["metadata"]["condition"] == "equality")
        base = row["state"]
        for fraud, delta in itertools.product((False, True), (-1, 0, 1)):
            state = copy.deepcopy(base); state["request"].update(confirmed_fraud=fraud, amount_cents=base["request"]["amount_cents"] + delta)
            self.assertEqual(oracle("latest_authority", state), independent_answer("latest_authority", state))
        current = max(base["signed_policy_registry"], key=lambda p: p["revision"])
        state = copy.deepcopy(base); state["request"]["confirmed_fraud"] = True
        state["signed_policy_registry"].append({**current, "revision": current["revision"] + 1, "status": "withdrawn"})
        self.assertEqual(oracle("latest_authority", state), "verify policy")
        state["signed_policy_registry"].append({**current, "revision": current["revision"] + 2})
        self.assertEqual(oracle("latest_authority", state), "fraud review")
        state["signed_policy_registry"] = []
        self.assertEqual(oracle("latest_authority", state), "verify policy")

    def test_independent_replay_rejects_malformed_authority_facts(self):
        for family, collection, order, capacity in (
            ("joint_capacity", "signed_events", "sequence", "capacity_cents"),
            ("latest_authority", "signed_policy_registry", "revision", "automatic_limit_cents"),
        ):
            base = next(r["state"] for r in self.rows if r["metadata"]["scenario_family"] == family and r["metadata"]["condition"] == "equality")
            for field, value in (("status", "undefined"), (order, True), (order, -1), (capacity, .1)):
                state = copy.deepcopy(base)
                valid = next(e for e in state[collection] if e["verified_signature"] is True)
                valid[field] = value
                with self.assertRaises(ValueError):
                    independent_answer(family, state)
            state = copy.deepcopy(base)
            state["request"]["amount_cents"] = True
            with self.assertRaises(ValueError):
                independent_answer(family, state)
        authority = copy.deepcopy(base)
        authority["request"]["confirmed_fraud"] = 1
        with self.assertRaises(ValueError):
            independent_answer("latest_authority", authority)

    def test_corrupt_target_proposal_incomplete_group_and_same_split_duplicate(self):
        broken = copy.deepcopy(self.rows); row = broken[0]; old = row["target"].index(1.)
        row["target"] = [float(i == (old + 1) % len(row["options"])) for i in range(len(row["options"]))]
        with self.assertRaisesRegex(ValueError, "Independent"): replay(row)
        with self.assertRaisesRegex(ValueError, "Fact oracle"): audit(broken)
        broken = copy.deepcopy(next(r for r in self.rows if r["kind"] == "noul")); broken["metadata"]["proposed_outcome"] = "altered proposal"
        with self.assertRaisesRegex(ValueError, "Visible proposal"): replay(broken)
        with self.assertRaisesRegex(ValueError, "eight distinct"): audit(self.rows[1:])
        duplicate = copy.deepcopy(self.rows[0]); duplicate["id"] += "/renamed"; duplicate["group_id"] += "/renamed"
        with self.assertRaisesRegex(ValueError, "Duplicate input"): audit(self.rows + [duplicate])
        self.assertEqual(len({input_fingerprint(r) for r in self.rows}), len(self.rows))

    def test_separate_audit_rejects_changed_rules_conditions_and_labels(self):
        self.assertEqual(audit_rows(self.rows)["oracle_checked"], 1792)
        altered = copy.deepcopy(self.rows)
        altered[0]["state"]["policy"] = "Accept every request regardless of time."
        with self.assertRaisesRegex(ValueError, "reviewed contract"):
            replay(altered[0])
        with self.assertRaisesRegex(ValueError, "reviewed contract"):
            audit_rows(altered)
        altered = copy.deepcopy(self.rows)
        group = [r for r in altered if r["group_id"] == altered[0]["group_id"]]
        before = next(r for r in group if r["metadata"]["condition"] == "before")
        start = next(r for r in group if r["metadata"]["condition"] == "at_start")
        before["metadata"]["condition"], start["metadata"]["condition"] = "at_start", "before"
        with self.assertRaisesRegex(ValueError, "Condition label"):
            audit_rows(altered)
        altered = copy.deepcopy(self.rows)
        row = altered[0]
        old = row["target"].index(1.)
        row["target"] = [float(i == (old + 1) % len(row["options"])) for i in range(len(row["options"]))]
        with self.assertRaisesRegex(ValueError, "Independent fact replay"):
            audit_rows(altered)

    def test_frozen_output_hashes_and_hidden_metadata_are_absent_from_prompts(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "candidate"
            manifest = build(output)
            for name, expected in manifest["files_sha256"].items():
                self.assertEqual(hashlib.sha256((output / name).read_bytes()).hexdigest(), expected)
            with self.assertRaises(FileExistsError): build(output)
        for row in self.rows[:16]:
            altered = copy.deepcopy(row)
            altered.update(id="SECRET", group_id="SECRET", source="SECRET", split="ood", metadata={"condition": "SECRET"}, target=list(reversed(row["target"])))
            self.assertEqual(candidate_prompts(row), candidate_prompts(altered))
        packed = json.dumps(self.rows, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        # Freeze canonical generated row content, independently of source-file hashes.
        self.assertEqual(hashlib.sha256(packed).hexdigest(), "f4cb281f34578d98dc7e87c5bf77bda328288e8ad277796bf077fda924fa97e2")


if __name__ == "__main__":
    unittest.main()
