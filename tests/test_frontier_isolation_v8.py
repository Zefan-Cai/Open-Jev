import copy
import unittest

from scripts.audit_frontier_isolation_v8 import (audit_rows, causal_projection, essential_witness,
                                               neutral_profiles, parent_neutral_padding, row_fingerprints)


def row(identity, split, state, *, group=None, kind="choice", options=None):
    return {"id": identity, "group_id": group or identity, "split": split,
            "source": "fixture-only", "state": state,
            "question": "Apply the visible rule and choose the outcome." if kind == "choice"
                else "Does the rule establish 'equal'?",
            "kind": kind, "options": options or (["below", "equal", "above"] if kind == "choice" else ["no", "yes"]),
            "target": [0., 1., 0.] if kind == "choice" else [0., 1.],
            "metadata": {"ancestry_hash": "untrusted-declaration"}}


def compare(left, right):
    return {"task": "integer_compare", "left_cents": left, "right_cents": right,
            "case_ref": "presentation-only", "policy": "Compare the supplied exact integers."}


def ledger(credit, debit):
    return {"task": "balance", "currency": "USD", "ledger": [
        {"direction": "credit", "amount_cents": credit},
        {"direction": "debit", "amount_cents": debit}],
        "policy": "Add credits and subtract debits."}


class FrontierIsolationTests(unittest.TestCase):
    def test_dynamic_topic_mapping_renaming_and_permutation_preserve_edges(self):
        state = {"case_ref": "case-A", "request": {"department": "sales", "amount_cents": 50, "confirmed_fraud": False},
            "required_topics": ["spending", "quality"], "policy_authorities": {"spending": "finance", "quality": "inspector"},
            "signed_policy_registry": [{"topic": topic, "department": "sales",
                "credential_scope": {"department": "sales", "topic": topic}, "issuer_role": authority,
                "verified_signature": True, "revision": 1, "status": "active", "automatic_limit_cents": limit}
                for topic, authority, limit in [("spending", "finance", 100), ("quality", "inspector", 200)]],
            "trust_contract": "Fixture authoritative topic conjunction."}
        first = row("one", "train", state)
        names = {"spending": "budget", "quality": "review", "finance": "treasurer", "inspector": "examiner", "sales": "operations"}
        def rename(value):
            if isinstance(value, dict): return {names.get(k, k): rename(v) for k, v in reversed(list(value.items()))}
            if isinstance(value, list): return [rename(v) for v in reversed(value)]
            return names.get(value, value) if isinstance(value, str) else value
        renamed = row("two", "test", rename(state))
        self.assertEqual(row_fingerprints(first, strict_v8=True)["wording_order_identity"],
                         row_fingerprints(renamed, strict_v8=True)["wording_order_identity"])
        wrong_edges = copy.deepcopy(state)
        wrong_edges["policy_authorities"] = {"spending": "inspector", "quality": "finance"}
        self.assertNotEqual(causal_projection(state)[0], causal_projection(wrong_edges)[0])
        wrong_edges["required_topics"].append("ghost")
        wrong_edges["policy_authorities"]["ghost"] = "ghost-authority"
        self.assertFalse(essential_witness([row("ghost", "test", wrong_edges)]))

    def test_account_membership_permutation_and_irrelevant_events(self):
        state = {"case_ref": "fixture", "required_accounts": ["first", "second"],
            "events": [{"account": "first", "sequence": 4, "status": "active"},
                       {"account": "second", "sequence": 9, "status": "paused"}], "rule": "Fixture aggregate rule."}
        moved = copy.deepcopy(state)
        moved["required_accounts"].reverse(); moved["events"].reverse()
        moved["events"].append({"account": "irrelevant", "sequence": 99, "status": "closed"})
        moved["note"] = "Irrelevant decoration cannot create an isolated parent."
        self.assertEqual(causal_projection(state)[0], causal_projection(moved)[0])
        moved["unreviewed_extra_condition"] = True
        with self.assertRaisesRegex(ValueError, "Unreviewed state field"):
            causal_projection(moved)

    def test_required_role_line_and_item_permutations_keep_causal_membership(self):
        scope = {"resource": "fixture-resource", "operation": "release", "currency": "USD"}
        joint = {"case_ref": "fixture", "request": {**scope, "amount_cents": 50},
            "trusted_policy": {"required_roles": ["guardian", "approver"], "rules": "Fixture conjunctive rule."},
            "signed_events": [{"issuer_role": role, "verified_signature": True, "credential_scope": dict(scope),
                "sequence": 1, "status": "grant", "capacity_cents": capacity}
                for role, capacity in [("guardian", 100), ("approver", 200)]]}
        moved = copy.deepcopy(joint)
        moved["trusted_policy"]["required_roles"].reverse(); moved["signed_events"].reverse()
        moved["signed_events"].append({**copy.deepcopy(moved["signed_events"][0]), "sequence": 99,
                                       "verified_signature": False, "status": "revoke"})
        self.assertEqual(causal_projection(joint), causal_projection(moved))
        self.assertEqual(causal_projection(joint)[1], 1)
        moved["trusted_policy"]["required_roles"].append("ghost")
        self.assertFalse(essential_witness([row("ghost", "train", moved)]))
        department = {"case_ref": "fixture", "request_lines": [
            {"department": "sales", "amount_cents": 30, "confirmed_fraud": False},
            {"department": "ops", "amount_cents": 50, "confirmed_fraud": False}], "policy_authority": "authority",
            "signed_policies": [{"department": d, "credential_scope": d, "issuer_role": "authority",
                "verified_signature": True, "revision": 1, "status": "active", "approval_limit_cents": 100}
                for d in ["sales", "ops"]], "policy": "Fixture line rule."}
        moved = copy.deepcopy(department)
        moved["request_lines"].reverse(); moved["signed_policies"].reverse()
        self.assertEqual(causal_projection(department), causal_projection(moved))
        refund = {"case_ref": "fixture", "purchase_document_verified": True,
            "items": [{"item_ref": name, "amount_cents": value, "defect_confirmed": True,
                       "seal_intact": False, "verified_recall": False} for name, value in [("a", 30), ("b", 40)]],
            "trusted_policy": {"currency": "USD", "automatic_limit_cents": 100, "ordered_rule": "Fixture refund rule."}}
        moved = copy.deepcopy(refund); moved["items"].reverse()
        for item in moved["items"]: item["item_ref"] += "-renamed"
        self.assertEqual(causal_projection(refund), causal_projection(moved))
        moved["items"].append({"item_ref": "zero", "amount_cents": 0, "defect_confirmed": True,
                               "seal_intact": False, "verified_recall": False})
        with self.assertRaisesRegex(ValueError, "Zero-value refund padding"):
            causal_projection(moved)

    def test_ignored_corrections_cannot_supply_temporal_width(self):
        state = {"case_ref": "C", "correction_authority": "A", "recorded_delivery_at": "2026-01-01T00:00:00Z",
            "request_received_at": "2026-01-01T00:00:07Z", "return_window_seconds": 3600,
            "exception_approved": False, "policy": "Fixture corrected-anchor rule.",
            "delivery_corrections": [{"sequence": index, "issuer_role": "A", "verified_signature": True,
                "case_ref": "C", "direction": direction, "seconds": seconds}
                for index, direction, seconds in [(1, "delay", 10), (2, "advance", 3)]]}
        moved = copy.deepcopy(state); moved["delivery_corrections"].reverse()
        moved["delivery_corrections"].append({"sequence": 99, "issuer_role": "A", "verified_signature": False,
            "case_ref": "C", "direction": "delay", "seconds": 999})
        self.assertEqual(causal_projection(state), causal_projection(moved))
        self.assertEqual(causal_projection(moved)[1], 2)

    def test_multi_entry_neutral_padding_is_blocked_but_whole_zero_balance_is_valid(self):
        self.assertEqual(neutral_profiles([3, 7, -4, -6]), set())
        self.assertIn((2, 1), neutral_profiles([21, -11, 3, 7, -10]))
        base = ledger(21, 11)
        padded = copy.deepcopy(base)
        for direction, value in [("credit", 3), ("credit", 7), ("debit", 10),
                                  ("credit", 12), ("debit", 4), ("debit", 8)]:
            padded["ledger"].append({"direction": direction, "amount_cents": value})
        other = copy.deepcopy(padded)
        other["ledger"][0]["amount_cents"] = 22
        self.assertTrue(parent_neutral_padding([row("one", "train", padded), row("two", "train", other)]))
        correction_state = {"case_ref": "C", "correction_authority": "A",
            "recorded_delivery_at": "2026-01-01T00:00:00Z", "request_received_at": "2026-01-01T00:00:23Z",
            "return_window_seconds": 3600, "exception_approved": False, "policy": "Fixture rule.",
            "delivery_corrections": [{"sequence": i, "case_ref": "C", "issuer_role": "A", "verified_signature": True,
                "direction": "delay" if value > 0 else "advance", "seconds": abs(value)}
                for i, value in enumerate([23, 3, 7, -10])]}
        self.assertEqual(causal_projection(correction_state)[1], 4)
        self.assertIn((2, 1), parent_neutral_padding([row("one", "validation", correction_state)]))

    def test_zero_operands_and_cancellation_pairs_cannot_inflate_v8_width(self):
        state = {"case_ref": "fixture", "task": "balance", "currency": "USD", "policy": "Fixture ledger rule.",
            "ledger": [{"entry_ref": str(i), "direction": direction, "amount_cents": value}
                for i, (direction, value) in enumerate([("credit", 3), ("credit", 7), ("debit", 4), ("debit", 6)])]}
        self.assertEqual(causal_projection(state)[1], 2)
        bad = copy.deepcopy(state)
        bad["ledger"][0]["amount_cents"] = 0
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            causal_projection(bad)
        bad = copy.deepcopy(state)
        bad["ledger"] += [{"entry_ref": "p1", "direction": "credit", "amount_cents": 55},
                          {"entry_ref": "p2", "direction": "debit", "amount_cents": 55}]
        with self.assertRaisesRegex(ValueError, "cancellation padding"):
            causal_projection(bad)

    def test_compiled_rotations_and_metadata_cannot_disguise_input(self):
        first = row("one", "train", compare(2, 2))
        second = copy.deepcopy(first)
        second.update(id="two", group_id="fabricated-other-parent", split="test")
        second["options"].reverse()
        second["metadata"] = {"ancestry_hash": "arbitrary-different-hash", "proposed_outcome": "forged"}
        result = audit_rows([first, second])
        self.assertIn("compiled_visible", {c["reason"] for c in result["collisions"]})

    def test_wording_identity_record_order_and_visible_noul_are_normalized(self):
        state = {"request": {"resource": "Alpha", "operation": "pay", "currency": "USD"},
            "trusted_policy": {"required_roles": ["keeper", "approver"], "rules": "Literal wording one."},
            "signed_events": [
                {"issuer_role": "keeper", "sequence": 1, "status": "grant", "verified_signature": True,
                 "capacity_cents": 900, "credential_scope": {"resource": "Alpha", "operation": "pay", "currency": "USD"}},
                {"issuer_role": "approver", "sequence": 2, "status": "grant", "verified_signature": True,
                 "capacity_cents": 900, "credential_scope": {"resource": "Alpha", "operation": "pay", "currency": "USD"}}]}
        first = row("one", "train", state, kind="noul")
        second = copy.deepcopy(first)
        second.update(id="two", group_id="two", split="test")
        second["question"] = "Under the specification, is 'equal' supported?"
        second["metadata"]["proposed_outcome"] = "forged-invisible-proposition"
        rendered = str(second["state"])
        for source, target in [("Alpha", "Beta"), ("keeper", "guardian"), ("approver", "signer")]:
            rendered = rendered.replace(source, target)
        import ast
        second["state"] = ast.literal_eval(rendered)
        second["state"]["signed_events"].reverse()
        second["state"]["trusted_policy"]["required_roles"].reverse()
        second["state"]["trusted_policy"]["rules"] = "Equivalent wording declared for the fixture."
        result = audit_rows([first, second])
        self.assertIn("wording_order_identity", {c["reason"] for c in result["collisions"]})

    def test_clock_representation_normalizes_absolute_instants(self):
        first = row("one", "train", {"delivered_at": "2026-01-01T00:00:00Z",
            "request_received_at": "2026-01-01T01:00:00Z", "return_window_seconds": 3600,
            "exception_approved": False})
        second = copy.deepcopy(first)
        second.update(id="two", group_id="two", split="test")
        second["state"]["delivered_at"] = "2026-01-01T02:00:00+02:00"
        second["state"]["request_received_at"] = "2026-01-01T03:00:00+02:00"
        self.assertEqual(row_fingerprints(first)["wording_order_identity"],
                         row_fingerprints(second)["wording_order_identity"])

    def test_full_menu_numeric_and_time_origin_clones_are_blocked(self):
        train = [row("a", "train", compare(1, 2), group="parent-A"),
                 row("b", "train", ledger(700, 200), group="parent-A")]
        test = [row("c", "test", compare(1001, 1002), group="parent-B"),
                row("d", "test", ledger(70000, 20000), group="parent-B")]
        result = audit_rows(train + test)
        self.assertIn("whole_parent_scaffold", {c["reason"] for c in result["collisions"]})
        original = {"delivered_at": "2026-01-01T00:00:00Z", "request_received_at": "2026-01-01T01:00:00Z",
                    "return_window_seconds": 3600, "exception_approved": False}
        moved = {**original, "delivered_at": "2027-04-03T00:00:00Z", "request_received_at": "2027-04-03T01:00:00Z"}
        result = audit_rows([row("e", "train", original), row("f", "ood", moved)])
        self.assertIn("whole_parent_scaffold", {c["reason"] for c in result["collisions"]})

    def test_repeated_rotations_and_noop_padding_do_not_make_new_parent(self):
        original = row("one", "train", ledger(700, 200))
        padded = row("two", "test", ledger(70000, 20000))
        padded["state"]["note"] = "Unrelated text cannot establish causal independence."
        padded["state"]["ledger"] += [{"direction": "credit", "amount_cents": 55},
                                       {"direction": "debit", "amount_cents": 55}]
        rotation = copy.deepcopy(padded)
        rotation["id"] = "rotation"
        rotation["options"].reverse()
        result = audit_rows([original, padded, rotation])
        self.assertIn("whole_parent_scaffold", {c["reason"] for c in result["collisions"]})

    def test_shared_numeric_atom_with_different_parent_menu_is_disclosed(self):
        rows = [row("a", "train", compare(1, 2), group="A"),
                row("b", "train", ledger(700, 200), group="A"),
                row("c", "test", compare(200, 900), group="B"),
                row("d", "test", {"delivered_at": "2026-01-01T00:00:00Z",
                    "request_received_at": "2026-01-01T02:00:00Z", "return_window_seconds": 3600,
                    "exception_approved": True}, group="B")]
        result = audit_rows(rows)
        self.assertEqual(result["collisions"], [])
        self.assertEqual(result["shared_atomic_scaffolds"]["magnitude_time_scaffold"]["cross_split_atomic_scaffolds"], 1)
        self.assertIn("pending_causal_contract_review", result["status"])

    def test_observed_overlap_exposes_only_opaque_fingerprints(self):
        old = row("PRIVATE_OLD_ID", "test", compare(8, 9))
        new = row("new", "train", compare(8, 9))
        result = audit_rows([new], observed_rows=[old])
        self.assertTrue(result["collisions"])
        rendered = str(result)
        self.assertNotIn("PRIVATE_OLD_ID", rendered)
        self.assertNotIn("left_cents", rendered)
        self.assertNotIn("target", rendered)
        self.assertEqual(result["quarantined_new_groups"], ["new"])

    def test_actual_parent_cannot_cross_splits_and_unknown_noul_is_not_guessed(self):
        with self.assertRaisesRegex(ValueError, "Actual parent group"):
            audit_rows([row("one", "train", compare(1, 2), group="same"),
                        row("two", "test", compare(9, 3), group="same")])
        unknown = row("noul", "train", compare(1, 2), kind="noul")
        unknown["question"] = "Unreviewed wording without an explicit proposition."
        with self.assertRaisesRegex(ValueError, "visible quoted proposition"):
            row_fingerprints(unknown)


if __name__ == "__main__":
    unittest.main()
