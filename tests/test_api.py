import copy
import json
import math
import subprocess
import sys
import unittest

from jev.api import candidate_prompts, compile_request, format_response


class TypedAPITest(unittest.TestCase):
    def setUp(self):
        self.questions = {
            "department": {"type": "choice", "instructions": "Which team handles this?",
                           "criteria": {"returns": "Exchanges and refunds", "shipping": "Delivery status"}},
            "severity": {"type": "score", "instructions": "How severe is the bug?",
                         "criteria": ["Cosmetic only", "Feature degraded but workaround exists", "Blocking without workaround"]},
            "refund": {"type": "noul", "instructions": "Is a refund requested?",
                       "criteria": {"true": "Explicitly asks for money back", "false": "Does not request money back"}},
        }

    def test_all_primitives_have_exact_typed_outputs(self):
        records = compile_request("Please refund the broken shoes.", self.questions)
        answers = format_response(records, [[0.9, 0.1], [0.0, 0.7, 0.3], [0.02, 0.98]])["answers"]
        self.assertEqual(answers["department"]["choice"], "returns")
        self.assertAlmostEqual(answers["department"]["confidence"], 0.8)
        self.assertAlmostEqual(answers["severity"]["score"], 1.3)
        self.assertEqual(answers["severity"]["legend"]["2"], "Blocking without workaround")
        self.assertEqual(answers["refund"], {"type": "noul", "noul": 0.98})
        self.assertNotIn("confidence", answers["refund"])
        json.dumps(answers, allow_nan=False)

    def test_question_ids_labels_targets_and_siblings_are_isolated(self):
        records = compile_request({"ticket": "A delivery question"}, self.questions)
        original = candidate_prompts(records[0])
        changed = copy.deepcopy(records[0])
        changed.update(id="HIDDEN_ID", target=[0.0, 1.0], split="HIDDEN_SPLIT", metadata={"secret": "HIDDEN_LABEL"})
        self.assertEqual(candidate_prompts(changed), original)
        alone = compile_request(records[0]["state"], {"different ID": self.questions["department"]})
        self.assertEqual(candidate_prompts(alone[0]), original)
        self.assertIn("returns: Exchanges and refunds", original[0])
        self.assertNotIn("Delivery status", original[0])
        reversed_record = copy.deepcopy(records[0])
        reversed_record["options"].reverse()
        self.assertEqual(candidate_prompts(reversed_record), original[::-1])

    def test_score_sees_own_description_without_neighbor_or_number(self):
        records = compile_request("A bug report", self.questions)
        prompts = candidate_prompts(records[1])
        self.assertIn("Cosmetic only", prompts[0])
        self.assertNotIn("workaround", prompts[0])
        self.assertNotIn("Proposed answer: 0", prompts[0])
        self.assertIn("Yes means: Explicitly asks", candidate_prompts(records[2])[0])
        self.assertIn("No means: Does not request", candidate_prompts(records[2])[0])

    def test_noul_accepts_optional_true_and_false_criteria(self):
        cases = [
            ({"true": "Refund requested"}, "Yes means: Refund requested", "No means:"),
            ({"false": "No refund requested"}, "No means: No refund requested", "Yes means:"),
            ({}, None, None),
        ]
        for criteria, expected_description, absent_description in cases:
            with self.subTest(criteria=criteria):
                record = compile_request("Refund please", {
                    "refund": {"type": "noul", "instructions": "Is a refund requested?", "criteria": criteria},
                })[0]
                self.assertEqual(record["options"], ["no", "yes"])
                self.assertEqual(record["answer_keys"], ["false", "true"])
                prompt = candidate_prompts(record)[0]
                if expected_description:
                    self.assertIn(expected_description, prompt)
                if absent_description:
                    self.assertNotIn(absent_description, prompt)
                if not criteria:
                    self.assertNotIn("Yes means:", prompt)
                    self.assertNotIn("No means:", prompt)

    def test_structured_descriptions_and_null_choice_descriptions(self):
        records = compile_request(["first", "second"], {
            "language": {"type": "choice", "instructions": {"task": "Identify language"},
                         "criteria": {"python": None, "other": {"includes": ["Go", "Rust"]}}},
        })
        self.assertEqual(records[0]["options"][0], "python")
        self.assertIn('"includes": ["Go", "Rust"]', records[0]["options"][1])
        self.assertIn('"task": "Identify language"', records[0]["question"])

    def test_input_mutation_does_not_modify_compiled_state(self):
        state = {"value": ["before"]}
        records = compile_request(state, self.questions)
        state["value"][0] = "after"
        self.assertEqual(records[0]["state"]["value"], ["before"])

    def test_cardinality_and_primitive_validation(self):
        for kind, criteria in [("choice", {}), ("choice", {str(i): None for i in range(256)}),
                               ("score", ["only"]), ("score", ["level"] * 11),
                               ("noul", {"other": "yes"}), ("unknown", None)]:
            with self.subTest(kind=kind, count=len(criteria) if criteria is not None else 0):
                with self.assertRaises(ValueError):
                    compile_request("state", {"q": {"type": kind, "instructions": "Question", "criteria": criteria}})
        records = compile_request("state", {"q": {"type": "choice", "instructions": "Question",
                                                     "criteria": {str(i): None for i in range(255)}}})
        self.assertEqual(len(records[0]["options"]), 255)

    def test_invalid_model_probabilities_are_never_formatted_as_answers(self):
        records = compile_request("state", {"q": self.questions["department"]})
        for probs in ([], [[1.0]], [[0.3, 0.3]], [[-0.1, 1.1]], [[math.nan, 0.0]], [[math.inf, 0.0]]):
            with self.subTest(probabilities=probs):
                with self.assertRaises(ValueError):
                    format_response(records, probs)
        with self.assertRaises(ValueError):
            format_response(records * 2, [[0.5, 0.5], [0.5, 0.5]])

    def test_api_cold_import_has_no_torch_dependency(self):
        subprocess.run([sys.executable, "-c", "import sys; import jev.api; assert 'torch' not in sys.modules"], check=True)


if __name__ == "__main__":
    unittest.main()
