"""CPU parity tests using real random Qwen hybrid attention and real PEFT LoRA.

No pretrained weights, network access, CUDA, or tokenizer downloads are needed.
The three pinned models all use qwen3_5_text (including Qwen3.8-27B).
Fixtures shrink dimensions/layers while retaining each model's full-attention
Q/KV and linear-attention V/K head ratios, the 3-linear/1-full block, and kernel 4.
Pinned config.json SHA256, in 2B / 9B / 27B order:
ed1c1723241f23f7f4e23430759cbd7dcfb4103cbdfe052bfe7626b57c2615b4
d0883072e01861ed0b2d47be3c16c36a8e81c224c7ffaa310c6558fb3f932b05
191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab
These test cache mechanics, not pretrained-model quality or GPU kernels.
"""

import copy
import gc
import importlib.util
import unittest
import weakref
from unittest.mock import patch

HAS_RUNTIME = all(importlib.util.find_spec(name) for name in ("torch", "transformers", "peft"))
if HAS_RUNTIME:
    import torch
    from torch import nn
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3_5TextConfig, Qwen3_5TextModel
    from transformers.cache_utils import DynamicCache, DynamicLayer, LinearAttentionLayer

    from jev.api import candidate_prompts, compile_request
    from jev.model import DecisionModel
    from jev.prefix_cache import RecurrentDtypeLayer, fork_cache, new_cache
    from jev.serving import Predictor, TorchScorer


PROFILES = {"qwen3.5-2b": (4, 1), "qwen3.5-9b": (4, 2), "qwen3.8-27b": (6, 3)}
CACHE_FIELDS = ("keys", "values", "conv_states", "recurrent_states")


class ByteTokenizer:
    """Lossless, deterministic small vocabulary; exercises complete chat inputs."""

    pad_token_id = 0
    padding_side = "right"

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert not tokenize and add_generation_prompt and not enable_thinking
        return "<user>\n" + messages[0]["content"] + "\n</user><assistant>\n"

    def encode(self, text):
        return [value + 1 for value in text.encode("utf-8")]

    def __call__(self, prompts, *, padding=False, truncation=False, return_tensors=None):
        assert not truncation
        rows = [self.encode(prompt) for prompt in prompts]
        masks = [[1] * len(row) for row in rows]
        if padding:
            maximum = max(map(len, rows))
            masks = [mask + [0] * (maximum - len(mask)) for mask in masks]
            rows = [row + [0] * (maximum - len(row)) for row in rows]
        result = {"input_ids": rows, "attention_mask": masks}
        return {key: torch.tensor(value) for key, value in result.items()} if return_tensors else result


def tiny_model(profile="qwen3.5-9b", *, lora=True, device="cpu"):
    query_ratio, linear_ratio = PROFILES[profile]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(4129)
        config = Qwen3_5TextConfig(
            vocab_size=257, hidden_size=32, intermediate_size=48, num_hidden_layers=4,
            num_attention_heads=query_ratio, num_key_value_heads=1, head_dim=16,
            linear_num_key_heads=1, linear_num_value_heads=linear_ratio,
            linear_key_head_dim=8, linear_value_head_dim=8, linear_conv_kernel_dim=4,
            layer_types=["linear_attention"] * 3 + ["full_attention"],
            max_position_embeddings=2048, initializer_range=0.06, pad_token_id=0,
            attention_dropout=0.0, use_cache=False, attn_implementation="sdpa",
            rope_parameters={"rope_type": "default", "rope_theta": 10000000.0,
                             "partial_rotary_factor": 0.25, "mrope_section": [1, 1, 0]},
        )
        backbone = Qwen3_5TextModel(config).requires_grad_(False)
        if lora:
            backbone = get_peft_model(backbone, LoraConfig(
                r=2, lora_alpha=4, lora_dropout=0.0, bias="none",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "out_proj"],
            ))
            # Fresh LoRA has zero B matrices; make adapters observably active.
            with torch.no_grad():
                for name, parameter in backbone.named_parameters():
                    if "lora_B" in name:
                        parameter.normal_(std=0.05)
            backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            backbone.enable_input_require_grads()
        # Avoid DecisionModel.__init__, which loads pretrained checkpoint weights.
        model = DecisionModel.__new__(DecisionModel)
        nn.Module.__init__(model)
        model.backbone, model.head = backbone, nn.Linear(32, 1)
        model.tokenizer, model.device_name, model.max_length = ByteTokenizer(), "cpu", 2048
        model.model_id, model.revision, model.lora_rank = profile, "random-cpu-fixture", 2 if lora else 0
        model.head.weight.data.normal_(std=0.25)
    model = model.eval()
    if device != "cpu":
        # Same seeded init as the CPU fixture, then relocated, so a CPU/device
        # comparison isolates the backend rather than the initialization.
        model.to(device)
        model.device_name = device
    return model


def mixed_request():
    choice = {"type": "choice", "instructions": "Which route is available?", "criteria": {
        "up": "go", "dn": "no", "east": "turn right through the long passage",
        "stay": "wait", "hold": "pause while another vehicle passes",
    }}
    return {"state": {"shared": "The vehicle approaches a junction. " * 3, "标记": "北"},
            "questions": {
                "route": choice,
                "confidence": {"type": "score", "instructions": "Rate route safety.",
                               "criteria": ["low", "moderate", "high with no visible obstacles"]},
                "clear": {"type": "noul", "instructions": "Is the route clear?"},
                "route_again": copy.deepcopy(choice),
            }}


def cache_snapshot(cache):
    return {(index, name): tensor.clone()
            for index, layer in enumerate(cache.layers) for name in CACHE_FIELDS
            if isinstance(tensor := getattr(layer, name, None), torch.Tensor)}


@unittest.skipUnless(HAS_RUNTIME, "requires optional torch/transformers/peft train runtime")
class PrefixCacheTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def assert_rows_close(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for left, right in zip(actual, expected):
            torch.testing.assert_close(left, right, atol=2e-5, rtol=2e-5)
            torch.testing.assert_close(left.softmax(-1), right.softmax(-1), atol=2e-6, rtol=2e-5)

    def assert_snapshot_unchanged(self, cache, snapshot):
        self.assertEqual(set(cache_snapshot(cache)), set(snapshot))
        for (index, name), expected in snapshot.items():
            torch.testing.assert_close(getattr(cache.layers[index], name), expected, atol=0, rtol=0)

    def test_real_hybrid_logits_probabilities_and_observed_token_reuse(self):
        request = mixed_request()
        records = compile_request(request["state"], request["questions"])
        original = copy.deepcopy(records)
        for profile in PROFILES:
            for lora in (False, True):
                model = tiny_model(profile, lora=lora)
                with torch.inference_mode():
                    expected = model(records)
                logical_tokens = model.last_input_tokens
                weights = {key: value.clone() for key, value in model.state_dict().items()}
                for batch_size in (1, 2):
                    with self.subTest(profile=profile, lora=lora, batch_size=batch_size):
                        calls = []

                        def observe(module, args, kwargs):
                            calls.append(tuple(kwargs["input_ids"].shape))
                            self.assertTrue(kwargs["use_cache"])
                            self.assertTrue(bool(kwargs["attention_mask"].all()))

                        hook = model.backbone.register_forward_pre_hook(observe, with_kwargs=True)
                        try:
                            actual, stats = model.score_cached(records, batch_size=batch_size)
                        finally:
                            hook.remove()
                        self.assert_rows_close(actual, expected)
                        self.assert_rows_close([actual[0]], [actual[3]])
                        self.assertEqual(actual[2][0].item(), 0.0)  # Noul false logit.
                        self.assertFalse(any(row.requires_grad for row in actual))
                        processed = sum(size * length for size, length in calls)
                        self.assertEqual(stats["logical_input_tokens"], logical_tokens)
                        self.assertEqual(stats["processed_input_tokens"], processed)
                        self.assertEqual(stats["reused_input_tokens"], logical_tokens - processed)
                        self.assertLess(processed, logical_tokens)
                        self.assertGreater(stats["shared_prefix_tokens"], 0)
                        self.assertGreater(stats["question_prefix_tokens"], 0)
                        self.assertLessEqual(max(size for size, _ in calls), batch_size)
                        self.assertEqual(stats["candidate_sequences"], 14)
                        self.assertEqual(records, original)
                        for name, value in model.state_dict().items():
                            torch.testing.assert_close(value, weights[name], atol=0, rtol=0)

    def test_candidate_and_question_permutations_remain_equivariant(self):
        model = tiny_model("qwen3.8-27b")
        request = mixed_request()
        records = compile_request(request["state"], request["questions"])
        expected, _ = model.score_cached(records, batch_size=1)
        permutation = [2, 3, 1, 0]
        changed = [copy.deepcopy(records[index]) for index in permutation]
        for record in changed:
            if record["kind"] != "noul":
                record["options"].reverse()
        actual, _ = model.score_cached(changed, batch_size=2)
        ordered = [expected[index] if records[index]["kind"] == "noul" else expected[index].flip(0)
                   for index in permutation]
        self.assert_rows_close(actual, ordered)

    def test_fork_isolates_every_cache_tensor_and_batched_continuations(self):
        for profile in PROFILES:
            model = tiny_model(profile)
            prefix = torch.tensor([[7, 9, 3, 12, 8, 4, 15, 10, 2]])
            with torch.inference_mode():
                saved = model.backbone(input_ids=prefix, past_key_values=new_cache(model.backbone.config),
                                       use_cache=True).past_key_values
                self.assertIs(type(saved), DynamicCache)
                self.assertEqual([type(layer) for layer in saved.layers],
                                 [RecurrentDtypeLayer] * 3 + [DynamicLayer])
                snapshot = cache_snapshot(saved)
                self.assertEqual({name for _, name in snapshot}, set(CACHE_FIELDS))
                for batch_size in (1, 2):
                    with self.subTest(profile=profile, batch_size=batch_size):
                        branch = fork_cache(saved, batch_size)
                        for (index, name), expected in snapshot.items():
                            actual = getattr(branch.layers[index], name)
                            self.assertNotEqual(actual.data_ptr(), getattr(saved.layers[index], name).data_ptr())
                            torch.testing.assert_close(actual, expected.repeat_interleave(batch_size, 0))
                            actual[0].add_(3)
                            if batch_size == 2:
                                torch.testing.assert_close(actual[1], expected[0], atol=0, rtol=0)
                        self.assert_snapshot_unchanged(saved, snapshot)
                        # Both one-token recurrent decode and chunk continuation.
                        for length in (1, 5):
                            suffix = torch.arange(20, 20 + batch_size * length).view(batch_size, length)
                            branch = fork_cache(saved, batch_size)
                            cached = model.backbone(input_ids=suffix, past_key_values=branch,
                                                    use_cache=True).last_hidden_state
                            independent = model.backbone(input_ids=torch.cat([
                                prefix.repeat(batch_size, 1), suffix], dim=-1),
                                use_cache=False).last_hidden_state[:, -length:]
                            torch.testing.assert_close(cached, independent, atol=2e-5, rtol=2e-5)
                            self.assert_snapshot_unchanged(saved, snapshot)

    def test_bfloat16_cache_keeps_kernel_recurrent_state_in_float32(self):
        model = tiny_model()
        model.backbone.to(dtype=torch.bfloat16)
        for name, parameter in model.backbone.named_parameters():
            if "lora_" in name:
                parameter.data = parameter.data.float()
        request = mixed_request()
        records = compile_request(request["state"], request["questions"])
        seen = []

        def inspect_states(module, args, output):
            for layer in output.past_key_values.layers:
                if isinstance(layer, LinearAttentionLayer):
                    seen.append((layer.conv_states.dtype, layer.recurrent_states.dtype))

        hook = model.backbone.register_forward_hook(inspect_states)
        try:
            model.score_cached(records, batch_size=2)
        finally:
            hook.remove()
        self.assertTrue(seen)
        self.assertTrue(all(conv == torch.bfloat16 and recurrent == torch.float32
                            for conv, recurrent in seen))

    def test_zero_short_and_identical_prefixes_match_independent_scoring(self):
        model = tiny_model()
        record = {"kind": "choice", "state": "", "question": "q", "options": ["a", "b", "c"]}
        chat = [model.tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False)
                for prompt in candidate_prompts(record)]
        cases = [
            [[3], [4, 8, 9], [5, 8]],  # No token shared; includes a one-token input.
            [[3, 4], [3, 8, 9], [3, 8, 10, 11]],  # Prefix shorter than convolution kernel.
            [[3, 4, 5], [3, 4, 5, 6], [3, 4, 9, 10]],  # One input ends inside another.
            [[3, 4, 5]] * 3,  # Exact duplicates must still leave a token to score.
        ]
        for sequences in cases:
            lookup = dict(zip(chat, sequences))
            with patch.object(model.tokenizer, "encode", side_effect=lambda text: lookup[text]):
                with torch.inference_mode():
                    expected = model([record])
                for batch_size in (1, 2):
                    with self.subTest(sequences=sequences, batch_size=batch_size):
                        actual, _ = model.score_cached([record], batch_size=batch_size)
                        self.assert_rows_close(actual, expected)

    def test_fixture_detects_losing_each_kind_of_cache_state(self):
        model = tiny_model("qwen3.8-27b")
        with torch.inference_mode():
            prefix, suffix = torch.tensor([[7, 9, 3, 12, 8, 4, 15, 10, 2]]), torch.tensor([[23]])
            saved = model.backbone(input_ids=prefix, use_cache=True).past_key_values
            expected = model.backbone(input_ids=torch.cat([prefix, suffix], -1),
                                      use_cache=False).last_hidden_state[:, -1]
            # Ensure parity tests are sensitive to K, V, conv, and recurrent errors.
            for name in CACHE_FIELDS:
                with self.subTest(lost_state=name):
                    broken = fork_cache(saved, 1)
                    for layer in broken.layers:
                        if isinstance(tensor := getattr(layer, name, None), torch.Tensor):
                            tensor.zero_()
                    actual = model.backbone(input_ids=suffix, past_key_values=broken,
                                            use_cache=True).last_hidden_state[:, -1]
                    self.assertGreater((actual - expected).abs().max().item(), 1e-3)
                    self.assertGreater((model.head(actual) - model.head(expected)).abs().max().item(), 1e-3)

    def test_lora_wrapper_is_active_on_cached_path(self):
        model = tiny_model("qwen3.8-27b")
        request = mixed_request()
        records = compile_request(request["state"], request["questions"])
        enabled, _ = model.score_cached(records, batch_size=2)
        with model.backbone.disable_adapter(), torch.inference_mode():
            disabled = model(records)
        self.assertGreater(max((left - right).abs().max().item()
                               for left, right in zip(enabled, disabled)), 1e-3)
        with torch.inference_mode():
            expected = model(records)
        self.assert_rows_close(enabled, expected)

    def test_requests_and_exception_paths_release_cache_and_preserve_inputs(self):
        model = tiny_model()
        request = mixed_request()
        records = compile_request(request["state"], request["questions"])
        original = copy.deepcopy(records)
        expected, _ = model.score_cached(records, batch_size=2)
        references = []

        def fail_after_cache_is_created(module, args, output):
            references.append(weakref.ref(output.past_key_values))
            if len(references) == 3:
                raise RuntimeError("injected continuation failure")

        hook = model.backbone.register_forward_hook(fail_after_cache_is_created)
        try:
            with self.assertRaisesRegex(RuntimeError, "injected continuation failure"):
                model.score_cached(records, batch_size=2)
        finally:
            hook.remove()
        gc.collect()
        self.assertEqual(len(references), 3)
        self.assertTrue(all(reference() is None for reference in references))
        self.assertEqual(records, original)
        self.assertFalse(model.training)
        self.assertFalse(model.backbone.config.use_cache)
        other = compile_request({"shared": "A wholly different second scene."}, request["questions"])
        model.score_cached(other, batch_size=1)
        references.clear()
        hook = model.backbone.register_forward_hook(
            lambda module, args, output: references.append(weakref.ref(output.past_key_values)))
        try:
            recovered, _ = model.score_cached(records, batch_size=2)
        finally:
            hook.remove()
        self.assert_rows_close(recovered, expected)
        gc.collect()
        self.assertTrue(references)
        self.assertTrue(all(reference() is None for reference in references))

    def test_overlength_and_training_rejected_before_any_cached_forward(self):
        model = tiny_model()
        request = mixed_request()
        records = compile_request(request["state"], request["questions"])
        with patch.object(model.backbone, "forward", side_effect=AssertionError("must fail before forward")):
            model.max_length = 1
            with self.assertRaisesRegex(ValueError, "max_length"):
                model.score_cached(records, batch_size=2)
            model.max_length = 2048
            model.train()
            with self.assertRaisesRegex(RuntimeError, "evaluation mode"):
                model.score_cached(records, batch_size=2)

    def test_training_forward_is_uncached_and_adapter_head_gradients_survive(self):
        model = tiny_model("qwen3.8-27b")
        request = mixed_request()
        records = compile_request(request["state"], request["questions"])
        model.score_cached(records, batch_size=2)
        model.train()
        calls = []
        hook = model.backbone.register_forward_pre_hook(
            lambda module, args, kwargs: calls.append(kwargs.get("use_cache")), with_kwargs=True)
        try:
            with patch.object(model, "score_cached", side_effect=AssertionError("training called cache")):
                logits = model(records)
                loss = sum(torch.nn.functional.cross_entropy(row[None], torch.tensor([1])) for row in logits)
                loss.backward()
        finally:
            hook.remove()
        self.assertEqual(calls, [False])
        self.assertGreater(model.head.weight.grad.abs().sum().item(), 0)
        adapter_grads = [parameter.grad for name, parameter in model.backbone.named_parameters() if "lora_" in name]
        self.assertTrue(adapter_grads)
        self.assertTrue(all(value is not None and bool(value.isfinite().all()) for value in adapter_grads))
        self.assertGreater(sum(value.abs().sum().item() for value in adapter_grads), 0)
        self.assertTrue(all(parameter.grad is None for name, parameter in model.backbone.named_parameters()
                            if "lora_" not in name))
        self.assertFalse(model.backbone.config.use_cache)

    def test_predictor_choice_noul_score_probabilities_and_usage_match(self):
        model = tiny_model()
        request = mixed_request()
        original = copy.deepcopy(request)
        for batch_size in (1, 2):
            with self.subTest(batch_size=batch_size):
                base = dict(scorer=TorchScorer(model), model_name="tiny", batch_size=batch_size, temperature=0.7)
                expected = Predictor(**base, prefix_cache=False).predict(request)
                actual = Predictor(**base, prefix_cache=True).predict(request)
                self.assertEqual(actual["usage"], expected["usage"])
                self.assertTrue(actual["metadata"]["prefix_cache"]["enabled"])
                for question_id, left in actual["answers"].items():
                    right = expected["answers"][question_id]
                    self.assertEqual(left["type"], right["type"])
                    if left["type"] == "noul":
                        self.assertAlmostEqual(left["noul"], right["noul"], places=5)
                    else:
                        if left["type"] == "choice":
                            self.assertEqual(left["choice"], right["choice"])
                        else:
                            self.assertAlmostEqual(left["score"], right["score"], places=5)
                        for key, value in left["probabilities"].items():
                            self.assertAlmostEqual(value, right["probabilities"][key], places=5)
                self.assertEqual(request, original)

    def test_checkpoint_benchmark_runner_uses_real_cpu_paths_and_checks_parity(self):
        from scripts.benchmark_prefix_cache import benchmark, compare_answers

        model, request = tiny_model(), mixed_request()
        original = copy.deepcopy(request)
        predictor = Predictor(TorchScorer(model), model_name="tiny", batch_size=2, prefix_cache=False)
        report = benchmark(predictor, [request], warmup=1, repetitions=2)
        self.assertEqual(report["status"], "passed")
        self.assertEqual([sample["mode"] for sample in report["samples"]],
                         ["uncached", "cached", "cached", "uncached"])
        self.assertTrue(all(sample["wall_seconds"] > 0 for sample in report["samples"]))
        self.assertTrue(all(sample["cuda_memory_bytes"] is None for sample in report["samples"]))
        self.assertGreater(report["samples"][0]["processed_input_tokens"],
                           report["samples"][1]["processed_input_tokens"])
        self.assertFalse(predictor.prefix_cache)
        self.assertEqual(request, original)
        responses = report["last_answers_by_request_and_mode"]["0"]
        baseline = {"answers": responses["uncached"], "usage": {"input_tokens": 4097}}
        broken = {"answers": copy.deepcopy(responses["cached"]), "usage": baseline["usage"]}
        broken["answers"]["clear"]["noul"] = 1 - baseline["answers"]["clear"]["noul"]
        check = compare_answers(baseline, broken, 1e-4)
        self.assertFalse(check["passed"])
        self.assertFalse(check["argmax_equal"])
        del broken["answers"]["route"]
        self.assertFalse(compare_answers(baseline, broken, 1e-4)["question_ids_equal"])
        boundary = copy.deepcopy(baseline)
        boundary["answers"]["clear"]["noul"] = 0.5
        baseline["answers"] = copy.deepcopy(baseline["answers"])
        baseline["answers"]["clear"]["noul"] = 0.49999
        check = compare_answers(baseline, boundary, 1e-4)
        self.assertLess(check["max_probability_error"], 1e-4)
        self.assertFalse(check["passed"])
        self.assertFalse(check["argmax_equal"])


if __name__ == "__main__":
    unittest.main()
