"""Apple Silicon (torch MPS) checks on the random hybrid Qwen fixture.

No pretrained weights or network access are needed. The MPS tests prove that
the hybrid backbone, LoRA adapter and scalar head execute on Metal with every
parameter resident there and no CPU fallback, and that MPS scores match the CPU
reference: float32 within the existing CPU parity bar, bfloat16 and float16 no
further from a float32 reference than CPU at the same dtype is. They skip on
machines without an MPS backend. The profiling regression runs everywhere.
"""
import json
import os
from pathlib import Path
import tempfile
import unittest
import warnings
from unittest.mock import patch

from tests import test_prefix_cache as base
from tests.test_prefix_cache import HAS_RUNTIME, PROFILES, mixed_request, tiny_model

if HAS_RUNTIME:
    import torch

    from jev.api import compile_request

HAS_MPS = HAS_RUNTIME and torch.backends.mps.is_available()


def records():
    request = mixed_request()
    return compile_request(request["state"], request["questions"])


def off_device(model, device_type):
    return [name for name, value in model.named_parameters() if value.device.type != device_type]


def score(model, rows, *, cached=False):
    with torch.inference_mode():
        logits = model.score_cached(rows, batch_size=2)[0] if cached else model(rows)
    return [row.float().cpu() for row in logits]


def max_error(actual, expected):
    return max((left - right).abs().max().item() for left, right in zip(actual, expected))


def cuda_must_not_synchronize(*args, **kwargs):
    raise AssertionError("CUDA synchronize called for a model that is not on CUDA")


@unittest.skipUnless(HAS_RUNTIME, "requires optional torch/transformers/peft train runtime")
class ProfileSynchronizationTest(unittest.TestCase):
    def test_cpu_profile_timing_never_synchronizes_cuda(self):
        # JEV_PROFILE=1 used to call torch.cuda.synchronize() unconditionally,
        # which raises on any machine without CUDA.
        with patch.dict(os.environ, {"JEV_PROFILE": "1"}), \
             patch("torch.cuda.synchronize", side_effect=cuda_must_not_synchronize):
            _, stats = tiny_model().score_cached(records(), batch_size=2)
        self.assertIn("shared_prefill", stats["profile_seconds"])


class ProfiledChildEnvironmentTest(unittest.TestCase):
    def test_profiled_child_loads_weights_on_one_thread(self):
        # MPSProfiler segfaulted inside transformers' threaded weight loading
        # on the released 2B checkpoint (M4, torch 2.14, transformers 5.10).
        from scripts.check_mps_engagement import PROFILE_LOG_OPTIONS, profiled_child_environment
        with patch.dict(os.environ, {"HF_HUB_OFFLINE": "1"}):
            environment = profiled_child_environment()
        self.assertEqual(environment["HF_DEACTIVATE_ASYNC_LOAD"], "1")
        self.assertEqual(environment["PYTORCH_MPS_LOG_PROFILE_INFO"], str(PROFILE_LOG_OPTIONS))
        self.assertEqual(environment["HF_HUB_OFFLINE"], "1")


@unittest.skipUnless(HAS_RUNTIME, "requires optional torch/transformers/peft train runtime")
class HttpTimeoutTest(unittest.TestCase):
    def test_loopback_timeout_defaults_to_120_seconds_and_is_configurable(self):
        # CPU bfloat16 took 172 s for one uncached request of the released 2B
        # checkpoint on an M4, beyond the fixed 120 s loopback timeout.
        import http.client
        from jev.serving import Predictor, TorchScorer
        from scripts.benchmark_inference_latency import benchmark, digest
        request = mixed_request()
        workloads = [{"id": "mixed", "request": request, "request_sha256": digest(request)}]
        predictor = Predictor(TorchScorer(tiny_model()), model_name="tiny", method="fixture")
        for keyword, expected in (({}, 120), ({"http_timeout": 900}, 900)):
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as directory, \
                 patch("http.client.HTTPConnection", wraps=http.client.HTTPConnection) as connection:
                result = benchmark(predictor, workloads, output=Path(directory), warmup=1, repetitions=1, **keyword)
                self.assertEqual(result["status"], "passed")
                self.assertEqual({call.kwargs["timeout"] for call in connection.call_args_list}, {expected})


@unittest.skipUnless(HAS_MPS, "requires a torch build with an available MPS backend")
class MpsSupportTest(unittest.TestCase):
    assert_rows_close = base.PrefixCacheTest.assert_rows_close

    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_every_parameter_is_resident_on_mps_and_nothing_falls_back_to_cpu(self):
        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
            self.skipTest("PYTORCH_ENABLE_MPS_FALLBACK=1 would hide unsupported MPS ops")
        rows = records()
        for profile in PROFILES:
            for lora in (False, True):
                with self.subTest(profile=profile, lora=lora):
                    model = tiny_model(profile, lora=lora, device="mps")
                    model.backbone.to(torch.bfloat16)  # the released dtype
                    self.assertEqual(off_device(model, "mps"), [])
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        with torch.inference_mode():
                            logits = model(rows) + model.score_cached(rows, batch_size=2)[0]
                        torch.mps.synchronize()
                    self.assertTrue(all(row.device.type == "mps" for row in logits))
                    fallbacks = [str(w.message) for w in caught if "fall back" in str(w.message)]
                    self.assertEqual(fallbacks, [])
        self.assertGreater(torch.mps.current_allocated_memory(), 0)

    def test_residency_check_detects_a_parameter_left_on_cpu(self):
        model = tiny_model(device="mps")
        model.head.to("cpu")
        self.assertEqual(off_device(model, "mps"), ["head.weight", "head.bias"])

    def test_float32_scores_match_cpu_within_the_existing_parity_bar(self):
        rows = records()
        for profile in PROFILES:
            for lora in (False, True):
                expected = score(tiny_model(profile, lora=lora), rows)
                model = tiny_model(profile, lora=lora, device="mps")
                for cached, ragged in ((False, False), (True, False), (True, True)):
                    with self.subTest(profile=profile, lora=lora, cached=cached, ragged=ragged), \
                         patch.dict(os.environ, {"JEV_RAGGED_SUFFIX": "1"} if ragged else {}):
                        self.assert_rows_close(score(model, rows, cached=cached), expected)

    def test_reduced_precision_error_is_no_worse_than_cpu_at_the_same_dtype(self):
        # bfloat16 and float16 are noisy on any backend; the MPS-specific claim
        # is only that Metal adds no error beyond what the dtype itself costs.
        rows = records()
        for dtype in (torch.bfloat16, torch.float16):
            for profile in PROFILES:
                for lora in (False, True):
                    with self.subTest(dtype=dtype, profile=profile, lora=lora):
                        cpu = tiny_model(profile, lora=lora)
                        reference = score(cpu, rows)
                        mps = tiny_model(profile, lora=lora, device="mps")
                        cpu.backbone.to(dtype)
                        mps.backbone.to(dtype)
                        actual = score(mps, rows)
                        self.assertTrue(all(torch.isfinite(row).all() for row in actual))
                        cpu_error = max_error(score(cpu, rows), reference)
                        self.assertLess(max_error(actual, reference), 4 * cpu_error + 1e-3)

    def test_mps_profile_timing_synchronizes_metal_not_cuda(self):
        model = tiny_model(device="mps")
        with patch.dict(os.environ, {"JEV_PROFILE": "1"}), \
             patch("torch.cuda.synchronize", side_effect=cuda_must_not_synchronize), \
             patch("torch.mps.synchronize", wraps=torch.mps.synchronize) as metal:
            _, stats = model.score_cached(records(), batch_size=2)
        self.assertGreater(metal.call_count, 0)
        self.assertIn("shared_prefill", stats["profile_seconds"])

    def test_latency_benchmark_synchronizes_metal_around_every_timed_call(self):
        # Metal dispatch is asynchronous: without a Metal synchronize the harness
        # would time kernel submission, not execution, and understate every call.
        from jev.serving import Predictor, TorchScorer
        from scripts.benchmark_inference_latency import benchmark, digest
        request = mixed_request()
        workloads = [{"id": "mixed", "request": request, "request_sha256": digest(request)}]
        predictor = Predictor(TorchScorer(tiny_model(device="mps")), model_name="tiny", method="fixture")
        with tempfile.TemporaryDirectory() as directory, \
             patch("torch.cuda.synchronize", side_effect=cuda_must_not_synchronize), \
             patch("torch.mps.synchronize", wraps=torch.mps.synchronize) as metal:
            result = benchmark(predictor, workloads, output=Path(directory), warmup=1, repetitions=1)
            samples = [json.loads(line) for line in (Path(directory) / "samples.jsonl").read_text().splitlines()]
        self.assertEqual(result["status"], "passed")
        self.assertEqual(len(samples), 8)  # (1 warmup + 1 measured) x cached/uncached x Predictor/HTTP
        self.assertGreaterEqual(metal.call_count, 2 * len(samples))
        for sample in samples:
            self.assertTrue(sample["success"])
            self.assertNotIn("peak_cuda_allocated_bytes", sample)
            for field in ("mps_allocated_bytes_at_start", "mps_allocated_bytes_at_end",
                          "mps_driver_allocated_bytes_at_end"):
                self.assertGreater(sample[field], 0)

    def test_cpu_and_mps_benchmark_runs_compare_as_equal_decisions(self):
        from jev.serving import Predictor, TorchScorer
        from scripts.benchmark_inference_latency import benchmark, digest
        from scripts.compare_device_reports import compare_runs
        request = mixed_request()
        workloads = [{"id": "mixed", "request": request, "request_sha256": digest(request)}]
        with tempfile.TemporaryDirectory() as directory:
            runs = {}
            for device in ("cpu", "mps"):
                runs[device] = Path(directory) / device
                runs[device].mkdir()
                predictor = Predictor(TorchScorer(tiny_model(device=device)), model_name="tiny", method="fixture")
                benchmark(predictor, workloads, output=runs[device], warmup=1, repetitions=1)
            result = compare_runs(runs["cpu"], runs["mps"])
            # A run that replayed different inputs must be refused, not compared.
            samples = runs["mps"] / "samples.jsonl"
            samples.write_text(samples.read_text().replace(digest(request), "0" * 64))
            with self.assertRaisesRegex(ValueError, "different request contents"):
                compare_runs(runs["cpu"], runs["mps"])
        self.assertEqual(result["requests"], 1)
        self.assertTrue(result["all_decisions_equal"])
        self.assertTrue(result["all_within_tolerance"])  # float32: Metal matches CPU closely
        self.assertLess(result["max_probability_error"], 1e-5)


if __name__ == "__main__":
    unittest.main()
