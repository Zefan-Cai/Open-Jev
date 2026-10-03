import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from jev import data, metrics, train


def arguments(root, deferred=True):
    return argparse.Namespace(model="fixture", revision="a" * 40, data=str(root / "data"),
                              output=str(root / "output"), steps=2, train_rows=1,
                              calibration_rows=1, eval_rows=1, max_length=128, lora_rank=8,
                              accumulation=1, lr=2e-5, head_lr=5e-5, brier_weight=0.1,
                              seed=42, training_sampling="shuffled", checkpoint_every=0,
                              resume_training=None, initial_checkpoint=None, defer_heldout=deferred)


def fixture_row(split, index):
    ident = f"{split}-{index}"
    return {"id": ident, "group_id": ident, "split": split, "source": "fixture",
            "state": {"fixture": ident}, "question": "Does the fixture establish yes?",
            "kind": "noul", "options": ["no", "yes"], "target": [0, 1],
            "metadata": {"family": "policy", "template_id": split,
                         "provenance": {"type": "synthetic", "license": "CC0-1.0",
                                        "split_policy": "fixture groups", "generator_version": "fixture",
                                        "seed": 42, "group_index": index, "variant": 0}}}


class Value:
    """Arithmetic stand-in: these tests check orchestration, not numerical training."""
    device = "mock-device"

    def float(self):
        return self

    def log_softmax(self, dimension):
        return self

    softmax = log_softmax

    def sum(self):
        return self

    def item(self):
        return 0.25

    def backward(self):
        pass

    def __neg__(self):
        return self

    def __mul__(self, other):
        return self

    __rmul__ = __mul__
    __add__ = __mul__
    __sub__ = __mul__
    __truediv__ = __mul__
    __pow__ = __mul__


class DeferredHeldoutTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.dataset = self.root / "data"
        self.dataset.mkdir()
        for split in ("train", "calibration", "validation", "test", "ood"):
            count = 2 if split == "train" else 1
            (self.dataset / f"{split}.jsonl").write_text(
                "".join(json.dumps(fixture_row(split, index)) + "\n" for index in range(count)))

    @contextlib.contextmanager
    def runtime(self, deferred=True, reload_logits=None, calibration_logits=None):
        self.forwards, self.saves, self.reads = [], [], []
        self.instances = []
        forwards, saves, instances = self.forwards, self.saves, self.instances
        parameter = types.SimpleNamespace(requires_grad=True, numel=lambda: 2)

        class Model:
            def __init__(model, *args, **kwargs):
                model.training = False
                model.backbone = model.head = types.SimpleNamespace(parameters=lambda: [parameter])
                instances.append(model)

            def parameters(model):
                return [parameter]

            def train(model):
                model.training = True

            def eval(model):
                model.training = False

            def __call__(model, rows):
                for row in rows:
                    if deferred and row["split"] not in ("train", "calibration"):
                        raise AssertionError("Forbidden heldout forward: " + row["split"])
                    forwards.append((row["split"], model.training, row["id"]))
                return [Value() for row in rows]

            def save(model, path):
                saves.append(Path(path))
                Path(path).mkdir()

            @classmethod
            def load(cls, path):
                return cls()

        def evaluate(model, rows, path):
            model.eval()
            model(rows)
            logits = [0.1, 0.2]
            if Path(path).name == "calibration.jsonl" and calibration_logits is not None:
                logits = calibration_logits
            if Path(path).name == "reload_check.jsonl" and reload_logits is not None:
                logits = reload_logits
            values = [{**{key: row[key] for key in ("id", "group_id", "source", "kind", "target")},
                       "question_id": row["kind"], "logits": logits, "latency_seconds": 0.01}
                      for row in rows]
            Path(path).write_text("".join(json.dumps(row) + "\n" for row in values))
            return values

        torch = types.ModuleType("torch")
        torch.__version__ = "mock-torch"
        torch.manual_seed = Mock()
        torch.inference_mode = contextlib.nullcontext
        torch.tensor = Mock(return_value=Value())
        torch.isfinite = Mock(return_value=True)
        torch.cuda = types.SimpleNamespace(synchronize=Mock(), reset_peak_memory_stats=Mock(),
                                          max_memory_allocated=Mock(return_value=3 * 2**30),
                                          max_memory_reserved=Mock(return_value=4 * 2**30),
                                          get_device_name=Mock(return_value="mock GPU"), empty_cache=Mock())
        optimizer = types.SimpleNamespace(zero_grad=Mock(), step=Mock())
        torch.optim = types.SimpleNamespace(AdamW=Mock(return_value=optimizer))
        torch.nn = types.SimpleNamespace(utils=types.SimpleNamespace(clip_grad_norm_=Mock(return_value=Value())))
        modules = {"torch": torch, "jev.model": types.ModuleType("jev.model")}
        modules["jev.model"].DecisionModel = Model
        for name in ("transformers", "transformers.models", "transformers.models.qwen3_5",
                     "transformers.models.qwen3_5.modeling_qwen3_5"):
            modules[name] = types.ModuleType(name)
        modules["transformers"].__version__ = "mock-transformers"
        modules["transformers.models.qwen3_5.modeling_qwen3_5"].is_fast_path_available = False

        original_path_open = Path.open
        original_open = open

        def guard(file):
            if isinstance(file, (str, Path)) and Path(file).parent == self.dataset:
                self.reads.append(Path(file).stem)
                if deferred and Path(file).stem in ("validation", "test", "ood"):
                    raise AssertionError("Forbidden heldout read: " + str(file))

        def guarded_path_open(path, *args, **kwargs):
            guard(path)
            return original_path_open(path, *args, **kwargs)

        def guarded_open(file, *args, **kwargs):
            guard(file)
            return original_open(file, *args, **kwargs)

        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, modules))
            stack.enter_context(patch.object(train, "source_checkout_commit", return_value="c" * 40))
            stack.enter_context(patch("importlib.metadata.version", return_value="mock-version"))
            stack.enter_context(patch.object(train, "evaluate", side_effect=evaluate))
            calibration_fit = stack.enter_context(patch.object(metrics, "fit_temperature", return_value=1.25))
            validation = stack.enter_context(patch.object(data, "validate_records", wraps=data.validate_records))
            directory_read = stack.enter_context(patch.object(data, "read_split_directory", wraps=data.read_split_directory))
            stack.enter_context(patch.object(Path, "open", guarded_path_open))
            stack.enter_context(patch("builtins.open", guarded_open))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            yield torch, optimizer, calibration_fit, validation, directory_read

    def test_deferred_run_never_reads_or_forwards_heldout_and_saves_fixed_final(self):
        for split in ("validation", "test", "ood"):
            (self.dataset / f"{split}.jsonl").write_text("not JSON; forbidden to parse\n")
        args = arguments(self.root)
        with self.runtime() as (torch, optimizer, fit, validation, directory_read):
            train.run(args)
            directory_read.assert_not_called()
            self.assertEqual({row["id"] for row in validation.call_args.args[0]},
                             {"train-0", "train-1", "calibration-0"})
            self.assertEqual(set(self.reads), {"train", "calibration"})
            self.assertEqual(optimizer.step.call_count, args.steps)
            self.assertEqual(fit.call_count, 1)
            self.assertEqual(self.forwards[0][0:2], ("train", False))
            self.assertEqual(sum(training for split, training, ident in self.forwards), args.steps)
            self.assertEqual({split for split, training, ident in self.forwards}, {"train", "calibration"})
            self.assertEqual(self.saves, [self.root / "output/checkpoint"])
            self.assertEqual(torch.cuda.reset_peak_memory_stats.call_count, 6)
        output = self.root / "output"
        summary, meta = [json.loads((output / name).read_text()) for name in ("summary.json", "run.json")]
        self.assertEqual(summary["metrics"], {})
        self.assertIsNone(summary["baseline_temperature"])
        self.assertEqual(summary["checkpoint_selection"], "fixed_final_step")
        self.assertEqual(summary["checkpoint_reload_split"], "calibration")
        self.assertEqual(summary["heldout_evaluation"],
                         {"status": "deferred", "splits": ["validation", "test", "ood"], "model_calls": 0})
        self.assertEqual(set(meta["data_sha256"]), {"train", "calibration"})
        self.assertEqual(meta["evaluation_ids"], [])
        self.assertEqual(meta["ood_ids"], [])
        self.assertEqual(meta["data_validation_splits"], ["train", "calibration"])
        self.assertEqual(set(summary["phase_metrics"]), {"model_loading", "warmup", "optimizer",
                                                       "checkpoint_save", "calibration_and_temperature", "checkpoint_reload"})
        for phase in summary["phase_metrics"].values():
            self.assertGreaterEqual(phase["elapsed_seconds"], 0)
            self.assertEqual(phase["peak_allocated_tensor_memory_gib"], 3)
            self.assertEqual(phase["peak_reserved_allocator_memory_gib"], 4)
        self.assertFalse(any((output / name).exists() for name in train.BASELINE_FILES))
        self.assertFalse((output / "trained_test.jsonl").exists())
        self.assertFalse((output / "trained_ood.jsonl").exists())

    def test_complete_allowed_splits_are_validated_before_sampling_or_model_construction(self):
        rows = [fixture_row("train", 0), fixture_row("train", 1)]
        rows[1]["target"] = [0, 0]
        (self.dataset / "train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        with self.runtime(), self.assertRaisesRegex(ValueError, "target must sum to one"):
            train.run(arguments(self.root))
        self.assertEqual(self.instances, [])

    def test_deferred_reload_rejects_nonfinite_width_and_value_mismatches_without_summary(self):
        cases = [([float("nan"), 0.2], None, "finite numbers"),
                 ([0.1, float("inf")], None, "finite numbers"),
                 ([0.1], None, "equal nonempty lengths"),
                 ([], None, "equal nonempty lengths"),
                 ([0.1, 0.2, 0.3], None, "equal nonempty lengths"),
                 ([0.3, 0.2], None, "reload mismatch"),
                 ([0.1, 0.2], [float("nan"), 0.2], "finite numbers")]
        for index, (reload_logits, calibration_logits, message) in enumerate(cases):
            args = arguments(self.root)
            args.output = str(self.root / f"failed-reload-{index}")
            with self.subTest(reload_logits=reload_logits, calibration_logits=calibration_logits), \
                    self.runtime(reload_logits=reload_logits, calibration_logits=calibration_logits), \
                    self.assertRaisesRegex(ValueError, message):
                train.run(args)
            self.assertFalse((Path(args.output) / "summary.json").exists())
            self.assertEqual({split for split, training, ident in self.forwards}, {"train", "calibration"})

    def test_deferred_run_requires_calibration(self):
        (self.dataset / "calibration.jsonl").write_text("")
        with self.runtime(), self.assertRaisesRegex(ValueError, "required split must be nonempty"):
            train.run(arguments(self.root))
        self.assertEqual(self.instances, [])

    def test_default_keeps_all_split_validation_baselines_and_heldout_evaluation(self):
        with self.runtime(deferred=False) as (torch, optimizer, fit, validation, directory_read):
            train.run(arguments(self.root, deferred=False))
            directory_read.assert_called_once()
            self.assertEqual(set(self.reads), {"train", "calibration", "validation", "test", "ood"})
            self.assertEqual(fit.call_count, 2)
            self.assertEqual(self.forwards[0][0:2], ("test", False))
            torch.cuda.reset_peak_memory_stats.assert_not_called()
            torch.cuda.max_memory_reserved.assert_not_called()
        summary = json.loads((self.root / "output/summary.json").read_text())
        self.assertEqual(len(summary["metrics"]), 8)
        self.assertNotIn("heldout_evaluation", summary)
        self.assertNotIn("phase_metrics", summary)
        self.assertTrue(all((self.root / "output" / name).exists() for name in train.BASELINE_FILES))

    def test_deferred_resume_and_snapshots_rejected_before_runtime_imports(self):
        for change in ({"checkpoint_every": 1}, {"resume_training": "old-snapshot"}):
            args = arguments(self.root)
            vars(args).update(change)
            with patch.object(train, "source_checkout_commit", return_value="c" * 40), \
                    patch.object(train, "initial_checkpoint_identity", side_effect=AssertionError("Too late")), \
                    self.assertRaisesRegex(ValueError, "defer-heldout requires"):
                train.run(args)

    def test_opt_in_changes_identity_without_changing_default_argument_schema(self):
        args = arguments(self.root, deferred=False)
        baseline = train.training_identity(args, {}, {}, {})
        self.assertNotIn("defer_heldout", baseline["arguments"])
        args.defer_heldout = True
        deferred = train.training_identity(args, {}, {}, {})
        self.assertIs(deferred["arguments"]["defer_heldout"], True)
        with self.assertRaisesRegex(ValueError, "identity differs"):
            train.validate_resume_identity(baseline, deferred)

    def test_cli_defaults_to_legacy_and_exposes_explicit_opt_in(self):
        argv = ["trainer", "--model", "fixture", "--revision", "pinned", "--data", "data", "--output", "out"]
        for suffix in ([], ["--defer-heldout"]):
            with patch.object(sys, "argv", argv + suffix), patch.object(train, "run") as run:
                train.main()
            self.assertEqual(run.call_args.args[0].defer_heldout, bool(suffix))


if __name__ == "__main__":
    unittest.main()
