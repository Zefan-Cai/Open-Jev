"""CPU regressions for pinned shard resolution and supported embedding offload."""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from tests.test_prefix_cache import HAS_RUNTIME

if HAS_RUNTIME:
    import torch
    from torch import nn
    from safetensors.torch import save_file
    from jev.model import DecisionModel, _checkpoint_tensor_file
    from jev.api import compile_request
    from tests.test_prefix_cache import mixed_request, tiny_model


@unittest.skipUnless(HAS_RUNTIME, "requires torch, transformers and peft")
class ConsumerGpuLoadingTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def checkpoint(self, tensors, *, indexed=True):
        shard = self.root / ("model-00001-of-00001.safetensors" if indexed else "model.safetensors")
        save_file(tensors, shard)
        index = self.root / "model.safetensors.index.json"
        if indexed:
            index.write_text(json.dumps({"weight_map": {name: shard.name for name in tensors}}))

        def resolve(model_id, filename, **kwargs):
            self.assertEqual(model_id, "Qwen/Qwen3.5-9B")
            self.assertEqual(kwargs["revision"], "a" * 40)
            candidate = self.root / filename
            return str(candidate) if candidate.exists() else None
        return resolve

    def full(self, *, meta_head=False, meta_layer=False, mapping=None):
        class Backbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(hidden_size=4, use_cache=False)
                self.embed_tokens = nn.Embedding(4, 4)
                self.layers = nn.ModuleList([nn.Linear(4, 4, device="meta" if meta_layer else "cpu")])
                self.norm = nn.LayerNorm(4)

            def get_input_embeddings(self):
                return self.embed_tokens

        output = nn.Embedding(4, 4, device="meta" if meta_head else "cpu", dtype=torch.bfloat16)
        return SimpleNamespace(
            get_output_embeddings=lambda: output,
            model=SimpleNamespace(language_model=Backbone()),
            hf_device_map=mapping or {"lm_head": "cpu", "model.language_model": 0})

    def construct(self, full, *, env=None):
        tokenizer = SimpleNamespace(pad_token_id=0,
                                    encode=lambda text, **kwargs: [1 if text == "Yes" else 2])
        with patch("jev.model.AutoTokenizer.from_pretrained", return_value=tokenizer), \
             patch("jev.model.AutoModelForImageTextToText.from_pretrained", return_value=full), \
             patch.dict(os.environ, env or {"JEV_DEVICE_MAP": '{"lm_head":"cpu","model.language_model":0}'}):
            return DecisionModel("Qwen/Qwen3.5-9B", "a" * 40, device="cpu", lora_rank=0)

    def test_hub_id_meta_head_uses_pinned_shard_and_original_subtraction_dtype(self):
        weights = torch.tensor([[0] * 4, [1.0078125] * 4, [-0.30078125] * 4, [0] * 4],
                               dtype=torch.bfloat16)
        with patch("jev.model.cached_file", side_effect=self.checkpoint({"lm_head.weight": weights})) as resolve:
            model = self.construct(self.full(meta_head=True))
        torch.testing.assert_close(model.head.weight[0], (weights[1] - weights[2]).float(), atol=0, rtol=0)
        self.assertEqual(resolve.call_count, 2)

    def test_unsharded_snapshot_resolution(self):
        with patch("jev.model.cached_file", side_effect=self.checkpoint({"lm_head.weight": torch.zeros(4, 4)}, indexed=False)):
            filename = _checkpoint_tensor_file("Qwen/Qwen3.5-9B", "a" * 40, "lm_head.weight")
        self.assertEqual(Path(filename).name, "model.safetensors")

    def test_local_snapshot_resolution(self):
        self.checkpoint({"lm_head.weight": torch.zeros(4, 4)})
        filename = _checkpoint_tensor_file(str(self.root), "unused-local-revision", "lm_head.weight")
        self.assertEqual(Path(filename).parent, self.root)

    def test_explicit_transformer_cpu_offload_is_rejected(self):
        mapping = {"model.language_model.layers.0": "cpu", "model.language_model.norm": 0}
        with self.assertRaisesRegex(RuntimeError, "refusing to serve"):
            self.construct(self.full(meta_layer=True, mapping=mapping),
                           env={"JEV_DEVICE_MAP": json.dumps(mapping)})

    def test_similar_embedding_module_name_does_not_bypass_guard(self):
        mapping = {"model.language_model.embed_tokens_extra": "cpu", "model.language_model": 0}
        with self.assertRaisesRegex(RuntimeError, "refusing to serve"):
            self.construct(self.full(mapping=mapping), env={"JEV_DEVICE_MAP": json.dumps(mapping)})

    def test_allow_offload_never_allows_unloaded_parameters(self):
        mapping = {"model.language_model.layers.0": "cpu", "model.language_model.norm": 0}
        with self.assertRaisesRegex(RuntimeError, "unloaded meta parameters"):
            self.construct(self.full(meta_layer=True, mapping=mapping),
                           env={"JEV_DEVICE_MAP": json.dumps(mapping), "JEV_ALLOW_OFFLOAD": "1"})

    def test_cpu_embedding_is_reused_by_uncached_and_cached_scoring(self):
        model = tiny_model()
        request = mixed_request()
        records = compile_request(request["state"], request["questions"])
        with torch.inference_mode():
            expected = model(records)
        core = model.backbone.get_base_model()
        weight = core.embed_tokens.weight.detach().clone()
        core.embed_tokens = nn.Embedding(weight.shape[0], weight.shape[1], device="meta")
        model.device_name = "meta"
        model.model_id, model.revision = "Qwen/Qwen3.5-9B", "a" * 40
        with patch("jev.model.cached_file", side_effect=self.checkpoint(
                {"model.language_model.embed_tokens.weight": weight})) as resolve:
            with torch.inference_mode():
                uncached = model(records)
            first, _ = model.score_cached(records, batch_size=2)
            second, _ = model.score_cached(records, batch_size=2)
            model._validate_loaded_parameters()
        self.assertEqual(resolve.call_count, 2, "the CPU embedding should load only once")
        for actual in (uncached, first, second):
            for left, right in zip(actual, expected):
                torch.testing.assert_close(left, right, atol=2e-5, rtol=2e-5)

    def test_load_revalidates_adapter_parameters(self):
        config = {"model_id": "mock", "revision": "mock", "lora_rank": 8, "max_length": 4096}
        (self.root / "model.json").write_text(json.dumps(config))
        torch.save(nn.Linear(4, 1).state_dict(), self.root / "head.pt")
        parent = self

        class StubModel(DecisionModel):
            def __init__(self, *args, **kwargs):
                nn.Module.__init__(self)
                self.backbone = parent.full().model.language_model
                self.head = nn.Linear(4, 1)
                self.device_name = "cpu"

        broken = self.full(meta_layer=True).model.language_model
        with patch("peft.PeftModel.from_pretrained", return_value=broken):
            with self.assertRaisesRegex(RuntimeError, "unloaded meta parameters"):
                StubModel.load(self.root, device="cpu")

    def load(self, *, device="cpu", env=None, loader=None):
        """Construct unsharded; the from_pretrained mock is returned for inspection."""
        loader = loader or MagicMock(return_value=self.full())
        tokenizer = SimpleNamespace(pad_token_id=0,
                                    encode=lambda text, **kwargs: [1 if text == "Yes" else 2])
        with patch("jev.model.AutoTokenizer.from_pretrained", return_value=tokenizer), \
             patch("jev.model.AutoModelForImageTextToText.from_pretrained", loader), \
             patch.dict(os.environ, env or {}):
            DecisionModel("Qwen/Qwen3.5-9B", "a" * 40, device=device, lora_rank=0)
        return loader

    def test_backbone_dtype_defaults_to_released_bfloat16_and_is_selectable(self):
        self.assertIs(self.load().call_args.kwargs["torch_dtype"], torch.bfloat16)
        for name, dtype in (("bfloat16", torch.bfloat16), ("float16", torch.float16),
                            ("float32", torch.float32)):
            with self.subTest(dtype=name):
                loader = self.load(env={"JEV_TORCH_DTYPE": name})
                self.assertIs(loader.call_args.kwargs["torch_dtype"], dtype)

    def test_unknown_backbone_dtype_is_rejected_before_loading(self):
        loader = MagicMock(return_value=self.full())
        with self.assertRaisesRegex(ValueError, "JEV_TORCH_DTYPE must be one of"):
            self.load(env={"JEV_TORCH_DTYPE": "float64"}, loader=loader)
        loader.assert_not_called()

    def test_quantized_loading_is_refused_on_apple_mps_before_loading(self):
        # torch.device("mps") needs no Apple hardware, so this runs everywhere.
        for flag in ("JEV_LOAD_8BIT", "JEV_LOAD_4BIT"):
            with self.subTest(flag=flag):
                loader = MagicMock(return_value=self.full())
                with self.assertRaisesRegex(ValueError, "no Apple MPS backend"):
                    self.load(device="mps", env={flag: "1"}, loader=loader)
                loader.assert_not_called()

    def test_provenance_records_loaded_rather_than_requested_backbone_dtypes(self):
        from jev.serving import load_predictor
        tokenizer = SimpleNamespace(pad_token_id=0,
                                    encode=lambda text, **kwargs: [1 if text == "Yes" else 2])
        with patch("jev.model.AutoTokenizer.from_pretrained", return_value=tokenizer), \
             patch("jev.model.AutoModelForImageTextToText.from_pretrained", return_value=self.full()):
            predictor = load_predictor(model_id="Qwen/Qwen3.5-9B", revision="a" * 40, device="cpu")
        # bfloat16 was requested, but this stand-in backbone is float32: record what loaded.
        self.assertEqual(predictor.provenance["backbone_parameter_dtypes"], ["torch.float32"])


if __name__ == "__main__":
    unittest.main()
