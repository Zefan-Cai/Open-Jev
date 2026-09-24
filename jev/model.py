"""Independent candidate scoring; no autoregressive generation or JSON decoding."""
from __future__ import annotations

import json
from pathlib import Path

import os
import torch
from torch import nn
from transformers import AutoModelForImageTextToText, AutoTokenizer
from transformers.utils.hub import cached_file

from .api import candidate_prompts


def _checkpoint_tensor_file(model_id, revision, tensor_key):
    """Resolve a pinned safetensors shard from a Hub repo ID or local snapshot."""
    index_path = cached_file(model_id, "model.safetensors.index.json", revision=revision,
                             _raise_exceptions_for_missing_entries=False)
    if index_path is None:
        filename = "model.safetensors"
    else:
        weight_map = json.loads(Path(index_path).read_text())["weight_map"]
        if tensor_key not in weight_map:
            raise ValueError(f"Checkpoint has no tensor {tensor_key}")
        filename = weight_map[tensor_key]
    return cached_file(model_id, filename, revision=revision)


def _module_under(name, parent):
    return name == parent or name.startswith(parent + ".")


# JEV_TORCH_DTYPE selects the backbone dtype; unset keeps the released bfloat16.
BACKBONE_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


class DecisionModel(nn.Module):
    def __init__(self, model_id, revision, device="cuda:0", lora_rank=8, max_length=384):
        super().__init__()
        self.model_id, self.revision = model_id, revision
        self.max_length, self.device_name = max_length, device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Optional quantized loading: JEV_LOAD_8BIT=1 or JEV_LOAD_4BIT=1.
        # bitsandbytes ships prebuilt kernels, so no CUDA toolkit is needed.
        # ⚠️ Quantization changes the weights the decision head was trained
        # against, so quality from this path is NOT comparable to the released
        # numbers. On 16 GB cards prefer the exact bf16 single-GPU layout below
        # (JEV_DEVICE_MAP + --prefix-cache); measured, it is also faster than
        # LLM.int8 (4.4 s vs 18.2 s per 21-question record on one card).
        # Optional backbone dtype: JEV_TORCH_DTYPE=bfloat16|float16|float32.
        # Unset keeps bfloat16, the dtype the released head and temperature were
        # fitted against; any other dtype changes numerics, so, as with
        # quantization, its quality is NOT comparable to the released numbers.
        _dtype = os.environ.get("JEV_TORCH_DTYPE") or "bfloat16"
        if _dtype not in BACKBONE_DTYPES:
            raise ValueError("JEV_TORCH_DTYPE must be one of: " + ", ".join(BACKBONE_DTYPES))
        if torch.device(device).type == "mps" and "1" in (os.environ.get("JEV_LOAD_8BIT"),
                                                          os.environ.get("JEV_LOAD_4BIT")):
            raise ValueError("bitsandbytes has no Apple MPS backend; unset JEV_LOAD_8BIT and JEV_LOAD_4BIT")
        _q = None
        if os.environ.get("JEV_LOAD_8BIT") == "1":
            from transformers import BitsAndBytesConfig
            _q = BitsAndBytesConfig(load_in_8bit=True)
        elif os.environ.get("JEV_LOAD_4BIT") == "1":
            from transformers import BitsAndBytesConfig
            _q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                    bnb_4bit_use_double_quant=True,
                                    bnb_4bit_compute_dtype=torch.bfloat16)
        # Device placement: JEV_DEVICE_MAP=auto|balanced|sequential hands
        # placement to accelerate (e.g. to span two cards); a JSON object gives
        # an explicit module -> device map. JEV_MAX_MEMORY="0=12GiB,1=11GiB"
        # caps each card. Note torch's device order can differ from nvidia-smi's.
        # When unset, the original single-device path is taken unchanged.
        _dm = os.environ.get("JEV_DEVICE_MAP")
        _kw = {}
        if _dm:
            # accept an explicit JSON device map,
            # not just "auto"/"balanced". Needed to fit the 9B on ONE card in
            # bf16: only `model.language_model` (12.89 GiB layers + 1.89 GiB
            # embeddings = 14.78 GiB) survives `del full` below. The vision
            # tower, `mtp` and `lm_head` (3.19 GiB together) are dropped right
            # after init — lm_head is read once for the Yes/No rows — so parking
            # them on CPU costs nothing and changes no output. Loading them onto
            # the GPU is the ONLY reason the full 17.98 GiB ever had to fit.
            if _dm.lstrip().startswith("{"):
                _parsed = json.loads(_dm)
                _kw["device_map"] = _parsed
            else:
                _kw["device_map"] = _dm
            _mm = os.environ.get("JEV_MAX_MEMORY")
            if _mm:
                _kw["max_memory"] = {
                    (int(k) if k.strip().isdigit() else k.strip()): v.strip()
                    for k, v in (part.split("=", 1) for part in _mm.split(","))}
        else:
            _kw["device_map"] = {"": device}
        full = AutoModelForImageTextToText.from_pretrained(
            model_id, revision=revision,
            **({"quantization_config": _q} if _q else
               {"torch_dtype": BACKBONE_DTYPES[_dtype]}),
            attn_implementation="sdpa", **_kw,
        )
        # Initialize a discriminative scalar from the pretrained Yes/No readout.
        yes = self.tokenizer.encode("Yes", add_special_tokens=False)
        no = self.tokenizer.encode("No", add_special_tokens=False)
        if len(yes) != 1 or len(no) != 1:
            raise ValueError("The Yes/No baseline requires single-token labels")
        _ow = full.get_output_embeddings().weight
        if _ow.device.type == "meta":
            # a device map that parks `lm_head` on
            # CPU leaves its weight a meta tensor, and the original expression
            # dies with "Cannot copy out of meta tensor". Read ONLY the two rows
            # the head is seeded from, straight out of the checkpoint shard —
            # identical bytes, no 1.89 GiB materialisation, no output change.
            from safetensors import safe_open
            _path = _checkpoint_tensor_file(model_id, revision, "lm_head.weight")
            with safe_open(_path, framework="pt") as _fh:
                _rows = _fh.get_slice("lm_head.weight")
                initial = (_rows[yes[0]:yes[0] + 1][0] -
                           _rows[no[0]:no[0] + 1][0]).detach().float().clone()
        else:
            initial = (_ow[yes[0]] - _ow[no[0]]).detach().float().clone()
        self.backbone = full.model.language_model
        hidden_size = self.backbone.config.hidden_size
        self.hf_device_map = dict(getattr(full, "hf_device_map", {}) or {})
        del full
        self.backbone.config.use_cache = False
        self.backbone.requires_grad_(False)
        if _dm:
            # Inputs must land where the embedding sits; the head must sit where
            # the LAST hidden state comes out. With one card those are the same
            # device and this is a no-op.
            _in = next(self.backbone.get_input_embeddings().parameters()).device
            _tail = getattr(self.backbone, "norm", None) or list(self.backbone.layers)[-1]
            _out = next(_tail.parameters()).device
            self.device_name, device = str(_in), str(_out)
            self.sharded = True
            # Print WHERE the shards actually landed. A device map that silently
            # put everything on one card looks identical to a working split from
            # the outside — the request succeeds either way.
            import collections as _c
            _cnt = _c.Counter(str(v) for v in self.hf_device_map.values())
            print("[jev] sharded: input=%s head=%s modules_per_device=%s"
                  % (_in, _out, dict(_cnt)), flush=True)
            # ⚠️ REFUSE TO SERVE A MODEL THAT IS NOT ACTUALLY LOADED.
            # device_map="auto" is BALANCED, not fill-first: with 18 GiB of bf16
            # weights and max_memory 0=14GiB,1=5GiB it put 13 module groups on
            # cuda:0, 12 on cuda:1 and *12 on disk*, left the trained decision
            # head on `meta`, and then copied the LoRA adapter into meta
            # parameters — "a no-op", i.e. a third of the adapter was never
            # applied. The server still answered /health with "ready" and would
            # have returned confident numbers from an unloaded model. Nothing
            # downstream could have told the difference. Use "sequential" to
            # fill card 0 before card 1; JEV_ALLOW_OFFLOAD=1 opts back in.
            # judge ONLY the modules that survive
            # `del full`. `self.backbone` is `full.model.language_model`; the
            # vision tower, `mtp` and `lm_head` are dropped right after init, so
            # parking THOSE on CPU is deliberate (it is what lets the 9B run in
            # bf16 on one card) and cannot leave a meta tensor in the forward
            # path. Anything under the backbone on cpu/disk/meta is still fatal
            # and still refuses, which is the case the guard was written for.
            _discarded = ("model.visual", "visual", "mtp", "lm_head")
            _embedding = "model.language_model.embed_tokens"
            _resident = _c.Counter(
                str(value) for name, value in self.hf_device_map.items()
                if not any(_module_under(name, parent) for parent in _discarded)
                and not (_module_under(name, _embedding) and str(value) == "cpu"))
            _bad = {d for d in _resident if d in ("cpu", "disk", "meta")}
            if (_bad or str(_out) == "meta") and os.environ.get("JEV_ALLOW_OFFLOAD") != "1":
                raise RuntimeError(
                    "refusing to serve: %s modules offloaded to %s and head on %s "
                    "— the decision head and part of the LoRA adapter would be "
                    "meta tensors. Raise JEV_MAX_MEMORY, use "
                    "JEV_DEVICE_MAP=sequential, or set JEV_ALLOW_OFFLOAD=1."
                    % (sum(_cnt[d] for d in _bad), sorted(_bad) or "-", _out))
        else:
            self.sharded = False
        self.head = nn.Linear(hidden_size, 1, bias=True, device=device, dtype=torch.float32)
        with torch.no_grad():
            self.head.weight.copy_(initial.unsqueeze(0))
            self.head.bias.zero_()
        del initial
        if lora_rank:
            from peft import LoraConfig, get_peft_model
            targets = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "out_proj"]
            available = {name.rsplit(".", 1)[-1] for name, _ in self.backbone.named_modules()}
            targets = [name for name in targets if name in available]
            if not targets:
                raise ValueError("No intended LoRA target modules found")
            self.backbone = get_peft_model(self.backbone, LoraConfig(
                r=lora_rank, lora_alpha=lora_rank * 2, target_modules=targets,
                lora_dropout=0.0, bias="none",
            ))
            self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.backbone.enable_input_require_grads()
        self.lora_rank = lora_rank
        if str(self.device_name) == "meta":
            self._cpu_embedding_weight()
        self._validate_loaded_parameters()

    def _cpu_embedding_weight(self):
        """Keep the supported CPU embedding fallback across cached requests."""
        if getattr(self, "_cpu_embed", None) is None:
            from safetensors import safe_open
            key = "model.language_model.embed_tokens.weight"
            path = _checkpoint_tensor_file(self.model_id, self.revision, key)
            with safe_open(path, framework="pt") as checkpoint:
                self._cpu_embed = checkpoint.get_tensor(key)
        return self._cpu_embed

    def _validate_loaded_parameters(self):
        # A CPU embedding is the only supported meta parameter: its actual
        # weights have been loaded above and both scoring paths bypass it.
        allowed = set()
        if str(self.device_name) == "meta" and getattr(self, "_cpu_embed", None) is not None:
            allowed = {id(value) for value in self.backbone.get_input_embeddings().parameters()}
        missing = [name for name, value in self.named_parameters()
                   if value.device.type == "meta" and id(value) not in allowed]
        if missing:
            raise RuntimeError("refusing to serve unloaded meta parameters: " + ", ".join(missing))

    def forward(self, records):
        prompts, counts = [], []
        for record in records:
            entries = candidate_prompts(record)
            counts.append(len(entries))
            for prompt in entries:
                prompts.append(self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}], tokenize=False,
                    add_generation_prompt=True, enable_thinking=False,
                ))
        encoded = self.tokenizer(prompts, padding=True, truncation=False, return_tensors="pt")
        lengths = encoded["attention_mask"].sum(-1)
        self.last_input_tokens = int(lengths.sum().item())
        if lengths.max().item() > self.max_length:
            raise ValueError(f"Input length {lengths.max().item()} exceeds max_length={self.max_length}; no silent truncation")
        if str(self.device_name) == "meta":
            # embedding offloaded to CPU. accelerate
            # would stream the 1.89 GiB matrix onto the GPU every call; look the
            # rows up on CPU and pass inputs_embeds instead. Same values -- an
            # embedding is a gather -- so no output changes. Mirrors the cached
            # path in prefix_cache.py.
            _cpu_embed = self._cpu_embedding_weight()
            _core = self.backbone.get_base_model() if hasattr(self.backbone, "get_base_model") else self.backbone
            _dev = next(_core.layers[0].parameters()).device
            _embeds = torch.nn.functional.embedding(encoded["input_ids"], _cpu_embed).to(_dev)
            outputs = self.backbone(inputs_embeds=_embeds, attention_mask=encoded["attention_mask"].to(_dev),
                                    use_cache=False, return_dict=True)
        else:
            encoded = {k: v.to(self.device_name) for k, v in encoded.items()}
            outputs = self.backbone(**encoded, use_cache=False, return_dict=True)
        # under sharding the last hidden state comes out on the
        # LAST device, not self.device_name. Indexing it with a tensor built on
        # the input device raises; the head then needs the same device again.
        hdev = outputs.last_hidden_state.device
        idx = torch.arange(len(prompts), device=hdev)
        hidden = outputs.last_hidden_state[idx, lengths.to(hdev) - 1]
        scores = self.head(hidden.float().to(self.head.weight.device)).squeeze(-1)
        logits, offset = [], 0
        for record, count in zip(records, counts):
            values = scores[offset:offset + count]
            if record["kind"] == "noul":
                values = torch.stack([torch.zeros_like(values[0]), values[0]])
            logits.append(values)
            offset += count
        return logits

    def save(self, output):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        if self.lora_rank:
            self.backbone.save_pretrained(output / "adapter")
        torch.save(self.head.state_dict(), output / "head.pt")
        (output / "model.json").write_text(json.dumps({
            "model_id": self.model_id, "revision": self.revision,
            "max_length": self.max_length, "lora_rank": self.lora_rank,
            "method": "independent_candidate_lora_nll_brier",
        }, indent=2) + "\n")

    def score_cached(self, records, *, batch_size=32):
        """Inference-only prefix reuse; training forward and saved weights stay unchanged."""
        from .prefix_cache import score_cached
        return score_cached(self, records, batch_size=batch_size)

    @classmethod
    def load(cls, output, device="cuda:0"):
        output = Path(output)
        config = json.loads((output / "model.json").read_text())
        model = cls(config["model_id"], config["revision"], device=device,
                    lora_rank=0, max_length=config["max_length"])
        if config["lora_rank"]:
            from peft import PeftModel
            model.backbone = PeftModel.from_pretrained(model.backbone, output / "adapter")
            model.lora_rank = config["lora_rank"]
        model.head.load_state_dict(torch.load(output / "head.pt", map_location=device, weights_only=True))
        model._validate_loaded_parameters()
        return model.eval()
