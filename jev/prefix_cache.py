"""Request-local token-prefix reuse for the pinned Qwen hybrid inference stack.

Full attention KV and linear-attention conv/recurrent states are all mutable.
Saved prefix caches remain immutable; each continuation gets its own tensors.
This saves prefix computation, not the memory copies required by HF branching.
"""
import collections
import os
import time
from collections import defaultdict
import copy

import torch
from transformers.cache_utils import DynamicCache, DynamicLayer, LinearAttentionLayer

from .api import candidate_prompts


class RecurrentDtypeLayer(LinearAttentionLayer):
    """Retain the recurrent kernel's dtype independently of convolution dtype.

    Transformers 5.10.2 allocates recurrent state with the convolution dtype.
    On BF16 models this rounds the FP32 delta-rule state at every prefix split;
    an unsplit forward keeps that state in FP32. Preserve the returned dtype.
    """

    def lazy_initialization(self, conv_states=None, recurrent_states=None):
        if conv_states is not None:
            super().lazy_initialization(conv_states=conv_states)
        if recurrent_states is not None:
            self.recurrent_states = torch.zeros_like(recurrent_states)
            self.is_recurrent_states_initialized = True


def new_cache(config):
    cache = DynamicCache(config=config)
    cache.layers = [RecurrentDtypeLayer(config) if type(layer) is LinearAttentionLayer else layer
                    for layer in cache.layers]
    return cache


def common_prefix_length(sequences):
    """Find an exact token LCP, leaving at least one final token to score."""
    if not sequences or any(not sequence for sequence in sequences):
        raise ValueError("Prefix scoring requires nonempty token sequences")
    first = sequences[0]
    limit = min(map(len, sequences)) - 1
    for index in range(limit):
        if any(sequence[index] != first[index] for sequence in sequences[1:]):
            return index
    return limit


def fork_cache(cache, batch_size):
    """Copy a one-sequence Qwen cache into independent candidate branches.

    DynamicCache.batch_repeat_interleave is insufficient in transformers 5.10.2:
    LinearAttentionLayer does not implement it. Shallow or KV-only copies also
    alias the in-place recurrent/conv updates and contaminate later candidates.
    """
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Cache branch batch size must be a positive integer")
    if type(cache) is not DynamicCache or cache.offloading:
        raise TypeError("Prefix reuse supports non-offloaded DynamicCache only")
    branch = copy.copy(cache)
    branch.layers = []
    for layer in cache.layers:
        if type(layer) is DynamicLayer:
            fields = ("keys", "values")
        elif type(layer) in (LinearAttentionLayer, RecurrentDtypeLayer):
            fields = ("conv_states", "recurrent_states")
        else:
            raise TypeError(f"Unsupported prefix cache layer: {type(layer).__name__}")
        cloned = copy.copy(layer)
        for name in fields:
            tensor = getattr(layer, name)
            if tensor is None:
                continue
            if tensor.shape[0] != 1:
                raise ValueError("Saved prefix cache must have exactly one sequence")
            # repeat_interleave allocates independent storage even for batch=1.
            setattr(cloned, name, tensor.repeat_interleave(batch_size, dim=0))
        if hasattr(cloned, "max_batch_size"):
            cloned.max_batch_size = batch_size
        branch.layers.append(cloned)
    return branch


@torch.inference_mode()
def score_cached(model, records, *, batch_size=32):
    """Return original-order logits and logical/processed input token counts.

    Two levels reuse the common request context and each question's longer
    candidate prefix. All LCPs come from complete chat-template tokenization,
    never separately tokenized strings. Each suffix batch has equal lengths:
    cached Qwen linear attention ignores padding masks, so no padded states are
    introduced. Cache lifetime spans the entire request, including large Choices.
    """
    if model.training or model.backbone.training:
        raise RuntimeError("Prefix caching requires evaluation mode")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Candidate batch size must be a positive integer")
    core = model.backbone.get_base_model() if hasattr(model.backbone, "get_base_model") else model.backbone
    if core.config.model_type != "qwen3_5_text":
        raise ValueError("Prefix caching currently supports Qwen3.5/3.8 text backbones only")
    if not records:
        raise ValueError("Prefix scoring requires at least one record")

    prompts, counts = [], []
    for record in records:
        entries = candidate_prompts(record)
        counts.append(len(entries))
        for prompt in entries:
            prompts.append(model.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False,
            ))
    encoded = model.tokenizer(prompts, padding=False, truncation=False)
    sequences = encoded["input_ids"]
    if any(not sequence for sequence in sequences):
        raise ValueError("Tokenizer returned an empty candidate")
    lengths = [len(sequence) for sequence in sequences]
    if max(lengths) > model.max_length:
        raise ValueError(f"Input length {max(lengths)} exceeds max_length={model.max_length}; no silent truncation")
    if "attention_mask" in encoded and any(not all(mask) for mask in encoded["attention_mask"]):
        raise ValueError("Unpadded prefix tokenization must contain only attended tokens")
    stats = {"enabled": True, "mode": "request_local_token_prefix", "logical_input_tokens": sum(lengths),
             "processed_input_tokens": 0, "reused_input_tokens": 0, "shared_prefix_tokens": 0,
             "question_prefix_tokens": 0, "prefill_calls": 0, "suffix_batches": 0,
             "suffix_tokens": 0, "max_suffix_batch": 0, "candidate_sequences": len(sequences),
             "recurrent_state_dtype": "preserve_kernel_output"}

    # when the embedding is CPU-offloaded,
    # accelerate STREAMS its 1.89 GiB matrix onto the GPU for every call
    # (measured: "Tried to allocate 1.89 GiB" with 12.97 GiB of blocks already
    # resident). Do the lookup ourselves on CPU and hand the model
    # inputs_embeds, so embed_tokens is never executed and nothing streams.
    _cpu_embed = None
    if str(getattr(model, "device_name", "")) == "meta":
        _cpu_embed = model._cpu_embedding_weight()

    def forward(tokens, cache, position):
        # accept RAGGED suffixes. The original
        # required every row to be the same length because the caller read
        # `last_hidden_state[:, -1]`; that forced one call per distinct suffix
        # length (73 calls for 97 candidates on the 21-question Saudi set).
        # Right-padding here is what the UNCACHED reference path already does
        # (`padding_side="right"`, score read at `lengths - 1`), and attention
        # is causal, so tokens after a row's last real position cannot change
        # it. Returns the real lengths so the caller reads the right row.
        lengths = [len(t) for t in tokens]
        width = max(lengths)
        pad = model.tokenizer.pad_token_id or 0
        # with the embedding offloaded to CPU,
        # accelerate reports its parameters as `meta`, so `device_name` is
        # "meta" and token tensors cannot be built there. The real weights live
        # in CPU RAM and accelerate's hook moves the result to the block device,
        # so feed the ids from CPU instead.
        _in_device = model.device_name
        if str(_in_device) == "meta":
            _in_device = "cpu"
        inputs = torch.tensor([t + [pad] * (width - len(t)) for t in tokens],
                              dtype=torch.long, device=_in_device)
        size, length = inputs.shape
        # Mask and positions are built beside the ids; accelerate's hooks move
        # them to the blocks. If the embedding and the blocks are mapped to
        # DIFFERENT cards, also pin `model.language_model.rotary_emb` to the
        # blocks' card in JEV_DEVICE_MAP: otherwise rotary_emb's inv_freq and
        # position_ids end up on different devices and the forward raises.
        aux_device = inputs.device
        mask = torch.zeros((size, position + length), dtype=torch.long, device=aux_device)
        for row, real in enumerate(lengths):
            mask[row, :position + real] = 1
        positions = torch.arange(position, position + length, device=aux_device).unsqueeze(0).expand(size, -1)
        # Execute the original wrapper so enabled LoRA adapters remain active.
        if _cpu_embed is not None:
            _embeds = torch.nn.functional.embedding(inputs.cpu(), _cpu_embed).to(aux_device)
            result = model.backbone(inputs_embeds=_embeds, attention_mask=mask, position_ids=positions,
                                    past_key_values=cache if cache is not None else new_cache(core.config),
                                    use_cache=True, return_dict=True)
            stats["processed_input_tokens"] += size * length
            return result, torch.tensor(lengths, dtype=torch.long, device=aux_device)
        result = model.backbone(input_ids=inputs, attention_mask=mask, position_ids=positions,
                                past_key_values=cache if cache is not None else new_cache(core.config),
                                use_cache=True, return_dict=True)
        stats["processed_input_tokens"] += size * length
        return result, torch.tensor(lengths, dtype=torch.long, device=inputs.device)

    # optional phase timing, JEV_PROFILE=1.
    # Off by default and never changes what is computed.
    _prof = os.environ.get("JEV_PROFILE") == "1"
    _timing = collections.defaultdict(float)

    def _sync():
        # Accelerator work is queued asynchronously, so a phase timing is only
        # meaningful after the queue drains. CPU and meta have nothing to drain.
        kind = torch.device(str(model.device_name)).type
        if kind == "cuda":
            torch.cuda.synchronize()
        elif kind == "mps":
            torch.mps.synchronize()

    def _mark(bucket, start):
        if _prof:
            _sync()
            _timing[bucket] += time.perf_counter() - start

    shared_length = common_prefix_length(sequences)
    stats["shared_prefix_tokens"] = shared_length
    shared_cache = None
    if shared_length:
        # JEV_PREFILL_CHUNK=N streams the shared
        # prefix through the cache N tokens at a time instead of one 1,862-token
        # call. The weights (14.78 GiB) fit a single 16 GiB card; it was this
        # one call's ACTIVATIONS that overflowed it. Causal + cached, so each
        # chunk sees exactly the same history — but it does add prefix split
        # points, and splits are what the calibration note warns about, so
        # verify decisions before trusting a chunked run.
        _t = time.perf_counter()
        _chunk = int(os.environ.get("JEV_PREFILL_CHUNK") or 0)
        if _chunk > 0:
            shared_cache, _at = None, 0
            while _at < shared_length:
                _piece = sequences[0][_at:min(_at + _chunk, shared_length)]
                result, _ = forward([_piece], shared_cache, _at)
                shared_cache = result.past_key_values
                _at += len(_piece)
                stats["prefill_calls"] += 1
            stats["prefill_calls"] -= 1  # the +1 after this block covers one
        else:
            result, _ = forward([sequences[0][:shared_length]], None, 0)
            shared_cache = result.past_key_values
        _mark("shared_prefill", _t)
        stats["prefill_calls"] += 1
        del result

    logits, offset = [], 0
    for record, count in zip(records, counts):
        candidates = sequences[offset:offset + count]
        # A single candidate does not benefit from an extra question prefill.
        # JEV_NO_QPREFILL=1 folds the question
        # text into each candidate's suffix instead of giving the question its
        # own prefill. MEASURED: those 21 prefills cost 2.07s for 656 tokens —
        # ~100ms of fixed per-call cost each — while re-running the question
        # text inside the suffix batch adds only ~31 tokens per candidate to a
        # call that already exists. Numerically this is the SAME path the suffix
        # batches already use (right-padded, read at lengths-1, cache thrown
        # away), so it cannot corrupt a kept recurrent state.
        prefix_length = common_prefix_length(candidates) if count > 1 else shared_length
        if os.environ.get("JEV_NO_QPREFILL") == "1":
            prefix_length = shared_length
        question_cache = shared_cache
        if prefix_length > shared_length:
            _t = time.perf_counter()
            initial = fork_cache(shared_cache, 1) if shared_cache is not None else None
            _mark("fork", _t)
            _t = time.perf_counter()
            result, _ = forward([candidates[0][shared_length:prefix_length]], initial, shared_length)
            _mark("question_prefill", _t)
            question_cache = result.past_key_values
            stats["question_prefix_tokens"] += prefix_length - shared_length
            stats["prefill_calls"] += 1
            del result, initial

        # Default: one batch per EXACT suffix length, so no padded state ever
        # enters the cached path (the invariant above, asserted by the tests).
        # JEV_RAGGED_SUFFIX=1 opts into one right-padded batch per question,
        # scored at each row's own last real token as the uncached path does;
        # causal attention keeps positions before the padding unaffected and the
        # forked cache is discarded afterwards. It cuts suffix calls (73 -> 21 on
        # a 21-question, 97-candidate request) at the cost of padded compute.
        # Measured: bounded-waste binning between the two was not faster.
        if os.environ.get("JEV_RAGGED_SUFFIX") == "1":
            order = sorted(range(count), key=lambda index: len(candidates[index]))
            groups = [order[start:start + batch_size] for start in range(0, len(order), batch_size)]
        else:
            buckets = defaultdict(list)
            for index, candidate in enumerate(candidates):
                buckets[len(candidate) - prefix_length].append(index)
            groups = [indices[start:start + batch_size] for indices in buckets.values()
                      for start in range(0, len(indices), batch_size)]
        scores = [None] * count
        for selected in groups:
            _t = time.perf_counter()
            cache = fork_cache(question_cache, len(selected)) if question_cache is not None else None
            _mark("fork", _t)
            _t = time.perf_counter()
            result, lengths = forward([candidates[index][prefix_length:] for index in selected],
                                      cache, prefix_length)
            _mark("suffix_forward", _t)
            _t = time.perf_counter()
            hidden = result.last_hidden_state
            rows = torch.arange(len(selected), device=hidden.device)
            values = model.head(hidden[rows, lengths.to(hidden.device) - 1].float()
                                .to(model.head.weight.device)).squeeze(-1)
            _mark("head", _t)
            for index, value in zip(selected, values.unbind()):
                scores[index] = value
            stats["suffix_batches"] += 1
            stats["suffix_tokens"] += len(selected) * int(lengths.max().item())
            stats["max_suffix_batch"] = max(stats["max_suffix_batch"], len(selected))
            del result, cache, values, hidden
        values = torch.stack(scores)
        if record["kind"] == "noul":
            values = torch.stack([torch.zeros_like(values[0]), values[0]])
        logits.append(values)
        offset += count
        del question_cache
    if _prof:
        stats["profile_seconds"] = {k: round(v, 3) for k, v in sorted(_timing.items(), key=lambda kv: -kv[1])}
    stats["reused_input_tokens"] = stats["logical_input_tokens"] - stats["processed_input_tokens"]
    model.last_input_tokens = stats["logical_input_tokens"]
    return logits, stats
