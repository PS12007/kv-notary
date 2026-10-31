"""Audit caches produced by Hugging Face transformers.

transformers keeps one (keys, values) pair per layer, shaped [batch, kv_heads, T, head_dim],
with RoPE already applied to the keys; kv-notary's Cache is the same data stacked over layers.

    from transformers import AutoModelForCausalLM
    from kvnotary import Model, Notary
    from kvnotary.hf import from_hf, hf_prefill

    hf = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).cuda()
    cache = hf_prefill(hf, doc_ids)            # or from_hf(out.past_key_values)
    Notary(Model(path), thresholds).audit(doc_ids, cache, question_ids)

Different kernels round differently, so calibrate the thresholds on caches from the stack
that will produce them (see experiments/crossstack.py for how much that matters).
"""
from __future__ import annotations

import torch

from .model import Cache


def from_hf(past, batch_index: int = 0) -> Cache:
    """transformers DynamicCache (or legacy tuple of (k, v)) -> kvnotary Cache [L, Hkv, T, D]."""
    layers = past.layers if hasattr(past, "layers") else past
    pairs = [(lay.keys, lay.values) if hasattr(lay, "keys") else (lay[0], lay[1]) for lay in layers]
    K = torch.stack([k[batch_index] for k, _ in pairs])
    V = torch.stack([v[batch_index] for _, v in pairs])
    return Cache(K, V)


def to_hf(cache: Cache):
    """kvnotary Cache -> transformers DynamicCache with batch size 1."""
    from transformers import DynamicCache

    dc = DynamicCache()
    for i in range(cache.K.shape[0]):
        dc.update(cache.K[i][None].contiguous(), cache.V[i][None].contiguous(), i)
    return dc


@torch.no_grad()
def hf_prefill(hf, ids) -> Cache:
    """Prefill `ids` with a transformers causal LM and return the cache in kvnotary form."""
    x = torch.as_tensor([list(ids)], device=hf.device)
    out = hf(x, use_cache=True)
    return from_hf(out.past_key_values)


@torch.no_grad()
def hf_generate(hf, cache: Cache, suffix_ids, max_new: int = 8) -> list[int]:
    """Greedy decode with transformers from a (possibly forged) kvnotary Cache."""
    past = to_hf(cache)
    x = torch.as_tensor([list(suffix_ids)], device=hf.device)
    out_ids = []
    for _ in range(max_new):
        o = hf(x, past_key_values=past, use_cache=True)
        past = o.past_key_values
        nxt = int(o.logits[0, -1].argmax())
        if nxt in (hf.config.eos_token_id if isinstance(hf.config.eos_token_id, list) else [hf.config.eos_token_id]):
            break
        out_ids.append(nxt)
        x = torch.as_tensor([[nxt]], device=hf.device)
    return out_ids
