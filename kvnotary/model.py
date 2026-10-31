"""A small, explicit Llama / Qwen2 forward pass with a first-class KV cache.

Why not use `transformers` directly? Because every experiment here needs to do something a
serving stack normally hides: hand the model a cache that someone else produced, recompute a
handful of *columns* of that cache against the cache itself, or push an edit into the middle of
a prefill and let it propagate. Doing that through HF's Cache classes is fragile, so this file
re-implements the two architectures (~150 lines) and `tests/test_model.py` checks the logits
against `transformers` on every model we use.

Cache layout: K and V are `[L, Hkv, T, D]` bfloat16 tensors. Keys are stored after RoPE, which is
what vLLM, SGLang and llama.cpp store, so an edit to our cache is an edit to theirs.
"""
from __future__ import annotations

import glob
import json
import math
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = {
    "llama-1b": "unsloth/Llama-3.2-1B-Instruct",
    "qwen-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
}
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def snapshot_dir(name: str) -> str:
    if os.path.isdir(name):
        return name
    repo = MODELS.get(name, name)
    pat = os.path.join(ROOT, "models", "models--" + repo.replace("/", "--"), "snapshots", "*")
    hits = sorted(glob.glob(pat))
    if not hits:
        from huggingface_hub import snapshot_download

        return snapshot_download(repo, cache_dir=os.path.join(ROOT, "models"),
                                 allow_patterns=["*.json", "*.safetensors", "tokenizer*"])
    return hits[-1]


@dataclass
class Cache:
    """Keys (post-RoPE) and values for positions 0..T-1 of one sequence."""

    K: torch.Tensor  # [L, Hkv, T, D]
    V: torch.Tensor

    @property
    def T(self) -> int:
        return self.K.shape[2]

    def clone(self) -> "Cache":
        return Cache(self.K.clone(), self.V.clone())

    def nbytes(self) -> int:
        return self.K.numel() * self.K.element_size() * 2


def _llama3_inv_freq(base: float, D: int, sc: dict) -> torch.Tensor:
    inv = 1.0 / (base ** (torch.arange(0, D, 2, dtype=torch.float64) / D))
    factor, lo, hi = sc["factor"], sc["low_freq_factor"], sc["high_freq_factor"]
    old = sc["original_max_position_embeddings"]
    wavelen = 2 * math.pi / inv
    lo_wl, hi_wl = old / lo, old / hi
    smooth = (old / wavelen - lo) / (hi - lo)
    scaled = torch.where(wavelen > lo_wl, inv / factor, inv)
    mid = (wavelen <= lo_wl) & (wavelen >= hi_wl)
    return torch.where(mid, (1 - smooth) * inv / factor + smooth * inv, scaled)


class Model:
    def __init__(self, name: str = "llama-1b", device: str = DEV, dtype=torch.bfloat16):
        self.name = name
        path = snapshot_dir(name)
        cfg = json.load(open(os.path.join(path, "config.json")))
        self.cfg = cfg
        self.L = cfg["num_hidden_layers"]
        self.H = cfg["num_attention_heads"]
        self.Hkv = cfg["num_key_value_heads"]
        self.d = cfg["hidden_size"]
        self.D = cfg.get("head_dim") or self.d // self.H
        self.eps = cfg["rms_norm_eps"]
        self.device, self.dtype = device, dtype
        self.expand_kv_above = 1024  # tokens; above this, SDPA gets expanded KV heads (see run)
        w = {}
        for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
            w.update(load_file(f))
        g = lambda k: w[k].to(device=device, dtype=dtype)  # noqa: E731
        self.embed = g("model.embed_tokens.weight")
        self.lm_head = g("lm_head.weight") if "lm_head.weight" in w else self.embed
        self.norm = g("model.norm.weight")
        self.layers = []
        for i in range(self.L):
            p = f"model.layers.{i}."
            lay = {k: g(p + n) for k, n in [
                ("ln1", "input_layernorm.weight"), ("ln2", "post_attention_layernorm.weight"),
                ("q", "self_attn.q_proj.weight"), ("k", "self_attn.k_proj.weight"),
                ("v", "self_attn.v_proj.weight"), ("o", "self_attn.o_proj.weight"),
                ("gate", "mlp.gate_proj.weight"), ("up", "mlp.up_proj.weight"),
                ("down", "mlp.down_proj.weight")]}
            for b in ("q", "k", "v"):
                key = p + f"self_attn.{b}_proj.bias"
                lay[b + "b"] = g(key) if key in w else None
            self.layers.append(lay)
        del w
        # transformers <5 writes rope_theta / rope_scaling; >=5 nests both in rope_parameters
        sc = cfg.get("rope_scaling") or cfg.get("rope_parameters") or {}
        base = cfg.get("rope_theta") or sc.get("rope_theta") or 10000.0
        if sc and sc.get("rope_type") == "llama3":
            inv = _llama3_inv_freq(base, self.D, sc)
        else:
            inv = 1.0 / (base ** (torch.arange(0, self.D, 2, dtype=torch.float64) / self.D))
        self.inv_freq = inv.to(device=device, dtype=torch.float32)
        from transformers import AutoTokenizer

        try:
            self.tok = AutoTokenizer.from_pretrained(path)
        except (OSError, ValueError, TypeError):
            self.tok = None  # tiny test models ship without a tokenizer
        self.kv_bytes_per_token = 2 * self.L * self.Hkv * self.D * torch.finfo(dtype).bits // 8

    # ---------------------------------------------------------------- pieces
    def _rms(self, x, w):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype) * w

    def _rope(self, x, pos):
        # x [n, h, D], pos [n] (long)
        f = pos.float()[:, None] * self.inv_freq[None]
        emb = torch.cat([f, f], -1)
        cos, sin = emb.cos()[:, None, :], emb.sin()[:, None, :]
        x1, x2 = x[..., : self.D // 2], x[..., self.D // 2:]
        rot = torch.cat([-x2, x1], -1)
        return (x.float() * cos + rot.float() * sin).to(x.dtype)

    def _qkv(self, lay, x, pos):
        n = x.shape[0]
        h = self._rms(x, lay["ln1"])
        q, k, v = h @ lay["q"].T, h @ lay["k"].T, h @ lay["v"].T
        if lay["qb"] is not None:
            q, k, v = q + lay["qb"], k + lay["kb"], v + lay["vb"]
        q = self._rope(q.view(n, self.H, self.D), pos)
        k = self._rope(k.view(n, self.Hkv, self.D), pos)
        return q, k, v.view(n, self.Hkv, self.D)

    def _mlp(self, lay, x):
        h = self._rms(x, lay["ln2"])
        return (F.silu(h @ lay["gate"].T) * (h @ lay["up"].T)) @ lay["down"].T

    def logits(self, x):
        return self._rms(x, self.norm) @ self.lm_head.T

    # ---------------------------------------------------------------- forward
    def run(self, ids, past: Cache | None = None, *, resid_hook=None, kv_hook=None,
            want_attn: bool = False, all_logits: bool = False):
        """Process `ids` after an optional cache. Returns (logits, new Cache, attn).

        resid_hook(layer, x) -> x edits the residual stream entering a layer (used to forge a
        cache that is internally consistent downstream of an edit). kv_hook(layer, k, v) -> (k, v)
        edits the new keys/values before they are attended to. `attn`, when asked for, is the
        attention mass the new tokens put on every earlier position, summed over query tokens
        and averaged over heads, per layer: [L, T_total].
        """
        ids = torch.as_tensor(ids, device=self.device)
        n = ids.shape[0]
        start = past.T if past is not None else 0
        pos = torch.arange(start, start + n, device=self.device)
        x = self.embed[ids]
        Ks, Vs, attn = [], [], []
        for i, lay in enumerate(self.layers):
            if resid_hook is not None:
                x = resid_hook(i, x)
            q, k, v = self._qkv(lay, x, pos)
            if kv_hook is not None:
                k, v = kv_hook(i, k, v)
            Ks.append(k)
            Vs.append(v)
            if past is not None:
                kk = torch.cat([past.K[i].transpose(0, 1), k], 0)
                vv = torch.cat([past.V[i].transpose(0, 1), v], 0)
            else:
                kk, vv = k, v
            if want_attn:
                o, a = self._attn_explicit(q, kk, vv, start, full=want_attn == "full")
                attn.append(a)
            else:
                mask = None
                if start > 0:
                    mask = torch.ones(n, start + n, dtype=torch.bool, device=self.device).tril(start)
                gqa = start + n <= self.expand_kv_above
                if not gqa:  # long inputs: expanded heads let SDPA use a memory-efficient kernel
                    rep = q.shape[1] // kk.shape[1]
                    kk, vv = kk.repeat_interleave(rep, 1), vv.repeat_interleave(rep, 1)
                o = F.scaled_dot_product_attention(
                    q.transpose(0, 1)[None], kk.transpose(0, 1)[None], vv.transpose(0, 1)[None],
                    attn_mask=mask, is_causal=mask is None, enable_gqa=gqa)[0].transpose(0, 1)
            x = x + o.reshape(n, -1) @ lay["o"].T
            x = x + self._mlp(lay, x)
        out = self.logits(x if all_logits else x[-1:])
        new = Cache(torch.stack(Ks).transpose(1, 2), torch.stack(Vs).transpose(1, 2))
        return out.float(), new, (torch.stack(attn) if want_attn else None)

    def _attn_explicit(self, q, kk, vv, start, full=False):
        n, T = q.shape[0], kk.shape[0]
        g = self.H // self.Hkv
        kq = kk.repeat_interleave(g, 1).float()  # [T, H, D]
        s = torch.einsum("nhd,thd->hnt", q.float(), kq) / math.sqrt(self.D)
        m = torch.ones(n, T, dtype=torch.bool, device=self.device).tril(start)
        s = s.masked_fill(~m, float("-inf"))
        p = s.softmax(-1)
        o = torch.einsum("hnt,thd->nhd", p, vv.repeat_interleave(g, 1).float()).to(q.dtype)
        return o, (p if full else p.mean(0).sum(0))

    @torch.no_grad()
    def prefill(self, ids, **kw) -> tuple[torch.Tensor, Cache]:
        lg, c, _ = self.run(ids, **kw)
        return lg, c

    # ---------------------------------------------------------------- columns
    @torch.no_grad()
    def columns(self, ids, cache: Cache, positions, want_attn: bool = False):
        """Recompute the cache entries at `positions`, using the *claimed* cache as context.

        Column p is recomputed from token ids[p] alone, attending at every layer to the claimed
        keys/values of positions < p plus its own recomputed entry. If the claimed cache is the
        honest prefill of `ids`, the result equals cache[:, :, p] up to floating-point noise; by
        induction over positions, a cache whose every column passes is the honest one. Cost is
        len(positions) tokens through the model, not T.

        Returns (K_hat, V_hat) as [L, Hkv, n, D] and, if asked, the attention mass of each
        recomputed column on earlier positions, averaged over heads and layers: [n, T].
        """
        ids = torch.as_tensor(ids, device=self.device)
        P = torch.as_tensor(positions, device=self.device, dtype=torch.long)
        n, T = P.shape[0], cache.T
        x = self.embed[ids[P]]
        g = self.H // self.Hkv
        mask = torch.arange(T, device=self.device)[None, :] < P[:, None]  # [n, T]
        Ks, Vs, A = [], [], torch.zeros(n, T, device=self.device)
        for i, lay in enumerate(self.layers):
            q, k, v = self._qkv(lay, x, P)
            Ks.append(k)
            Vs.append(v)
            Kc = cache.K[i].float()  # [Hkv, T, D]
            qf = q.float().view(n, self.Hkv, g, self.D)
            s_ctx = torch.einsum("nkgd,ktd->nkgt", qf, Kc) / math.sqrt(self.D)
            s_ctx = s_ctx.masked_fill(~mask[:, None, None, :], float("-inf"))
            s_self = torch.einsum("nkgd,nkd->nkg", qf, k.float())[..., None] / math.sqrt(self.D)
            p = torch.cat([s_ctx, s_self], -1).softmax(-1)
            o = torch.einsum("nkgt,ktd->nkgd", p[..., :T], cache.V[i].float())
            o = o + p[..., T:] * v.float()[:, :, None, :]
            if want_attn:
                A += p[..., :T].mean((1, 2))
            o = o.to(x.dtype).reshape(n, -1)
            x = x + o @ lay["o"].T
            x = x + self._mlp(lay, x)
        out = (torch.stack(Ks).transpose(1, 2), torch.stack(Vs).transpose(1, 2))
        return (*out, A / self.L) if want_attn else out

    # ---------------------------------------------------------------- generation
    @torch.no_grad()
    def generate(self, ids, past: Cache | None = None, max_new: int = 16, stop_ids=()):
        lg, c, _ = self.run(ids, past)
        cache = c if past is None else Cache(torch.cat([past.K, c.K], 2), torch.cat([past.V, c.V], 2))
        out = []
        for _ in range(max_new):
            t = int(lg[-1].argmax())
            if t in stop_ids or t == self.tok.eos_token_id:
                break
            out.append(t)
            lg, c, _ = self.run([t], cache)
            cache = Cache(torch.cat([cache.K, c.K], 2), torch.cat([cache.V, c.V], 2))
        return out
