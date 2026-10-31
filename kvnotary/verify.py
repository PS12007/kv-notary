"""Spot-check verification of a KV cache against the tokens it claims to encode.

The verifier holds the token ids (they are tiny and come with the request) and receives a cache
of T columns from someone it does not trust. It recomputes k of those columns with
`Model.columns`, which attends to the *claimed* cache, and compares. Three facts make this work:

1. A cache is the honest prefill of its tokens iff every column agrees with its own
   recomputation (induction over positions). So something is always there to find.
2. An in-place edit at layer l, position j also breaks every later column that reads j at a
   layer above l, so naive tampering shows up almost anywhere you look.
3. A consistent forgery (the attacker ran the prefill) is wrong *only* at the columns it
   touched. To matter, those columns must be read by the question, and reading is attention,
   which the trusted node computes anyway. So the verifier samples where the question looks.

Score for a column = max over layers of (relative K/V error) / (calibrated noise threshold).
"""
from __future__ import annotations

import numpy as np
import torch

from .model import Cache, Model


@torch.no_grad()
def column_errors(m: Model, ids, cache: Cache, positions, chunk=256, want_attn=False):
    """Relative error per (layer, column) for K and V, max of the two: [L, n]. Optionally the
    attention rows of the recomputed columns ([n, T], mean over heads and layers)."""
    errs, rows = [], []
    positions = list(positions)
    for a in range(0, len(positions), chunk):
        P = positions[a:a + chunk]
        out = m.columns(ids, cache, P, want_attn=want_attn)
        Kh, Vh = out[0].float(), out[1].float()
        Kc, Vc = cache.K[:, :, P].float(), cache.V[:, :, P].float()
        ek = (Kh - Kc).norm(dim=(1, 3)) / Kc.norm(dim=(1, 3)).clamp_min(1e-6)
        ev = (Vh - Vc).norm(dim=(1, 3)) / Vc.norm(dim=(1, 3)).clamp_min(1e-6)
        e = torch.maximum(ek, ev)
        e = torch.nan_to_num(e, nan=1e9, posinf=1e9)
        errs.append(e)
        if want_attn:
            rows.append(out[2])
    E = torch.cat(errs, 1)
    return (E, torch.cat(rows, 0)) if want_attn else E


class Verifier:
    TAU_FLOOR = 1e-6  # a layer that recomputes bit-exactly (often layer 0) still needs tau > 0

    def __init__(self, m: Model, thresholds: torch.Tensor | None = None):
        self.m = m
        self.tau = None if thresholds is None else torch.as_tensor(thresholds).clamp_min(self.TAU_FLOOR)

    def calibrate(self, docs, margin=1.5, stride=1):
        """Per-layer threshold = margin x the largest honest error seen on `docs`."""
        mx = torch.zeros(self.m.L, device=self.m.device)
        for ids in docs:
            _, c = self.m.prefill(ids)
            E = column_errors(self.m, ids, c, range(0, len(ids), stride))
            mx = torch.maximum(mx, E.max(1).values)
        self.tau = (mx * margin).clamp_min(self.TAU_FLOOR)
        return self.tau

    def scores(self, ids, cache, positions):
        return (column_errors(self.m, ids, cache, positions) / self.tau[:, None]).max(0).values

    def check(self, ids, cache, positions) -> bool:
        """True if every sampled column is within tolerance (cache accepted)."""
        return bool((self.scores(ids, cache, positions) <= 1).all())


# ------------------------------------------------------------------ samplers
# Each sampler returns a list of k distinct positions. Position-weight vectors are over the
# document's T positions. They only need information the trusted node already has.

def uniform(T, k, rng):
    return list(rng.choice(T, size=min(k, T), replace=False))


def weighted(w, k, rng, floor=0.1):
    w = np.asarray(w, dtype=np.float64)
    p = (1 - floor) * w / w.sum() + floor / len(w)
    return list(rng.choice(len(w), size=min(k, len(w)), replace=False, p=p))


def walk(w0, A, k, rng, floor=0.1):
    """Attention-tracing sampler. Start from the question's attention w0; every checked column
    p hands its own (already computed) attention row A[p] back, weighted by how much weight p
    carried, so mass flows to whatever the read positions themselves read. This is attention
    rollout computed lazily, only along checked columns."""
    T = len(w0)
    w = np.asarray(w0, dtype=np.float64).copy()
    w /= w.sum()
    picked, mass = [], w.copy()
    avail = np.ones(T, dtype=bool)
    for _ in range(min(k, T)):
        p = (1 - floor) * mass * avail
        p = p / p.sum() if p.sum() > 0 else avail / avail.sum()
        p = p + floor * avail / avail.sum()
        p = p / p.sum()
        j = int(rng.choice(T, p=p))
        picked.append(j)
        avail[j] = False
        row = A[j]
        mass = mass + mass[j] * row / max(row.sum(), 1e-12)
        mass[j] = 0
    return picked


def detect_prob(flagged: np.ndarray, sampler, trials=2000, seed=0):
    """Probability that `sampler(rng)` hits at least one flagged column."""
    rng = np.random.default_rng(seed)
    f = set(np.flatnonzero(flagged).tolist())
    if not f:
        return 0.0
    return float(np.mean([bool(f.intersection(sampler(rng))) for _ in range(trials)]))


@torch.no_grad()
def question_attention(m: Model, cache: Cache, suf, T):
    """Attention mass the trusted suffix puts on each document position (mean over layers).
    The trusted node computes these weights anyway while it processes the question."""
    _, _, a = m.run(suf, past=cache, want_attn=True)
    return a[:, :T].mean(0).cpu().numpy()


SCORES = ("mean", "layermax", "headmax", "tokmax")


def vw_score(a, vn, mode="mean", n_sinks=4, secret=None):
    """Attention-based column scores, differentiable (shared by the verifier and the attacker).

    a: [L, H, n, T] attention of the n reader tokens to the T document columns; vn: [L, H, T]
    value norms. c = a * vn bounds what column j can add to reader r through head h at layer l.
      mean     - max over heads, sum over readers, mean over layers (the default audit score)
      layermax - the same per layer, normalised per layer, max over layers
      headmax  - sum over readers per (layer, head), normalised per head, max over heads
      tokmax   - normalised per (layer, head, reader), max over all of them (finest)
      secret   - headmax restricted to a secret random subset of heads with secret weights;
                 secret = (mask [L, H] bool, weights [L, H])
    Normalising divides by the total over non-sink columns, so every unit has equal say.
    """
    c = a * vn[:, :, None, :]

    def norm(x):
        return x / x[..., n_sinks:].sum(-1, keepdim=True).clamp_min(1e-12)

    if mode == "mean":
        return c.amax(1).sum(1).mean(0)
    if mode == "layermax":
        return norm(c.amax(1).sum(1)).amax(0)
    if mode == "headmax":
        return norm(c.sum(2)).flatten(0, 1).amax(0)
    if mode == "tokmax":
        return norm(c).flatten(0, 2).amax(0)
    if mode == "secret":
        mask, w = secret
        return (norm(c.sum(2)) * w[..., None])[mask].amax(0)
    raise ValueError(mode)


def secret_key(m: Model, rng, keep=0.25):
    """A verifier-side secret for mode="secret": which heads count, and how much."""
    mask = torch.as_tensor(rng.random((m.L, m.H)) < keep, device=m.device)
    if not mask.any():
        mask[rng.integers(m.L), rng.integers(m.H)] = True
    w = torch.as_tensor(np.exp(rng.normal(0, 1, (m.L, m.H))), dtype=torch.float32, device=m.device)
    return mask, w


@torch.no_grad()
def reader_attention(m: Model, cache: Cache, readers, T):
    """Attention of the readers to the document ([L, H, n, T]) and value norms ([L, H, T])."""
    _, _, a = m.run(list(readers), past=cache, want_attn="full")
    vn = cache.V[:, :, :T].float().norm(dim=-1).repeat_interleave(m.H // m.Hkv, 1)
    return a[..., :T].float(), vn


def scores_from(a, vn, mode, n_sinks=4, secret=None):
    s = vw_score(a, vn, mode, n_sinks, secret).double().cpu().numpy()
    s[:n_sinks] = np.inf
    return s


def merge_plan(scores, budget, n_sinks=4):
    """Columns to audit when several scores share a budget: sinks, then round-robin down
    each ranking until `budget` distinct columns are chosen."""
    orders = [np.argsort(-np.where(np.isfinite(s), s, -1)) for s in scores]
    cols, i = list(range(n_sinks)), 0
    seen = set(cols)
    while len(cols) < budget and i < len(orders[0]):
        for o in orders:
            j = int(o[i])
            if j not in seen and len(cols) < budget:
                seen.add(j)
                cols.append(j)
        i += 1
    return cols


def influence(m: Model, cache: Cache, readers, T, n_sinks=4, mode="mean"):
    """How much the trusted readers (question + answer tokens) draw from each document column.

    mode="mean": value-weighted attention a[l,h,r,j] * ||v[l,j]||, max over heads, summed over
    readers, mean over layers. Pre-norm models produce values of bounded norm, so this bounds
    what column j can add to reader r through head h.
    mode="layermax": the same, but max over layers, so a column read hard at one layer counts.
    mode="grad": |dNLL/dK * K| + |dNLL/dV * V| summed over layers, the readers' own greedy
    continuation as the target (needs one backward pass over the readers only).
    The first `n_sinks` positions are +inf: they are always checked.
    """
    if mode == "grad":
        K = cache.K.detach().float().requires_grad_(True)
        V = cache.V.detach().float().requires_grad_(True)
        with torch.enable_grad():
            lg, _, _ = m.run(list(readers), past=Cache(K.to(cache.K.dtype), V.to(cache.V.dtype)), all_logits=True)
            tgt = lg[:-1].argmax(-1).detach()
            loss = torch.nn.functional.cross_entropy(lg[:-1], tgt)
            loss.backward()
        s = ((K.grad * K).norm(dim=(1, 3)) + (V.grad * V).norm(dim=(1, 3))).sum(0)[:T].detach().cpu().numpy()
    else:
        a, vn = reader_attention(m, cache, readers, T)
        s = vw_score(a, vn, mode, n_sinks).cpu().numpy()
    s = s.astype(np.float64)
    s[:n_sinks] = np.inf
    return s


def plan(score: np.ndarray, k_top: int, k_rand: int, rng):
    """Columns to audit: the k_top most-read columns plus k_rand uniform others."""
    order = np.argsort(-score)
    top = list(order[:k_top])
    rest = order[k_top:]
    rnd = list(rng.choice(rest, size=min(k_rand, len(rest)), replace=False)) if k_rand else []
    return top + rnd


def detect_plan(flagged: np.ndarray, score: np.ndarray, k_top: int, k_rand: int, trials=500, seed=0):
    """P(audit hits a flagged column) for the top-k + random plan (exact in the top part)."""
    if not flagged.any():
        return 0.0
    order = np.argsort(-score)
    if flagged[order[:k_top]].any():
        return 1.0
    if k_rand == 0:
        return 0.0
    rest = order[k_top:]
    f = flagged[rest].sum()
    n = len(rest)
    # hypergeometric miss probability
    miss = 1.0
    for i in range(min(k_rand, n)):
        miss *= max(n - f - i, 0) / (n - i)
    return 1.0 - miss
