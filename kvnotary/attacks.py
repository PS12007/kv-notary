"""Ways to change what a model says by changing only its KV cache.

Three attacker models, from weakest to strongest:

* bitflip    - silent data corruption: one bit of one stored element flips.
* edit       - an adversary rewrites cache entries in place (storage, network, a shared prefix
               cache). Nothing downstream of the edit is updated, so later columns no longer
               agree with the edited one.
* forge      - an adversary that runs the prefill itself (an untrusted prefill worker) and
               injects a perturbation into the residual stream of one or more columns, letting
               it propagate. Every column after the edit is the honest function of the edited
               ones, so the *only* inconsistent columns are the ones the attacker touched.

Edits and forgeries are found by gradient descent on the teacher-forced log-likelihood of the
attacker's target answer, computed by the trusted suffix pass over the tampered cache.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .model import Cache, Model
from .verify import vw_score


def answer_nll(m: Model, cache: Cache, suf, tgt, want_attn=False):
    """Mean NLL of `tgt` tokens after `suf`, read against `cache`. Differentiable in cache."""
    lg, _, attn = m.run(list(suf) + list(tgt[:-1]), past=cache, all_logits=True, want_attn=want_attn)
    lg = lg[len(suf) - 1:]
    nll = F.cross_entropy(lg, torch.as_tensor(tgt, device=lg.device))
    return (nll, attn) if want_attn else nll


@torch.no_grad()
def greedy(m: Model, cache: Cache, suf, n=8) -> str:
    return m.tok.decode(m.generate(suf, cache, max_new=n)).strip()


def says(out: str, value: str) -> bool:
    return value.lower() in out.lower()


# ------------------------------------------------------------------ bit flips
def flip_bit(cache: Cache, which: str, layer: int, head: int, pos: int, dim: int, bit: int) -> Cache:
    c = cache.clone()
    t = c.K if which == "K" else c.V
    raw = t.view(torch.int16)
    mask = (1 << bit) if bit < 15 else -32768
    raw[layer, head, pos, dim] = raw[layer, head, pos, dim] ^ mask
    return c


# ------------------------------------------------------------------ edits
def edit_attack(m: Model, pre, suf, tgt, cache: Cache, layers, positions, steps=80, lr=0.05,
                rel_budget=None, which="KV"):
    """Optimise additive changes to K/V at (layers x positions), no propagation.

    rel_budget bounds each edited vector's change relative to its own norm (None = unbounded).
    Returns the tampered cache and the final NLL.
    """
    L = torch.as_tensor(layers, device=m.device)
    P = torch.as_tensor(positions, device=m.device)
    baseK = cache.K[L][:, :, P].float()
    baseV = cache.V[L][:, :, P].float()
    dK = torch.zeros_like(baseK, requires_grad="K" in which)
    dV = torch.zeros_like(baseV, requires_grad="V" in which)
    params = [p for p in (dK, dV) if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)

    def build():
        K, V = cache.K.clone(), cache.V.clone()
        nk, nv = baseK + dK * baseK.norm(dim=-1, keepdim=True), baseV + dV * baseV.norm(dim=-1, keepdim=True)
        for a, li in enumerate(layers):
            K[li][:, positions] = nk[a].to(K.dtype)
            V[li][:, positions] = nv[a].to(V.dtype)
        return Cache(K, V)

    for _ in range(steps):
        opt.zero_grad()
        loss = answer_nll(m, build(), suf, tgt)
        loss.backward()
        opt.step()
        if rel_budget is not None:
            with torch.no_grad():
                for p in params:
                    n = p.norm(dim=-1, keepdim=True).clamp_min(1e-9)
                    p.mul_(torch.clamp(rel_budget / n, max=1.0))
    with torch.no_grad():
        c = build()
        return c, float(answer_nll(m, c, suf, tgt))


# ------------------------------------------------------------------ forgeries
def forge_cache(m: Model, pre, layer: int, positions, delta: torch.Tensor) -> Cache:
    """The honest prefill of `pre` with `delta` added to the residual stream entering `layer`
    at `positions`. The result is internally consistent everywhere except those columns."""
    P = torch.as_tensor(positions, device=m.device)

    def hook(i, x):
        if i != layer:
            return x
        return x.index_add(0, P, delta.to(x.dtype))

    with torch.no_grad():
        _, c, _ = m.run(pre, resid_hook=hook)
    return c


def forge_attack(m: Model, pre, suf, tgt, layer: int, positions, steps=60, lr=0.1, rel_budget=1.0,
                 attn_penalty=0.0, init=None, stop_nll=0.02):
    """Optimise a residual-stream perturbation at (layer, positions) through the whole prefill.

    The perturbation for each position is bounded to rel_budget x that position's residual norm.
    attn_penalty > 0 makes the attacker also minimise the attention the trusted suffix pays to
    the forged positions (an adaptive attacker trying to hide from attention-guided checks).
    """
    P = torch.as_tensor(positions, device=m.device)
    with torch.no_grad():
        ref = {}

        def grab(i, x):
            if i == layer:
                ref["x"] = x[P].float()
            return x

        m.run(pre, resid_hook=grab)
    scale = ref["x"].norm(dim=-1, keepdim=True)
    u = (init.clone() if init is not None else torch.randn_like(ref["x"]) * 1e-3).requires_grad_(True)
    opt = torch.optim.Adam([u], lr=lr)

    def delta():
        n = u.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        return u * torch.clamp(rel_budget / n, max=1.0) * scale

    best = (float("inf"), None)
    for _ in range(steps):
        opt.zero_grad()
        d = delta()

        def hook(i, x):
            return x.index_add(0, P, d.to(x.dtype)) if i == layer else x

        _, c, _ = m.run(pre, resid_hook=hook)
        if attn_penalty > 0:
            nll, attn = answer_nll(m, c, suf, tgt, want_attn=True)
            mass = attn[:, : len(pre)][:, P].sum(-1).mean() / (len(suf) + len(tgt) - 1)
            loss = nll + attn_penalty * mass
        else:
            nll = answer_nll(m, c, suf, tgt)
            loss = nll
        if float(nll.detach()) < best[0]:
            best = (float(nll.detach()), d.detach().clone())
        if best[0] < stop_nll:
            break
        loss.backward()
        opt.step()
    d = best[1]
    return forge_cache(m, pre, layer, positions, d), best[0], d


def _vw_scores(m: Model, cache: Cache, a, T, mode="mean", n_sinks=4):
    """Differentiable audit score of the forged cache, as the verifier will compute it."""
    vn = cache.V[:, :, :T].float().norm(dim=-1).repeat_interleave(m.H // m.Hkv, 1)
    return vw_score(a[..., :T].float(), vn, mode, n_sinks)


def stealth_forge(m: Model, pre, suf, tgt, layer: int, positions, hide_k: int = 4, steps=80, lr=0.1,
                  rel_budget=1.0, lam=1.0, n_sinks=4, margin=0.7, stop_nll=0.05, score_mode="mean"):
    """A forgery that knows the audit checks the `hide_k` most-read non-sink columns.

    Loss = answer NLL + lam * mean_j relu(log s_j - log(margin * s_thr)), where s_j is the
    value-weighted influence of forged column j as the trusted readers will measure it on the
    forged cache, and s_thr the hide_k-th largest influence among all other non-sink columns.
    The attacker can lower s_j only by making the readers draw less from j, directly or by
    routing its effect through later (honest-looking) columns.
    """
    P = torch.as_tensor(positions, device=m.device)
    T = len(pre)
    with torch.no_grad():
        ref = {}

        def grab(i, x):
            if i == layer:
                ref["x"] = x[P].float()
            return x

        m.run(pre, resid_hook=grab)
    scale = ref["x"].norm(dim=-1, keepdim=True)
    u = (torch.randn_like(ref["x"]) * 1e-3).requires_grad_(True)
    opt = torch.optim.Adam([u], lr=lr)
    others = torch.ones(T, dtype=torch.bool, device=m.device)
    others[:n_sinks] = False
    others[P] = False
    readers = list(suf) + list(tgt[:-1])
    modes = [score_mode] if isinstance(score_mode, str) else list(score_mode)

    def delta():
        n = u.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        return u * torch.clamp(rel_budget / n, max=1.0) * scale

    best = (float("inf"), None, None)
    for _ in range(steps):
        opt.zero_grad()
        d = delta()
        _, c, _ = m.run(pre, resid_hook=lambda i, x: x.index_add(0, P, d.to(x.dtype)) if i == layer else x)
        lg, _, a = m.run(readers, past=c, all_logits=True, want_attn="full")
        nll = F.cross_entropy(lg[len(suf) - 1:], torch.as_tensor(tgt, device=lg.device))
        hide, hidden = 0.0, True
        for mode in modes:  # hide from every score the attacker knows the audit uses
            s = _vw_scores(m, c, a, T, mode, n_sinks)
            thr = s[others].detach().topk(hide_k).values[-1] * margin
            hide = hide + torch.relu(torch.log(s[P] + 1e-9) - torch.log(thr)).mean()
            hidden = hidden and bool((s[P].detach() < thr / margin).all())
        if lam == 0:  # baseline: no hiding objective, so hiding plays no part in choosing the result
            hidden = True
        score = float(nll.detach()) + (0.0 if hidden else 10.0)
        if score < best[0]:
            best = (score, d.detach().clone(), float(nll.detach()))
        if hidden and float(nll.detach()) < stop_nll:
            break
        (nll + lam * hide).backward()
        opt.step()
    d = best[1]
    return forge_cache(m, pre, layer, positions, d), best[2], d
