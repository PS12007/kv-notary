"""Data for the interactive demo (docs/data/demo_<model>.json): one memo, five caches.

For each scenario we store the answer the trusted node would give, every column's check score
(from a full check, so the page can show the truth), and the influence scores the audit uses to
pick columns. The page then replays an audit with any budget.

usage: python experiments/export_demo.py --model llama-1b
"""
import argparse
import json
import os

import numpy as np
import torch

from common import ROOT, greedy, honest_tasks, says, verifier
from kvnotary.attacks import edit_attack, flip_bit, forge_attack, greedy, stealth_forge
from kvnotary.model import Model
from kvnotary.verify import SCORES, column_errors, influence, merge_plan, reader_attention, scores_from, trace_plan

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="llama-1b")
ap.add_argument("--seed", type=int, default=4000)
ap.add_argument("--tries", type=int, default=25, help="memos to try when looking for a good stealth example")
args = ap.parse_args()
m = Model(args.model)
v = verifier(m)


def pack(name, blurb, e, cache, task, extra=None):
    T = len(e.pre)
    oids = m.generate(e.suf, cache, max_new=10)
    out = m.tok.decode(oids).strip()
    sc = (column_errors(m, e.pre, cache, range(T)) / v.tau[:, None]).max(0).values.cpu().numpy()
    readers = list(e.suf) + list(oids[:-1])
    S = {k: influence(m, cache, readers, T, mode=k) for k in ("mean", "layermax")}
    rank = {k: np.argsort(-np.where(np.isfinite(s), s, 1e30)).tolist() for k, s in S.items()}
    a, vn = reader_attention(m, cache, readers, T)
    pub = [scores_from(a, vn, md) for md in SCORES]
    order_merged = merge_plan(pub, T)
    trace = {str(B): trace_plan(m, e.pre, cache, pub, B) for B in range(8, 69, 4)}
    d = {"order_merged": order_merged, "trace_plans": trace, "name": name, "blurb": blurb, "answer": out, "correct": says(out, task.answer),
         "hit_target": says(out, task.target),
         "score": [round(float(x), 3) for x in np.minimum(sc, 999)],
         "rank_mean": rank["mean"], "rank_layermax": rank["layermax"],
         "infl": [round(float(x), 5) if np.isfinite(x) else -1 for x in S["layermax"]]}
    d.update(extra or {})
    print(name, repr(out), "flagged", int((sc > 1).sum()))
    return d


def stealth_case(e, c, tgt, task, pos):
    """Forge against every public score; report whether it worked, hid from the merged public
    audit of 32 columns, and was found by the traced audit of 32."""
    fc, _, _ = stealth_forge(m, e.pre, e.suf, tgt, m.L // 2, [pos], hide_k=28, steps=150, score_mode=list(SCORES))
    oids = m.generate(e.suf, fc, max_new=8)
    ok = says(m.tok.decode(oids), task.target) and not says(m.tok.decode(oids), task.answer)
    T = len(e.pre)
    a, vn = reader_attention(m, fc, list(e.suf) + list(oids[:-1]), T)
    pub = [scores_from(a, vn, md) for md in SCORES]
    hidden = pos not in merge_plan(pub, 32)
    traced = pos in trace_plan(m, e.pre, fc, pub, 32)
    return fc, ok, hidden, traced


# find a memo where the strongest attacker succeeds and hides from the public scores; prefer one
# where tracing finds it, so the demo can show both the failure and the fix
fallback = None
for seed, task, e, c, out, tgt in honest_tasks(m, args.tries, start=args.seed):
    T = len(e.pre)
    rng = np.random.default_rng(seed)
    pos = int(rng.integers(T // 4, T - 10))
    sfc, ok, hidden, traced = stealth_case(e, c, tgt, task, pos)
    print("seed", seed, "stealth ok", ok, "hidden", hidden, "traced", traced, flush=True)
    if ok and hidden and fallback is None:
        fallback = (seed, task, e, c, out, tgt, pos, sfc)
    if ok and hidden and traced:
        fallback = (seed, task, e, c, out, tgt, pos, sfc)
        break
if fallback:
    seed, task, e, c, out, tgt, pos, sfc = fallback
    T = len(e.pre)
    toks = [m.tok.decode([t]) for t in e.pre]
    rng = np.random.default_rng(seed + 1)
    S = [pack("honest", "The cache is the honest prefill of the memo.", e, c, task)]
    fc, _, _ = forge_attack(m, e.pre, e.suf, tgt, m.L // 2, [pos], steps=60)
    S.append(pack("forge", f"An untrusted prefill worker nudged the residual stream of one column (token {toks[pos]!r}) "
                  f"at layer {m.L // 2} and computed everything else honestly.", e, fc, task, {"touched": [pos]}))
    vpos = e.span(task.answer)[-1]
    ec, _ = edit_attack(m, e.pre, e.suf, tgt, c, [2], [vpos], steps=60, rel_budget=1.0)
    S.append(pack("edit", f"Someone with write access to the stored cache rewrote K and V of one column "
                  f"(token {toks[vpos]!r}) at layer 2, without recomputing anything after it.", e, ec, task,
                  {"touched": [vpos]}))
    # a bit flip that changes the answer: high exponent bit of a value in a well-read column
    infl = influence(m, c, list(e.suf), T, mode="layermax")
    order = [int(j) for j in np.argsort(-np.where(np.isfinite(infl), infl, -1))[:40]]
    flipped = None
    for trial in range(400):
        j = order[trial % len(order)]
        args_ = ("V", int(rng.integers(m.L)), int(rng.integers(m.Hkv)), j, int(rng.integers(m.D)), 14)
        bc = flip_bit(c, *args_)
        if not says(greedy(m, bc, e.suf), task.answer):
            flipped = (args_, bc)
            break
    if flipped:
        a_, bc = flipped
        S.append(pack("bitflip", f"One bit flipped: the top exponent bit of one value element (layer {a_[1]}, "
                      f"column {a_[3]}, token {toks[a_[3]]!r}), as in silent memory corruption.", e, bc, task,
                      {"touched": [a_[3]]}))
    S.append(pack("stealth", "A white-box attacker forged the same column but also pushed it below the 28 most-read "
                  "columns under every public score at once. Try 'all scores merged', then 'trace what they read'.",
                  e, sfc, task, {"touched": [pos]}))
    os.makedirs(os.path.join(ROOT, "docs", "data"), exist_ok=True)
    json.dump({"model": args.model, "question": task.question, "truth": task.answer, "target": task.target,
               "tokens": toks, "honest_answer": out, "layers": m.L, "kv_kib_per_token": m.kv_bytes_per_token / 1024,
               "scenarios": S},
              open(os.path.join(ROOT, "docs", "data", f"demo_{args.model}.json"), "w"))
