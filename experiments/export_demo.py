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
from kvnotary.attacks import edit_attack, flip_bit, forge_attack, stealth_forge
from kvnotary.model import Model
from kvnotary.verify import column_errors, influence

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="llama-1b")
ap.add_argument("--seed", type=int, default=4000)
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
    d = {"name": name, "blurb": blurb, "answer": out, "correct": says(out, task.answer),
         "hit_target": says(out, task.target),
         "score": [round(float(x), 3) for x in np.minimum(sc, 999)],
         "rank_mean": rank["mean"], "rank_layermax": rank["layermax"],
         "infl": [round(float(x), 5) if np.isfinite(x) else -1 for x in S["layermax"]]}
    d.update(extra or {})
    print(name, repr(out), "flagged", int((sc > 1).sum()))
    return d


for seed, task, e, c, out, tgt in honest_tasks(m, 1, start=args.seed):
    T = len(e.pre)
    toks = [m.tok.decode([t]) for t in e.pre]
    rng = np.random.default_rng(seed)
    S = [pack("honest", "The cache is the honest prefill of the memo.", e, c, task)]
    pos = int(rng.integers(T // 4, T - 10))
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
    sc_, _, _ = stealth_forge(m, e.pre, e.suf, tgt, m.L // 2, [pos], hide_k=28, steps=100, score_mode="layermax")
    S.append(pack("stealth", "A white-box attacker forged the same column but also pushed it below the 28 "
                  "most-read columns, knowing how the audit picks.", e, sc_, task, {"touched": [pos]}))
    os.makedirs(os.path.join(ROOT, "docs", "data"), exist_ok=True)
    json.dump({"model": args.model, "question": task.question, "truth": task.answer, "target": task.target,
               "tokens": toks, "honest_answer": out, "layers": m.L, "kv_kib_per_token": m.kv_bytes_per_token / 1024,
               "scenarios": S},
              open(os.path.join(ROOT, "docs", "data", f"demo_{args.model}.json"), "w"))
