"""In-place edits and random bit flips: what they do to answers, and how far they spread.

Part A (edits): an adversary with write access to stored caches optimises an additive change to
K and V of one column at one layer, without recomputing anything downstream. Every later column
that reads the edited one at a higher layer now disagrees with its own recomputation.

Part B (bit flips): one random bit of one bf16 element flips, as in silent data corruption on a
GPU, in host RAM, or on disk. We record whether the answer changes, how many columns the full
check flags, and whether a cheap range check (|x| above the largest honest magnitude) fires.

usage: python experiments/corrupt.py --model llama-1b --n 12 --flips 192
"""
import argparse
import time

import numpy as np
import torch

from common import greedy, honest_tasks, save, says, verifier
from kvnotary.attacks import edit_attack, flip_bit
from kvnotary.model import Model
from kvnotary.verify import column_errors, detect_plan, influence

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="llama-1b")
ap.add_argument("--n", type=int, default=12)
ap.add_argument("--flips", type=int, default=192)
args = ap.parse_args()

m = Model(args.model)
v = verifier(m)
L = m.L
layers = sorted({0, L // 4, L // 2, 3 * L // 4, L - 1})
edits, flips = [], []
t_start = time.time()


def full_scores(ids, cache):
    return (column_errors(m, ids, cache, range(len(ids))) / v.tau[:, None]).max(0).values.cpu().numpy()


for seed, task, e, c, out, tgt in honest_tasks(m, args.n, start=1000):
    T = len(e.pre)
    rng = np.random.default_rng(seed)
    # per-layer honest magnitude bound for the range check
    bound = torch.maximum(c.K.float().abs().amax((1, 2, 3)), c.V.float().abs().amax((1, 2, 3)))
    for kind, pos in (("value", e.span(task.answer)[-1]), ("random", int(rng.integers(4, T)))):
        for layer in layers:
            t0 = time.time()
            ec, nll = edit_attack(m, e.pre, e.suf, tgt, c, [layer], [pos], steps=60, rel_budget=1.0)
            o = greedy(m, ec, e.suf)
            ok = says(o, task.target) and not says(o, task.answer)
            sc = full_scores(e.pre, ec)
            fl = sc > 1
            edits.append({"seed": seed, "kind": kind, "layer": layer, "pos": pos, "T": T, "success": ok,
                          "out": o, "nll": nll, "n_flagged": int(fl.sum()),
                          "n_after": T - pos, "self_flagged": bool(fl[pos]),
                          "rand": {k: detect_plan(fl, np.zeros(T), 0, k) for k in (1, 4, 16)},
                          "sec": time.time() - t0})
            print(f"edit {seed} {kind} L{layer} ok={ok} '{o}' flagged={int(fl.sum())}/{T - pos} after", flush=True)
    honest_ids = m.generate(e.suf, c, max_new=8)
    for _ in range(args.flips):
        which = "K" if rng.random() < 0.5 else "V"
        layer, head = int(rng.integers(L)), int(rng.integers(m.Hkv))
        pos, dim, bit = int(rng.integers(T)), int(rng.integers(m.D)), int(rng.integers(16))
        fc = flip_bit(c, which, layer, head, pos, dim, bit)
        before = float((c.K if which == "K" else c.V)[layer, head, pos, dim])
        after = float((fc.K if which == "K" else fc.V)[layer, head, pos, dim])
        oids = m.generate(e.suf, fc, max_new=8)
        changed = oids != honest_ids
        sc = full_scores(e.pre, fc)
        fl = sc > 1
        rng_fire = bool(not np.isfinite(after) or abs(after) > 1.0 * float(bound[layer]))
        infl = influence(m, fc, list(e.suf) + list(oids[:-1]), T) if len(oids) else np.zeros(T)
        flips.append({"seed": seed, "which": which, "layer": layer, "pos": pos, "bit": bit,
                      "before": before, "after": after if np.isfinite(after) else str(after),
                      "changed": bool(changed), "wrong": not says(m.tok.decode(oids), task.answer),
                      "n_flagged": int(fl.sum()), "self_flagged": bool(fl[pos]), "range_check": rng_fire,
                      "top8": detect_plan(fl, infl, 8, 0) if len(oids) else 0.0,
                      "rand16": detect_plan(fl, np.zeros(T), 0, 16)})
    print(f"flips {seed}: changed {np.mean([f['changed'] for f in flips[-args.flips:]]):.3f}", flush=True)
    save(f"corrupt_{args.model}.json", {"model": args.model, "layers": layers, "edits": edits,
                                         "flips": flips, "minutes": (time.time() - t_start) / 60})
print("done", (time.time() - t_start) / 60)
