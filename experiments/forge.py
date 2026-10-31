"""Consistent single-column forgeries: can they flip answers, and which audits catch them?

For each honest task, an untrusted prefill worker perturbs the residual stream of ONE document
column at one layer and runs the rest of the prefill honestly. Then:
  * success  - the trusted decode node, reading the forged cache, says the attacker's value;
  * flagged  - which columns fail a full column check (should be exactly the forged one);
  * detect   - probability each audit plan catches it, at a range of column budgets.

usage: python experiments/forge.py --model llama-1b --n 24
"""
import argparse
import time

import numpy as np
import torch

from common import greedy, honest_tasks, save, says, verifier
from kvnotary.attacks import forge_attack
from kvnotary.model import Model
from kvnotary.verify import column_errors, detect_plan, influence

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="llama-1b")
ap.add_argument("--n", type=int, default=24)
ap.add_argument("--steps", type=int, default=40)
ap.add_argument("--long", type=int, default=0, help="haystack characters (0 = short memo)")
args = ap.parse_args()

m = Model(args.model)
v = verifier(m)
L = m.L
layers = sorted({1, L // 4, L // 2, 3 * L // 4, L - 1})
configs = [(l, 1.0) for l in layers] + [(L // 2, b) for b in (0.25, 0.5, 2.0)]
if args.long:
    configs = [(L // 4, 1.0), (L // 2, 1.0), (3 * L // 4, 1.0)]
tag = f"{args.model}_long{args.long}" if args.long else args.model
KT = [1, 2, 4, 8, 16, 32]
KR = [4, 16, 64]
recs, honest = [], []
t_start = time.time()
for seed, task, e, c, out, tgt in honest_tasks(m, args.n, start=3000 if args.long else 0, long_chars=args.long or None):
    T = len(e.pre)
    sc_h = (column_errors(m, e.pre, c, range(T)) / v.tau[:, None]).max(0).values.cpu().numpy()
    honest.append({"seed": seed, "T": T, "max_score": float(sc_h.max()), "n_flagged": int((sc_h > 1).sum())})
    rng = np.random.default_rng(seed)
    kinds = {"value": e.span(task.answer)[-1], "random": int(rng.integers(4, T)), "last": T - 1}
    for kind, pos in kinds.items():
        for layer, budget in configs:
            t0 = time.time()
            fc, nll, d = forge_attack(m, e.pre, e.suf, tgt, layer, [pos], steps=args.steps, rel_budget=budget)
            oids = m.generate(e.suf, fc, max_new=8)
            o = m.tok.decode(oids).strip()
            ok = says(o, task.target) and not says(o, task.answer)
            sc = (column_errors(m, e.pre, fc, range(T)) / v.tau[:, None]).max(0).values.cpu().numpy()
            flagged = sc > 1
            readers = list(e.suf) + list(oids[:-1])
            infl = influence(m, fc, readers, T)
            r = {"seed": seed, "kind": kind, "pos": pos, "layer": layer, "budget": budget, "T": T,
                 "nll": nll, "out": o, "honest_out": out, "target": task.target, "success": ok,
                 "flagged": np.flatnonzero(flagged).tolist(), "pos_score": float(sc[pos]),
                 "infl_rank": int((infl > infl[pos]).sum()),
                 "top": {k: detect_plan(flagged, infl, k, 0) for k in KT},
                 "top8_rand8": detect_plan(flagged, infl, 8, 8),
                 "uniform": {k: detect_plan(flagged, np.zeros(T), 0, k) for k in KR},
                 "sec": time.time() - t0}
            recs.append(r)
            print(f"{seed} {kind:6s} L{layer:2d} b{budget:<4} ok={ok!s:5} nll={nll:.2f} '{o}' "
                  f"flag={r['flagged'][:6]} rank={r['infl_rank']} {r['sec']:.1f}s", flush=True)
    save(f"forge_{tag}.json", {"model": args.model, "layers": layers, "configs": configs,
                                       "records": recs, "honest": honest, "tau": v.tau,
                                       "minutes": (time.time() - t_start) / 60})
print("done", (time.time() - t_start) / 60, "min")
