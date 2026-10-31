"""Attacks that hide inside the verifier's numerical tolerance.

A bf16 serving stack cannot reproduce its own cache bit for bit, so the verifier must accept
relative errors up to a calibrated threshold tau_l per layer. Two attackers try to live inside it:

* edit  - change every stored entry in place by at most rho x (largest honest error). Each edit
          also shifts the recomputation of every later column, so the errors compound.
* forge - nudge the residual stream of EVERY document column at one layer by a relative eps and
          compute the rest honestly: each column then carries only its own small inconsistency.

If either flips answers with every column under tolerance, the tolerance is the attack surface,
and the only fix is a tighter reference (deterministic kernels or a higher-precision check).

usage: python experiments/innoise.py --model llama-1b --dtype bf16 --n 12
"""
import argparse
import time

import numpy as np
import torch

from common import greedy, honest_tasks, save, says, verifier
from kvnotary.attacks import edit_attack, forge_attack
from kvnotary.model import Model
from kvnotary.verify import Verifier, column_errors

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="llama-1b")
ap.add_argument("--dtype", default="bf16")
ap.add_argument("--n", type=int, default=12)
args = ap.parse_args()
m = Model(args.model, dtype=torch.bfloat16 if args.dtype == "bf16" else torch.float32)
v = verifier(m)
recs = []
t_start = time.time()
edit_layers = [i for i in range(m.L) if float(v.tau[i]) > Verifier.TAU_FLOOR * 10]
EPS = (0.001, 0.003, 0.01, 0.03, 0.1) if args.dtype == "bf16" else (1e-5, 1e-4, 1e-3, 0.01, 0.1)
for seed, task, e, c, out, tgt in honest_tasks(m, args.n, start=2000):
    T = len(e.pre)
    trials = [("edit", r) for r in (0.1, 0.3, 1.0)] + [("forge", eps) for eps in EPS]
    for mode, x in trials:
        t0 = time.time()
        if mode == "edit":
            budget = (x * v.tau[edit_layers] / 1.5).view(-1, 1, 1, 1)
            ac, nll = edit_attack(m, e.pre, e.suf, tgt, c, edit_layers, list(range(T)),
                                  steps=100, lr=0.02, rel_budget=budget)
        else:
            ac, nll, _ = forge_attack(m, e.pre, e.suf, tgt, m.L // 2, list(range(4, T)),
                                      steps=60, lr=0.1, rel_budget=x)
        o = greedy(m, ac, e.suf)
        ok = says(o, task.target) and not says(o, task.answer)
        sc = (column_errors(m, e.pre, ac, range(T)) / v.tau[:, None]).max(0).values.cpu().numpy()
        recs.append({"seed": seed, "mode": mode, "x": x, "success": ok, "out": o, "nll": nll, "T": T,
                     "max_score": float(sc.max()), "n_flagged": int((sc > 1).sum()),
                     "q99_score": float(np.quantile(sc, 0.99)), "sec": time.time() - t0})
        print(f"{seed} {mode} x={x} ok={ok} nll={nll:.2f} '{o}' max_score={sc.max():.2f} "
              f"flagged={int((sc > 1).sum())}/{T}", flush=True)
    save(f"innoise_{args.model}_{args.dtype}.json", {"model": args.model, "dtype": args.dtype, "tau": v.tau,
                                                     "records": recs, "minutes": (time.time() - t_start) / 60})
print("done")
