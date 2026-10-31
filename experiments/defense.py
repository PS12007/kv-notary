"""Round two of the arms race: can the verifier pick columns in a way the attacker cannot game?

Each attacker forges one random column at the middle layer (beta = 1) and hides it below the 28
most-read columns of one or more public scores (or of all of them at once). Then every defender
plan audits the same number of columns:

  * one public score (mean, layermax, headmax, tokmax, grad), top-B
  * all public scores merged round-robin into one budget of B
  * a SECRET score: headmax over a random quarter of heads with random weights, drawn by the
    verifier per audit and never shown to the attacker (averaged over 32 secret draws)
  * the secret score merged with the public ones
  * half the budget on the merged public scores, half uniformly random
  * TRACE: half the budget on the merged public scores, the rest following what those
    audited columns themselves read (attention back-tracing, see verify.trace_plan)
  * trace on half the budget, the other half uniformly random
  * B uniformly random columns

Detection probabilities are exact for random parts (hypergeometric) and averaged over secrets.

usage: python experiments/defense.py --model llama-1b --n 12
"""
import argparse
import math
import time

import numpy as np

from common import honest_tasks, save, says, verifier
from kvnotary.attacks import stealth_forge
from kvnotary.model import Model
from kvnotary.verify import (SCORES, column_errors, influence, merge_plan, reader_attention,
                             scores_from, secret_key, trace_plan)

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="llama-1b")
ap.add_argument("--n", type=int, default=12)
ap.add_argument("--steps", type=int, default=150)
ap.add_argument("--hide", type=int, default=28)
ap.add_argument("--secrets", type=int, default=32)
args = ap.parse_args()

m = Model(args.model)
v = verifier(m)
layer = m.L // 2
BUDGETS = (16, 32, 64)
ATTACKERS = [("none", ["mean"], 0.0), ("mean", ["mean"], 1.0), ("layermax", ["layermax"], 1.0),
             ("headmax", ["headmax"], 1.0), ("tokmax", ["tokmax"], 1.0),
             ("all public", list(SCORES), 1.0), ("all public + trace", list(SCORES) + ["trace"], 1.0)]
key_rng = np.random.default_rng(12345)
KEYS = [secret_key(m, key_rng) for _ in range(args.secrets)]  # the verifier's secrets
recs = []
t_start = time.time()


def p_random(n_flag_left, n_left, r):
    """P(at least one of n_flag_left bad columns among r drawn uniformly from n_left)."""
    if n_flag_left == 0 or r <= 0:
        return 0.0
    if r > n_left - n_flag_left:
        return 1.0
    return 1.0 - math.exp(math.lgamma(n_left - n_flag_left + 1) - math.lgamma(n_left - n_flag_left - r + 1)
                          - math.lgamma(n_left + 1) + math.lgamma(n_left - r + 1))


def hit(flagged, cols):
    return float(flagged[cols].any())


def mix(flagged, cols, B, T):
    """cols chosen deterministically, then the rest of the budget B uniformly at random."""
    if flagged[cols].any():
        return 1.0
    return p_random(int(flagged.sum()), T - len(cols), B - len(cols))


for seed, task, e, c, out, tgt in honest_tasks(m, args.n, start=600):
    T = len(e.pre)
    rng = np.random.default_rng(seed)
    pos = [int(rng.integers(4, T))]
    for name, modes, lam in ATTACKERS:
        t0 = time.time()
        fc, nll, _ = stealth_forge(m, e.pre, e.suf, tgt, layer, pos, hide_k=args.hide, lam=lam,
                                   steps=args.steps, score_mode=modes)
        oids = m.generate(e.suf, fc, max_new=8)
        o = m.tok.decode(oids).strip()
        ok = says(o, task.target) and not says(o, task.answer)
        r = {"seed": seed, "attacker": name, "pos": pos, "T": T, "success": ok, "nll": nll, "out": o,
             "target": task.target}
        if ok:
            sc = (column_errors(m, e.pre, fc, range(T)) / v.tau[:, None]).max(0).values.cpu().numpy()
            flagged = sc > 1
            readers = list(e.suf) + list(oids[:-1])
            a, vn = reader_attention(m, fc, readers, T)
            S = {mode: scores_from(a, vn, mode) for mode in SCORES}
            S["grad"] = influence(m, fc, readers, T, mode="grad")
            sec = [scores_from(a, vn, "secret", secret=k) for k in KEYS]
            rank = lambda s, p: int((s[4:] > s[p]).sum())  # among non-sink columns, 0 = top
            r.update({
                "n_flagged": int(flagged.sum()),
                "ranks": {k: rank(s, pos[0]) for k, s in S.items()},
                "secret_ranks": [rank(s, pos[0]) for s in sec],
                "detect": {B: {
                    **{k: hit(flagged, merge_plan([s], B)) for k, s in S.items()},
                    "all public": hit(flagged, merge_plan(list(S.values()), B)),
                    "secret": float(np.mean([hit(flagged, merge_plan([s], B)) for s in sec])),
                    "secret + public": float(np.mean([hit(flagged, merge_plan([s] + list(S.values()), B)) for s in sec])),
                    "half public + half random": mix(flagged, merge_plan(list(S.values()), B // 2), B, T),
                    "trace": hit(flagged, trace_plan(m, e.pre, fc, list(S.values()), B)),
                    "trace + random": mix(flagged, trace_plan(m, e.pre, fc, list(S.values()), B // 2), B, T),
                    "random": p_random(int(flagged.sum()), T, B),
                } for B in BUDGETS},
            })
        r["sec"] = time.time() - t0
        recs.append(r)
        d = r.get("detect", {}).get(32, {})
        print(f"{seed} vs {name:10s} ok={ok!s:5} nll={nll:.2f} '{o}' ranks={r.get('ranks')} "
              f"secret_med={np.median(r['secret_ranks']) if ok else '-'} "
              f"d32: all={d.get('all public', '-')} secret={d.get('secret', '-')} trace={d.get('trace', '-')} {r['sec']:.0f}s", flush=True)
    save(f"defense_{args.model}.json", {"model": args.model, "layer": layer, "hide": args.hide,
                                         "budgets": BUDGETS, "records": recs,
                                         "minutes": (time.time() - t_start) / 60})
print("done", (time.time() - t_start) / 60)
