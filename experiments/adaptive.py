"""Adaptive attackers against the top-k + random audit.

The attacker knows the audit checks the 4 sink columns, the `hide_k` most-read other columns,
and some uniformly random ones. It forges m columns at once (m = 1 concentrates the edit, larger
m spreads it thin) and is penalised for any forged column ranking among the most-read. We record
whether it still flips the answer, where its columns land in the influence ranking, and the
probability each audit plan catches it. Plans also include a trace step: recompute the top
columns, follow their own attention one hop back, and check the columns they read most.

usage: python experiments/adaptive.py --model llama-1b --n 16
"""
import argparse
import time

import numpy as np
import torch

from common import greedy, honest_tasks, load, save, says, verifier
from kvnotary.attacks import stealth_forge
from kvnotary.model import Model
from kvnotary.verify import column_errors, detect_plan, influence

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="llama-1b")
ap.add_argument("--n", type=int, default=16)
ap.add_argument("--steps", type=int, default=100)
ap.add_argument("--baseline_only", action="store_true", help="rerun only the lam=0 trials into an existing result")
args = ap.parse_args()

m = Model(args.model)
v = verifier(m)
layer = m.L // 2
recs = []
t_start = time.time()


def trace_scores(e, fc, infl, k_top, T):
    """One hop of attention tracing from the top columns (uses rows the audit computed anyway)."""
    order = [int(i) for i in np.argsort(-infl)[:k_top]]
    _, A = column_errors(m, e.pre, fc, order, want_attn=True)
    A = A.cpu().numpy()
    w = np.where(np.isfinite(infl), infl, 0)
    hop = (w[order][:, None] * A).sum(0)
    hop[order] = -1
    hop[:4] = -1
    return hop


for seed, task, e, c, out, tgt in honest_tasks(m, args.n, start=500):
    T = len(e.pre)
    rng = np.random.default_rng(seed)
    trials = ([(1, 0.0, 4, "mean")] + [(1, 1.0, k, "mean") for k in (4, 28)]
              + [(1, 1.0, k, "layermax") for k in (4, 12, 28, 60)]
              + [(mc, 1.0, 12, "layermax") for mc in (4, 16, 64)])
    for mcols, lam, hide_k, smode in trials:
        pos = sorted(rng.choice(np.arange(4, T), size=mcols, replace=False).tolist())
        if not (args.baseline_only and lam != 0):
            t0 = time.time()
            fc, nll, d = stealth_forge(m, e.pre, e.suf, tgt, layer, pos, hide_k=hide_k,
                                       lam=lam, rel_budget=1.0, steps=args.steps, score_mode=smode)
            oids = m.generate(e.suf, fc, max_new=8)
            o = m.tok.decode(oids).strip()
            ok = says(o, task.target) and not says(o, task.answer)
            sc = (column_errors(m, e.pre, fc, range(T)) / v.tau[:, None]).max(0).values.cpu().numpy()
            flagged = sc > 1
            infl = influence(m, fc, list(e.suf) + list(oids[:-1]), T)
            ranks = [int((infl > infl[p]).sum()) - 4 for p in pos]
            alt, S = {}, {"mean": infl}
            for mode in ("layermax", "grad"):
                S[mode] = influence(m, fc, list(e.suf) + list(oids[:-1]), T, mode=mode)
                alt[mode] = [int((S[mode] > S[mode][p]).sum()) - 4 for p in pos]

            def ens(k):  # union of the top-k (non-sink) columns under each score, plus sinks
                cols = set(range(4))
                for sc_ in S.values():
                    cols |= set(int(i) for i in np.argsort(-sc_)[: 4 + k])
                return float(flagged[sorted(cols)].any()), len(cols)

            # does putting the honest forged columns back undo the attack? (direct vs laundered)
            rc = fc.clone()
            rc.K[:, :, pos], rc.V[:, :, pos] = c.K[:, :, pos], c.V[:, :, pos]
            direct = not says(greedy(m, rc, e.suf), task.target)
            hop = trace_scores(e, fc, infl, 4 + hide_k, T)
            hop_rank = [int((hop > hop[p]).sum()) for p in pos]
            lm = S["layermax"]
            r = {"direct": direct, "ens": {k: ens(k) for k in (4, 8, 16, 32)},"seed": seed, "m": mcols, "lam": lam, "hide_k": hide_k, "against": smode, "T": T, "alt_ranks": alt, "success": ok, "nll": nll, "out": o,
                 "target": task.target, "pos": pos, "ranks": ranks, "hop_ranks": hop_rank,
                 "n_flagged": int(flagged.sum()), "forged_flagged": int(flagged[pos].sum()),
                 "plans": {
                     "top8": detect_plan(flagged, infl, 8, 0),
                     "top32": detect_plan(flagged, infl, 32, 0),
                     "top64": detect_plan(flagged, infl, 64, 0),
                     "top32+rand32": detect_plan(flagged, infl, 32, 32),
                     "lm8": detect_plan(flagged, lm, 8, 0),
                     "lm16": detect_plan(flagged, lm, 16, 0),
                     "lm32": detect_plan(flagged, lm, 32, 0),
                     "lm16+rand16": detect_plan(flagged, lm, 16, 16),
                     "lm32+rand32": detect_plan(flagged, lm, 32, 32),
                     "top16": detect_plan(flagged, infl, 16, 0),
                     "top8+rand8": detect_plan(flagged, infl, 8, 8),
                     "top8+rand32": detect_plan(flagged, infl, 8, 32),
                     "rand16": detect_plan(flagged, np.zeros(T), 0, 16),
                     "rand64": detect_plan(flagged, np.zeros(T), 0, 64),
                     "top8+trace8": 1.0 if (detect_plan(flagged, infl, 8, 0) == 1.0 or
                                            flagged[np.argsort(-hop)[:8]].any()) else 0.0,
                 },
                 "sec": time.time() - t0}
            recs.append(r)
            print(f"{seed} m={mcols:2d} lam={lam} vs={smode} hide={hide_k} alt={ {k: min(v) for k, v in alt.items()} } ok={ok!s:5} nll={nll:.2f} '{o}' min_rank={min(ranks)} "
                  f"hop={min(hop_rank)} plans={r['plans']} {r['sec']:.0f}s", flush=True)
    if args.baseline_only:
        continue
    save(f"adaptive_{args.model}.json", {"model": args.model, "layer": layer,
                                          "records": recs, "minutes": (time.time() - t_start) / 60})
if args.baseline_only:  # swap the new lam=0 records into the saved result
    d = load(f"adaptive_{args.model}.json")
    d["records"] = [r for r in d["records"] if r["lam"] != 0] + recs
    save(f"adaptive_{args.model}.json", d)
print("done", (time.time() - t_start) / 60)
