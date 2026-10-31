"""One cache, many questions: collateral damage, and audits driven by other questions.

The attacker targets one question. A decode node that keeps a cache usually answers several
questions from it, and each one can drive an audit. For every successful forgery we ask the
memo's other nine questions and record:

  * collateral - does the forged cache change any other answer? (a targeted attack should not)
  * audit by another question - does a 32-column audit driven by question j's attention
    (all public scores merged) contain the forged column? And an audit driven by the union of
    all nine other questions, at the same 32-column budget?

Two attackers: a plain forger (§4) and a stealth forger that hides from all public scores of
the TARGET question (§6.1). The question is whether hiding from one reader hides from others.

usage: python experiments/multiq.py --model llama-1b --n 12
"""
import argparse
import dataclasses
import time

import numpy as np

from common import encode, greedy, honest_tasks, save, says, verifier
from kvnotary.attacks import forge_attack, stealth_forge
from kvnotary.model import Model
from kvnotary.verify import SCORES, column_errors, merge_plan, reader_attention, scores_from

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="llama-1b")
ap.add_argument("--n", type=int, default=12)
args = ap.parse_args()
m = Model(args.model)
v = verifier(m)
layer = m.L // 2
recs = []
t0 = time.time()


def public_scores(cache, suf, ans_ids, T):
    a, vn = reader_attention(m, cache, list(suf) + list(ans_ids[:-1]), T)
    return [scores_from(a, vn, md) for md in SCORES]


for seed, task, e, c, out, tgt in honest_tasks(m, args.n, start=6000):
    T = len(e.pre)
    rng = np.random.default_rng(seed)
    pos = int(rng.integers(4, T))
    others = [(k, q, a) for k, q, a in task.questions if q != task.question]
    sufs = [encode(m, dataclasses.replace(task, question=q)).suf for _, q, _ in others]
    honest_out = [greedy(m, c, s) for s in sufs]
    for name in ("plain", "stealth"):
        if name == "plain":
            fc, _, _ = forge_attack(m, e.pre, e.suf, tgt, layer, [pos], steps=40)
        else:
            fc, _, _ = stealth_forge(m, e.pre, e.suf, tgt, layer, [pos], hide_k=28, steps=150, score_mode=list(SCORES))
        oids = m.generate(e.suf, fc, max_new=8)
        o = m.tok.decode(oids).strip()
        ok = says(o, task.target) and not says(o, task.answer)
        r = {"seed": seed, "attacker": name, "pos": pos, "T": T, "success": ok, "out": o}
        if ok:
            flagged = (column_errors(m, e.pre, fc, range(T)) / v.tau[:, None]).max(0).values.cpu().numpy() > 1
            own = public_scores(fc, e.suf, oids, T)
            per_q, all_scores = [], []
            for (k, q, a), suf, h in zip(others, sufs, honest_out):
                fo = m.generate(suf, fc, max_new=8)
                f_txt = m.tok.decode(fo).strip()
                S = public_scores(fc, suf, fo, T)
                all_scores += S
                per_q.append({"kind": k, "honest": h, "forged": f_txt, "changed": f_txt != h,
                              "still_right": says(f_txt, a), "caught32": bool(flagged[merge_plan(S, 32)].any()),
                              "rank_layermax": int((S[1][4:] > S[1][pos]).sum())})
            r.update({"n_flagged": int(flagged.sum()), "own_caught32": bool(flagged[merge_plan(own, 32)].any()),
                      "others": per_q, "union_caught32": bool(flagged[merge_plan(all_scores, 32)].any()),
                      "union_all_caught32": bool(flagged[merge_plan(own + all_scores, 32)].any())})
            print(f"{seed} {name:7s} own={r['own_caught32']} union={r['union_caught32']} "
                  f"per-q caught {sum(x['caught32'] for x in per_q)}/{len(per_q)} collateral {sum(x['changed'] for x in per_q)}/{len(per_q)}",
                  flush=True)
        else:
            print(f"{seed} {name:7s} failed '{o}'", flush=True)
        recs.append(r)
    save(f"multiq_{args.model}.json", {"model": args.model, "layer": layer, "records": recs,
                                        "minutes": (time.time() - t0) / 60})
print("done", (time.time() - t0) / 60)
