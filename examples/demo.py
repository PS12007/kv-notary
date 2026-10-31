"""End to end: a poisoned prefix cache changes an answer, and the notary catches it.

    python examples/demo.py [--model llama-1b] [--seed 3]

1. A memo is prefilled honestly, and the model answers a question about it.
2. An untrusted prefill worker forges the cache: it nudges the residual stream of ONE column
   and computes everything else honestly. The text is unchanged; the answer is not.
3. The notary audits ~6% of the columns and rejects the forged cache. It accepts the honest one.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))

from common import honest_tasks, verifier  # noqa: E402
from kvnotary import Model, Notary  # noqa: E402
from kvnotary.attacks import forge_attack, greedy  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="llama-1b")
ap.add_argument("--seed", type=int, default=3)
args = ap.parse_args()

m = Model(args.model)
notary = Notary(m, verifier(m).tau)
seed, task, e, cache, out, tgt = next(honest_tasks(m, 1, start=args.seed))
T = len(e.pre)
print(f"\nQ: {task.question}\nhonest cache ({T} columns, {cache.nbytes() / 2**20:.1f} MiB) -> {out!r}")
print("  audit:", notary.audit(e.pre, cache, e.suf, m.generate(e.suf, cache, max_new=8)))

pos = T // 3
fc, nll, _ = forge_attack(m, e.pre, e.suf, tgt, m.L // 2, [pos], steps=60)
ans = m.generate(e.suf, fc, max_new=8)
print(f"\nforged at column {pos} (token {m.tok.decode([e.pre[pos]])!r}), layer {m.L // 2} -> {greedy(m, fc, e.suf)!r}")
print("  audit:", notary.audit(e.pre, fc, e.suf, ans))
