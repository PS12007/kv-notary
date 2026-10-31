"""What an audit costs next to the prefill it replaces, from 512 to 16K tokens.

usage: python experiments/cost.py --model llama-1b
"""
import argparse
import os
import time

import numpy as np
import torch

from common import ROOT, save
from kvnotary.fast import GraphedAudit
from kvnotary.model import Cache, Model
from kvnotary.verify import column_errors, influence, trace_plan

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="llama-1b")
ap.add_argument("--max_t", type=int, default=16384)
args = ap.parse_args()
m = Model(args.model)
m.expand_kv_above = 0  # same (memory-efficient) attention kernel at every length; outputs are identical
text = open(os.path.join(ROOT, "data", "moby_dick.txt"), encoding="utf8").read()
allids = m.tok(text, add_special_tokens=False)["input_ids"]
readers = m.tok("\n\nQuestion: who is the captain of the ship, and what is he hunting?", add_special_tokens=False)["input_ids"]


def timeit(f, reps=5):
    f()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        f()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


rows = []
T = 512
while T <= args.max_t:
    ids = allids[:T]
    _, c = m.prefill(ids)
    rng = np.random.default_rng(0)
    r = {"T": T, "prefill": timeit(lambda: m.prefill(ids)),
         "influence": timeit(lambda: influence(m, c, readers, T)),
         "kv_mb": c.nbytes() / 2**20}
    for k in (8, 16, 64):
        P = sorted(rng.choice(T, size=k, replace=False).tolist())
        r[f"cols{k}"] = timeit(lambda: column_errors(m, ids, c, P))
    if T <= 4096:  # CUDA-graph audit, cache landed in its buffers (zero copy)
        for k in (16, 64):
            g = GraphedAudit(m, T, k)
            g.K.copy_(c.K)
            g.V.copy_(c.V)
            landed, P = Cache(g.K, g.V), sorted(rng.choice(T, size=k, replace=False).tolist())
            r[f"graph{k}"] = timeit(lambda: g(ids, landed, P))
            del g, landed
        torch.cuda.empty_cache()
    infl = influence(m, c, readers, T)
    r["trace32"] = timeit(lambda: trace_plan(m, ids, c, [infl], 32))  # picks 32 columns, audits 28 of them
    if T <= 4096:
        r["full_check"] = timeit(lambda: column_errors(m, ids, c, range(T)), reps=2)
    rows.append(r)
    print(r, flush=True)
    del c
    torch.cuda.empty_cache()
    T *= 2
save(f"cost_{args.model}.json", {"model": args.model, "gpu": torch.cuda.get_device_name(), "rows": rows})
