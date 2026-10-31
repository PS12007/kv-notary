"""Prefill with one stack, verify with another: Hugging Face transformers vs kv-notary's forward.

A real verifier rarely runs the same kernels as the prefill worker. Here the cache comes from
transformers (bf16, SDPA) and is checked by kv-notary's own column recomputation.

  1. honest transformers caches, scored with thresholds calibrated on kv-notary's own caches
     (false alarms if the stacks round differently);
  2. thresholds calibrated on transformers caches instead (how much the tolerance must grow);
  3. single-column forgeries (built with kv-notary's forward, as in forge.py) under both
     thresholds, and whether transformers' own decoding is fooled by them.

usage: python experiments/crossstack.py --model llama-1b --n 24
"""
import argparse
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from common import CAL_SEEDS, encode, honest_tasks, make_task, save, says, verifier
from kvnotary.attacks import forge_attack
from kvnotary.hf import hf_generate, hf_prefill
from kvnotary.model import Model, snapshot_dir
from kvnotary.verify import column_errors

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="llama-1b")
ap.add_argument("--n", type=int, default=24)
args = ap.parse_args()

m = Model(args.model)
v = verifier(m)
hf = AutoModelForCausalLM.from_pretrained(snapshot_dir(args.model), dtype=torch.bfloat16).to(m.device).eval()
t0 = time.time()


def errors(ids, cache):
    return column_errors(m, ids, cache, range(len(ids)))  # [L, T]


# thresholds calibrated on transformers caches, same documents and margin as the native ones
raw = None
for s in CAL_SEEDS:
    ids = encode(m, make_task(s)).pre
    e = errors(ids, hf_prefill(hf, ids)).max(1).values
    raw = e if raw is None else torch.maximum(raw, e)
tau_hf = (1.5 * raw).clamp_min(1e-6)
print("tau ratio hf/native per layer", (tau_hf / v.tau).cpu().numpy().round(2), flush=True)

recs = []
for seed, task, e, c, out, tgt in honest_tasks(m, args.n, start=5000):
    T = len(e.pre)
    hc = hf_prefill(hf, e.pre)
    E = errors(e.pre, hc)
    s_nat = (E / v.tau[:, None]).max(0).values
    s_hf = (E / tau_hf[:, None]).max(0).values
    hf_out = m.tok.decode(hf_generate(hf, hc, e.suf)).strip()
    rng = np.random.default_rng(seed)
    pos = int(rng.integers(4, T))
    fc, nll, _ = forge_attack(m, e.pre, e.suf, tgt, m.L // 2, [pos], steps=40)
    Ef = errors(e.pre, fc)
    f_nat = (Ef / v.tau[:, None]).max(0).values
    f_hf = (Ef / tau_hf[:, None]).max(0).values
    ours = m.tok.decode(m.generate(e.suf, fc, max_new=8)).strip()
    theirs = m.tok.decode(hf_generate(hf, fc, e.suf)).strip()
    r = {"seed": seed, "T": T, "hf_out": hf_out, "hf_correct": says(hf_out, task.answer),
         "honest_max_native": float(s_nat.max()), "honest_flagged_native": int((s_nat > 1).sum()),
         "honest_max_hf": float(s_hf.max()), "honest_flagged_hf": int((s_hf > 1).sum()),
         "forge_ok_ours": says(ours, task.target) and not says(ours, task.answer),
         "forge_ok_hf": says(theirs, task.target) and not says(theirs, task.answer),
         "forged_score_native": float(f_nat[pos]), "forged_score_hf": float(f_hf[pos]),
         "forge_flagged_native": int((f_nat > 1).sum()), "forge_flagged_hf": int((f_hf > 1).sum()),
         "out_ours": ours, "out_hf": theirs}
    recs.append(r)
    print(f"{seed} honest max {r['honest_max_native']:.2f}/{r['honest_max_hf']:.2f} flagged "
          f"{r['honest_flagged_native']}/{r['honest_flagged_hf']} | forge ours={r['forge_ok_ours']} hf={r['forge_ok_hf']} "
          f"score {r['forged_score_native']:.1f}/{r['forged_score_hf']:.1f}", flush=True)
    save(f"crossstack_{args.model}.json", {"model": args.model, "tau_native": v.tau, "tau_hf": tau_hf,
                                            "records": recs, "minutes": (time.time() - t0) / 60})
print("done", (time.time() - t0) / 60)
