"""Shared setup: models, honest tasks, calibrated verifiers. Results go to results/*.json."""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kvnotary.attacks import greedy, says  # noqa: E402
from kvnotary.model import Model  # noqa: E402
from kvnotary.tasks import answer_ids, encode, make_long_task, make_task  # noqa: E402
from kvnotary.verify import Verifier  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
CAL_SEEDS = range(100000, 100020)  # calibration documents, disjoint from evaluation seeds


def save(name, obj):
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, name), "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))


def load(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


def verifier(m, margin=1.5):
    """Thresholds calibrated on CAL_SEEDS and cached in results/tau_<model>_<dtype>.json."""
    tag = f"tau_{m.name}_{str(m.dtype).split('.')[-1]}.json"
    v = Verifier(m)
    p = os.path.join(RES, tag)
    if os.path.exists(p):
        v.tau = torch.tensor(load(tag)["tau"], device=m.device).clamp_min(Verifier.TAU_FLOOR)
    else:
        t0 = time.time()
        raw = v.calibrate([encode(m, make_task(s)).pre for s in CAL_SEEDS], margin=1.0)
        v.tau = (raw * margin).clamp_min(Verifier.TAU_FLOOR)
        save(tag, {"tau": v.tau, "raw_max": raw, "margin": margin, "seconds": time.time() - t0})
    return v


def honest_tasks(m, n, start=0, long_chars=None, **kw):
    """Yield (seed, task, encoded, cache, honest_output, target_ids) for tasks the model gets
    right with an honest cache. The target is the honest output with the answer swapped, so
    the attacker asks for exactly the format the model would use."""
    got, s = 0, start
    while got < n:
        t = make_long_task(s, long_chars) if long_chars else make_task(s, **kw)
        e = encode(m, t)
        _, c = m.prefill(e.pre)
        out = greedy(m, c, e.suf)
        if says(out, t.answer) and t.answer in out:
            yield s, t, e, c, out, answer_ids(m, out.replace(t.answer, t.target))
            got += 1
        s += 1
