"""The one-call API: audit a KV cache you did not compute before you trust its answer.

    from kvnotary import Model, Notary
    m = Model("llama-1b")
    notary = Notary(m)                      # loads or calibrates per-layer tolerances
    verdict = notary.audit(doc_ids, cache, question_ids, answer_ids)
    if not verdict.ok: recompute the prefill yourself

The audit checks the attention-sink columns, the columns the question and answer read most
(value-weighted attention, which bounds how much any column can change them), and a few random
columns, which catch attackers who spread an edit thin to stay out of the top. It returns which
columns failed and how long it took.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from .model import Cache, Model
from .verify import Verifier, column_errors, influence, plan


@dataclass
class Verdict:
    ok: bool
    checked: list
    failed: list
    worst: float  # largest score among checked columns (1.0 = at tolerance)
    seconds: float
    scores: dict = field(default_factory=dict)

    def __str__(self):
        s = "ACCEPT" if self.ok else f"REJECT (columns {self.failed[:8]} disagree with their tokens)"
        return f"{s}; checked {len(self.checked)} columns in {self.seconds * 1e3:.0f} ms, worst score {self.worst:.2f}"


class Notary:
    def __init__(self, m: Model, thresholds=None, k_top: int = 12, k_rand: int = 16, seed: int = 0):
        self.m = m
        self.v = Verifier(m, thresholds)
        self.k_top, self.k_rand = k_top, k_rand
        self.rng = np.random.default_rng(seed)

    def calibrate(self, docs, margin=1.5):
        return self.v.calibrate(docs, margin=margin)

    @torch.no_grad()
    def audit(self, ids, cache: Cache, question_ids, answer_ids=()) -> Verdict:
        if self.v.tau is None:
            raise RuntimeError("calibrate() first, or pass thresholds")
        t0 = time.perf_counter()
        T = len(ids)
        readers = list(question_ids) + list(answer_ids)[:-1] if len(answer_ids) else list(question_ids)
        infl = influence(self.m, cache, readers, T)
        cols = sorted(int(c) for c in plan(infl, self.k_top, self.k_rand, self.rng))
        E = column_errors(self.m, ids, cache, cols)
        sc = (E / self.v.tau[:, None]).max(0).values.cpu().numpy()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        failed = [c for c, s in zip(cols, sc) if s > 1]
        return Verdict(not failed, cols, failed, float(sc.max()), time.perf_counter() - t0,
                       {c: float(s) for c, s in zip(cols, sc)})
