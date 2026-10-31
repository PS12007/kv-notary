"""CUDA-graph audits: the same column check, launched once instead of ~1000 times.

An audit of a few columns does little GPU work (about one read of the weights), so plain PyTorch
spends most of its time launching small kernels. `GraphedAudit` captures `column_errors` for a
fixed document length T and column count n, then replays it: each audit copies the claimed cache
and the column positions into static buffers and launches the whole computation as one graph.

    audit = GraphedAudit(m, T=len(ids), n=16)
    E = audit(ids, cache, positions)       # [L, n] relative errors, same as verify.column_errors

A receiver can land incoming caches straight in `audit.K` / `audit.V` (e.g. the transfer's
destination buffer) and pass `Cache(audit.K, audit.V)`; the audit then copies nothing.
"""
from __future__ import annotations

import torch

from .model import Cache, Model


class GraphedAudit:
    def __init__(self, m: Model, T: int, n: int, warmup: int = 2):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA graphs need a GPU")
        self.m, self.T, self.n = m, T, n
        d = m.device
        self.ids = torch.zeros(T, dtype=torch.long, device=d)
        self.P = torch.zeros(n, dtype=torch.long, device=d)
        shape = (m.L, m.Hkv, T, m.D)
        self.K = torch.zeros(shape, dtype=m.dtype, device=d)
        self.V = torch.zeros(shape, dtype=m.dtype, device=d)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(warmup):  # let allocators and cuBLAS settle before capture
                self._compute()
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.out = self._compute()

    @torch.no_grad()
    def _compute(self):
        cache = Cache(self.K, self.V)
        Kh, Vh = self.m.columns(self.ids, cache, self.P)
        Kh, Vh = Kh.float(), Vh.float()
        Kc = self.K.index_select(2, self.P).float()
        Vc = self.V.index_select(2, self.P).float()
        ek = (Kh - Kc).norm(dim=(1, 3)) / Kc.norm(dim=(1, 3)).clamp_min(1e-6)
        ev = (Vh - Vc).norm(dim=(1, 3)) / Vc.norm(dim=(1, 3)).clamp_min(1e-6)
        return torch.nan_to_num(torch.maximum(ek, ev), nan=1e9, posinf=1e9)

    @torch.no_grad()
    def __call__(self, ids, cache: Cache, positions) -> torch.Tensor:
        positions = list(positions)
        if len(ids) != self.T or len(positions) != self.n:
            raise ValueError(f"captured for T={self.T}, n={self.n}")
        self.ids.copy_(torch.as_tensor(ids, device=self.m.device))
        self.P.copy_(torch.as_tensor(positions, device=self.m.device))
        if cache.K.data_ptr() != self.K.data_ptr():  # zero-copy when the cache already lives in .K/.V
            self.K.copy_(cache.K[:, :, : self.T])
            self.V.copy_(cache.V[:, :, : self.T])
        self.graph.replay()
        return self.out.clone()
