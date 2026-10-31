"""Exactness tests on tiny random Llama / Qwen2 models (CPU, fp32, no downloads)."""
import numpy as np
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM

from kvnotary.attacks import flip_bit, forge_cache
from kvnotary.model import Cache, Model
from kvnotary.verify import column_errors, detect_plan, detect_prob, uniform

COMMON = dict(vocab_size=97, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
              num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
              tie_word_embeddings=True)


def _build(kind, path):
    torch.manual_seed(0)
    if kind == "llama":
        cfg = LlamaConfig(**COMMON, rope_theta=500000.0, head_dim=16,
                          rope_scaling={"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
                                        "high_freq_factor": 4.0, "original_max_position_embeddings": 64})
        hf = LlamaForCausalLM(cfg)
    else:
        cfg = Qwen2Config(**COMMON, rope_theta=1e6)
        hf = Qwen2ForCausalLM(cfg)
        with torch.no_grad():  # random non-zero biases so the bias path is exercised
            for lay in hf.model.layers:
                for p in (lay.self_attn.q_proj, lay.self_attn.k_proj, lay.self_attn.v_proj):
                    p.bias.normal_(0, 0.5)
    hf.save_pretrained(path, safe_serialization=True)
    return hf.eval()


@pytest.fixture(scope="module", params=["llama", "qwen"])
def pair(request, tmp_path_factory):
    path = str(tmp_path_factory.mktemp(request.param))
    hf = _build(request.param, path)
    return hf, Model(path, device="cpu", dtype=torch.float32)


IDS = list(np.random.default_rng(0).integers(0, 97, size=120))


def test_logits_match_transformers(pair):
    hf, m = pair
    lg, _, _ = m.run(IDS, all_logits=True)
    with torch.no_grad():
        ref = hf(torch.tensor([IDS])).logits[0]
    assert (lg - ref).abs().max() < 1e-4


def test_incremental_equals_full(pair):
    _, m = pair
    _, c = m.prefill(IDS)
    _, c2 = m.prefill(IDS[70:], past=Cache(c.K[:, :, :70], c.V[:, :, :70]))
    assert (c2.K - c.K[:, :, 70:]).abs().max() < 1e-4
    assert (c2.V - c.V[:, :, 70:]).abs().max() < 1e-4


def test_columns_reproduce_honest_cache(pair):
    _, m = pair
    _, c = m.prefill(IDS)
    E = column_errors(m, IDS, c, range(len(IDS)))
    assert E.max() < 1e-4


def test_forgery_breaks_only_forged_column(pair):
    _, m = pair
    torch.manual_seed(1)
    fc = forge_cache(m, IDS, layer=1, positions=[50], delta=torch.randn(1, m.d) * 3)
    E = column_errors(m, IDS, fc, range(len(IDS))).max(0).values
    bad = torch.nonzero(E > 1e-3).flatten().tolist()
    assert bad == [50]


def test_inplace_edit_contaminates_later_columns(pair):
    _, m = pair
    _, c = m.prefill(IDS)
    ec = c.clone()
    ec.V[0, :, 30] += 5.0
    E = column_errors(m, IDS, ec, range(len(IDS))).max(0).values
    assert E[30] > 1e-3 and (E[31:] > 1e-3).float().mean() > 0.9 and (E[:30] < 1e-4).all()


def test_flip_bit_changes_one_element(pair):
    _, m = pair
    _, c = m.prefill(IDS)
    c = Cache(c.K.bfloat16(), c.V.bfloat16())
    fc = flip_bit(c, "K", 1, 0, 5, 3, 14)
    diff = (fc.K.float() != c.K.float())
    assert diff.sum() == 1 and diff[1, 0, 5, 3]
    assert (flip_bit(fc, "K", 1, 0, 5, 3, 14).K.view(torch.int16) == c.K.view(torch.int16)).all()


def test_detect_plan_matches_monte_carlo():
    T = 200
    flagged = np.zeros(T, dtype=bool)
    flagged[[10, 99, 150]] = True
    exact = detect_plan(flagged, np.zeros(T), 0, 20)
    mc = detect_prob(flagged, lambda r: uniform(T, 20, r), trials=20000)
    assert abs(exact - mc) < 0.02
    score = np.arange(T, dtype=float)
    score[99] = 1e9
    assert detect_plan(flagged, score, 1, 0) == 1.0


def test_expanded_kv_path_matches_gqa_path(pair):
    _, m = pair
    lg, c = m.prefill(IDS)
    m.expand_kv_above = 0  # force the long-input path
    try:
        lg2, c2 = m.prefill(IDS)
        _, c3 = m.prefill(IDS[70:], past=Cache(c.K[:, :, :70], c.V[:, :, :70]))
    finally:
        m.expand_kv_above = 1024
    assert (lg - lg2).abs().max() < 1e-4
    assert (c2.K - c.K).abs().max() < 1e-4
    assert (c3.V - c.V[:, :, 70:]).abs().max() < 1e-4


def test_score_family():
    from kvnotary.verify import merge_plan, vw_score
    torch.manual_seed(0)
    L, H, n, T = 3, 4, 5, 40
    a = torch.rand(L, H, n, T).softmax(-1).requires_grad_(True)
    vn = torch.rand(L, H, T) + 0.5
    c = a * vn[:, :, None, :]
    per_layer = c.amax(1).sum(1)
    assert torch.allclose(vw_score(a, vn, "mean"), per_layer.mean(0))
    lm = per_layer / per_layer[:, 4:].sum(-1, keepdim=True)
    assert torch.allclose(vw_score(a, vn, "layermax"), lm.amax(0))
    mask = torch.zeros(L, H, dtype=torch.bool)
    mask[1, 2] = True
    w = torch.ones(L, H)
    hm = c.sum(2) / c.sum(2)[..., 4:].sum(-1, keepdim=True)
    assert torch.allclose(vw_score(a, vn, "secret", secret=(mask, w)), hm[1, 2])
    for mode in ("mean", "layermax", "headmax", "tokmax"):
        a.grad = None
        vw_score(a, vn, mode)[10].backward()
        assert a.grad.abs().sum() > 0
    s1, s2 = np.arange(T, dtype=float), np.arange(T, dtype=float)[::-1].copy()
    cols = merge_plan([s1, s2], 10)
    assert cols[:4] == [0, 1, 2, 3] and len(set(cols)) == 10 and T - 1 in cols and 4 in cols


def test_trace_rows(pair):
    from kvnotary.verify import trace_plan
    _, m = pair
    _, c = m.prefill(IDS)
    P = [10, 40, 90]
    _, _, tr = m.columns(IDS, c, P, want_attn="trace")
    assert tr.shape == (3, len(IDS))
    assert (tr[:, :4] == 0).all()
    for i, p in enumerate(P):
        assert (tr[i, p:] == 0).all() and tr[i, 4:p].max() <= 1 + 1e-6
    s = np.random.default_rng(0).random(len(IDS))
    cols = trace_plan(m, IDS, c, [s], 20)
    assert len(cols) == len(set(cols)) <= 20 and cols[:4] == [0, 1, 2, 3]


def test_notary_rejects_forgery_accepts_honest(pair):
    from kvnotary import Notary
    _, m = pair
    _, c = m.prefill(IDS)
    notary = Notary(m, thresholds=torch.full((m.L,), 1e-3), k_top=len(IDS), k_rand=0)
    q = [5, 6, 7]
    assert notary.audit(IDS, c, q).ok
    fc = forge_cache(m, IDS, 1, [50], torch.randn(1, m.d) * 5)
    v = notary.audit(IDS, fc, q)
    assert not v.ok and v.failed == [50]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")
def test_graphed_audit_matches_eager(tmp_path_factory):
    from kvnotary.fast import GraphedAudit
    path = tmp_path_factory.mktemp("g") / "llama"
    _build("llama", str(path))
    m = Model(str(path), device="cuda", dtype=torch.float32)
    _, c = m.prefill(IDS)
    P = [5, 40, 90, 110]
    g = GraphedAudit(m, len(IDS), len(P))
    assert torch.equal(g(IDS, c, P), column_errors(m, IDS, c, P))
    fc = forge_cache(m, IDS, 1, [40], torch.randn(1, m.d, device="cuda") * 5)
    assert g(IDS, fc, P)[:, 1].max() > 1e-2
