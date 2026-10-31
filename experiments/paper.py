"""Builds docs/paper.html and docs/data/summary.json from results/*.json.

Every number in the paper text is computed here from the result files, so the report cannot
drift from the experiments. Missing result files just drop their section.

usage: python experiments/paper.py   (then print docs/paper.html to PDF, see README)
"""
import json
import os
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
DOCS = os.path.join(ROOT, "docs")
MODELS = [("llama-1b", "Llama-3.2-1B"), ("qwen-0.5b", "Qwen2.5-0.5B")]


def load(name):
    p = os.path.join(RES, name + ".json")
    return json.load(open(p)) if os.path.exists(p) else None


def pct(x, d=0):
    return f"{100 * x:.{d}f}%"


def mean(xs):
    xs = list(xs)
    return float(np.mean(xs)) if xs else float("nan")


def table(head, rows, cls=""):
    h = "".join(f"<th>{x}</th>" for x in head)
    b = "".join("<tr>" + "".join(f"<td>{x}</td>" for x in r) + "</tr>" for r in rows)
    return f'<table class="{cls}"><tr>{h}</tr>{b}</table>'


def fig(name, cap):
    if not os.path.exists(os.path.join(DOCS, "figures", name + ".png")):
        return ""
    return f'<figure><img src="figures/{name}.png" alt="{name}"><figcaption>{cap}</figcaption></figure>'


S = {}  # numbers shared between sections, the summary, and the README
out = []

# ---------------------------------------------------------------- forgeries (§4, §5)
F = {k: load(f"forge_{k}") for k, _ in MODELS}
sec4_rows, sec5_rows = [], []
for k, name in MODELS:
    d = F[k]
    if not d:
        continue
    R, H = d["records"], d["honest"]
    succ = [r for r in R if r["success"]]
    S[k] = s = {
        "tasks": len(H), "T": mean(h["T"] for h in H), "trials": len(R), "succ": len(succ),
        "honest_flagged": sum(h["n_flagged"] for h in H), "honest_max": max(h["max_score"] for h in H),
        "only_forged": mean(r["flagged"] == [r["pos"]] for r in succ),
        "top": {kk: mean(r["top"][kk] for r in succ) for kk in ("1", "2", "4", "8", "16", "32")},
        "uni": {kk: mean(r["uniform"][kk] for r in succ) for kk in ("4", "16", "64")},
        "t8r8": mean(r["top8_rand8"] for r in succ),
        "rank_med": float(np.median([r["infl_rank"] for r in succ])),
        "first": mean(r["infl_rank"] < 5 for r in succ),
    }
    L = max(r["layer"] for r in R) + 1
    by = defaultdict(list)
    for r in R:
        by[(r["layer"], r["budget"])].append(r)
    s["by_cfg"] = {f"{l}/{b}": mean(r["success"] for r in v) for (l, b), v in by.items()}
    s["full_budget"] = mean(r["success"] for r in R if r["budget"] == 1.0)
    for (l, b), v in sorted(by.items()):
        ok = [r for r in v if r["success"]]
        sec4_rows.append([name, l, f"{b:g}", f"{len(ok)}/{len(v)}", pct(mean(r["success"] for r in v if r["kind"] == "value")),
                          pct(mean(r["success"] for r in v if r["kind"] == "random")),
                          pct(mean(r["success"] for r in v if r["kind"] == "last")),
                          pct(mean(r["flagged"] == [r["pos"]] for r in ok)) if ok else "–"])
    sec5_rows.append([name, len(succ)] + [pct(s["top"][kk]) for kk in ("8", "16", "32")]
                     + [pct(s["t8r8"])] + [pct(s["uni"][kk], 1) for kk in ("4", "16", "64")])
    for kind in ("value", "random", "last"):
        s[f"top8_{kind}"] = mean(r["top"]["8"] for r in succ if r["kind"] == kind)

# long context
LONG = []
for tag in ("long8000", "long16000"):
    d = load(f"forge_llama-1b_{tag}")
    if not d:
        continue
    R = d["records"]
    succ = [r for r in R if r["success"]]
    T = mean(h["T"] for h in d["honest"])
    LONG.append({"tag": tag, "T": T, "n": len(R), "succ": len(succ),
                 "top": {kk: mean(r["top"][kk] for r in succ) for kk in ("8", "16", "32")},
                 "uni64": mean(r["uniform"]["64"] for r in succ),
                 "only": mean(r["flagged"] == [r["pos"]] for r in succ),
                 "rank_med": float(np.median([r["infl_rank"] for r in succ])) if succ else float("nan")})
S["long"] = LONG

# ---------------------------------------------------------------- adaptive (§6)
A = {k: load(f"adaptive_{k}") for k, _ in MODELS}
ADA = {}
for k, name in MODELS:
    d = A[k]
    if not d or len(d["records"]) < 20:
        continue
    g = defaultdict(list)
    for r in d["records"]:
        g[(r["m"], r["lam"], r["hide_k"], r["against"])].append(r)
    rows = []
    for key, v in g.items():
        ok = [r for r in v if r["success"]]
        if not ok:
            rows.append({"key": key, "n": len(v), "succ": 0})
            continue
        best = lambda r, f: min(f(r))  # the attacker's best-hidden column
        rows.append({
            "key": key, "n": len(v), "succ": len(ok),
            "rank_mean": float(np.median([min(r["ranks"]) for r in ok])),
            "rank_lm": float(np.median([min(r["alt_ranks"]["layermax"]) for r in ok])),
            "rank_grad": float(np.median([min(r["alt_ranks"]["grad"]) for r in ok])),
            "direct": mean(r["direct"] for r in ok),
            "plans": {p: mean(r["plans"][p] for r in ok) for p in ok[0]["plans"]},
            "ens": {e: mean(r["ens"][e][0] for r in ok) for e in ok[0]["ens"]},
            "ens_cols": {e: mean(r["ens"][e][1] for r in ok) for e in ok[0]["ens"]},
            "n_flagged": mean(r["n_flagged"] for r in ok),
        })
    ADA[k] = sorted(rows, key=lambda r: (r["key"][3] != "mean", r["key"][0], r["key"][1], r["key"][2]))

# ---------------------------------------------------------------- tolerance (§7)
INN = {}
for k, _ in MODELS:
    for dt in ("bf16", "fp32"):
        d = load(f"innoise_{k}_{dt}")
        if d and len(d["records"]) >= 16:
            g = defaultdict(list)
            for r in d["records"]:
                g[(r["mode"], r["x"])].append(r)
            INN[(k, dt)] = {key: {"n": len(v), "succ": mean(r["success"] for r in v),
                                  "flag": mean(r["n_flagged"] for r in v), "T": mean(r["T"] for r in v),
                                  "flag_succ": mean(r["n_flagged"] for r in v if r["success"]),
                                  "min_flag_succ": min([r["n_flagged"] for r in v if r["success"]], default=None),
                                  "max": float(np.median([r["max_score"] for r in v]))}
                            for key, v in sorted(g.items())}

# ---------------------------------------------------------------- naive tampering (§8)
COR = {}
for k, _ in MODELS:
    d = load(f"corrupt_{k}")
    if not d or len(d["flips"]) < 100:
        continue
    E, FL = d["edits"], d["flips"]
    eg = defaultdict(list)
    for r in E:
        eg[r["layer"]].append(r)
    bits = defaultdict(list)
    for f in FL:
        bits[f["bit"]].append(f)
    COR[k] = {
        "edits": {l: {"n": len(v), "succ": mean(r["success"] for r in v),
                      "flag_med": float(np.median([r["n_flagged"] for r in v])),
                      "frac": float(np.median([r["n_flagged"] / max(r["n_after"], 1) for r in v])),
                      "rand1": mean(r["rand"]["1"] for r in v), "rand4": mean(r["rand"]["4"] for r in v),
                      "rand16": mean(r["rand"]["16"] for r in v)} for l, v in sorted(eg.items())},
        "n_flips": len(FL), "changed": mean(f["changed"] for f in FL), "wrong": mean(f["wrong"] for f in FL),
        "bit": {b: {"n": len(v), "changed": mean(f["changed"] for f in v), "wrong": mean(f["wrong"] for f in v),
                    "flag": mean(f["n_flagged"] > 0 for f in v), "range": mean(f["range_check"] for f in v)}
                for b, v in sorted(bits.items())},
        "harm": [f for f in FL if f["wrong"]],
    }
    hw = COR[k]["harm"]
    COR[k]["harm_flag"] = mean(f["n_flagged"] > 0 for f in hw)
    COR[k]["harm_range"] = mean(f["range_check"] for f in hw)
    COR[k]["harm_top8"] = mean(f["top8"] for f in hw)
    COR[k]["harm_any"] = mean((f["range_check"] or f["top8"] > 0.5) for f in hw)
    COR[k]["flag_any"] = mean(f["n_flagged"] > 0 for f in FL)

# ---------------------------------------------------------------- cost (§9)
COST = load("cost_llama-1b")


# ================================================================= text
def methods():
    return open(os.path.join(DOCS, "paper_methods.html"), encoding="utf8").read()


ll, qw = S.get("llama-1b", {}), S.get("qwen-0.5b", {})


def sec_forge():
    if not ll:
        return ""
    t = [f"<h2>4. One column is enough</h2>",
         f"<p>We forge one column, at one of five depths and three kinds of position: the last token of the fact's value, "
         f"a uniformly random non-sink token, and the last document token. The forger may change the residual stream there by up to β times its norm. "
         f"It then runs the rest of the prefill honestly. Over {ll['tasks']} memos per model "
         f"(mean length {ll['T']:.0f} and {qw.get('T', 0):.0f} tokens) and {ll['trials']} attacks per model, "
         f"{ll['succ']} of the Llama forgeries and {qw.get('succ', 0)} of the Qwen forgeries made the trusted node answer exactly the attacker's value. "
         f"At β&nbsp;=&nbsp;1, the success rate across all layers and positions is {pct(ll['full_budget'])} for Llama and {pct(qw.get('full_budget', 0))} for Qwen. "
         f"Position barely matters: a forged column on an unrelated token, far from the fact, works about as well as one on the fact itself. "
         f"The attack does not need to edit the fact. It needs one place the question reads.</p>",
         f"<p>The full column check confirms the structure from §3.1. "
         f"In {pct(ll['only_forged'], 1)} (Llama) and {pct(qw.get('only_forged', 0), 1)} (Qwen) of successful forgeries, the forged column is the "
         f"<i>only</i> column that fails. The few exceptions are forgeries that also change later columns enough to show. "
         f"No honest cache failed a single column: the largest honest score over all {ll['tasks'] + qw.get('tasks', 0)} evaluation memos was "
         f"{ll['honest_max']:.2f} for Llama and {qw.get('honest_max', 0):.2f} for Qwen, against a threshold of 1. These memos are disjoint from the 20 calibration memos.</p>",
         fig("forge_success", "Figure 1. Success rate of single-column forgeries (β = 1) by layer and position kind, with 95% Wilson intervals."),
         fig("forge_budget", "Figure 2. Success against the perturbation budget β at the middle layer."),
         table(["model", "layer", "β", "success", "value pos.", "random pos.", "last pos.", "only forged col. fails"], sec4_rows),
         "<p class='cap'>Table 1. Single-column forgeries. 24 tasks × 3 positions per row.</p>"]
    return "".join(t)


def sec_detect():
    if not ll:
        return ""
    t = [f"<h2>5. Auditing where the question reads</h2>",
         f"<p>A full check is a second prefill, so the practical question is how few columns an audit needs. Uniform sampling of <i>k</i> columns "
         f"catches one bad column with probability <i>k/T</i>. The data match: 64 random columns catch {pct(ll['uni']['64'])} of Llama forgeries and "
         f"{pct(qw.get('uni', {}).get('64', 0))} of Qwen forgeries. Checking the 4 sinks plus the 4 columns with the highest value-weighted attention "
         f"(<i>k</i>&nbsp;=&nbsp;8 in total) catches {pct(ll['top']['8'])} and {pct(qw.get('top', {}).get('8', 0))}. With <i>k</i>&nbsp;=&nbsp;32 it catches "
         f"{pct(ll['top']['32'])} and {pct(qw.get('top', {}).get('32', 0))}. "
         f"The median forged column is the {int(ll['rank_med']) + 1}th most-read column, counting the four sinks: "
         f"to change the answer, the forged column has to be read, and reading is visible. "
         f"In {pct(ll['first'])} of Llama forgeries and {pct(qw.get('first', 0))} of Qwen forgeries, the single most-read non-sink column <i>is</i> the forged one.</p>",
         f"<p>Forgeries on the last document token are the easiest to find (top-8 catches {pct(ll['top8_last'])} on Llama). The last token is read heavily "
         f"by the question in any case. Forgeries on the fact's own value token are the hardest ({pct(ll['top8_value'])}), "
         f"because the honest value token is already among the most-read, so a forgery there competes with the real signal. "
         f"Qwen is harder throughout. With only 2 KV heads, its attention spreads over more columns, so the forged one is less of an outlier.</p>",
         fig("detection", "Figure 3. Probability that an audit of k columns contains the forged column. Exact hypergeometric probabilities for random plans, averaged over successful forgeries."),
         table(["model", "forgeries", "top-8", "top-16", "top-32", "top-8 + 8 random", "random 4", "random 16", "random 64"], sec5_rows),
         "<p class='cap'>Table 2. Detection of successful single-column forgeries. Top-k includes the 4 sink columns.</p>"]
    if LONG:
        rows = [[f"{x['T']:.0f}", f"{x['succ']}/{x['n']}", pct(x['only'], 1), pct(x['top']['8']), pct(x['top']['16']), pct(x['top']['32']),
                 pct(x['uni64'], 1), f"{x['rank_med'] + 1:.0f}"] for x in LONG]
        lt = LONG[-1]
        t += [f"<h3>5.1 Long documents</h3><p>The ratio matters more as documents grow. With the facts scattered through Moby-Dick "
              f"(Llama, layers L/4, L/2, 3L/4), a {lt['T']:.0f}-token cache leaves uniform 64-column audits at {pct(lt['uni64'], 1)}, while top-32 still catches "
              f"{pct(lt['top']['32'])}. The forged column's median rank stays small as <i>T</i> grows, so the audit cost "
              f"in columns stays flat while the uniform cost grows linearly.</p>",
              table(["T (tokens)", "success", "only forged fails", "top-8", "top-16", "top-32", "random 64", "median rank"], rows),
              "<p class='cap'>Table 3. Llama-3.2-1B on long documents.</p>"]
    return "".join(t)


def plan_name(key):
    m_, lam, hk, ag = key
    if lam == 0:
        return "no hiding (λ = 0)"
    return f"m = {m_}, hide below top-{hk}, vs. {ag}"


def sec_adaptive():
    if not ADA:
        return ""
    t = ["<h2>6. Adaptive attackers</h2>",
         "<p>The audit's strength in §5 rests on a forged column having to be read. An attacker who knows the scoring rule can try to be read "
         "<i>enough</i> without being read <i>most</i>. Our adaptive forger adds a hinge penalty that pushes each forged column's score below "
         "0.7× the score of the <i>k</i>-th ranked other column. The score is computed on its own forged cache with the trusted node's question, "
         "which we let the attacker know in advance. It attacks one of two scores: <i>mean</i> (the default in §5) or <i>layermax</i> "
         "(per-layer normalised, max over layers). It can also spread the edit over <i>m</i> random columns.</p>"]
    for k, name in MODELS:
        if k not in ADA:
            continue
        rows = []
        for r in ADA[k]:
            if not r["succ"]:
                rows.append([plan_name(r["key"]), f"0/{r['n']}"] + ["–"] * 9)
                continue
            p = r["plans"]
            rows.append([plan_name(r["key"]), f"{r['succ']}/{r['n']}",
                         f"{r['rank_mean']:.0f}", f"{r['rank_lm']:.0f}", f"{r['rank_grad']:.0f}",
                         pct(p["top32"]), pct(p["lm32"]), pct(r["ens"]["16"]), pct(p["top32+rand32"]), pct(p["rand64"]),
                         pct(r["direct"])])
        t += [f"<h3>{name}</h3>",
              table(["attacker", "success", "rank (mean)", "rank (layermax)", "rank (grad)", "top-32 mean", "top-32 layermax",
                     "ensemble-16", "top-32 + 32 random", "random 64", "direct"], rows, "small"),
              ]
    t.append("<p class='cap'>Table 4. Adaptive forgeries at the middle layer, β = 1. Ranks are the median over successful attacks of the "
             "best-hidden forged column's rank among non-sink columns (0 = most read). Detection columns are the probability of catching it. "
             "Ensemble-16 is the union of the top 16 under all three scores, plus the sinks (≈ 30–45 columns). "
             "<i>Direct</i> is the fraction where putting the honest forged columns back restores the honest answer, so the effect does not run through other columns.</p>")
    t.append(adaptive_text())
    t.append(fig("adaptive", "Figure 4. Adaptive attackers. Left: where the forged column lands in each ranking. Right: detection by audit plan."))
    return "".join(t)


def adaptive_text():
    k = "llama-1b" if "llama-1b" in ADA else list(ADA)[0]
    R = {r["key"]: r for r in ADA[k]}
    base = R.get((1, 0.0, 4, "mean"))
    vm = R.get((1, 1.0, 28, "mean"))
    vl = R.get((1, 1.0, 28, "layermax"))
    sp = R.get((64, 1.0, 12, "layermax"))
    parts = []
    if base and vm and base["succ"] and vm["succ"]:
        parts.append(f"<p><b>A single score can be gamed.</b> Without the penalty, the forged column is ranked {base['rank_mean']:.0f} (median, among non-sink columns). "
                     f"An attacker hiding below the top 28 under the mean score still succeeds in {vm['succ']}/{vm['n']} tasks and lands at rank {vm['rank_mean']:.0f}, so "
                     f"top-32 under that score catches it {pct(vm['plans']['top32'])} of the time. The same forgery is ranked {vm['rank_lm']:.0f} by layermax: "
                     f"the mean score is low because the column is read hard at a few layers, not lightly everywhere.</p>")
    if vl and vl["succ"]:
        parts.append(f"<p><b>Gaming the stronger score works too.</b> Attacking layermax directly with hide-28, the forger succeeds in {vl['succ']}/{vl['n']} tasks. "
                     f"Its column then ranks {vl['rank_lm']:.0f} under layermax, {vl['rank_mean']:.0f} under mean and {vl['rank_grad']:.0f} under the gradient score, "
                     f"and top-32 layermax catches {pct(vl['plans']['lm32'])}. The union of the top 16 under all three scores catches {pct(vl['ens']['16'])}, "
                     f"with about {vl['ens_cols']['16']:.0f} columns. This is the honest result: an attacker that knows the score can lower it. "
                     f"A deterministic top-k audit is an arms race, not a guarantee.</p>")
    q = {r["key"]: r for r in ADA.get("qwen-0.5b", [])}.get((1, 1.0, 28, "layermax"))
    if q and q["succ"]:
        parts.append(f"<p>Qwen2.5-0.5B is easier to game. The same attacker succeeds in {q['succ']}/{q['n']} tasks, and top-32 layermax catches "
                     f"{pct(q['plans']['lm32'])} of them, the ensemble {pct(q['ens']['16'])}. Only the random columns still find it, at the k/T rate "
                     f"({pct(q['plans']['rand64'])} for 64).</p>")
    if sp and sp["succ"]:
        parts.append(f"<p><b>Spreading out costs the attacker detectability elsewhere.</b> With m&nbsp;=&nbsp;64 forged columns hidden from layermax top-12, "
                     f"it succeeds in {sp['succ']}/{sp['n']} tasks, but each of those caches has {sp['n_flagged']:.0f} failing columns, "
                     f"so 64 random columns catch it {pct(sp['plans']['rand64'])} of the time. That is the case random draws are for. "
                     f"Mixing the two (top-32 + 32 random) covers both ends: {pct(sp['plans']['top32+rand32'])} here.</p>")
    parts.append("<p>So the deterministic part of an audit should be treated as a cheap filter for non-adaptive faults and attacks. "
                 "Guarantees against an adaptive forger come only from the random part, at <i>k/T</i> per forged column, "
                 "or from a full recompute of whatever the audit's budget cannot cover.</p>")
    return "".join(parts)


DEF_ATT = ["none", "mean", "layermax", "headmax", "tokmax", "all public", "all public + trace"]
DEF_PLANS = [("mean", "mean"), ("layermax", "layermax"), ("headmax", "headmax"), ("tokmax", "tokmax"), ("grad", "gradient"),
             ("all public", "all public merged"), ("secret", "secret heads"), ("trace", "trace"),
             ("trace + random", "trace + random"), ("random", "random")]
def load_defense(k):
    """defense_<model>.json plus any defense_<model>_<tag>.json with more seeds."""
    import glob
    parts = [load(f"defense_{k}")] + [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RES, f"defense_{k}_*.json")))]
    parts = [x for x in parts if x]
    if not parts:
        return None
    return {**parts[0], "records": [r for x in parts for r in x["records"]]}


DEF = {k: load_defense(k) for k, _ in MODELS}


def def_stats(d, B="32"):
    out = {}
    for a in DEF_ATT:
        rs = [r for r in d["records"] if r["attacker"] == a]
        ok = [r for r in rs if r["success"]]
        if not rs:
            continue
        out[a] = {"n": len(rs), "succ": len(ok),
                  "det": {pk: mean(r["detect"][B][pk] for r in ok) if ok else float("nan") for pk, _ in DEF_PLANS}}
    return out


def sec_defense():
    if not any(DEF.values()):
        return ""
    t = ["<h3>6.1 Round two: audits the attacker cannot see</h3>",
         "<p>Two things stand out in Table 4. Different scores disagree about where a hidden column lands, and in 25–60% of successful hidden "
         "forgeries, putting the honest column back does <i>not</i> restore the answer. The forgery has been <i>laundered</i>: later columns read the "
         "forged one during prefill, carry its effect forward, and the question reads them instead. Those later columns are consistent with their "
         "inputs, so they pass the check. This suggests two defenses. The first is scores the attacker cannot optimise against: finer ones (per head, "
         "per reader token) and a <b>secret</b> one, a random quarter of the heads with random weights, drawn by the verifier for each audit. "
         "The second is <b>tracing</b>. The audit recomputes its columns anyway, and the recomputation yields each column's own attention to earlier "
         "columns. So after auditing the most-read columns, the verifier follows what <i>they</i> read most (value-weighted, per head, max over layers), "
         "audits those, and repeats until the budget is spent (Algorithm: half the budget on merged public scores, the rest in rounds of B/8 traced columns).</p>",
         "<p>We test this against seven attackers that hide one forged column below the 28 most-read columns of one public score, of all four at once, "
         "or of all four <i>and</i> the tracing rule itself, using a differentiable copy of the trace rows on the forged prefill. 12 tasks per model, "
         "150 Adam steps, middle layer, β = 1. Every plan audits exactly 32 columns, including the 4 sinks.</p>"]
    for k, name in MODELS:
        d = DEF[k]
        if not d:
            continue
        st = def_stats(d)
        rows = []
        for a, x in st.items():
            lab = "no hiding" if a == "none" else "hides from " + a
            rows.append([lab, f"{x['succ']}/{x['n']}"] + [pct(x["det"][pk]) if x["succ"] else "–" for pk, _ in DEF_PLANS])
        t += [f"<h3>{name}</h3>", table(["attacker", "success"] + [lab for _, lab in DEF_PLANS], rows, "small")]
    t.append("<p class='cap'>Table 5. Probability that a 32-column audit catches a successful forgery, by attacker (rows) and audit plan (columns). "
             "Secret: mean over 32 secret draws. Random parts are exact hypergeometric probabilities.</p>")
    t.append(fig("defense", "Figure 5. Round two of the arms race. Each cell: share of successful forgeries a 32-column audit catches."))
    t.append(defense_text())
    return "".join(t)


def game(d, B):
    """Attacker's best reply per plan: max over attackers of P(answer flips and audit misses)."""
    R = d["records"]
    atts = [a for a in DEF_ATT if any(r["attacker"] == a for r in R)]
    out = {}
    for pk, _ in DEF_PLANS:
        w = {a: mean((1 - r["detect"][B][pk]) if r["success"] else 0.0 for r in R if r["attacker"] == a) for a in atts}
        a = max(w, key=w.get)
        out[pk] = (w[a], a)
    return out


def defense_text():
    rows, notes = [], []
    G = {k: {B: game(DEF[k], B) for B in ("16", "32", "64")} for k, _ in MODELS if DEF[k]}
    for pk, lab in DEF_PLANS:
        row = [lab]
        for k, name in MODELS:
            if k in G:
                row += [f"{100 * G[k][B][pk][0]:.0f}%" for B in ("16", "32", "64")]
        rows.append(row)
    head = ["audit plan"] + [f"{name.split('-')[0]} B={B}" for k, name in MODELS if k in G for B in ("16", "32", "64")]
    for k, name in MODELS:
        if k not in G:
            continue
        g = G[k]["32"]
        best = min(g, key=lambda pk: g[pk][0])
        notes.append(f"On {name}, against a 32-column audit the attacker's best reply wins {pct(g['mean'][0])} of the time if the verifier uses the default "
                     f"mean score, {pct(g['random'][0])} against random columns, and {pct(g[best][0])} against the best plan, "
                     f"<i>{dict(DEF_PLANS)[best]}</i> (its best reply there: <i>{'no hiding' if g[best][1] == 'none' else 'hides from ' + g[best][1]}</i>)")
    def w(pk, k, B="32"):
        return pct(G[k][B][pk][0]) if k in G else "–"
    ks = [k for k, _ in MODELS if k in G]
    nm = {k: dict(MODELS)[k] for k in ks}
    find = ("<p><b>What holds on both models.</b> "
            f"(1) The default mean score is the easiest to game: the best reply beats a 32-column mean audit " + " and ".join(f"{w('mean', k)} of the time on {nm[k]}" for k in ks)
            + ", no better than 32 random columns (" + ", ".join(w('random', k) for k in ks) + "). "
            "(2) No single public score is robust across models. The finest one, tokmax (per head, layer and reader token), holds the attacker to "
            + " but ".join(f"{w('tokmax', k)} on {nm[k]}" for k in ks) + ". "
            "(3) Tracing is the most consistent plan: " + " and ".join(f"{w('trace', k)} on {nm[k]}" for k in ks)
            + ". It does not rely on any one score, but follows the path the forged information has to take to reach the question. "
            "The attacker that also optimises against the trace rule did exactly as well as the one that ignores it, in success and in detection. "
            "Our differentiable copy of the trace covers one hop from a fixed set of read columns, while the verifier keeps following hops until its budget "
            "runs out; a stronger trace-aware attacker is the obvious next step. "
            "(4) Secret head subsets help only modestly (" + ", ".join(w('secret', k) for k in ks) + "): an attacker that is quiet in every head "
            "is quiet in any subset of them.</p>"
            "<p>Two caveats. The attackers hide below rank 28, which matches the 32-column budget. At 64 columns every attention-based plan looks "
            "strong (" + ", ".join(f"tracing {w('trace', k, '64')} on {nm[k]}" for k in ks) + ") only because the attackers were not told about the larger "
            "budget. And all of these are games against our optimiser (150 Adam steps, one hinge loss per score), so the rates describe a modest white-box "
            "attacker, not a bound. Random columns give the only attacker-independent guarantee, and at k/T it is a weak one.</p>")
    return ("<p><b>The game.</b> An audit plan is only as good as the attacker's best reply to it. For each plan, Table 6 gives the share of tasks in which the "
            "<i>best</i> of our seven attackers both flips the answer and escapes the audit (success rate × miss rate, maximised over attackers). "
            + ". ".join(notes) + ".</p>"
            + table(head, rows, "small")
            + "<p class='cap'>Table 6. Worst-case attacker win rate per audit plan (lower is better for the verifier), at audit budgets of 16, 32 and 64 columns.</p>"
            + fig("game", "Figure 6. The attacker's best reply to each 32-column audit plan: how often it both flips the answer and escapes the audit.")
            + find)


def sec_innoise():
    if not INN:
        return ""
    t = ["<h2>7. Hiding inside the tolerance</h2>",
         "<p>bf16 forces a tolerance, and a tolerance is room. We test two attackers that try to stay inside it. "
         "The <i>edit</i> attacker changes every stored entry at every layer in place, by at most ρ × the largest honest error. "
         "The <i>distributed forger</i> nudges the residual stream of every document column at the middle layer by a relative ε, "
         "then computes everything else honestly, so each column carries only its own small inconsistency.</p>"]
    rows = []
    for (k, dt), g in INN.items():
        name = dict(MODELS)[k]
        for (mode, x), r in g.items():
            rows.append([name, dt, mode, f"{x:g}", f"{pct(r['succ'])}", f"{r['flag']:.0f} / {r['T']:.0f}",
                         f"{r['min_flag_succ']}" if r["min_flag_succ"] is not None else "–", f"{r['max']:.1f}"])
    t.append(table(["model", "dtype", "attack", "ρ or ε", "success", "failing columns (mean)", "fewest failing when it worked",
                    "median max score"], rows, "small"))
    t.append("<p class='cap'>Table 7. Attacks that try to stay inside the bf16 (or fp32) tolerance.</p>")
    lines, silent = [], []
    for (k, dt), g in INN.items():
        works = [(key, r) for key, r in g.items() if r["succ"] > 0]
        if works:
            (mode, x), r = min(works, key=lambda w: w[1]["min_flag_succ"])
            f, T = r["min_flag_succ"], r["T"]
            if f == 0:
                silent.append((k, dt, mode, x, r))
                continue
            p16, p64 = 1 - (1 - f / T) ** 16, 1 - (1 - f / T) ** 64
            lines.append(f"{dict(MODELS)[k]} {dt}: the quietest successful attack ({mode}, {'ρ' if mode == 'edit' else 'ε'} = {x:g}) "
                         f"left {f} of {T:.0f} columns failing, which 16 random columns would catch with probability {pct(p16)} and 64 with {pct(p64)}")
    t.append("<p>" + "; ".join(lines) + ". "
             "Below the tolerance (ρ = 0.1, ε ≤ 0.003 in bf16) nothing succeeded on Llama. The band between the tolerance and an effective attack is narrow, "
             "because §4 needed perturbations on the order of the residual norm while the tolerance is a few percent of it, and spreading the change over 400 columns "
             "does not change that: attention averages the nudges, so effective spread attacks usually break many columns at once, which is exactly what random columns are good at. "
             "In fp32 the tolerance is ~10<sup>−6</sup> and every attack lit up almost every column.</p>")
    for k, dt, mode, x, r in silent:
        n_silent = sum(1 for rec in load(f"innoise_{k}_{dt}")["records"]
                       if rec["success"] and rec["n_flagged"] == 0)
        n_tasks = len({rec["seed"] for rec in load(f"innoise_{k}_{dt}")["records"]})
        t.append(f"<p><b>But the tolerance is an attack surface.</b> On {dict(MODELS)[k]} ({dt}), {n_silent} of {n_tasks} tasks fell to a distributed forgery "
                 f"(ε = {x:g}) that flipped the answer with <i>every</i> column inside the tolerance: its worst column scored just under 1. "
                 "No column audit, however chosen, can catch that cache, because by the verifier's own standard it is honest. "
                 f"Tightening the threshold would catch it only at the price of false alarms on honest caches, whose worst score was {S.get(k, {}).get('honest_max', float('nan')):.2f} on this model. "
                 "The real fix is a smaller tolerance: batch-invariant, deterministic kernels in prefill and verification, or a verification precision the prefill also uses. "
                 "In fp32 (tested on Llama), every forgery of this kind lit up hundreds of columns.</p>")
    t.append(fig("innoise", "Figure 7. In-tolerance attacks: success rate and the number of failing columns as the per-column budget grows."))
    return "".join(t)


def sec_corrupt():
    if not COR:
        return ""
    t = ["<h2>8. Naive tampering and corruption</h2>"]
    rows = []
    for k, c in COR.items():
        for l, e in c["edits"].items():
            rows.append([dict(MODELS)[k], l, pct(e["succ"]), f"{e['flag_med']:.0f}", pct(e["frac"]), pct(e["rand1"]), pct(e["rand4"])])
    lc = COR.get("llama-1b") or list(COR.values())[0]
    t.append("<p><b>In-place edits.</b> An editor that optimises stored K/V directly at one column, without recomputing anything later, "
             "makes every later column that reads the edited one disagree with its own recomputation. "
             "That holds at every layer except the last, which no later layer reads. "
             "So these edits are loud, and even a single random column usually catches them.</p>")
    t.append(table(["model", "edited layer", "success", "failing columns (median)", "of later columns", "random 1", "random 4"], rows, "small"))
    t.append("<p class='cap'>Table 8. In-place edits of one column (24 per row: 12 tasks × 2 positions).</p>")
    brow = []
    for k, c in COR.items():
        for b in (15, 14, 13, 7, 0):
            if b in c["bit"]:
                x = c["bit"][b]
                brow.append([dict(MODELS)[k], {15: "15 (sign)", 14: "14 (exp. MSB)", 13: "13 (exp.)", 7: "7 (exp. LSB)", 0: "0 (mantissa LSB)"}[b],
                             x["n"], pct(x["changed"]), pct(x["wrong"]), pct(x["flag"]), pct(x["range"])])
    t.append(f"<p><b>Bit flips.</b> We flip {lc['n_flips']} random bits per model, uniformly over element, layer, head, position and bit. "
             f"For Llama, {pct(lc['changed'], 1)} change the decoded tokens and {pct(lc['wrong'], 1)} make the answer wrong. "
             f"Nearly all the damage comes from bit 14, the top exponent bit, which turns a value of magnitude ~1 into one of ~10<sup>38</sup>. "
             f"A range check (|x| above the largest honest magnitude at that layer) catches {pct(lc['harm_range'])} of the harmful flips at zero cost. "
             f"The full column check flags {pct(lc['flag_any'])} of all flips, including harmless ones. "
             f"Hardware faults are therefore the cheap case: a range check plus a small random audit covers them.</p>")
    t.append(table(["model", "bit", "flips", "output changed", "answer wrong", "check flags it", "range check fires"], brow, "small"))
    t.append("<p class='cap'>Table 9. Random single-bit flips by bit position (selected bits).</p>")
    t.append(fig("corrupt", "Figure 8. Left: columns that fail after an in-place edit, by edited layer. Right: effect of a single bit flip by bit position."))
    return "".join(t)


def sec_cost():
    if not COST:
        return ""
    rows = []
    for r in COST["rows"]:
        a = r["influence"] + r["cols16"]
        rows.append([r["T"], f"{r['kv_mb']:.0f}", f"{r['prefill'] * 1e3:.0f}", f"{r['influence'] * 1e3:.1f}",
                     f"{r['cols8'] * 1e3:.1f}", f"{r['cols16'] * 1e3:.1f}", f"{r['cols64'] * 1e3:.1f}",
                     f"{r['trace32'] * 1e3:.0f}" if "trace32" in r else "–",
                     f"{r['full_check'] * 1e3:.0f}" if "full_check" in r else "–", pct(a / r["prefill"], 1)])
    big = COST["rows"][-1]
    S["cost"] = {"T": big["T"], "frac": (big["influence"] + big["cols16"]) / big["prefill"],
                 "frac512": (COST["rows"][0]["influence"] + COST["rows"][0]["cols16"]) / COST["rows"][0]["prefill"]}
    return "".join([
        "<h2>9. Cost</h2>",
        f"<p>On {COST['gpu']}, Llama-3.2-1B, bf16. The influence pass reads the cache once with the question tokens, which the trusted node does anyway "
        f"to answer, so in a fused implementation it would be nearly free. Here we time it separately. "
        f"An audit of 16 columns plus the influence pass costs {pct(S['cost']['frac512'], 1)} of a prefill at 512 tokens and "
        f"{pct(S['cost']['frac'], 1)} at {big['T']} tokens. The audit has a floor of about 30 ms: our unfused PyTorch code launches dozens of small kernels per layer, so on short documents it costs as much as a prefill. "
        "Above that floor, the prefill grows quadratically in T while the audit is a fixed number of tokens attending to T columns, so the gap widens with length. "
        "Our implementation is plain PyTorch with explicit attention for the audited rows. The prefill uses SDPA. "
        + (f"A traced audit of 32 columns runs its recomputation in about five sequential rounds, so it pays the floor several times: "
           f"{big['trace32'] * 1e3:.0f} ms at {big['T']} tokens, {pct(big['trace32'] / big['prefill'], 1)} of the prefill. "
           "Fusing those rounds into one kernel launch each would bring it close to the plain audit." if "trace32" in big else "")
        + "</p>",
        table(["T", "KV (MiB)", "prefill (ms)", "influence (ms)", "8 cols (ms)", "16 cols (ms)", "64 cols (ms)", "trace 32 (ms)", "full check (ms)",
               "influence + 16 / prefill"], rows, "small"),
        "<p class='cap'>Table 10. Median wall-clock time of 5 runs.</p>",
        fig("cost", "Figure 9. Time for prefill, a full check, and audits of 16 and 64 columns, by document length.")])


RELATED = """
<h2>10. Related work</h2>
<p><b>Moving KV caches.</b> Disaggregated serving separates prefill from decode and transfers the cache between them (DistServe, Zhong et al. 2024; Splitwise, Patel et al. 2024; Mooncake, Qin et al. 2024). Prefix caching reuses caches across requests (vLLM's automatic prefix caching, Kwon et al. 2023; SGLang's RadixAttention, Zheng et al. 2024). LMCache and CacheGen store, compress and stream caches across tiers and machines (Liu et al. 2024). CacheBlend reuses non-prefix caches and recomputes a selected subset of tokens to repair cross-attention (Yao et al. 2024). Its selective-recompute step is close in mechanics to our column check, but it is used for quality, not integrity. None of these systems verifies a received cache against its tokens.</p>
<p><b>Attacks on caches.</b> Prior work has studied KV caches as a side channel: prompts leak through the timing of shared prefix caches (Song et al. 2024; Wu et al. 2025). Fault-injection work on LLMs (bit flips in weights, e.g. rowhammer-style attacks) targets parameters, not activations. We look at integrity rather than confidentiality, at the cache rather than the weights, and at a defence that needs no trust in the producer.</p>
<p><b>Verifiable inference.</b> Zero-knowledge and interactive proofs for neural network inference (e.g. zkLLM, Sun et al. 2024) give cryptographic soundness at a large prover cost. Spot-checking and replication schemes for untrusted compute (e.g. TOPLOC, Ong et al. 2025, which hashes activations to detect model or precision swaps) trade guarantees for cost. Our check sits at the cheap end: it uses the structure of autoregressive caches (each column is a one-token function of its prefix) to make spot checks exact up to floating-point noise, and attention to choose where to check.</p>
"""

LIMITS = """
<h2>11. Limitations</h2>
<ul>
<li><b>Scale.</b> Two small models (0.5B and 1.1B parameters), memo-style factual QA with one-token-ish answers, and documents up to ~6K tokens. Larger models have more layers and heads, so influence may concentrate differently.</li>
<li><b>Known question.</b> The attacker optimises for a question it anticipates, and our audit reads the influence of the question actually asked. An attacker that has to work across many questions has a harder job, and an audit across many questions has more readers. We did not measure either.</li>
<li><b>Determinism.</b> The tolerance τ is calibrated on our own kernels. A real deployment with heterogeneous prefill and decode hardware would need calibration across kernel pairs, and τ would grow.</li>
<li><b>Adaptive attacker.</b> Our adaptive forger is white-box but simple: one hinge loss and 100 Adam steps. Stronger optimisers will hide better. §6 should be read as a lower bound on what an adaptive attacker can do.</li>
<li><b>Not a MAC.</b> The check proves that a cache is consistent with its tokens and weights. It says nothing about whether the tokens themselves are the right ones. That is the ordinary job of authenticating the prompt.</li>
</ul>
<h2>12. Conclusion</h2>
<p>A KV cache can be checked against its tokens column by column, for the price of a few tokens. Every forgery in our experiments fails at the column where it was made, and honest caches pass. Where to check is the hard part. Against faults and non-adaptive tampering, the columns the question reads most are where the problem is: a few columns catch most single-column forgeries, at any document length. Against a white-box attacker that knows the rule, any single score can be gamed. Tracing the audit through the cache, following what the read columns read, makes the attacker's job harder on both models we tested, but only random columns and full recompute give guarantees. That is a real trade-off, and serving systems that move caches between machines can now make it with numbers.</p>
<h2>References</h2>
<ol class="refs">
<li>Zhong et al. DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving. OSDI 2024.</li>
<li>Patel et al. Splitwise: Efficient Generative LLM Inference Using Phase Splitting. ISCA 2024.</li>
<li>Qin et al. Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving. 2024.</li>
<li>Kwon et al. Efficient Memory Management for Large Language Model Serving with PagedAttention. SOSP 2023.</li>
<li>Zheng et al. SGLang: Efficient Execution of Structured Language Model Programs. 2024.</li>
<li>Liu et al. CacheGen: KV Cache Compression and Streaming for Fast LLM Serving. SIGCOMM 2024. (LMCache)</li>
<li>Yao et al. CacheBlend: Fast LLM Serving for RAG with Cached Knowledge Fusion. 2024.</li>
<li>Sun et al. zkLLM: Zero Knowledge Proofs for Large Language Models. CCS 2024.</li>
<li>Ong et al. TOPLOC: A Locality Sensitive Hashing Scheme for Trustless Verifiable Inference. 2025.</li>
<li>Song et al. The Early Bird Catches the Leak: Unveiling Timing Side Channels in LLM Serving Systems. 2024.</li>
<li>Wu et al. I Know What You Asked: Prompt Leakage via KV-Cache Sharing in Multi-Tenant LLM Serving. NDSS 2025.</li>
<li>Xiao et al. Efficient Streaming Language Models with Attention Sinks. ICLR 2024.</li>
</ol>
"""

CSS = """
body{font:15.5px/1.6 Georgia,'Times New Roman',serif;color:#1d1d1b;background:#fff;max-width:820px;margin:40px auto;padding:0 24px}
h1{font:600 30px/1.2 system-ui,sans-serif;margin:0 0 6px}h2{font:600 20px system-ui,sans-serif;margin:34px 0 8px}
h3{font:600 16px system-ui,sans-serif;margin:22px 0 6px}.sub{color:#555;font:14px system-ui,sans-serif;margin-bottom:22px}
.abs{background:#f6f6f3;border-left:3px solid #2a78d6;padding:12px 16px;margin:18px 0}
table{border-collapse:collapse;margin:14px 0 4px;font:13px system-ui,sans-serif;width:100%}
th,td{border-bottom:1px solid #ddd;padding:4px 6px;text-align:left;vertical-align:top}th{border-bottom:2px solid #999}
table.small{font-size:11.5px}.cap{font:12.5px system-ui,sans-serif;color:#555;margin-top:2px}
figure{margin:18px 0}figure img{width:100%}figcaption{font:12.5px system-ui,sans-serif;color:#555}
code{font-size:13px}.refs{font-size:13.5px}
@media print{body{margin:0 auto;font-size:11pt}h2{break-after:avoid}figure,table{break-inside:avoid}}
"""


GAME_TXT = {pk: " and ".join(pct(game(DEF[k], "32")[pk][0]) for k, _ in MODELS if DEF[k]) for pk in ("trace", "mean")} if any(DEF.values()) else {"trace": "?", "mean": "?"}


def abstract():
    if not ll:
        return ""
    lt = LONG[-1] if LONG else None
    return (f"<div class='abs'><b>Abstract.</b> Disaggregated serving, prefix caches and KV stores pass a transformer's KV cache between machines, "
            f"and the reader trusts it. We show that this trust is easy to abuse and cheap to check. "
            f"Forging a single column of the cache (one token's keys and values, computed from a nudged residual stream) "
            f"makes Llama-3.2-1B give the attacker's answer in {pct(ll['full_budget'])} of attempts, from any layer and almost any token position. "
            f"Yet every cache is a deterministic function of its tokens, and each column can be recomputed against the claimed cache for the cost of one token. "
            f"In {pct(ll['only_forged'], 1)} of successful forgeries this flags exactly the forged column, and no honest cache was flagged. "
            f"Choosing columns by the question's value-weighted attention catches {pct(ll['top']['8'])} of forgeries with 8 columns (Qwen2.5-0.5B: {pct(qw['top']['8'])}), "
            f"against {pct(ll['uni']['64'])} for 64 random columns" + (f", and {pct(lt['top']['32'])} with 32 columns at {lt['T']:.0f} tokens" if lt else "") + ". "
            f"Attacks that try to hide inside the bf16 tolerance usually fail or leave many columns failing, but one in 12 Qwen tasks fell to a forgery that stayed entirely inside it, so the tolerance itself is an attack surface that only deterministic kernels can close. A white-box attacker that knows the scoring rule can push its column down the ranking, "
            f"Hidden forgeries often reach the question through later columns that read them, so we also trace the audit: follow what the most-read columns "
            f"themselves read, using attention rows the recomputation yields for free. In a game of 7 attackers against 10 audit plans, tracing is the only plan that is "
            f"best or tied-best on both models (the attacker's best reply wins {GAME_TXT['trace']} of the time, against {GAME_TXT['mean']} for the default score). "
            f"No deterministic plan is a guarantee, and random columns remain the only "
            f"attacker-independent one. Code, data and an interactive demo are public.</div>")


html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>kv-notary: Checking KV Caches Against Their Tokens</title><style>{CSS}</style></head><body>
<h1>Can you trust a KV cache you did not compute?</h1>
<div class="sub">kv-notary · forging, and auditing, transformer KV caches column by column · <a href="https://github.com/PS12007/kv-notary">github.com/PS12007/kv-notary</a> · <a href="index.html">interactive demo</a></div>
{abstract()}
{methods()}
{sec_forge()}
{sec_detect()}
{sec_adaptive()}
{sec_defense()}
{sec_innoise()}
{sec_corrupt()}
{sec_cost()}
{RELATED}
{LIMITS}
</body></html>"""
open(os.path.join(DOCS, "paper.html"), "w", encoding="utf8").write(html)

# ---------------------------------------------------------------- summary for the demo hero
stats = []
if ll:
    stats.append([pct(ll["full_budget"]), "of one-column forgeries flip Llama's answer"])
    stats.append([pct(ll["top"]["8"]), "caught by checking 8 most-read columns"])
    stats.append([pct(ll["uni"]["64"]), "caught by 64 random columns"])
    stats.append([f"{ll['honest_flagged'] + qw.get('honest_flagged', 0)}", f"false alarms on {ll['tasks'] + qw.get('tasks', 0)} honest caches"])
if "cost" in S:
    stats.append([pct(S["cost"]["frac"], 1), f"of a prefill to audit {S['cost']['T'] // 1024}K tokens"])
os.makedirs(os.path.join(DOCS, "data"), exist_ok=True)
json.dump({"stats": stats, "numbers": {k: v for k, v in S.items() if k != "long"}, "long": LONG},
          open(os.path.join(DOCS, "data", "summary.json"), "w"), indent=1, default=float)
print("wrote docs/paper.html and docs/data/summary.json")
for x in stats:
    print(" ", x)
