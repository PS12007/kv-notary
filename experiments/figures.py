"""Every figure in the README and paper, from results/*.json. Missing results are skipped.

usage: python experiments/figures.py
"""
import collections
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "docs", "figures")
os.makedirs(OUT, exist_ok=True)

C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
     "magenta": "#e87ba4", "violet": "#6250d6", "red": "#e34948", "ink": "#0b0b0b",
     "ink2": "#52514e", "muted": "#8a8984", "grid": "#e4e3df", "surface": "#fcfcfb"}
plt.rcParams.update({
    "figure.facecolor": C["surface"], "axes.facecolor": C["surface"], "savefig.facecolor": C["surface"],
    "axes.edgecolor": C["muted"], "axes.labelcolor": C["ink2"], "xtick.color": C["ink2"],
    "ytick.color": C["ink2"], "text.color": C["ink"], "axes.grid": True, "grid.color": C["grid"],
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
    "lines.linewidth": 2, "font.family": "DejaVu Sans"})
MODELS = [("llama-1b", "Llama-3.2-1B"), ("qwen-0.5b", "Qwen2.5-0.5B")]


def load(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else None


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def wilson(k, n, z=1.96):
    if n == 0:
        return 0, 0, 0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


# ------------------------------------------------------------------ fig: forge success
def fig_forge_success():
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4), sharey=True)
    for ax, (mk, label) in zip(axes, MODELS):
        d = load(f"forge_{mk}.json")
        if not d:
            continue
        R = d["records"]
        L = max(d["layers"]) + 1
        kinds = [("value", "the fact's own token", C["blue"]), ("random", "a random token", C["orange"]),
                 ("last", "the last token", C["aqua"])]
        layers = d["layers"]
        w = 0.26
        for i, (k, lab, col) in enumerate(kinds):
            ys, lo, hi = [], [], []
            for l in layers:
                rs = [r for r in R if r["kind"] == k and r["layer"] == l and r["budget"] == 1.0]
                p, a, b = wilson(sum(r["success"] for r in rs), len(rs))
                ys.append(p), lo.append(p - a), hi.append(b - p)
            x = np.arange(len(layers)) + (i - 1) * w
            ax.bar(x, ys, w * 0.92, color=col, label=lab, yerr=[lo, hi], ecolor=C["ink2"],
                   error_kw={"elinewidth": 1, "capsize": 2})
        ax.set_xticks(range(len(layers)), [f"{l}" for l in layers])
        ax.set_xlabel(f"layer forged (of {L})")
        ax.set_title(label)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("answer flipped to target")
    axes[0].legend(title="forged column", loc="lower left", fontsize=8.5, title_fontsize=8.5)
    fig.suptitle("One forged column (one residual vector, norm ≤ the residual's) flips the answer", y=1.03)
    save(fig, "forge_success")


def fig_budget():
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for (mk, label), col in zip(MODELS, (C["blue"], C["orange"])):
        d = load(f"forge_{mk}.json")
        if not d:
            continue
        R = d["records"]
        mid = max(d["layers"]) // 2 if False else sorted(set(r["layer"] for r in R if r["budget"] != 1.0) or [0])[0]
        bs = sorted(set(r["budget"] for r in R if r["layer"] == mid))
        ys = [np.mean([r["success"] for r in R if r["layer"] == mid and r["budget"] == b]) for b in bs]
        ax.plot(bs, ys, "-o", color=col, label=f"{label} (layer {mid})", markersize=6)
    ax.set_xscale("log", base=2)
    ax.set_xticks([0.25, 0.5, 1, 2], ["¼", "½", "1", "2"])
    ax.set_xlabel("perturbation norm / residual norm")
    ax.set_ylabel("answer flipped")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8.5)
    ax.set_title("How big the one-column nudge must be")
    save(fig, "forge_budget")


# ------------------------------------------------------------------ fig: detection curve
def fig_detection():
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.5), sharey=True)
    for ax, (mk, label) in zip(axes, MODELS):
        d = load(f"forge_{mk}.json")
        if not d:
            continue
        R = [r for r in d["records"] if r["success"]]
        T = np.mean([r["T"] for r in R])
        ku = sorted(int(k) for k in R[0]["uniform"])
        uni = [np.mean([r["uniform"][str(k)] for r in R]) for k in ku]
        kk = np.arange(1, 65)
        ax.plot(kk, 1 - (1 - 1 / T) ** kk, color=C["muted"], lw=1.2, ls="--", label="uniform, theory")
        ax.plot(ku, uni, "o", color=C["orange"], markersize=7, label="uniform random (measured)")
        # top-k catches the forgery iff the forged column ranks among the k most-read (k counts the
        # 4 sinks); the stored rank gives the whole curve, and matches the stored top-k exactly
        kd = np.arange(5, 65)
        rk = np.array([r["infl_rank"] for r in R])
        ax.plot(kd, [(rk < k).mean() for k in kd], color=C["blue"], label=f"4 sinks + most-read, T≈{T:.0f}")
        ax.plot([8, 16, 32], [np.mean([r["top"][str(k)] for r in R]) for k in (8, 16, 32)], "o", color=C["blue"], markersize=6)
        if mk == "llama-1b":
            for tag, col in (("long8000", C["violet"]), ("long16000", C["magenta"])):
                dl = load(f"forge_{mk}_{tag}.json")
                if not dl:
                    continue
                Rl = [r for r in dl["records"] if r["success"]]
                Tl = np.mean([r["T"] for r in Rl])
                rl = np.array([r["infl_rank"] for r in Rl])
                ax.plot(kd, [(rl < k).mean() for k in kd], color=col, label=f"4 sinks + most-read, T≈{Tl:.0f}")
                ax.plot(kk, 1 - (1 - 1 / Tl) ** kk, color=col, lw=1, ls=":", alpha=0.9)
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16, 32, 64], ["1", "2", "4", "8", "16", "32", "64"])
        ax.set_xlabel(f"columns audited (document ≈ {T:.0f} columns)")
        ax.set_title(f"{label}: {len(R)} successful forgeries")
        ax.set_ylim(-0.02, 1.05)
    axes[0].set_ylabel("forgery detected")
    axes[0].legend(fontsize=8, loc="center right")
    fig.suptitle("Checking the most-read columns finds non-adaptive forgeries; uniform sampling scales as k/T", y=1.03)
    save(fig, "detection")


# ------------------------------------------------------------------ fig: corruption
def fig_corrupt():
    d = load("corrupt_llama-1b.json")
    if not d:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.3))
    ax = axes[0]
    E = d["edits"]
    layers = d["layers"]
    for kind, col, lab in (("value", C["blue"], "fact token"), ("random", C["orange"], "random token")):
        ys = [np.mean([e["n_flagged"] / max(e["n_after"], 1) for e in E if e["layer"] == l and e["kind"] == kind]) for l in layers]
        ax.plot(layers, ys, "-o", color=col, label=lab, markersize=6)
    ax.set_xlabel("layer edited in place")
    ax.set_xticks(layers, [str(l) for l in layers])
    ax.set_ylabel("share of later columns that fail")
    ax.set_title("In-place edits break later columns")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8.5)
    ax = axes[1]
    F = d["flips"]
    bits = range(16)
    ch = [np.mean([f["changed"] for f in F if f["bit"] == b]) for b in bits]
    fl = [np.mean([f["n_flagged"] > 0 for f in F if f["bit"] == b]) for b in bits]
    x = np.arange(16)
    ax.bar(x - 0.2, fl, 0.38, color=C["aqua"], label="full check flags it")
    ax.bar(x + 0.2, ch, 0.38, color=C["red"], label="answer changes")
    ax.set_xticks(x, [str(b) for b in bits], fontsize=8)
    ax.set_xlabel("bf16 bit flipped (15 sign, 14-7 exponent, 6-0 mantissa)")
    ax.set_title("Bit flips: the top exponent bit matters")
    ax.legend(fontsize=8.5, loc="upper left")
    ax.set_ylim(0, 1.05)
    save(fig, "corrupt")


# ------------------------------------------------------------------ fig: adaptive
def fig_adaptive():
    ds = [(mk, lab, load(f"adaptive_{mk}.json")) for mk, lab in MODELS]
    ds = [x for x in ds if x[2]]
    if not ds:
        return
    fig, axes = plt.subplots(1, len(ds), figsize=(4.9 * len(ds), 3.6), sharey=True, squeeze=False)
    for ax, (mk, lab, d) in zip(axes[0], ds):
        R = [r for r in d["records"] if r["m"] == 1 and r["lam"] > 0 and r["against"] == "layermax"]
        hk = sorted(set(r["hide_k"] for r in R))
        succ = [np.mean([r["success"] for r in R if r["hide_k"] == k]) for k in hk]
        ax.plot(hk, succ, "-o", color=C["red"], label="attack still succeeds", markersize=6)
        for key, col, name in (("lm32", C["blue"], "audit: top 32 (layermax)"),
                               ("ens", C["aqua"], "audit: ensemble, ~top 16 x 3 scores"),
                               ("lm32+rand32", C["violet"], "audit: top 32 + 32 random")):
            ys = []
            for k in hk:
                rs = [r for r in R if r["hide_k"] == k and r["success"]]
                if key == "ens":
                    ys.append(np.mean([r["ens"]["16"][0] for r in rs]) if rs else np.nan)
                else:
                    ys.append(np.mean([r["plans"][key] for r in rs]) if rs else np.nan)
            ax.plot(hk, ys, "-o", color=col, label=name, markersize=5, lw=1.6)
        ax.set_xscale("log", base=2)
        ax.set_xticks(hk, [str(k) for k in hk])
        ax.set_xlabel("attacker hides below rank (layermax score)")
        ax.set_title(lab)
        ax.set_ylim(-0.02, 1.05)
    axes[0][0].set_ylabel("rate (detection: among successful)")
    axes[0][0].legend(fontsize=8, loc="lower left")
    fig.suptitle("A white-box attacker can hide from the score it optimises against", y=1.03)
    save(fig, "adaptive")


# ------------------------------------------------------------------ fig: in-noise
def fig_innoise():
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    any_ = False
    for (mk, label), col in zip(MODELS, (C["blue"], C["orange"])):
        d = load(f"innoise_{mk}_bf16.json")
        if not d:
            continue
        any_ = True
        R = [r for r in d["records"] if r["mode"] == "forge"]
        xs = sorted(set(r["x"] for r in R))
        s = [np.mean([r["success"] for r in R if r["x"] == x]) for x in xs]
        u = [np.mean([r["success"] and r["n_flagged"] == 0 for r in R if r["x"] == x]) for x in xs]
        ax.plot(xs, s, "-o", color=col, label=f"{label}: flips answer", markersize=6)
        ax.plot(xs, u, "--s", color=col, label=f"{label}: flips and passes full check", markersize=5, lw=1.4)
    if not any_:
        plt.close(fig)
        return
    ax.set_xscale("log")
    ax.set_xlabel("nudge at every column / residual norm")
    ax.set_ylabel("share of tasks")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=8)
    ax.set_title("Hiding inside bf16 tolerance (every column forged)")
    save(fig, "innoise")


def fig_cost():
    d = load("cost_llama-1b.json")
    if not d:
        return
    rows = d["rows"]
    T = [r["T"] for r in rows]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(T, [r["prefill"] * 1e3 for r in rows], "-o", color=C["ink2"], label="full prefill (recompute)")
    ax.plot(T, [(r["cols16"] + r["influence"]) * 1e3 for r in rows], "-o", color=C["blue"], label="audit: 16 columns + scoring")
    ax.plot(T, [(r["cols64"] + r["influence"]) * 1e3 for r in rows], "-o", color=C["aqua"], label="audit: 64 columns + scoring")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(T, [f"{t // 1024}K" if t >= 1024 else str(t) for t in T])
    ax.set_xlabel("document length (tokens)")
    ax.set_ylabel("milliseconds (RTX 5060 Laptop)")
    ax.legend(fontsize=8.5)
    ax.set_title("What an audit costs (Llama-3.2-1B, bf16)")
    save(fig, "cost")

# ------------------------------------------------------------------ fig: defenses vs adaptive forgers
def load_defense(mk):
    import glob
    parts = [load(f"defense_{mk}.json")] + [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RES, f"defense_{mk}_*.json")))]
    parts = [x for x in parts if x]
    return {**parts[0], "records": [r for x in parts for r in x["records"]]} if parts else None


DEF_ATT = ["none", "mean", "layermax", "headmax", "tokmax", "all public", "all public + trace"]
DEF_PLANS = [("mean", "mean"), ("layermax", "layermax"), ("headmax", "headmax"), ("tokmax", "tokmax"),
             ("grad", "gradient"), ("all public", "all public|merged"), ("secret", "secret|heads"),
             ("trace", "trace"), ("trace + random", "trace +|random"), ("random", "random")]


def fig_defense(B="32"):
    ds = [(mk, lab, load_defense(mk)) for mk, lab in MODELS]
    ds = [x for x in ds if x[2]]
    if not ds:
        return
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", ["#f2f6fc", "#9cc0ec", C["blue"], "#16437a"])
    fig, axes = plt.subplots(len(ds), 1, figsize=(8.6, 3.9 * len(ds)), squeeze=False)
    for ax, (mk, lab, d) in zip(axes[:, 0], ds):
        R = d["records"]
        atts = [a for a in DEF_ATT if any(r["attacker"] == a for r in R)]
        M = np.full((len(atts), len(DEF_PLANS)), np.nan)
        ylab = []
        for i, a in enumerate(atts):
            rs = [r for r in R if r["attacker"] == a]
            ok = [r for r in rs if r["success"]]
            ylab.append(f"{'no hiding' if a == 'none' else 'hides from ' + a}  ({len(ok)}/{len(rs)})")
            for j, (pk, _) in enumerate(DEF_PLANS):
                if ok:
                    M[i, j] = np.mean([r["detect"][B][pk] for r in ok])
        ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{100 * M[i, j]:.0f}", ha="center", va="center", fontsize=8,
                            color="white" if M[i, j] > 0.55 else C["ink"])
        ax.set_xticks(range(len(DEF_PLANS)), [x[1].replace("|", " ") for x in DEF_PLANS], fontsize=7.5,
                      rotation=35, ha="right", rotation_mode="anchor")
        ax.set_yticks(range(len(atts)), ylab, fontsize=8)
        ax.set_xlabel(f"how the verifier picks {B} columns", fontsize=9)
        ax.set_title(f"{lab}: % of successful forgeries caught", fontsize=10)
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.suptitle("Round two: audits against forgers that know the public scores (32 columns audited)", y=1.0)
    fig.tight_layout()
    save(fig, "defense")



def fig_game(B="32"):
    ds = [(mk, lab, load_defense(mk)) for mk, lab in MODELS]
    ds = [x for x in ds if x[2]]
    if not ds:
        return
    plans = [p for p in DEF_PLANS]
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    y = np.arange(len(plans))
    h = 0.38
    for i, ((mk, lab, d), col) in enumerate(zip(ds, (C["blue"], C["orange"]))):
        R = d["records"]
        atts = sorted(set(r["attacker"] for r in R))
        wins = []
        for pk, _ in plans:
            wins.append(max(np.mean([(1 - r["detect"][B][pk]) if r["success"] else 0.0 for r in R if r["attacker"] == a]) for a in atts))
        ax.barh(y + (i - 0.5) * h, wins, h * 0.92, color=col, label=lab)
        for yy, w in zip(y + (i - 0.5) * h, wins):
            ax.text(w + 0.01, yy, f"{100 * w:.0f}%", va="center", fontsize=7.5, color=C["ink2"])
    ax.set_yticks(y, [x[1].replace("|", " ") for x in plans], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.08)
    ax.set_xlabel(f"attacker's best reply: P(answer flipped and audit of {B} columns misses it)")
    ax.grid(axis="y", visible=False)
    ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
    ax.set_title("Which audit is hardest to game? (lower is better)")
    save(fig, "game")


if __name__ == "__main__":
    for f in (fig_forge_success, fig_budget, fig_detection, fig_corrupt, fig_adaptive, fig_innoise, fig_cost, fig_defense, fig_game):
        try:
            f()
        except Exception as ex:  # keep going: partial results are normal mid-run
            print("skip", f.__name__, type(ex).__name__, ex)
