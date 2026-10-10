"""Mean potential outcome figure (Section 5.2): bias, ARB, MSE and coverage from bias/MSE result JSONs.

Usage:
    python -m model.visualization.mean_potential_outcome small_n.json large_n.json [out.jpg] [--fixed-model]

Inputs are bias_mse_study outputs at two sample sizes (rows are labelled by
each study's n). Targets without any estimate are marked n/a. Coverage labels use
the retraining bootstrap (`retrain_bootstrap`) unless --fixed-model selects the
fixed-model network-block bootstrap (`bootstrap`).
"""

import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

PATHS = [("never", "Never treated"), ("early", "Early"), ("late", "Late"),
         ("intermittent", "Intermittent"), ("frequent", "Frequent")]
DLEV = [("high", "High d"), ("mid", "Mid d"), ("low", "Low d")]
METH = [("lstm_kipw", "Proposed", "#3A7CB8", "o"),
        ("oracle_kipw", "Oracle Propensity", "#4CA64C", "^"),
        ("ols", "OLS", "#E4622B", "s")]
TEXT_FRACTION = 0.44  # share of panel width left of the label column


def summarize(path, boot_key):
    """Per (path, d-level) target: (bias, ARB, MSE) per method over replications, and bootstrap CI coverage (%) of the proposed estimator; returns (n, cells)."""
    d = json.load(open(path))
    truth, reps = d["truth_with_contrasts"], d["replications"]
    out = {}
    for p, _ in PATHS:
        for dl, _ in DLEV:
            t = f"mu__{p}__{dl}"
            cell = {}
            for m, _, _, _ in METH:
                b = np.array([r["point_estimates"][m][t] - truth[t] for r in reps
                              if r["point_estimates"][m].get(t) is not None])
                if b.size:
                    cell[m] = (b.mean(), abs(b.mean()) / abs(truth[t]), (b ** 2).mean())
                else:
                    cell[m] = (np.nan, np.nan, np.nan)
            hit = n = 0
            for r in reps:
                s = ((r.get(boot_key) or {}).get("lstm_kipw") or {}).get("summaries", {}).get(t)
                if s is None or s.get("ci_lower") is None:
                    continue
                n += 1
                hit += s["ci_lower"] <= truth[t] <= s["ci_upper"]
            cell["cov"] = 100 * hit / n if n else np.nan
            out[(p, dl)] = cell
    return d["arguments"]["n"], out


def main(argv):
    """Parse the command line and write the 2 x 5 panel figure (rows: sample sizes, columns: own-treatment paths)."""
    if {"-h", "--help"} & set(argv):
        print(__doc__)
        return
    boot_key = "bootstrap" if "--fixed-model" in argv else "retrain_bootstrap"
    argv = [a for a in argv if not a.startswith("--")]
    out = argv[2] if len(argv) > 2 else "mean_potential_outcome.jpg"
    if len(argv) < 2:
        sys.exit(__doc__)
    D = dict(summarize(f, boot_key) for f in argv[:2])
    files = list(D)
    if len(files) != 2:
        sys.exit("The two studies must have different n.")

    fig = plt.figure(figsize=(20, 8.2), dpi=240)
    L, R, B, T = 0.055, 0.965, 0.075, 0.885
    gapx, gapy = 0.012, 0.045
    W = (R - L - 4 * gapx) / 5
    H = (T - B - gapy) / 2
    ys = {(di, mi): 8 - (di * 3 + mi) for di in range(3) for mi in range(3)}

    for ci, (p, plab) in enumerate(PATHS):
        vals = [v for v in (D[n][(p, dl)][m][0] for n in files for dl, _ in DLEV
                            for m, _, _, _ in METH) if np.isfinite(v)]
        lo, hi = min(vals + [0]), max(vals + [0])
        span = hi - lo if hi > lo else 1.0
        x0 = lo - 0.10 * span
        x1 = x0 + (hi - x0) / TEXT_FRACTION
        for ri, n in enumerate(files):
            ax = fig.add_axes([L + ci * (W + gapx), T - H - ri * (H + gapy), W, H])
            ax.set_xlim(x0, x1)
            ax.set_ylim(-0.7, 8.7)
            ax.set_axisbelow(True)
            ax.grid(axis="x", color="#ededed", lw=0.7)
            for di in range(3):
                ax.axhline(ys[(di, 1)], color="#ededed", lw=0.7)
            ax.axvline(0, color="#1a1a1a", lw=1.3, zorder=2.5)
            for di, (dl, _) in enumerate(DLEV):
                cell = D[n][(p, dl)]
                for mi, (m, _, col, mk) in enumerate(METH):
                    bias, arb, mse = cell[m]
                    y = ys[(di, mi)]
                    if not np.isfinite(bias):
                        ax.annotate("n/a (no estimate)", (0, y), xytext=(9, 0),
                                    textcoords="offset points", va="center", ha="left",
                                    fontsize=7.3, color=col, zorder=4)
                        continue
                    ax.plot([bias], [y], marker=mk, color=col, ms=7.5, mec="white",
                            mew=0.8, zorder=3, clip_on=False)
                    lab = f"ARB {arb * 100:.1f}%  $\\cdot$  MSE {mse:.2f}"
                    if m == "lstm_kipw":
                        cov = cell["cov"]
                        lab += f"  $\\cdot$  Cov {cov:.0f}%" if np.isfinite(cov) else "  $\\cdot$  Cov n/a"
                    ax.annotate(lab, (bias, y), xytext=(9, 0), textcoords="offset points",
                                va="center", ha="left", fontsize=7.3, color=col, zorder=4)
            ax.set_yticks([ys[(di, 1)] for di in range(3)])
            ax.set_yticklabels([dlab for _, dlab in DLEV] if ci == 0 else [], fontsize=10)
            ax.tick_params(axis="y", length=0)
            for s in ("top", "right", "left"):
                ax.spines[s].set_visible(False)
            ax.spines["bottom"].set_color("#999999")
            if ri == 1:
                ax.tick_params(axis="x", labelsize=9, colors="#333333")
            else:
                ax.set_xticklabels([])
                ax.add_patch(Rectangle((0, 1.005), 1, 0.085, transform=ax.transAxes,
                                       facecolor="#eeeeee", edgecolor="#cccccc", lw=0.6,
                                       clip_on=False, zorder=5))
                ax.text(0.5, 1.048, plab, transform=ax.transAxes, ha="center", va="center",
                        fontsize=11.5, fontweight="bold", zorder=6)
            if ci == 4:
                ax.add_patch(Rectangle((1.008, 0), 0.055, 1, transform=ax.transAxes,
                                       facecolor="#eeeeee", edgecolor="#cccccc", lw=0.6,
                                       clip_on=False, zorder=5))
                ax.text(1.0355, 0.5, f"n = {n:,}", transform=ax.transAxes, ha="center",
                        va="center", rotation=270, fontsize=11, zorder=6)

    handles = [plt.Line2D([], [], marker=mk, color=col, ls="", ms=9, label=lb)
               for _, lb, col, mk in METH]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=3,
               frameon=False, fontsize=11.5, handletextpad=0.4, columnspacing=2.6)
    fig.text(0.5, 0.018, "Bias (estimate $-$ ground truth)", ha="center", fontsize=12.5)
    fig.savefig(out, format="jpg", pil_kwargs={"quality": 94})
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1:])
