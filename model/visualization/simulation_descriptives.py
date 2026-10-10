"""Regenerate the Supplementary Material descriptive figures of the simulated data.

Usage:
    python -m model.visualization.simulation_descriptives study.json OUT_DIR [--rep 0]
    python -m model.visualization.simulation_descriptives --overlap fit_n50000.npz fit_n5000.npz OUT_DIR

The first form regenerates replication REP's analysis network from the study's
recorded seed and DGP arguments and draws covariate_baselines.png,
covariate_trajectories.png, own_treatment_family_distribution.png,
spillover_by_cluster_pooled.png and outcome_trajectories.png.

The second form draws overlap_own_propensity_50000_vs_5000.png from .npz files
holding the fitted own-treatment propensities `e_hat` (n, 4) and the realized
treatments `X` (n, 4) at the decision times.
"""

import argparse
import contextlib
import copy
import io
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FAMILIES = {
    "never": {"0000"},
    "early": {"1000", "1100", "1110"},
    "late": {"0001", "0011", "0111"},
    "intermittent": {"0010", "0100", "0101", "0110", "1001", "1010", "1011", "1101"},
    "frequent": {"1111"},
}
COLORS = {"never": "#3B4A5C", "early": "#1E8A64", "late": "#2E86C1",
          "intermittent": "#8E44AD", "frequent": "#D4691E"}
NAVY, GREY = "#1F2A44", "#5F6B7A"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#E3E3E3", "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.edgecolor": "#333333", "xtick.color": "#333333", "ytick.color": "#333333",
})


def config_from_study(study):
    """Copy the default CONFIG and override the DGP settings recorded in the study's arguments."""
    from model.config import CONFIG
    a = study["arguments"]
    c = copy.deepcopy(CONFIG)
    c["num_workers"] = 0
    c["latent_nonlinear_variant"] = a["dgp_variant"]
    c["latent_spillover_burden_eta"] = float(a["latent_spillover_burden_eta"])
    c["latent_beta_XD"] = float(a["latent_beta_xd"])
    c["treatment_nonlinear_variant"] = a["treatment_variant"]
    c["treatment_theta_w2"] = float(a["treatment_theta_w2"])
    return c


def regenerate(study, rep_index):
    """Rebuild replication rep_index's analysis dataset from its recorded n and seed; check the final-time outcome mean against the stored value."""
    from model.pipeline.bias_mse_study import make_analysis_and_validation
    rep = study["replications"][rep_index]
    with contextlib.redirect_stdout(io.StringIO()):
        ds, _ = make_analysis_and_validation(config_from_study(study), rep["n"], rep["seed"],
                                             no_validation=True)
    y_t = ds.y.numpy()[:, -1, 0]
    stored = rep["observed_data_diagnostics"]["outcome_T"]["mean"]
    if abs(y_t.mean() - stored) > 1e-3:
        raise RuntimeError(f"regenerated Y_T mean {y_t.mean():.5f} != stored {stored:.5f}")
    return ds


def families_of(ds):
    """Label each unit with its own-treatment family from its treatment pattern at the decision times."""
    steps = list(ds.decision_steps)
    x = ds.x.numpy()[:, steps, 0].astype(int)
    patterns = np.array(["".join(map(str, row)) for row in x])
    labels = np.empty(len(patterns), dtype=object)
    for fam, members in FAMILIES.items():
        labels[np.isin(patterns, list(members))] = fam
    return labels


def header(fig, title, subtitle, top=0.985):
    """Draw a left-aligned title and subtitle at the top of the figure."""
    fig.text(0.008, top, title, ha="left", va="top", fontsize=15, fontweight="bold", color=NAVY)
    fig.text(0.008, top - 0.035, subtitle, ha="left", va="top", fontsize=10.5, color=GREY)


def covariate_baselines(ds, out):
    """Histogram each baseline covariate dimension V^(k) at t=0."""
    v0 = np.asarray(ds.v_0_baseline)
    fig, axes = plt.subplots(5, 2, figsize=(12, 15))
    for k, ax in enumerate(axes.flat):
        ax.hist(v0[:, k], bins=30, color="#4A78B5", edgecolor="white", linewidth=0.5)
        ax.axvline(v0[:, k].mean(), color="#D0342C", ls="--", lw=1.6)
        ax.set_title(f"$V^{{({k + 1})}}$", fontsize=12, color=NAVY)
    header(fig, "Baseline Covariate Distributions",
           f"Baseline (t=0) draws across the {v0.shape[1]} covariate dimensions; "
           "dashed line marks the mean")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "covariate_baselines.png", dpi=150)
    plt.close(fig)


def covariate_trajectories(ds, out):
    """Plot the population mean +/- 1 SD of each covariate over time."""
    v = ds.v.numpy()
    t = np.arange(1, v.shape[1] + 1)
    fig, axes = plt.subplots(5, 2, figsize=(12, 17))
    for k, ax in enumerate(axes.flat):
        m, s = v[:, :, k].mean(0), v[:, :, k].std(0)
        ax.fill_between(t, m - s, m + s, color="#A8DCCF", alpha=0.6, lw=0)
        ax.plot(t, m, color="#1F7A6B", lw=2.2)
        ax.set_title(f"$V^{{({k + 1})}}$", fontsize=12, color=NAVY)
        ax.set_xlabel("Time")
        ax.set_ylabel("Latent score")
    header(fig, "Covariate Trajectories",
           f"Population mean ± 1 SD over the full T={v.shape[1]} processing-time horizon")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "covariate_trajectories.png", dpi=150)
    plt.close(fig)


def family_distribution(fam, out):
    """Bar chart of the share of units in each own-treatment family; returns the shares in percent."""
    names = list(FAMILIES)
    share = np.array([(fam == f).mean() for f in names]) * 100
    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.bar(names, share, width=0.62, color=[COLORS[f] for f in names])
    for b, s in zip(bars, share):
        ax.text(b.get_x() + b.get_width() / 2, s + 0.6, f"{s:.1f}%", ha="center", va="bottom",
                fontsize=13, fontweight="bold", color=NAVY)
    ax.grid(axis="x", visible=False)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(decimals=1))
    ax.set_ylim(0, max(share) * 1.15)
    ax.set_xlabel("Own-treatment family", fontsize=12)
    ax.set_ylabel("Share of units", fontsize=12)
    ax.tick_params(labelsize=12)
    header(fig, "Own-Treatment Family Distribution",
           "Share of units realizing each of the five canonical own-treatment paths")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(out / "own_treatment_family_distribution.png", dpi=150)
    plt.close(fig)
    return dict(zip(names, share))


def spillover_by_family(ds, fam, out):
    """Violin and box plots of the spillover exposure D_it by own-treatment family, pooled over time."""
    d = ds.d_xs.numpy()[:, :, 0]
    names = list(FAMILIES)
    data = [d[fam == f].ravel() for f in names]
    fig, ax = plt.subplots(figsize=(11, 7))
    parts = ax.violinplot(data, showextrema=False, widths=0.8)
    for body, f in zip(parts["bodies"], names):
        body.set_facecolor(COLORS[f])
        body.set_edgecolor(COLORS[f])
        body.set_alpha(0.6)
    ax.boxplot(data, widths=0.14, patch_artist=True, showfliers=False,
               boxprops=dict(facecolor="white", edgecolor="#222222"),
               medianprops=dict(color="#C0392B", lw=2), whiskerprops=dict(color="#222222"),
               capprops=dict(color="#222222"))
    ax.set_xticks(range(1, len(names) + 1), names)
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("Own-treatment family", fontsize=12)
    ax.set_ylabel("Spillover treatment exposure $D_{it}$", fontsize=12)
    ax.tick_params(labelsize=12)
    header(fig, "Spillover Exposure by Own-Treatment Family",
           f"Pooled across all T={d.shape[1]} processing times; box shows quartiles")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(out / "spillover_by_cluster_pooled.png", dpi=150)
    plt.close(fig)


def outcome_trajectories(ds, fam, out):
    """Plot overall and by-family mean outcome trajectories; returns final-time means by family and the overall final-time (mean, SD)."""
    y = ds.y.numpy()[:, :, 0]
    t = np.arange(1, y.shape[1] + 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.6))
    m, s = y.mean(0), y.std(0)
    a1.fill_between(t, m - s, m + s, color="#F5CBA7", alpha=0.75, lw=0)
    a1.plot(t, m, color="#8B4513", lw=2.4)
    a1.set_title("Overall", fontsize=13, fontweight="bold", color=NAVY)
    for f in FAMILIES:
        a2.plot(t, y[fam == f].mean(0), color=COLORS[f], lw=2.4, label=f)
    a2.legend(frameon=False, fontsize=12, loc="lower right")
    a2.set_title("By own-treatment family", fontsize=13, fontweight="bold", color=NAVY)
    for ax in (a1, a2):
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Outcome $Y_{it}$", fontsize=12)
        ax.tick_params(labelsize=11)
    header(fig, "Outcome Trajectories", "Population mean ± 1 SD (left) and by-family means (right)")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(out / "outcome_trajectories.png", dpi=150)
    plt.close(fig)
    return {f: float(y[fam == f, -1].mean()) for f in FAMILIES}, (float(m[-1]), float(s[-1]))


def overlap(files, out):
    """Histogram the fitted own-treatment propensities by realized treatment, one panel per decision time and sample size."""
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    bins = np.linspace(0, 1, 41)
    for row, (n, path) in enumerate(files):
        z = np.load(path)
        e, x, steps = z["e_hat"], z["X"], z["steps"]
        for c in range(e.shape[1]):
            ax = axes[row, c]
            ax.hist(e[x[:, c] == 0, c], bins=bins, density=True, alpha=0.55, color="#9DB0D6",
                    label="actual X=0")
            ax.hist(e[x[:, c] == 1, c], bins=bins, density=True, alpha=0.55, color="#E8A87C",
                    label="actual X=1")
            ax.set_title(f"n={n}, decision t={int(steps[c]) + 1}")
            ax.set_xlabel("Estimated own propensity")
            ax.grid(False)
            if c == 0:
                ax.set_ylabel("Density")
            if row == 0 and c == 0:
                ax.legend()
            for side in ("top", "right"):
                ax.spines[side].set_visible(True)
    fig.suptitle("Step-specific own-treatment propensity by actual treatment status", fontsize=14)
    fig.tight_layout()
    fig.savefig(out / "overlap_own_propensity_50000_vs_5000.png", dpi=150)
    plt.close(fig)


def main(argv):
    """Parse the command line and draw either the descriptive figures or the overlap figure."""
    p = argparse.ArgumentParser()
    p.add_argument("inputs", nargs="+")
    p.add_argument("--rep", type=int, default=0)
    p.add_argument("--overlap", action="store_true")
    a = p.parse_args(argv)
    out = Path(a.inputs[-1])
    out.mkdir(parents=True, exist_ok=True)
    if a.overlap:
        overlap([(50000, a.inputs[0]), (5000, a.inputs[1])], out)
        return
    study = json.load(open(a.inputs[0]))
    ds = regenerate(study, a.rep)
    fam = families_of(ds)
    covariate_baselines(ds, out)
    covariate_trajectories(ds, out)
    shares = family_distribution(fam, out)
    spillover_by_family(ds, fam, out)
    final_by_family, (m, s) = outcome_trajectories(ds, fam, out)
    d = ds.d_xs.numpy()[:, :, 0]
    print(json.dumps({
        "family_share_pct": {k: round(v, 1) for k, v in shares.items()},
        "Y_T_mean_sd": [round(m, 2), round(s, 2)],
        "Y_T_by_family": {k: round(v, 2) for k, v in final_by_family.items()},
        "D_median_by_family": {f: round(float(np.median(d[fam == f])), 3) for f in FAMILIES},
        "D_mean_by_family": {f: round(float(d[fam == f].mean()), 3) for f in FAMILIES},
    }, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
