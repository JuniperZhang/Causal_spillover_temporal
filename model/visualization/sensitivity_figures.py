"""Draw the three sensitivity-analysis PNG figures from a complete study.json.

Usage:
    python -m model.visualization.sensitivity_figures --study study.json --out OUT_DIR

Reads the stored sensitivity_summary only: no training, no weights, no
recomputed bounds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PATHS = ("never", "early", "late", "intermittent", "frequent")
MEANS = tuple(f"mu__{path}__{level}" for path in PATHS for level in ("low", "mid", "high"))
EFFECTS = tuple(f"{kind}__{path}" for kind in ("DE", "SE", "TE") for path in PATHS
                if kind == "SE" or path != "never")
METHODS = ("lstm_kipw", "oracle_kipw")


def validate_study(payload):
    """Check that the study is complete and its sensitivity_summary is well formed; raise ValueError otherwise."""
    args = payload["arguments"]
    n, count = args["n"], args["R"]
    if not isinstance(n, int) or not isinstance(count, int) or n <= 0 or count <= 0:
        raise ValueError("n and R must be positive integers")
    indices = sorted(r["rep_index"] for r in payload["replications"])
    if indices != list(range(count)):
        raise ValueError("Incomplete or duplicate replications: merge all shards before plotting")
    grid = args["sensitivity_gammas"]
    if not grid or grid[0] != 1 or any(not math.isfinite(g) for g in grid) or any(b <= a for a, b in zip(grid, grid[1:])):
        raise ValueError("Gamma grid must start at 1 and increase strictly")
    for method in METHODS:
        for name in MEANS + EFFECTS:
            rows = payload["sensitivity_summary"][method][name]
            if [row["gamma"] for row in rows] != grid:
                raise ValueError(f"Inconsistent Gamma grid: {method}/{name}")
            for row in rows:
                if row["n_total"] != count or not 0 <= row["n_valid"] <= count:
                    raise ValueError(f"Invalid replication denominator: {method}/{name}")
                if row["n_valid"]:
                    lo, hi, zero = row["mean_lower"], row["mean_upper"], row["zero_inclusion_rate"]
                    if not all(math.isfinite(v) for v in (lo, hi, zero)) or lo > hi or not 0 <= zero <= 1:
                        raise ValueError(f"Invalid endpoints or inclusion rate: {method}/{name}")
            if not math.isfinite(payload["truth_with_contrasts"][name]):
                raise ValueError(f"Non-finite truth: {name}")


def ticks(grid, preferred):
    """Gamma-axis tick positions: the preferred ticks, plus the grid endpoints when the grid is not [1, 5]."""
    if grid[0] == 1 and grid[-1] == 5:
        return preferred
    # Keep endpoints visible for regridded results without pretending Gamma ends at 5.
    return sorted(set([grid[0], grid[-1]] + [g for g in preferred if grid[0] <= g <= grid[-1]]))


def endpoint_figures(payload, directory):
    """Draw mean lower/upper sensitivity endpoints against Gamma for the causal effects and for the potential-outcome means, fitted vs oracle weights."""
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.titlesize": 8,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    summary = payload["sensitivity_summary"]
    xticks = ticks(payload["arguments"]["sensitivity_gammas"], [1, 2, 5])
    for kind, columns in (("effect", ("DE", "SE", "TE")), ("mean", ("low", "mid", "high"))):
        fig, axes = plt.subplots(5, 3, figsize=(7.2, 8.1))
        handles = labels = None
        for i, path in enumerate(PATHS):
            for j, column in enumerate(columns):
                ax = axes[i, j]
                name = f"{column}__{path}" if kind == "effect" else f"mu__{path}__{column}"
                if name not in summary["lstm_kipw"]:
                    ax.set_visible(False)
                    continue
                for method, label, color, style in (("lstm_kipw", "DL-KIPW", "#0072B2", "-"),
                                                    ("oracle_kipw", "Oracle KIPW", "#D55E00", "--")):
                    values = summary[method][name]
                    x = [v["gamma"] for v in values]
                    lo = [v["mean_lower"] if v["n_valid"] else math.nan for v in values]
                    hi = [v["mean_upper"] if v["n_valid"] else math.nan for v in values]
                    ax.plot(x, lo, color=color, ls=style, lw=.9, label=label)
                    ax.plot(x, hi, color=color, ls=style, lw=.9)
                    ax.fill_between(x, lo, hi, color=color, alpha=.06)
                ax.axhline(payload["truth_with_contrasts"][name], color=".4", ls=":", lw=.8,
                           label="Local-MC truth")
                if kind == "effect":
                    ax.axhline(0, color="black", ls="--", lw=.5)
                ax.set_title(name.replace("mu__", "").replace("__", ": "))
                ax.set_xscale("log")
                ax.set_xticks(xticks, [f"{g:g}" for g in xticks])
                ax.minorticks_off()
                ax.set_xlabel(r"$\Gamma$", labelpad=1)
                ax.grid(alpha=.15, lw=.5)
                handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
        fig.tight_layout(rect=(0, .025, 1, 1), pad=.7, h_pad=1, w_pad=1)
        fig.savefig(directory / f"sensitivity_{kind}_bounds_n{payload['arguments']['n']}.png",
                    dpi=300, bbox_inches="tight")
        plt.close(fig)


def zero_inclusion_figure(payload, directory):
    """Draw the share of DL-KIPW sensitivity intervals containing zero against Gamma for DE, SE and TE."""
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.titlesize": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    colors = {"never": "#777777", "early": "#0072B2", "late": "#D55E00",
              "intermittent": "#009E73", "frequent": "#CC79A7"}
    styles = {"never": ("-", "o"), "early": ("--", "s"), "late": ("-.", "^"),
              "intermittent": (":", "D"), "frequent": ("-", "v")}
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), sharex=True, sharey=True, squeeze=False)
    summary = payload["sensitivity_summary"]["lstm_kipw"]
    xticks = ticks(payload["arguments"]["sensitivity_gammas"], [1, 1.5, 2, 3, 5])
    for col, (prefix, title) in enumerate((("DE", "Direct"), ("SE", "Spillover"), ("TE", "Total"))):
        ax = axes[0, col]
        for path, color in colors.items():
            name = f"{prefix}__{path}"
            if name not in summary:
                continue
            values = summary[name]
            ax.plot([v["gamma"] for v in values],
                    [v["zero_inclusion_rate"] if v["n_valid"] else math.nan for v in values],
                    color=color, linestyle=styles[path][0], marker=styles[path][1],
                    markersize=2.5, linewidth=1.0, label=path)
        ax.set_title(f"{chr(97+col)}) {title}, n = {payload['arguments']['n']:,}")
        ax.set_xscale("log")
        ax.set_xticks(xticks, [f"{g:g}" for g in xticks])
        ax.minorticks_off()
        ax.set_ylim(-.03, 1.03)
        ax.set_yticks([0, .25, .5, .75, 1], ["0", "25", "50", "75", "100"])
        ax.grid(alpha=.2, linewidth=.5)
        ax.set_xlabel(r"Joint sensitivity parameter $\Gamma$")
        if col == 0:
            ax.set_ylabel("Intervals containing zero (%)")
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, .06, 1, 1), pad=.8)
    fig.savefig(directory / f"sensitivity_zero_inclusion_n{payload['arguments']['n']}.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)


def latex_fragment(n):
    """Return a LaTeX fragment that includes the three figures with captions and labels."""
    number = f"{n:,}".replace(",", "{,}")
    sections = [r"% Preamble: \usepackage{graphicx}",
                r"% Set \SensitivityFigureDir before \input to change the PNG directory.",
                r"\providecommand{\SensitivityFigureDir}{figs}"]
    for stem, caption, label in (
        ("zero_inclusion", "Percentage of DL-kernel-IPW sensitivity intervals containing zero, "
         r"by joint sensitivity parameter \(\Gamma\) and own-treatment path. Panels show direct, "
         "spillover and total causal effects. Percentages use valid-replication denominators.", f"fig:sensitivity_n{n}"),
        ("effect_bounds", "Mean sensitivity endpoints for the 13 causal effects. "
         "Blue solid lines use fitted DL-kernel-IPW weights; orange dashed lines use oracle weights. "
         "Dotted lines mark local-Monte-Carlo approximate truth. Shading spans mean endpoints, "
         "not a confidence band.", f"fig:sensitivity_effect_bounds_n{n}"),
        ("mean_bounds", "Mean sensitivity endpoints for the 15 potential-outcome means. "
         "Blue solid lines use fitted DL-kernel-IPW weights; orange dashed lines use oracle weights. "
         "Dotted lines mark local-Monte-Carlo approximate truth. Shading spans mean endpoints, "
         "not a confidence band.", f"fig:sensitivity_mean_bounds_n{n}"),
    ):
        size = r"width=\linewidth" if stem == "zero_inclusion" else r"width=\linewidth,height=0.76\textheight,keepaspectratio"
        sections += ["", r"\begin{figure}[p]", r"\centering",
                     rf"\includegraphics[{size}]{{\SensitivityFigureDir/sensitivity_{stem}_n{n}.png}}",
                     rf"\caption{{Sensitivity analysis for \(n={number}\). {caption}}}",
                     rf"\label{{{label}}}", r"\end{figure}", r"\clearpage"]
    return "\n".join(sections) + "\n"


def main(argv=None):
    """Parse the command line, validate the study and write the figures, LaTeX fragment and hash manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="New or empty output directory")
    args = parser.parse_args(argv)
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("Output directory is not empty; existing paper figures will not be overwritten")
    raw = args.study.read_bytes()
    payload = json.loads(raw)
    validate_study(payload)
    args.out.mkdir(parents=True, exist_ok=True)
    with plt.rc_context():
        endpoint_figures(payload, args.out)
        zero_inclusion_figure(payload, args.out)
    (args.out / "sensitivity_figures.tex").write_text(latex_fragment(payload['arguments']['n']))
    manifest = {"input_study": str(args.study.resolve()), "input_sha256": hashlib.sha256(raw).hexdigest(),
                "plot_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "n": payload["arguments"]["n"], "R": payload["arguments"]["R"],
                "matplotlib": matplotlib.__version__, "dpi": 300,
                "data_source": "stored sensitivity_summary; no endpoints recomputed",
                "training_performed": False, "weights_loaded": False,
                "output_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted(args.out.iterdir()) if p.is_file()}}
    (args.out / "plot_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved three PNG figures, LaTeX fragment and plot manifest in {args.out}")


if __name__ == "__main__":
    main()
