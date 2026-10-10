"""Overlay several complete study.json files in the sensitivity-analysis figures.

Usage:
    python -m model.visualization.sensitivity_comparison --studies a/study.json b/study.json --out OUT_DIR

Reads the stored sensitivity_summary only: no training, no weights, no
recomputed bounds. The studies must share the Gamma grid, the fixed reference
paths, the target sequences and the approximate truth, or they are not
comparable and nothing is drawn.
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

from . import sensitivity_figures as base

PATHS = base.PATHS
# Colour keeps its meaning from the single-study figures; dashes carry sample size.
# The two estimators nearly coincide, so the oracle is a wide translucent line
# underneath and the fitted estimator a thin opaque one on top: overlap stays visible.
METHOD_STYLE = (("oracle_kipw", "Oracle KIPW", "#D55E00", 2.2, .34),
                ("lstm_kipw", "DL-KIPW", "#0072B2", .9, 1.))
SIZE_DASHES = ((4, 2), (), (1, 1.4), (6, 2, 1, 2))


def close_enough(left, right, rel_tol=1e-9):
    """Structural equality with a float tolerance; endpoints move by a few ulp."""
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(close_enough(left[k], right[k], rel_tol) for k in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(close_enough(a, b, rel_tol) for a, b in zip(left, right))
    if isinstance(left, float) or isinstance(right, float):
        return (left is not None and right is not None
                and math.isclose(left, right, rel_tol=rel_tol, abs_tol=1e-12))
    return left == right


def check_comparable(payloads):
    """Raise ValueError unless the studies have distinct n and share the Gamma grid, reference paths, target sequences and truth."""
    if len(payloads) < 2 or len(payloads) > len(SIZE_DASHES):
        raise ValueError(f"Overlay needs 2 to {len(SIZE_DASHES)} studies")
    sizes = [p["arguments"]["n"] for p in payloads]
    if len(set(sizes)) != len(sizes):
        raise ValueError(f"Studies must have distinct sample sizes: {sizes}")
    first = payloads[0]
    for payload in payloads[1:]:
        if payload["arguments"]["sensitivity_gammas"] != first["arguments"]["sensitivity_gammas"]:
            raise ValueError("Studies use different Gamma grids")
        for block in ("reference_paths", "expanded_target_sequences", "truth_with_contrasts"):
            if not close_enough(payload.get(block), first.get(block)):
                raise ValueError(f"Studies disagree on {block}; they are not comparable")


def suffix(payloads):
    """File-name suffix listing the studies' sample sizes, e.g. n5000_n50000."""
    return "_".join(f"n{p['arguments']['n']}" for p in payloads)


def endpoint_figures(payloads, directory):
    """Draw mean sensitivity endpoints against Gamma for all studies overlaid, sample size encoded by dash pattern."""
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.titlesize": 8,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    xticks = base.ticks(payloads[0]["arguments"]["sensitivity_gammas"], [1, 2, 5])
    truth = payloads[0]["truth_with_contrasts"]
    for kind, columns in (("effect", ("DE", "SE", "TE")), ("mean", ("low", "mid", "high"))):
        fig, axes = plt.subplots(5, 3, figsize=(7.2, 8.1))
        handles = labels = None
        for i, path in enumerate(PATHS):
            for j, column in enumerate(columns):
                ax = axes[i, j]
                name = f"{column}__{path}" if kind == "effect" else f"mu__{path}__{column}"
                if any(name not in p["sensitivity_summary"]["lstm_kipw"] for p in payloads):
                    ax.set_visible(False)
                    continue
                for payload, dashes in zip(payloads, SIZE_DASHES):
                    size = payload["arguments"]["n"]
                    for method, method_label, color, width, alpha in METHOD_STYLE:
                        values = payload["sensitivity_summary"][method][name]
                        x = [v["gamma"] for v in values]
                        lo = [v["mean_lower"] if v["n_valid"] else math.nan for v in values]
                        hi = [v["mean_upper"] if v["n_valid"] else math.nan for v in values]
                        ax.plot(x, lo, color=color, dashes=dashes, lw=width, alpha=alpha,
                                label=f"{method_label}, n = {size:,}")
                        ax.plot(x, hi, color=color, dashes=dashes, lw=width, alpha=alpha)
                ax.axhline(truth[name], color=".4", ls=":", lw=.8, label="Local-MC truth")
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
        fig.tight_layout(rect=(0, .045, 1, 1), pad=.7, h_pad=1, w_pad=1)
        fig.savefig(directory / f"sensitivity_{kind}_bounds_{suffix(payloads)}.png",
                    dpi=300, bbox_inches="tight")
        plt.close(fig)


def zero_inclusion_figure(payloads, directory):
    """Draw zero-inclusion rates against Gamma, one row per study and one column per effect type."""
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.titlesize": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    colors = {"never": "#777777", "early": "#0072B2", "late": "#D55E00",
              "intermittent": "#009E73", "frequent": "#CC79A7"}
    styles = {"never": ("-", "o"), "early": ("--", "s"), "late": ("-.", "^"),
              "intermittent": (":", "D"), "frequent": ("-", "v")}
    rows = len(payloads)
    fig, axes = plt.subplots(rows, 3, figsize=(7.2, 2.55 * rows),
                             sharex=True, sharey=True, squeeze=False)
    xticks = base.ticks(payloads[0]["arguments"]["sensitivity_gammas"], [1, 1.5, 2, 3, 5])
    for row, payload in enumerate(payloads):
        summary = payload["sensitivity_summary"]["lstm_kipw"]
        for col, (prefix, title) in enumerate((("DE", "Direct"), ("SE", "Spillover"), ("TE", "Total"))):
            ax = axes[row, col]
            for path, color in colors.items():
                name = f"{prefix}__{path}"
                if name not in summary:
                    continue
                values = summary[name]
                ax.plot([v["gamma"] for v in values],
                        [v["zero_inclusion_rate"] if v["n_valid"] else math.nan for v in values],
                        color=color, linestyle=styles[path][0], marker=styles[path][1],
                        markersize=2.5, linewidth=1.0, label=path)
            ax.set_title(f"{chr(97 + row * 3 + col)}) {title}, n = {payload['arguments']['n']:,}")
            ax.set_xscale("log")
            ax.set_xticks(xticks, [f"{g:g}" for g in xticks])
            ax.minorticks_off()
            ax.set_ylim(-.03, 1.03)
            ax.set_yticks([0, .25, .5, .75, 1], ["0", "25", "50", "75", "100"])
            ax.grid(alpha=.2, linewidth=.5)
            if row == rows - 1:
                ax.set_xlabel(r"Joint sensitivity parameter $\Gamma$")
            if col == 0:
                ax.set_ylabel("Intervals containing zero (%)")
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, .05 / rows * 2, 1, 1), pad=.8)
    fig.savefig(directory / f"sensitivity_zero_inclusion_{suffix(payloads)}.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)


def latex_fragment(payloads):
    """Return a LaTeX fragment that includes the three overlay figures with captions and labels."""
    sizes = [f"{p['arguments']['n']:,}".replace(",", "{,}") for p in payloads]
    listed = r" and ".join([", ".join(sizes[:-1]), sizes[-1]] if len(sizes) > 2 else sizes)
    styles = ("Dashed and solid lines distinguish the sample sizes, in the order listed. "
              "Wide translucent orange lines use oracle weights and thin blue lines fitted "
              "DL-kernel-IPW weights, drawn on top because the two nearly coincide. Dotted "
              "lines mark the local-Monte-Carlo approximate truth, which is common to all "
              "sample sizes.")
    stem = suffix(payloads)
    sections = [r"% Preamble: \usepackage{graphicx}",
                r"% Set \SensitivityFigureDir before \input to change the PNG directory.",
                r"\providecommand{\SensitivityFigureDir}{figs}"]
    for name, caption, label in (
        ("zero_inclusion", "Percentage of DL-kernel-IPW sensitivity intervals containing zero, "
         r"by joint sensitivity parameter \(\Gamma\) and own-treatment path. Rows give the sample "
         "sizes and columns the direct, spillover and total causal effects. Percentages use "
         "valid-replication denominators.", f"fig:sensitivity_{stem}"),
        ("effect_bounds", "Mean sensitivity endpoints for the 13 causal effects, averaged over "
         f"valid replications. {styles} Shading is omitted so the sample sizes stay legible.",
         f"fig:sensitivity_effect_bounds_{stem}"),
        ("mean_bounds", "Mean sensitivity endpoints for the 15 potential-outcome means, averaged "
         f"over valid replications. {styles}", f"fig:sensitivity_mean_bounds_{stem}"),
    ):
        size = r"width=\linewidth" if name == "zero_inclusion" else r"width=\linewidth,height=0.76\textheight,keepaspectratio"
        sections += ["", r"\begin{figure}[p]", r"\centering",
                     rf"\includegraphics[{size}]{{\SensitivityFigureDir/sensitivity_{name}_{stem}.png}}",
                     rf"\caption{{Sensitivity analysis for \(n={listed}\). {caption}}}",
                     rf"\label{{{label}}}", r"\end{figure}", r"\clearpage"]
    return "\n".join(sections) + "\n"


def main(argv=None):
    """Parse the command line, validate and compare the studies and write the figures, LaTeX fragment and hash manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--studies", type=Path, nargs="+", required=True,
                        help="Complete study.json files, in the order they should be drawn")
    parser.add_argument("--out", type=Path, required=True, help="New or empty output directory")
    args = parser.parse_args(argv)
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("Output directory is not empty; existing paper figures will not be overwritten")
    raws = [path.read_bytes() for path in args.studies]
    payloads = [json.loads(raw) for raw in raws]
    for payload in payloads:
        base.validate_study(payload)
    check_comparable(payloads)
    args.out.mkdir(parents=True, exist_ok=True)
    with plt.rc_context():
        endpoint_figures(payloads, args.out)
        zero_inclusion_figure(payloads, args.out)
    (args.out / "sensitivity_figures.tex").write_text(latex_fragment(payloads))
    manifest = {"input_studies": [{"path": str(p.resolve()), "sha256": hashlib.sha256(r).hexdigest(),
                                   "n": q["arguments"]["n"], "R": q["arguments"]["R"]}
                                  for p, r, q in zip(args.studies, raws, payloads)],
                "plot_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "matplotlib": matplotlib.__version__, "dpi": 300,
                "data_source": "stored sensitivity_summary; no endpoints recomputed",
                "training_performed": False, "weights_loaded": False,
                "output_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted(args.out.iterdir()) if p.is_file()}}
    (args.out / "plot_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved three PNG figures, LaTeX fragment and plot manifest in {args.out}")


if __name__ == "__main__":
    main()
