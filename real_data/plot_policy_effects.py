"""
Figure 3 of the paper: estimated effects of policy paths relative to the joint
low-intensity baseline, one panel per policy domain and outcome month.

    python real_data/plot_policy_effects.py --results-root real_data/pooled --out real_data/policy_effects.png

Each panel shows the treated months of each path (left) and the effect with
its 95% interval for the last-week and month-average case rates (right).
--results-root holds <domain>/estimates/outcome_expectations_<summary>_<month>.json,
as written by pool_seeds.py (or by estimate.py for a single seed).
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

SERIES = ["Last week", "Month average"]
COLORS = {"Last week": "#F7776C", "Month average": "#433641"}
OFFSETS = {"Last week": -0.12, "Month average": 0.12}
CELL_CMAP = ListedColormap(["#F0E0D3", "#DE476A"])

PANELS = [
    ("Business", "Sep 2020 endpoint", "Business: Sep 2020", "Business_Economic_Restrictions", "2020-09"),
    ("Business", "Oct 2020 endpoint", "Business: Oct 2020", "Business_Economic_Restrictions", "2020-10"),
    ("Education", "Jul 2020 endpoint", "Education: Jul 2020", "Education_Childcare", "2020-07"),
    ("Education", "Aug 2020 endpoint", "Education: Aug 2020", "Education_Childcare", "2020-08"),
]
AGG_FILE = {"Last week": "last_week",
            "Month average": "final_month_average"}


BUSINESS_DISPLAY_PERIODS = {
    "Sep 2020 endpoint": ["2020-04", "2020-05", "2020-06", "2020-07", "2020-08", "2020-09"],
    "Oct 2020 endpoint": ["2020-04", "2020-05", "2020-06", "2020-07", "2020-08", "2020-09", "2020-10"],
}
EDUCATION_DISPLAY_PERIODS = {
    "Jul 2020 endpoint": ["2020-04", "2020-05", "2020-06", "2020-07"],
    "Aug 2020 endpoint": ["2020-04", "2020-05", "2020-06", "2020-07", "2020-08"],
}
BUSINESS_PERIODS = ["2020-04", "2020-05", "2020-06", "2020-07", "2020-08", "2020-09", "2020-10"]
# Business paths not shown in Figure 3 (Sep 2020 own-treatment patterns, April-September).
BUSINESS_EXCLUDED = {
    (1, 0, 0, 0, 0, 0),
    (0, 1, 0, 0, 0, 0),
    (0, 0, 1, 1, 0, 0),
    (1, 1, 1, 1, 1, 0),
    (1, 1, 1, 1, 1, 1),
}


def load_json(path: Path) -> dict:
    """Read a JSON file."""
    return json.loads(path.read_text())


def parse_x(seq_key: str, periods: list[str]) -> tuple[int, ...]:
    """Own-treatment pattern of a path key, padded with zeros to len(periods)."""
    seq = ast.literal_eval(seq_key)
    x = [int(seq[i]) for i in range(0, len(seq), 2)]
    return tuple(x + [0] * (len(periods) - len(x)))


def business_rows(cases) -> dict[str, list[tuple[int, ...]]]:
    """
    Business rows: every estimated non-baseline September path except
    BUSINESS_EXCLUDED, ordered by number of treated months, then later start,
    then larger |effect|. October rows extend each September path by its last month.
    """
    by_subtitle = {subtitle: paths for policy, subtitle, paths in cases if policy == "Business"}
    sep_periods = BUSINESS_PERIODS[:-1]
    scores: dict[tuple[int, ...], float] = {}
    for path in by_subtitle["Sep 2020 endpoint"].values():
        for seq_key, info in load_json(path)["ate_results"].items():
            ate = info.get("ate_estimate")
            if ate is None:
                continue
            x_full = parse_x(seq_key, sep_periods)
            if sum(x_full) == 0:
                continue
            scores[x_full] = max(scores.get(x_full, 0.0), abs(float(ate)))
    sep_rows = sorted(scores, key=lambda x: (sum(x), -next((i for i, v in enumerate(x) if v == 1), 999),
                                             -scores[x]))
    sep_rows = [x for x in sep_rows if x not in BUSINESS_EXCLUDED]
    oct_rows = [tuple(list(x) + [x[-1]]) for x in sep_rows]
    oct_rows = [x for x in oct_rows if x != (1, 1, 1, 1, 1, 1, 1)]
    return {"Sep 2020 endpoint": sep_rows, "Oct 2020 endpoint": oct_rows}


def parse_args() -> argparse.Namespace:
    """Command-line options."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-root", required=True,
                   help="Folder holding <domain>/estimates/outcome_expectations_*.json")
    p.add_argument("--drop", action="append", default=[], metavar="TITLE::ROW",
                   help='Omit one row, e.g. "Education: Jul 2020::07 | 1 mo"; repeatable')
    p.add_argument("--out", required=True)
    p.add_argument("--ci-note", default=None,
                   help="Caption line under the figure describing the interval")
    p.add_argument("--dpi", type=int, default=200)
    return p.parse_args()


def build_cases(root: Path) -> list[tuple[str, str, dict[str, Path]]]:
    """Estimate file of each panel and series; fails if one is missing."""
    cases = []
    for policy, subtitle, _title, family, end in PANELS:
        paths = {name: root / family / "estimates" /
                 f"outcome_expectations_{AGG_FILE[name]}_{end}.json"
                 for name in SERIES}
        for path in paths.values():
            if not path.exists():
                raise FileNotFoundError(path)
        cases.append((policy, subtitle, paths))
    return cases


def row_label(pattern: tuple[int, ...], display_periods: list[str]) -> str:
    """Row label: first-last treated month and number of treated months."""
    on = [i for i, v in enumerate(pattern) if v == 1]
    months = [p[5:] for p in display_periods]
    if len(on) == 1:
        return f"{months[on[0]]} | 1 mo"
    return f"{months[on[0]]}-{months[on[-1]]} | {len(on)} mo"


def series_points(obj: dict, policy: str, patterns, display_periods):
    """Point estimate and CI per row, using the module's own row matching."""
    out = []
    for pattern in patterns:
        info = next((cand for key, cand in obj["ate_results"].items()
                     if parse_x(key, display_periods) == pattern), None)
        if info is None or info.get("ate_estimate") is None:
            out.append(None)
        else:
            out.append((info["ate_estimate"], info["ci_lower"], info["ci_upper"]))
    return out


def education_rows(cases) -> dict[str, list[tuple[int, ...]]]:
    """Every estimated non-never path at each Education endpoint, whatever its sign."""
    rows = {}
    for policy, subtitle, paths in cases:
        if policy != "Education":
            continue
        periods = EDUCATION_DISPLAY_PERIODS[subtitle]
        found = set()
        for path in paths.values():
            for key, info in load_json(path)["ate_results"].items():
                if info.get("ate_estimate") is None:
                    continue
                pattern = parse_x(key, periods)
                if sum(pattern):
                    found.add(pattern)
        rows[subtitle] = sorted(found, key=lambda x: (sum(x), -next(i for i, v in enumerate(x) if v)))
    return rows


def limits(values, pad_frac: float = 0.06) -> tuple[float, float]:
    """Shared x-axis range with padding, always including 0."""
    flat = [v for group in values for item in group if item is not None for v in item]
    lo, hi = min(flat + [0.0]), max(flat + [0.0])
    pad = pad_frac * (hi - lo) if hi > lo else 1.0
    return lo - pad, hi + pad


def draw_panel(fig, spec, title, display_periods, patterns, panel_data, xlim):
    """One panel: treated-month grid on the left, effects with intervals on the right."""
    frame = fig.add_subplot(spec)
    frame.set_axis_off()
    frame.set_title(title, fontsize=15, weight="bold", pad=18)

    sub = spec.subgridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.20)
    ax_t = fig.add_subplot(sub[0, 0])
    ax_a = fig.add_subplot(sub[0, 1], sharey=ax_t)

    matrix = np.array([list(p) for p in patterns], dtype=float)
    ax_t.imshow(matrix, aspect="auto", cmap=CELL_CMAP, interpolation="none", vmin=0, vmax=1)
    ax_t.set_xticks(np.arange(len(display_periods)))
    ax_t.set_xticklabels([p[5:] for p in display_periods], fontsize=11)
    ax_t.set_yticks(np.arange(len(patterns)))
    ax_t.set_yticklabels([row_label(p, display_periods) for p in patterns], fontsize=11)
    ax_t.set_xlabel("Month", fontsize=12)
    ax_t.set_ylabel("Policy path", fontsize=12)
    ax_t.set_xticks(np.arange(-0.5, len(display_periods), 1), minor=True)
    ax_t.set_yticks(np.arange(-0.5, len(patterns), 1), minor=True)
    ax_t.grid(which="minor", color="white", linewidth=1.4)
    ax_t.tick_params(which="minor", bottom=False, left=False)
    ax_t.tick_params(which="major", length=0)
    for side in ax_t.spines.values():
        side.set_visible(False)

    ax_a.axvline(0.0, color="#3A3A3A", lw=1.2, zorder=1)
    for name in SERIES:
        xs, los, his, ys = [], [], [], []
        for row_idx, item in enumerate(panel_data[name]):
            if item is None:
                continue
            est, lo, hi = item
            xs.append(est)
            los.append(lo)
            his.append(hi)
            ys.append(row_idx + OFFSETS[name])
        if not xs:
            continue
        ax_a.hlines(ys, los, his, color=COLORS[name], lw=2.6, zorder=2)
        ax_a.scatter(xs, ys, color=COLORS[name], s=46, zorder=3, edgecolors="none")

    ax_a.set_xlim(*xlim)
    ax_a.set_ylim(len(patterns) - 0.5, -0.5)
    ax_a.set_xlabel("Effect (cases per 10,000)", fontsize=12)
    ax_a.tick_params(axis="x", labelsize=11)
    # sharey also shares ax_t's minor ticks, so silence both levels here.
    ax_a.tick_params(axis="y", which="both", left=False, labelleft=False)
    ax_a.grid(axis="x", color="#DDDDDD", lw=0.9)
    ax_a.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax_a.spines[side].set_visible(False)
    ax_a.spines["bottom"].set_color("#BBBBBB")


def main() -> None:
    """Draw the four panels and save the figure."""
    args = parse_args()
    cases = build_cases(Path(args.results_root))

    orders = {}
    orders.update(business_rows(cases))
    orders.update(education_rows(cases))

    panels = []
    for (policy, subtitle, paths), (_p, _s, title, _f, _e) in zip(cases, PANELS):
        display_periods = (EDUCATION_DISPLAY_PERIODS[subtitle] if policy == "Education"
                           else BUSINESS_DISPLAY_PERIODS[subtitle])
        patterns = orders[subtitle]
        data = {name: series_points(load_json(path), policy, patterns, display_periods)
                for name, path in paths.items()}
        dropped = {d.split("::", 1)[1].strip() for d in args.drop if d.split("::", 1)[0].strip() == title}
        keep = [i for i in range(len(patterns)) if any(data[n][i] is not None for n in SERIES)
                and row_label(patterns[i], display_periods) not in dropped]
        patterns = [patterns[i] for i in keep]
        data = {n: [data[n][i] for i in keep] for n in SERIES}
        panels.append((policy, title, display_periods, patterns, data))

    # Education panels share an x range; Business panels get their own.
    edu_xlim = limits([data[name] for policy, _title, _periods, _patterns, data in panels
                       if policy == "Education" for name in SERIES])

    fig = plt.figure(figsize=(20, 14.8), dpi=args.dpi)
    outer = fig.add_gridspec(2, 2, wspace=0.20, hspace=0.30,
                             left=0.055, right=0.985, top=0.905, bottom=0.065)

    for spec, (policy, title, display_periods, patterns, data) in zip(outer, panels):
        xlim = edu_xlim if policy == "Education" else limits([data[n] for n in SERIES])
        draw_panel(fig, spec, title, display_periods, patterns, data, xlim)

    handles = [Line2D([0], [0], color=COLORS[n], marker="o", lw=2.6, markersize=9,
                      markeredgecolor="none", label=n) for n in SERIES]
    fig.legend(handles=handles, loc="upper center", ncol=len(SERIES), frameon=False,
               fontsize=14, bbox_to_anchor=(0.5, 0.985), columnspacing=2.6,
               handletextpad=0.6)

    if args.ci_note:
        fig.text(0.5, 0.018, args.ci_note, ha="center", fontsize=12, color="#444444")

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
