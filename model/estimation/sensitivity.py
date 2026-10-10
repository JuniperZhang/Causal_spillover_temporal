"""Sensitivity bounds for the Hajek (self-normalized) kernel IPW estimator (Section 4).

Each decision-time joint assignment density ratio between the true and
fitted models lies in [1/Gamma, Gamma]; over M decisions the per-unit weight
multiplier r_i lies in [Gamma**-M, Gamma**M]. For fixed fitted weights w_i
the bounds are the minimum and maximum of sum_i r_i w_i Y_i / sum_i r_i w_i
over such r. These are hidden-bias ranges, not confidence intervals.

With outcomes sorted, the minimum puts the upper multiplier on outcomes below
a threshold and the lower multiplier above it (reverse for the maximum).
Enumerating all n+1 thresholds solves the same linear-fractional program as
the Charnes--Cooper LP in the Supplementary Material.
"""
from __future__ import annotations

import numpy as np


DEFAULT_GAMMAS = (1., 1.01, 1.025, 1.05, 1.1, 1.2, 1.3, 1.5, 2., 3., 5.)


def validate_gammas(gammas):
    """Return the Gamma grid as an array; it must be finite, strictly increasing and start at 1."""
    grid = np.asarray(gammas, dtype=float)
    if (grid.ndim != 1 or not len(grid) or not np.isfinite(grid).all()
            or grid[0] != 1 or np.any(np.diff(grid) <= 0)):
        raise ValueError("Gamma grid must be finite, strictly increasing and start at 1")
    return grid


def hajek_sensitivity_bounds(weights, outcomes, gammas=DEFAULT_GAMMAS, M=4):
    """Lower and upper Hajek bounds for each Gamma.

    Args:
        weights: (n,) nonnegative fitted IPW weights w_i (kernel times inverse
            assignment density).
        outcomes: (n,) outcomes Y_i; entries with zero weight may be nonfinite.
        gammas: Gamma grid starting at 1.
        M: number of decision times.

    Returns:
        (len(gammas), 2) array of [lower, upper]; all NaN if no weight is positive.
        Zero-weight observations are dropped. The bounds are invariant to the
        scale of the weights.
    """
    grid = validate_gammas(gammas)
    if isinstance(M, bool) or int(M) != M or M < 1:
        raise ValueError("M must be a positive integer")
    w, y = np.asarray(weights, dtype=float), np.asarray(outcomes, dtype=float)
    if w.ndim != 1 or y.shape != w.shape:
        raise ValueError("weights and outcomes must be matching one-dimensional arrays")
    if not np.isfinite(w).all() or np.any(w < 0):
        raise ValueError("weights must be finite and nonnegative")
    keep = w > 0
    if not np.any(keep):
        return np.full((len(grid), 2), np.nan)
    w, y = w[keep], y[keep]
    if not np.isfinite(y).all():
        raise ValueError("positive-weight outcomes must be finite")
    order = np.argsort(y)
    w, y = w[order] / w.max(), y[order]
    # Prefix sums for each threshold; computing suffix sums separately (not
    # as total minus prefix) avoids cancellation in small tail weights.
    left_w = np.r_[0., np.cumsum(w)]
    right_w = np.r_[np.cumsum(w[::-1])[::-1], 0.]
    center = y[0]
    wy = w * (y - center)
    left_y = np.r_[0., np.cumsum(wy)]
    right_y = np.r_[np.cumsum(wy[::-1])[::-1], 0.]
    bounds = []
    for gamma in grid:
        # Multipliers Gamma**M and Gamma**-M rescaled to 1 and Gamma**(-2M).
        ratio = np.exp(-2 * int(M) * np.log(gamma))
        if ratio == 0:
            raise ValueError("Gamma**(2*M) exceeds numerical range")
        lower_candidates = (left_y + ratio * right_y) / (left_w + ratio * right_w)
        upper_candidates = (ratio * left_y + right_y) / (ratio * left_w + right_w)
        bounds.append((center + lower_candidates.min(), center + upper_candidates.max()))
    return np.asarray(bounds)


def sensitivity_analysis(inputs, contrasts, gammas=DEFAULT_GAMMAS, M=4):
    """Bounds for named level targets and for contrasts between them.

    Args:
        inputs: {target: (weights, outcomes)} for each target exposure path.
        contrasts: {name: (a, b)}; the contrast a - b gets the conservative
            interval [lower_a - upper_b, upper_a - lower_b].
        gammas, M: as in hajek_sensitivity_bounds.

    Returns:
        JSON-ready dict with, per target, lower/upper endpoints per Gamma
        (None where unsupported), whether each interval contains zero, and
        gamma_star, the smallest grid Gamma whose interval contains zero.
    """
    grid = validate_gammas(gammas)
    endpoints = {name: hajek_sensitivity_bounds(w, y, grid, M)
                 for name, (w, y) in inputs.items()}
    for name, (a, b) in contrasts.items():
        if a in endpoints and b in endpoints:
            endpoints[name] = np.column_stack((endpoints[a][:, 0] - endpoints[b][:, 1],
                                               endpoints[a][:, 1] - endpoints[b][:, 0]))
        else:
            endpoints[name] = np.full((len(grid), 2), np.nan)
    output = {}
    for name, bounds in endpoints.items():
        valid = np.isfinite(bounds).all(axis=1)
        includes = valid & (bounds[:, 0] <= 0) & (bounds[:, 1] >= 0)
        indices = np.flatnonzero(includes)
        output[name] = {
            "lower": [float(x) if np.isfinite(x) else None for x in bounds[:, 0]],
            "upper": [float(x) if np.isfinite(x) else None for x in bounds[:, 1]],
            "contains_zero": [bool(x) if v else None for x, v in zip(includes, valid)],
            "gamma_star": float(grid[indices[0]]) if len(indices) else None,
            "gamma_star_status": ("observed_on_grid" if len(indices) else
                                  "above_grid" if valid.all() else "unsupported"),
        }
    return {"gammas": grid.tolist(), "M": int(M), "targets": output,
            "interval_type": "conditional_hidden_bias_bounds_not_confidence_intervals"}


def summarize_sensitivity(replicates, truth):
    """Average sensitivity results over simulation replications.

    For each method and target, reports per Gamma the mean endpoints and
    width, the rates of zero inclusion and sign retention, Monte Carlo
    standard errors of the endpoints, and, when the true value is known,
    the rate at which the interval contains it. That rate is descriptive,
    not confidence-interval coverage.
    """
    result = {}
    for method in ("lstm_kipw", "oracle_kipw"):
        runs = [r["sensitivity"][method] for r in replicates if r.get("sensitivity")]
        if not runs:
            continue
        grid = runs[0]["gammas"]
        if any(r["gammas"] != grid or r["M"] != runs[0]["M"] for r in runs):
            raise ValueError("Cannot aggregate different sensitivity designs")
        targets = sorted(set().union(*(r["targets"] for r in runs)))
        summaries = {}
        for name in targets:
            rows = []
            for k, gamma in enumerate(grid):
                pairs = [(r["targets"][name]["lower"][k], r["targets"][name]["upper"][k])
                         for r in runs if name in r["targets"]]
                valid = np.asarray([(lo, hi) for lo, hi in pairs
                                    if lo is not None and hi is not None], dtype=float)
                row = {"gamma": gamma, "n_valid": len(valid), "n_total": len(runs)}
                if len(valid):
                    lo, hi = valid.T
                    row.update(mean_lower=float(lo.mean()), mean_upper=float(hi.mean()),
                               mean_width=float((hi-lo).mean()),
                               zero_inclusion_rate=float(np.mean((lo <= 0) & (hi >= 0))),
                               negative_sign_retention=float(np.mean(hi < 0)),
                               positive_sign_retention=float(np.mean(lo > 0)),
                               lower_mcse=float(lo.std(ddof=1)/np.sqrt(len(lo))) if len(lo)>1 else None,
                               upper_mcse=float(hi.std(ddof=1)/np.sqrt(len(hi))) if len(hi)>1 else None)
                    if name in truth:
                        row["truth_inclusion_rate"] = float(np.mean((lo <= truth[name]) & (hi >= truth[name])))
                rows.append(row)
            summaries[name] = rows
        result[method] = summaries
    return result
