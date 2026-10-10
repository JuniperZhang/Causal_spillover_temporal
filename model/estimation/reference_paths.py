"""
Fixed reference spillover paths (low / mid / high) for the simulation study.

For each reference level, a grid of quantile levels is given; each decision
time's target d_tau is chosen among the population quantiles of the observed
spillover proportion D at that time. The chosen combination maximizes the
minimum, over the canonical own-treatment paths, of the oracle K-IPW
effective sample size (ESS) at (own path, d path).

The resulting d path d_{t;K} = (d_tau_1, ..., d_tau_K) can vary across
decision times and is shared by all own-treatment paths, so DE/TE contrasts
compare own paths at the same d path and SE contrasts vary the d path
within a fixed own path (Section 2). The returned paths are concrete D
values, not quantile labels.
"""
import itertools
from typing import Dict, Sequence, Tuple

import numpy as np

from .oracle import compute_oracle_own_propensity, compute_oracle_spillover_pmf
from .gaussian_kernel import spillover_numerator

DEFAULT_LEVEL_RANGES = {
    'low':     np.arange(0.05, 0.40, 0.05),
    'mid':     np.arange(0.35, 0.70, 0.05),
    'midhigh': np.arange(0.55, 0.90, 0.05),
    'high':    np.arange(0.60, 1.00, 0.05),
}

def find_reference_spillover_paths(
    dataset,
    config: Dict,
    canonical_paths: Dict[str, Tuple[int, ...]],
    decision_steps: Sequence[int],
    level_ranges: Dict[str, np.ndarray] = None,
    bandwidth: float = None,
    weight_truncation_percentile: float = 100.0,
) -> Dict[str, dict]:
    """
    Exhaustive search over per-decision-time quantile levels for each
    reference level.

    Args:
        dataset: NetworkTemporalCausalDataset instance
        config: DGP parameters, for the oracle propensity
        canonical_paths: name -> length-K own-treatment path (x_tau_1, ..., x_tau_K)
        decision_steps: 0-based processing-time indices of the K decision times
        level_ranges: level name -> candidate quantile levels; defaults to
            DEFAULT_LEVEL_RANGES
        bandwidth: kernel bandwidth h; defaults to dataset.spillover_bandwidth
        weight_truncation_percentile: percentile at which to clip the weights

    Returns:
        For each level name, a dict with
          'd_path': length-K list of D values, shared by all canonical paths
          'percentiles': quantile level chosen at each decision time (diagnostic)
          'ess_by_path': oracle K-IPW ESS of each canonical path at d_path
          'min_ess': minimum of 'ess_by_path', the maximized objective
    """
    if level_ranges is None:
        level_ranges = DEFAULT_LEVEL_RANGES
    if bandwidth is None:
        bandwidth = dataset.spillover_bandwidth

    K = len(decision_steps)
    own_propensity = compute_oracle_own_propensity(dataset, config)
    e_hat = own_propensity['e_hat']
    pmf_table = compute_oracle_spillover_pmf(dataset, e_hat, decision_steps)

    x_obs = dataset.x.numpy()[:, decision_steps, 0]
    d_obs_prop = dataset.d_xs.numpy()[:, decision_steps, 0]
    d_count_obs = dataset.d_xs_count.numpy()[:, decision_steps, 0]
    n = dataset.n_samples

    p_d_obs = np.zeros((n, K))
    for pos in range(K):
        for i in range(n):
            pmf = pmf_table[pos][i]
            d = int(round(d_count_obs[i, pos]))
            d = min(max(d, 0), len(pmf) - 1)
            p_d_obs[i, pos] = max(pmf[d], 1e-12)

    # Candidate D values: population percentiles at each decision time.
    all_pcts = sorted(set(round(p, 2) for r in level_ranges.values() for p in r))
    d_cand = {pos: {pct: float(np.quantile(d_obs_prop[:, pos], pct)) for pct in all_pcts}
              for pos in range(K)}

    step_weight = {name: {pos: {} for pos in range(K)} for name in canonical_paths}
    for name, xpath in canonical_paths.items():
        for pos in range(K):
            x_tau = xpath[pos]
            indicator = (x_obs[:, pos] == x_tau)
            e = e_hat[:, pos]
            own_prob = np.where(x_tau == 1, e, 1.0 - e)
            denom = np.clip(own_prob * p_d_obs[:, pos], 1e-12, None)
            for pct in all_pcts:
                numerator, _ = spillover_numerator(
                    d_obs_prop[:, pos], d_cand[pos][pct], bandwidth
                )
                step_weight[name][pos][pct] = np.where(indicator, numerator / denom, 0.0)

    def joint_ess(name, pct_tuple):
        """Kish ESS of the full-window oracle weights for one own path and quantile-level tuple."""
        w = np.ones(n)
        for pos, pct in enumerate(pct_tuple):
            w = w * step_weight[name][pos][pct]
        if weight_truncation_percentile < 100 and (w > 0).any():
            w_upper = np.percentile(w[w > 0], weight_truncation_percentile)
            w = np.clip(w, 0, w_upper)
        s = w.sum()
        ss = (w ** 2).sum()
        return (s ** 2 / ss) if ss > 0 else 0.0

    results = {}
    for level, rng in level_ranges.items():
        pcts = [round(p, 2) for p in rng]
        best_combo, best_min_ess, best_ess_by_path = None, -1.0, None
        for combo in itertools.product(pcts, repeat=K):
            ess_by_path = {name: joint_ess(name, combo) for name in canonical_paths}
            m = min(ess_by_path.values())
            if m > best_min_ess:
                best_min_ess = m
                best_combo = combo
                best_ess_by_path = ess_by_path
        d_path = [d_cand[pos][pct] for pos, pct in enumerate(best_combo)]
        results[level] = {
            'd_path': d_path,
            'percentiles': list(best_combo),
            'ess_by_path': best_ess_by_path,
            'min_ess': best_min_ess,
        }
    return results
