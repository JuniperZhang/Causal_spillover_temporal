"""Disjoint network-block bootstrap for path-level means and causal contrasts.

The observed graph and graph-derived histories are held fixed. The graph is
partitioned into disjoint neighborhood-like clusters by randomized k-hop-max
clustering; clusters are sampled with replacement, and every Hajek mean is
recomputed with the same cluster multipliers. Sharing multipliers across all
targets preserves the covariance between the two means in a contrast.

The bootstrap conditions on the fitted propensity model, consistent with the
condition in Section 4 that propensity estimation error is asymptotically
negligible. It does not capture first-stage training uncertainty (see
retrain_bootstrap for a version that refits the model on each draw).
"""
from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.sparse import issparse


def build_neighbor_lists(adj_matrix):
    """Return first-order neighbor indices for every unit."""
    n = adj_matrix.shape[0]
    if issparse(adj_matrix):
        csr = adj_matrix.tocsr()
        return [
            csr.indices[csr.indptr[i]:csr.indptr[i + 1]].copy()
            for i in range(n)
        ]
    return [np.flatnonzero(np.asarray(adj_matrix[i]) > 0) for i in range(n)]


def k_hop_max_partition(
    adj_matrix,
    k_hops: int = 2,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Randomized k-hop-max partition of the graph into disjoint clusters.

    Each unit draws a Uniform(0,1) score and joins the cluster of the
    highest-scoring unit within its k-hop neighborhood (itself included).

    Returns:
        (n,) int64 cluster labels in 0..C-1.
    """
    if k_hops < 0:
        raise ValueError("k_hops must be nonnegative")
    if rng is None:
        rng = np.random.default_rng()

    neighbor_lists = build_neighbor_lists(adj_matrix)
    n = len(neighbor_lists)
    scores = rng.random(n)
    centers = np.empty(n, dtype=np.int64)

    for i in range(n):
        visited = {i}
        frontier = {i}
        for _ in range(k_hops):
            next_frontier = set()
            for u in frontier:
                next_frontier.update(int(v) for v in neighbor_lists[u])
            next_frontier.difference_update(visited)
            if not next_frontier:
                break
            visited.update(next_frontier)
            frontier = next_frontier
        candidates = np.fromiter(visited, dtype=np.int64)
        centers[i] = candidates[np.argmax(scores[candidates])]

    _, labels = np.unique(centers, return_inverse=True)
    return labels.astype(np.int64)


def _summarize_replicates(
    values: Sequence[float],
    n_requested: int,
    ci_level: float,
    point: Optional[float],
) -> Dict:
    """Percentile CI and standard error from bootstrap draws, ignoring non-finite draws.

    'reliable' is True when at least 90% of the requested draws are finite
    and there are at least 100 of them.
    """
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    valid_fraction = float(len(vals) / max(n_requested, 1))
    if len(vals) == 0:
        return {
            "point": point,
            "ci_lower": None,
            "ci_upper": None,
            "se": None,
            "n_valid": 0,
            "n_requested": int(n_requested),
            "valid_fraction": valid_fraction,
            "reliable": False,
        }
    alpha = 1.0 - ci_level
    lo, hi = np.quantile(vals, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {
        "point": point,
        "ci_lower": float(lo),
        "ci_upper": float(hi),
        "se": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
        "n_valid": int(len(vals)),
        "n_requested": int(n_requested),
        "valid_fraction": valid_fraction,
        "reliable": bool(valid_fraction >= 0.90 and len(vals) >= 100),
    }


def joint_network_block_bootstrap(
    target_inputs: Mapping[str, Tuple[np.ndarray, np.ndarray]],
    contrast_pairs: Optional[Mapping[str, Tuple[str, str]]],
    adj_matrix,
    n_boot: int = 1000,
    ci_level: float = 0.95,
    k_hops: int = 2,
    n_partitions: int = 3,
    partitions: Optional[Sequence[np.ndarray]] = None,
    rng: Optional[np.random.Generator] = None,
    return_draws: bool = False,
) -> Dict[str, Dict]:
    """Joint percentile CIs for weighted means and their contrasts.

    target_inputs maps a target name to (weights, values), each of length n;
    the point estimate is sum(weights * values) / sum(weights). OLS plug-in
    means use unit weights and target-specific predictions. contrast_pairs
    maps a contrast name to (target_a, target_b), estimated as a - b.

    In draw b, cluster c is drawn m_c times and every unit in c gets
    multiplier m_c; the draw's mean is sum(m * w * y) / sum(m * w). Draws
    cycle over n_partitions random partitions (or the supplied `partitions`).

    Returns:
        {"summaries": {name: _summarize_replicates output},
         "metadata": {...}}, plus "draws" (name -> list of n_boot values)
        when return_draws=True.
    """
    if n_boot <= 0:
        return {}
    if n_partitions <= 0 and partitions is None:
        raise ValueError("n_partitions must be positive")
    if rng is None:
        rng = np.random.default_rng()
    contrast_pairs = dict(contrast_pairs or {})

    n = adj_matrix.shape[0]
    clean_inputs = {}
    points = {}
    for name, pair in target_inputs.items():
        weights = np.asarray(pair[0], dtype=np.float64).reshape(-1)
        values = np.asarray(pair[1], dtype=np.float64).reshape(-1)
        if len(weights) != n or len(values) != n:
            raise ValueError(f"{name}: weights/values must both have length {n}")
        clean_inputs[name] = (weights, values)
        denom = weights.sum()
        points[name] = float(np.dot(weights, values) / denom) if denom > 1e-12 else None

    for cname, (a, b) in contrast_pairs.items():
        if a not in clean_inputs or b not in clean_inputs:
            raise KeyError(f"{cname}: unknown contrast target(s) {a}, {b}")
        pa, pb = points[a], points[b]
        points[cname] = (pa - pb) if pa is not None and pb is not None else None

    if partitions is None:
        partitions = [
            k_hop_max_partition(adj_matrix, k_hops=k_hops, rng=rng)
            for _ in range(n_partitions)
        ]
    else:
        partitions = [np.asarray(labels, dtype=np.int64) for labels in partitions]
        if not partitions or any(len(labels) != n for labels in partitions):
            raise ValueError("Every supplied partition must have one label per unit")
        n_partitions = len(partitions)
    draws = {name: [] for name in list(clean_inputs) + list(contrast_pairs)}
    cluster_counts = []

    for b_idx in range(n_boot):
        labels = partitions[b_idx % n_partitions]
        n_clusters = int(labels.max()) + 1 if len(labels) else 0
        cluster_counts.append(n_clusters)
        if n_clusters == 0:
            continue
        sampled = rng.integers(0, n_clusters, size=n_clusters)
        multiplicity_by_cluster = np.bincount(sampled, minlength=n_clusters)
        unit_multiplier = multiplicity_by_cluster[labels].astype(np.float64)

        level_estimates = {}
        for name, (weights, values) in clean_inputs.items():
            w_b = unit_multiplier * weights
            denom = w_b.sum()
            if denom <= 1e-12:
                level_estimates[name] = np.nan
            else:
                level_estimates[name] = float(np.dot(w_b, values) / denom)
            draws[name].append(level_estimates[name])

        for cname, (a, b) in contrast_pairs.items():
            ea, eb = level_estimates[a], level_estimates[b]
            draws[cname].append(ea - eb if np.isfinite(ea) and np.isfinite(eb) else np.nan)

    summaries = {
        name: _summarize_replicates(vals, n_boot, ci_level, points[name])
        for name, vals in draws.items()
    }
    metadata = {
        "method": "disjoint_k_hop_max_cluster_bootstrap",
        "n_boot": int(n_boot),
        "n_partitions": int(n_partitions),
        "k_hops": int(k_hops),
        "mean_n_clusters": float(np.mean(cluster_counts)) if cluster_counts else 0.0,
        "conditions_on_fitted_nuisance": True,
    }
    result = {"summaries": summaries, "metadata": metadata}
    if return_draws:
        result["draws"] = draws
    return result
