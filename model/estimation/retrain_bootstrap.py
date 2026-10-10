"""Retrain bootstrap: refit the LSTM propensity model on every network-block
resample (an alternative described in the Supplementary Material).

Unlike bootstrap.py, which reweights the outputs of one fitted model, each
draw retrains the model, so the CI also reflects propensity re-estimation
uncertainty. Each draw costs a full LSTM fit. Enabled with
--bootstrap-retrain in pipeline/bias_mse_study.py.

Resampling unit: the k-hop-max clusters of bootstrap.k_hop_max_partition.
Clusters are drawn with replacement; each drawn cluster contributes a copy of
its member units with their observed histories unchanged (the DGP is not
re-simulated) and only its within-cluster edges, so copies are mutually
disconnected and cross-cluster edges are dropped.
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.sparse import issparse, block_diag, csr_matrix

from .bootstrap import (
    k_hop_max_partition,
    joint_network_block_bootstrap,
    _summarize_replicates,
)


def build_cluster_resample(
    adj_matrix,
    labels: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, "csr_matrix"]:
    """Resampled unit indices and block-diagonal adjacency for one draw.

    Draws C clusters with replacement from the C clusters in `labels`.

    Returns:
        unit_indices: (n_boot,) indices into the source dataset, repeated
            for clusters drawn more than once; n_boot varies across draws.
        adj_boot: (n_boot, n_boot) CSR adjacency, block-diagonal by cluster
            copy.
    """
    n_clusters = int(labels.max()) + 1 if len(labels) else 0
    if n_clusters == 0:
        return np.array([], dtype=np.int64), csr_matrix((0, 0))

    cluster_members = [np.flatnonzero(labels == c) for c in range(n_clusters)]
    sampled = rng.integers(0, n_clusters, size=n_clusters)

    unit_blocks: List[np.ndarray] = []
    adj_blocks = []
    for c in sampled:
        members = cluster_members[c]
        if len(members) == 0:
            continue
        unit_blocks.append(members)
        sub = adj_matrix[members][:, members]
        adj_blocks.append(sub.tocsr() if issparse(sub) else csr_matrix(sub))

    if not unit_blocks:
        return np.array([], dtype=np.int64), csr_matrix((0, 0))

    unit_indices = np.concatenate(unit_blocks)
    adj_boot = block_diag(adj_blocks, format="csr")
    return unit_indices, adj_boot


def retrain_bootstrap_ipw(
    analysis,
    validation,
    config: Mapping,
    device,
    sequence_list: Sequence[Tuple],
    sequence_names: Mapping[str, Tuple],
    contrast_pairs: Mapping[str, Tuple[str, str]],
    bandwidth: float,
    weight_truncation_percentile: float,
    n_boot: int,
    k_hops: int,
    n_partitions: int,
    ci_level: float,
    seed: int,
    point_estimates: Mapping[str, Optional[float]],
    n_reweight_per_retrain: int = 1,
    log_fn=None,
) -> Dict:
    """Retrain-bootstrap percentile CIs for the LSTM K-IPW targets.

    For each of `n_boot` draws, resamples the training network (`analysis`)
    and, independently with its own clusters, the early-stopping validation
    network (`validation`); trains a new LSTM; and recomputes the K-IPW
    Hajek estimates. Resampling the validation network as well lets
    early-stopping variability enter the CI.

    With n_reweight_per_retrain = 1, each retrained model contributes one
    draw per target. With n_reweight_per_retrain = r > 1, each retrained
    model instead contributes r draws of joint_network_block_bootstrap on its
    own resampled network and K-IPW weights, giving n_boot * r draws in
    total (a nested bootstrap).

    Returns the same {"summaries", "metadata"} structure as
    bootstrap.joint_network_block_bootstrap.
    """
    from ..training.train import train_model_spillover_distributional
    from .gaussian_kernel import estimate_ate_ipw_gaussian_kernel
    from ..data.dataset import NetworkTemporalCausalDataset

    if n_boot <= 0:
        return {}

    rng = np.random.default_rng(seed)
    partitions = [
        k_hop_max_partition(analysis.adj_matrix, k_hops=k_hops, rng=rng)
        for _ in range(max(n_partitions, 1))
    ]
    has_validation = validation is not None
    val_partitions = (
        [
            k_hop_max_partition(validation.adj_matrix, k_hops=k_hops, rng=rng)
            for _ in range(max(n_partitions, 1))
        ]
        if has_validation
        else []
    )

    n_reweight_per_retrain = max(int(n_reweight_per_retrain), 1)
    n_boot_total = n_boot * n_reweight_per_retrain

    draws: Dict[str, List[float]] = {
        name: [] for name in list(sequence_names) + list(contrast_pairs)
    }
    n_clusters_draw: List[int] = []

    for b_idx in range(n_boot):
        if log_fn is not None:
            log_fn(f"  retrain bootstrap draw {b_idx + 1}/{n_boot}")
        labels = partitions[b_idx % len(partitions)]
        unit_indices, adj_boot = build_cluster_resample(analysis.adj_matrix, labels, rng)
        n_clusters_draw.append(int(labels.max()) + 1 if len(labels) else 0)

        if has_validation:
            val_labels = val_partitions[b_idx % len(val_partitions)]
            val_unit_indices, val_adj_boot = build_cluster_resample(
                validation.adj_matrix, val_labels, rng
            )
            val_ok = len(val_unit_indices) > 0
        else:
            val_ok = True

        if len(unit_indices) == 0 or not val_ok:
            for name in draws:
                draws[name].extend([np.nan] * n_reweight_per_retrain)
            continue

        boot_dataset = NetworkTemporalCausalDataset.from_unit_resample(
            analysis, unit_indices, adj_boot, seed=seed + 1_000_000 + b_idx,
        )
        boot_dataset.spillover_bandwidth = bandwidth
        if has_validation:
            boot_validation = NetworkTemporalCausalDataset.from_unit_resample(
                validation, val_unit_indices, val_adj_boot, seed=seed + 2_000_000 + b_idx,
            )
            boot_validation.spillover_bandwidth = bandwidth
        else:
            boot_validation = None

        # Per-draw seed for training randomness (initialization, dropout,
        # minibatch order), so each draw is reproducible on its own.
        torch.manual_seed(seed + 900_000 + b_idx)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + 900_000 + b_idx)
        boot_model = train_model_spillover_distributional(
            boot_dataset, boot_validation, dict(config), device,
        )
        boot_result = estimate_ate_ipw_gaussian_kernel(
            boot_model,
            boot_dataset,
            list(sequence_list),
            device,
            bandwidth=bandwidth,
            weight_truncation_percentile=weight_truncation_percentile,
            return_weights=(n_reweight_per_retrain > 1),
        )

        if n_reweight_per_retrain == 1:
            level_estimates = {}
            for name, seq in sequence_names.items():
                entry = boot_result["estimates"][tuple(seq)]
                level_estimates[name] = entry.get("estimate")
                draws[name].append(
                    level_estimates[name] if level_estimates[name] is not None else np.nan
                )
            for cname, (a, b) in contrast_pairs.items():
                ea, eb = level_estimates.get(a), level_estimates.get(b)
                draws[cname].append(
                    ea - eb if ea is not None and eb is not None else np.nan
                )
            continue

        # n_reweight_per_retrain > 1: network-block reweighting of this
        # model's per-unit K-IPW weights on this draw's resampled network.
        reweight_inputs = {}
        for name, seq in sequence_names.items():
            entry = boot_result["estimates"][tuple(seq)]
            if entry.get("weights") is not None:
                reweight_inputs[name] = (entry["weights"], entry["y"])
        reweight_pairs = {
            cname: (a, b)
            for cname, (a, b) in contrast_pairs.items()
            if a in reweight_inputs and b in reweight_inputs
        }
        reweighted = joint_network_block_bootstrap(
            reweight_inputs,
            reweight_pairs,
            boot_dataset.adj_matrix,
            n_boot=n_reweight_per_retrain,
            ci_level=ci_level,
            k_hops=k_hops,
            n_partitions=n_partitions,
            rng=rng,
            return_draws=True,
        )
        sub_draws = reweighted.get("draws", {})
        for name in draws:
            vals = sub_draws.get(name)
            if vals is None:
                draws[name].extend([np.nan] * n_reweight_per_retrain)
            else:
                draws[name].extend(vals)

    summaries = {
        name: _summarize_replicates(vals, n_boot_total, ci_level, point_estimates.get(name))
        for name, vals in draws.items()
    }
    metadata = {
        "method": "retrain_cluster_bootstrap",
        "n_boot": int(n_boot_total),
        "n_retrain": int(n_boot),
        "n_reweight_per_retrain": int(n_reweight_per_retrain),
        "n_partitions": int(n_partitions),
        "k_hops": int(k_hops),
        "mean_n_clusters": float(np.mean(n_clusters_draw)) if n_clusters_draw else 0.0,
        "conditions_on_fitted_nuisance": False,
    }
    return {"summaries": summaries, "metadata": metadata}
