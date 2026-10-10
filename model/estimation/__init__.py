"""
Causal estimation (paper Sections 3.4 and 4, Supplementary Material).

Modules:
    ipw:               fitted propensities from the trained LSTM model
    gaussian_kernel:   self-normalized kernel IPW (K-IPW) estimator (Section 3.4)
    oracle:            K-IPW with the true DGP propensity
    ols_baseline:      outcome-regression baseline
    reference_paths:   fixed low/mid/high reference spillover paths
    bootstrap:         disjoint network-block bootstrap
    retrain_bootstrap: network-block bootstrap that refits the LSTM on every draw
    sensitivity:       sensitivity bounds for unmeasured confounding (Section 4)
"""

from .ipw import compute_propensity_scores_lstm
from .gaussian_kernel import (
    zoib_binomial_pmf,
    gaussian_kernel_weight,
    estimate_ate_ipw_gaussian_kernel,
)
from .oracle import (
    compute_oracle_own_propensity,
    poisson_binomial_pmf,
    compute_oracle_spillover_pmf,
    estimate_ate_oracle_kipw,
)
from .ols_baseline import fit_predict_ols_baseline
from .bootstrap import (
    build_neighbor_lists,
    k_hop_max_partition,
    joint_network_block_bootstrap,
)
from .retrain_bootstrap import (
    build_cluster_resample,
    retrain_bootstrap_ipw,
)

__all__ = [
    'compute_propensity_scores_lstm',
    'zoib_binomial_pmf',
    'gaussian_kernel_weight',
    'estimate_ate_ipw_gaussian_kernel',
    'compute_oracle_own_propensity',
    'poisson_binomial_pmf',
    'compute_oracle_spillover_pmf',
    'estimate_ate_oracle_kipw',
    'fit_predict_ols_baseline',
    'build_neighbor_lists',
    'k_hop_max_partition',
    'joint_network_block_bootstrap',
    'build_cluster_resample',
    'retrain_bootstrap_ipw',
]
