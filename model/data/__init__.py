"""Simulation data-generating process, dataset and Monte Carlo ground truth."""

from .generation import (
    generate_erdos_renyi_network,
    normalize_adjacency_matrix,
    generate_baseline_covariates_with_smoothing,
    assign_baseline_families,
    get_family_multipliers,
    compute_baseline_risk_score,
    generate_treatment_logistic,
    generate_spillover_treatment_proportion,
    evolve_covariates,
    generate_baseline_outcome,
    generate_latent_disease_activity,
    generate_outcome,
    FAMILY_LABELS,
)
from .dataset import NetworkTemporalCausalDataset
from .ground_truth import compute_ground_truth_monte_carlo_arbitrary_T

__all__ = [
    'generate_erdos_renyi_network',
    'normalize_adjacency_matrix',
    'generate_baseline_covariates_with_smoothing',
    'assign_baseline_families',
    'get_family_multipliers',
    'compute_baseline_risk_score',
    'generate_treatment_logistic',
    'generate_spillover_treatment_proportion',
    'evolve_covariates',
    'generate_baseline_outcome',
    'generate_latent_disease_activity',
    'generate_outcome',
    'FAMILY_LABELS',
    'NetworkTemporalCausalDataset',
    'compute_ground_truth_monte_carlo_arbitrary_T',
]
