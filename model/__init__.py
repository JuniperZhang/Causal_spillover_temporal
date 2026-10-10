"""
Neural spatiotemporal spillover estimation.

Subpackages:
    data:          simulation DGP, dataset and Monte Carlo ground truth
    models:        LSTM encoder, decision-time latent map g_Z and assignment models f_X, f_D
    training:      joint training of the encoder and assignment models
    estimation:    kernel IPW, oracle and OLS estimators, bootstrap, sensitivity analysis
    pipeline:      simulation and sensitivity runners (run with `python -m`)
    visualization: paper figures (run with `python -m`)
"""

from .config import CONFIG, device
from .data import NetworkTemporalCausalDataset, compute_ground_truth_monte_carlo_arbitrary_T
from .models import TemporalCausalModelSpillover
from .training import train_model_spillover_distributional
from .estimation import (
    compute_propensity_scores_lstm,
    estimate_ate_ipw_gaussian_kernel,
    estimate_ate_oracle_kipw,
    fit_predict_ols_baseline,
)

__version__ = "1.0.0"

__all__ = [
    'CONFIG',
    'device',
    'NetworkTemporalCausalDataset',
    'compute_ground_truth_monte_carlo_arbitrary_T',
    'TemporalCausalModelSpillover',
    'train_model_spillover_distributional',
    'compute_propensity_scores_lstm',
    'estimate_ate_ipw_gaussian_kernel',
    'estimate_ate_oracle_kipw',
    'fit_predict_ols_baseline',
]
