"""
Outcome-regression (OLS) benchmark for the simulation study.

Fits one linear model of the final outcome Y_T on the observed decision-time
own-treatment and spillover paths, the baseline covariates V_0 and an
intercept, then predicts E[Y_T(x, d)] by plugging in the target
(x_tau, d_tau) path and averaging over the units' baseline covariates. It
does not adjust for time-varying confounding after baseline.
"""
import numpy as np
from typing import Dict, List, Tuple

from .gaussian_kernel import _extract_target_path


def fit_predict_ols_baseline(
    dataset,
    treatment_sequences: List[Tuple],
    decision_steps: List[int] = None,
    return_predictions: bool = False,
) -> Dict:
    """
    Fit the OLS benchmark and return its plug-in mean for each target path.

    Args:
        dataset: NetworkTemporalCausalDataset instance
        treatment_sequences: length-2T (x1,d1,...,xT,dT) target tuples, as in
            estimate_ate_ipw_gaussian_kernel
        decision_steps: 0-based processing-time indices of the K decision
            times; inferred from dataset.config if None
        return_predictions: also return per-unit predictions and unit weights

    Returns:
        {seq: {'estimate': float}}; with return_predictions also
        'predictions' (n,) and 'weights' (n,) of ones.
    """
    if decision_steps is None:
        update_interval = dataset.config['treatment_update_interval']
        decision_steps = [t - 1 for t in range(1, dataset.T + 1, update_interval)]

    X_obs = dataset.x.numpy()[:, decision_steps, 0]        # (n, K)
    D_obs = dataset.d_xs.numpy()[:, decision_steps, 0]     # (n, K)
    V0 = dataset.v[:, 0, :].numpy()                        # (n, v_dim)
    Y_T = dataset.y.numpy()[:, -1, 0]                      # (n,)
    n = len(Y_T)

    design = np.concatenate(
        [X_obs, D_obs, V0, np.ones((n, 1))], axis=1
    ).astype(np.float64)
    beta, *_ = np.linalg.lstsq(design, Y_T.astype(np.float64), rcond=None)

    results = {}
    for seq in treatment_sequences:
        x_t, d_t = _extract_target_path(seq, decision_steps)
        design_target = np.concatenate(
            [np.tile(x_t, (n, 1)), np.tile(d_t, (n, 1)), V0, np.ones((n, 1))], axis=1
        )
        preds = design_target @ beta
        results[seq] = {'estimate': float(preds.mean())}
        if return_predictions:
            results[seq]['predictions'] = preds
            results[seq]['weights'] = np.ones(n)

    return results
