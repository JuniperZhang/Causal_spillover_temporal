"""
Oracle K-IPW estimator: the Section 3.4 estimator with the true DGP
propensity (config parameters, no model fitting) in place of the fitted LSTM
propensity.

Own-treatment propensity: e_i,tau = E[sigmoid(eta_i,tau + eps^X)],
eps^X ~ N(0, sigma_X^2), with eta_i,tau the DGP linear predictor in
theta_0/theta_R/theta_prev/theta_w/theta_i/theta_v and the unit's observed
history (R_i^0, U_prev, X_prev, W, I), as in
generation.generate_treatment_logistic. The expectation over the logit noise
is taken by Gauss-Hermite quadrature; sigmoid(eta) alone would be biased.

Spillover propensity: D_i = sum_j A_ij X_j with the neighbors' X_j
independent Bernoulli(e_j) given the history, so D_i | H is Poisson-Binomial
with success probabilities (e_j)_{j in N(i)}.

Functions:
    compute_oracle_own_propensity: e_hat_i,tau via the true DGP theta's
    poisson_binomial_pmf: exact PMF of a sum of independent non-identical Bernoullis
    compute_oracle_spillover_pmf: per-unit, per-decision-time D_i PMF
    estimate_ate_oracle_kipw: K-IPW/Hajek estimator using the oracle propensity
"""
import numpy as np
from scipy.sparse import issparse
from typing import Dict, List, Tuple, Optional

from ..data.generation import get_burden_weights, compute_burden
from .gaussian_kernel import spillover_numerator

_GH_DEGREE = 30
_GH_NODES, _GH_WEIGHTS = np.polynomial.hermite.hermgauss(_GH_DEGREE)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Logistic function."""
    return 1.0 / (1.0 + np.exp(-z))


def _expected_sigmoid_gaussian(eta: np.ndarray, sigma: float) -> np.ndarray:
    """
    E[sigmoid(eta + eps)], eps ~ N(0, sigma^2), via Gauss-Hermite quadrature:
        E[f(Z)] = (1/sqrt(pi)) sum_k w_k f(eta + sqrt(2) sigma x_k)
    """
    if sigma <= 0:
        return _sigmoid(eta)
    z = eta[:, None] + np.sqrt(2.0) * sigma * _GH_NODES[None, :]
    vals = _sigmoid(z)
    return (vals * _GH_WEIGHTS[None, :]).sum(axis=1) / np.sqrt(np.pi)


def compute_oracle_own_propensity(dataset, config: Dict) -> Dict:
    """
    True own-treatment propensity P(X_i,tau = 1 | H_i,tau) for every unit and
    decision time, computed from the DGP parameters in `config` as in
    generation.generate_treatment_logistic and marginalized over the logit
    noise.

    Returns
    -------
    Dict with:
        'e_hat'          : (n_samples, K) E[X_i,tau=1 | H_i,tau]
        'decision_steps' : list[K] 0-based processing-time indices
    """
    T = dataset.T
    update_interval = int(config.get('treatment_update_interval', 1))
    decision_times = set(range(1, T + 1, update_interval))
    decision_steps = sorted(t - 1 for t in decision_times)
    K = len(decision_steps)
    n = dataset.n_samples

    theta_0 = config.get('treatment_theta_0', -3.85)
    theta_r = config.get('treatment_theta_R', 1.00)
    theta_p = config.get('treatment_theta_prev', 0.28)
    theta_w = config.get('treatment_theta_w', 0.19)
    theta_i = config.get('treatment_theta_i', 0.08)
    theta_v = config.get('treatment_theta_v', 0.06)
    sigma_x = config.get('treatment_sigma_X', 0.45)
    variant = config.get('treatment_nonlinear_variant', 'baseline')
    theta_w2 = config.get('treatment_theta_w2', 0.05)

    weights = get_burden_weights(config, dataset.v_dim)
    V = dataset.v.numpy()             # (n, T, v_dim)
    Y = dataset.y.numpy()[:, :, 0]    # (n, T)
    Y0 = dataset.y_0_baseline         # (n,)
    R0 = dataset.baseline_risk_score  # (n,)  standardized R_i^0, fixed at baseline
    X = dataset.x.numpy()[:, :, 0]    # (n, T)

    e_hat = np.zeros((n, K))
    x_prev_running = np.zeros(n)

    for pos, j in enumerate(decision_steps):
        # V[:, j, :] holds V_{t-1} for the decision at processing time t=j+1.
        U_prev = compute_burden(V[:, j, :], weights)

        if j == 0:
            y_prev, y_prev2 = Y0, None
        else:
            y_prev = Y[:, j - 1]
            y_prev2 = Y[:, j - 2] if j >= 2 else None

        if y_prev2 is not None:
            delta = y_prev - y_prev2
            W = np.maximum(delta, 0.0)
            I = np.maximum(-delta, 0.0)
        else:
            W = np.zeros(n)
            I = np.zeros(n)

        eta = (theta_0 + theta_r * R0 + theta_p * x_prev_running
               + theta_w * W + theta_i * I + theta_v * U_prev)
        if variant == 'worsening_quadratic':
            eta = eta + theta_w2 * W ** 2
        e_hat[:, pos] = _expected_sigmoid_gaussian(eta, sigma_x)

        x_prev_running = X[:, j]

    print(f"Oracle own-treatment propensity: mean={e_hat.mean():.4f}, "
          f"min={e_hat.min():.4f}, max={e_hat.max():.4f}")

    return {'e_hat': e_hat, 'decision_steps': decision_steps}


def poisson_binomial_pmf(probs: np.ndarray) -> np.ndarray:
    """
    Exact pmf of D = sum of independent Bernoulli(probs[k]), k=1..n, by
    successive convolution of the Bernoulli pmfs. O(n^2).

    Returns
    -------
    pmf : (n+1,) array, pmf[d] = P(D=d)
    """
    pmf = np.array([1.0])
    for p in probs:
        pmf = np.convolve(pmf, [1.0 - p, p])
    return pmf


def compute_oracle_spillover_pmf(dataset,
                                 e_hat: np.ndarray,
                                 decision_steps: List[int]) -> List[List[np.ndarray]]:
    """
    Exact Poisson-Binomial pmf of the treated-neighbor count D_i,tau over
    {0,...,n_i} for every unit i and decision-time position pos, from the
    neighbors' oracle propensities e_hat[neighbor, pos].

    Returns
    -------
    pmf_table[pos][i] : np.ndarray of length n_i+1, pmf_table[pos][i][d] = P(D_i=d)
    """
    adj = dataset.adj_matrix
    n = dataset.n_samples
    K = len(decision_steps)

    if issparse(adj):
        adj_csr = adj.tocsr()
        neighbor_lists = [adj_csr.indices[adj_csr.indptr[i]:adj_csr.indptr[i + 1]]
                          for i in range(n)]
    else:
        neighbor_lists = [np.nonzero(adj[i])[0] for i in range(n)]

    pmf_table = [[None] * n for _ in range(K)]
    for pos in range(K):
        e_col = e_hat[:, pos]
        for i in range(n):
            nb = neighbor_lists[i]
            if len(nb) == 0:
                pmf_table[pos][i] = np.array([1.0])
            else:
                pmf_table[pos][i] = poisson_binomial_pmf(e_col[nb])
    return pmf_table


def estimate_ate_oracle_kipw(
    dataset,
    config: Dict,
    treatment_sequences: List[Tuple],
    bandwidth: float = None,
    weight_truncation_percentile: float = 100.0,
    M: Optional[int] = None,
    outcome_time: Optional[int] = None,
    hajek: bool = True,
    return_weights: bool = False,
) -> Dict:
    """
    Estimate E[Y_t(x_target, d_target)] for each treatment sequence with the
    Section 3.4 K-IPW (Hajek or Horvitz-Thompson) estimator, using the true
    DGP propensity in the denominator.

    Arguments and per-sequence outputs are as in
    gaussian_kernel.estimate_ate_ipw_gaussian_kernel, with `config` (the DGP
    parameters) in place of the fitted model. The returned dict carries the
    oracle own propensities under 'e_hat'.
    """
    print("\n" + "="*80)
    print(f"ORACLE K-IPW {'Hajek' if hajek else 'Horvitz-Thompson'} ATE ESTIMATION")
    print("="*80)

    if bandwidth is None:
        bandwidth = dataset.spillover_bandwidth
    print(f"Bandwidth: {bandwidth:.3f}")

    own_propensity = compute_oracle_own_propensity(dataset, config)
    e_hat = own_propensity['e_hat']                 # (n, K)
    decision_steps = own_propensity['decision_steps']
    K = len(decision_steps)

    pmf_table = compute_oracle_spillover_pmf(dataset, e_hat, decision_steps)

    x_obs = dataset.x.numpy()[:, decision_steps, 0]          # (n, K)
    d_obs_prop = dataset.d_xs.numpy()[:, decision_steps, 0]  # (n, K) proportion
    d_count_obs = dataset.d_xs_count.numpy()[:, decision_steps, 0]  # (n, K) count
    Y = dataset.y.numpy()
    n_samples = dataset.n_samples

    if outcome_time is None:
        outcome_time = dataset.T - 1
    M_t = K - 1
    m0 = 0 if (M is None or M > M_t + 1) else M_t - M + 1
    window = list(range(m0, M_t + 1))
    print(f"Outcome time index: {outcome_time} (0-based); decision window positions: {window} / K={K}")

    # p_d_obs[i, pos] = P(D_i,tau = observed count | neighbours' oracle e_j)
    p_d_obs = np.zeros((n_samples, K))
    for pos in range(K):
        for i in range(n_samples):
            pmf = pmf_table[pos][i]
            d = int(round(d_count_obs[i, pos]))
            d = min(max(d, 0), len(pmf) - 1)
            p_d_obs[i, pos] = max(pmf[d], 1e-12)

    ipw_estimates = {}

    for seq in treatment_sequences:
        seq_values = list(seq)
        x_targets = np.array([seq_values[2 * j] for j in decision_steps], dtype=np.float64)
        d_targets = np.array([seq_values[2 * j + 1] for j in decision_steps], dtype=np.float64)

        log_weight = np.zeros(n_samples)
        valid = np.ones(n_samples, dtype=bool)
        local_support = np.ones(n_samples, dtype=bool)

        for pos in window:
            x_tau = x_targets[pos]
            d_tau = d_targets[pos]

            indicator = (x_obs[:, pos] == x_tau)
            numerator, d_support = spillover_numerator(d_obs_prop[:, pos], d_tau, bandwidth)
            local_support &= indicator & d_support

            e = e_hat[:, pos]
            own_prob = np.where(x_tau == 1, e, 1.0 - e)
            denom = np.clip(own_prob * p_d_obs[:, pos], 1e-12, None)

            step_weight = np.where(indicator, numerator / denom, 0.0)
            valid &= indicator
            with np.errstate(divide='ignore'):
                log_weight = log_weight + np.log(np.clip(step_weight, 1e-300, None))

        weights = np.where(valid, np.exp(log_weight), 0.0)

        if weight_truncation_percentile < 100 and (weights > 0).any():
            weight_upper = np.percentile(weights[weights > 0], weight_truncation_percentile)
            weights = np.clip(weights, 0, weight_upper)

        weight_sum = weights.sum()
        ess = (weight_sum ** 2 / (weights ** 2).sum()) if (weights ** 2).sum() > 0 else 0.0
        raw_path_count = int(valid.sum())
        local_support_count = int(local_support.sum())

        if weight_sum < 1e-8:
            ipw_estimates[seq] = {
                'estimate': None,
                'effective_n': float(ess),
                'raw_path_count': raw_path_count,
                'local_support_count': local_support_count,
                'weight_sum': float(weight_sum),
                'max_normalized_weight': None,
                'weighted_outcome_sd': None,
            }
            print(f"  E[Y{seq}]: insufficient weight (ESS={ess:.1f})")
            continue

        y_t = Y[:, outcome_time, 0]
        if hajek:
            estimate = (weights * y_t).sum() / weight_sum
        else:
            estimate = (weights * y_t).sum() / n_samples

        normalized_weights = weights / weight_sum
        weighted_var = float(np.sum(normalized_weights * (y_t - estimate) ** 2))
        ipw_estimates[seq] = {
            'estimate': float(estimate),
            'effective_n': float(ess),
            'raw_path_count': raw_path_count,
            'local_support_count': local_support_count,
            'weight_sum': float(weight_sum),
            'max_normalized_weight': float(normalized_weights.max()),
            'weighted_outcome_sd': float(np.sqrt(max(weighted_var, 0.0))),
        }
        if return_weights:
            ipw_estimates[seq]['weights'] = weights
            ipw_estimates[seq]['y'] = y_t
        print(f"  Ê_oracle[Y{seq}] = {estimate:.6f} (ESS={ess:.1f})")

    print("="*80)

    return {
        'estimates': ipw_estimates,
        'e_hat': e_hat,
        'bandwidth': bandwidth,
        'M': M,
        'outcome_time': outcome_time,
    }
