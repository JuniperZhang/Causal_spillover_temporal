"""
Self-normalized kernel-weighted IPW (K-IPW) estimator (paper Section 3.4).

For an outcome time t with most recent decision index M_t and window length
M <= M_t, set m0 = M_t - M + 1. The localized weight of unit i is

    omega_hat_{i,t;M}(h) = prod_{l=m0}^{M_t}
        1{X_i,tau_l = x_tau_l} K_h(D_i,tau_l, d_tau_l)
        / p_hat_tau_l(x_tau_l, D_i,tau_l | H_i,tau_l)

with K_h the Gaussian kernel on the spillover proportion and the fitted joint
propensity p_hat(x, d | H) = e_hat^x (1 - e_hat)^(1-x) p_hat_D(d | H), where
p_hat_D is the fitted ZOIB-Binomial pmf evaluated at the unit's observed
exposure D_i,tau_l; the target d_tau_l enters only through K_h. The
Horvitz-Thompson and Hajek estimators are

    mu_hat_HT,t = (1/n) sum_i omega_hat_i Y_i,t
    mu_hat_H,t  = sum_i omega_hat_i Y_i,t / sum_i omega_hat_i

M = M_t (the default) gives the full-history estimator.
"""
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional

from .ipw import compute_propensity_scores_lstm


def gaussian_kernel_weight(d_obs: np.ndarray, d_target: float, bandwidth: float) -> np.ndarray:
    """Gaussian kernel K_h(d_obs, d_target) = exp(-(d_obs - d_target)^2 / (2 h^2)), elementwise."""
    distance = d_obs - d_target
    return np.exp(-(distance ** 2) / (2.0 * bandwidth ** 2))


def spillover_numerator(
    d_prop: np.ndarray,
    d_target: float,
    bandwidth: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Kernel factor K_h(D_i, d_target) of one step weight and its support |D_i - d_target| <= h."""
    return (gaussian_kernel_weight(d_prop, d_target, bandwidth),
            np.abs(d_prop - d_target) <= bandwidth)


def _binomial_pmf(d: np.ndarray, n: np.ndarray, q: np.ndarray) -> np.ndarray:
    """P(D=d | n, q) for D ~ Binomial(n, q), vectorized (numpy, via lgamma)."""
    from scipy.special import gammaln
    q_c = np.clip(q, 1e-6, 1.0 - 1e-6)
    n_c = np.maximum(n, 1e-8)
    log_coef = gammaln(n_c + 1.0) - gammaln(d + 1.0) - gammaln(n_c - d + 1.0)
    return np.exp(log_coef + d * np.log(q_c) + (n_c - d) * np.log(1.0 - q_c))


def zoib_binomial_pmf(d_logit: np.ndarray, d_count: np.ndarray, n_neighbors: np.ndarray) -> np.ndarray:
    """
    Fitted ZOIB-Binomial spillover pmf p_hat_D(d | H, n_i) from raw f_D output,
    the same parameterization as combined_model._zoib_binomial_loss:

        P(D=0) = pi_0,  P(D=n) = pi_1,
        P(D=d) = pi_c Binomial(d; n, q) / Z(n, q),  0 < d < n,

    with (pi_0, pi_1, pi_c) = softmax of the first three logits, q = sigmoid
    of the fourth, and Z(n, q) = 1 - (1-q)^n - q^n renormalizing the Binomial
    to {1, ..., n-1} so the pmf sums to 1 over {0, ..., n}.

    Args:
        d_logit     : (n,4) raw f_D output [logit0, logit1, logit_cont, q_raw]
        d_count     : (n,)  treated-neighbor count at which to evaluate the pmf
        n_neighbors : (n,)  n_i, number of neighbors

    Returns:
        pmf: (n,) P(D=d_count | H, n_i), floored at 1e-10
    """
    logits = d_logit - d_logit.max(axis=1, keepdims=True)
    exp_l = np.exp(logits[:, :3])
    softmax3 = exp_l / exp_l.sum(axis=1, keepdims=True)
    pi0, pi1, p_cont = softmax3[:, 0], softmax3[:, 1], softmax3[:, 2]
    q = 1.0 / (1.0 + np.exp(-d_logit[:, 3]))
    q_c = np.clip(q, 1e-6, 1.0 - 1e-6)

    is_zero = d_count <= 0.5
    is_full = (d_count >= n_neighbors - 0.5) & (n_neighbors > 0.5)

    n_c = np.maximum(n_neighbors, 1.0)
    binom = _binomial_pmf(d_count, n_c, q)
    edge_mass = np.power(1.0 - q_c, n_c) + np.power(q_c, n_c)
    z_interior = np.clip(1.0 - edge_mass, 1e-10, None)

    pmf = np.where(is_zero, pi0, np.where(is_full, pi1, p_cont * binom / z_interior))
    return np.clip(pmf, 1e-10, None)


def _extract_target_path(seq: Tuple, decision_steps: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Target (x_tau, d_tau) path at the decision times from a length-2T
    (x1,d1,...,xT,dT) treatment-sequence tuple.

    Returns
    -------
    x_targets, d_targets : each (K,) arrays over the K decision times
    """
    seq_values = list(seq)
    x_targets = np.array([seq_values[2 * j] for j in decision_steps], dtype=np.float64)
    d_targets = np.array([seq_values[2 * j + 1] for j in decision_steps], dtype=np.float64)
    return x_targets, d_targets


def estimate_ate_ipw_gaussian_kernel(
    model,
    dataset,
    treatment_sequences: List[Tuple],
    device: torch.device,
    bandwidth: float = None,
    weight_truncation_percentile: float = 100.0,
    M: Optional[int] = None,
    outcome_time: Optional[int] = None,
    hajek: bool = True,
    return_weights: bool = False,
) -> Dict:
    """
    Estimate E[Y_t(x_target, d_target)] for each treatment sequence with the
    self-normalized K-IPW (Hajek) estimator of Section 3.4, or its
    Horvitz-Thompson counterpart if hajek=False.

    Step weights are multiplied in log space; a unit whose observed own
    treatment differs from the target at any window time gets weight 0.

    Args:
        model: trained ContinuousLSTMCausalModel (TemporalCausalModelSpillover)
        dataset: NetworkTemporalCausalDataset instance
        treatment_sequences: List of (x1,d1,...,xT,dT) length-2T tuples;
            only the decision-time entries are used as the target path
        device: torch device
        bandwidth: kernel bandwidth h; defaults to dataset.spillover_bandwidth
        weight_truncation_percentile: percentile at which to clip omega_hat
        M: window length (number of most-recent decision times to weight
            over); None means the full-history estimator, M = M_t
        outcome_time: 0-based processing-time index of the outcome time t;
            None means the final processing time T-1
        hajek: if True (default) return the self-normalized Hajek estimator;
            if False return the Horvitz-Thompson estimator
        return_weights: also return each unit's weight and outcome

    Returns:
        Dict with 'estimates': {seq: {'estimate', 'effective_n' (Kish ESS),
        'raw_path_count' (units matching the own-treatment path),
        'local_support_count' (of those, units with |D - d_tau| <= h at every
        window time), 'weight_sum', 'max_normalized_weight',
        'weighted_outcome_sd'}}, plus 'propensity_scores', 'bandwidth', 'M',
        'outcome_time'. 'estimate' is None when the weights sum to ~0.
    """
    print("\n" + "="*80)
    print(f"K-IPW {'Hajek' if hajek else 'Horvitz-Thompson'} ATE ESTIMATION")
    print("="*80)

    if bandwidth is None:
        bandwidth = dataset.spillover_bandwidth
    print(f"Bandwidth: {bandwidth:.3f}")
    print(f"Weight truncation: {weight_truncation_percentile}th percentile")

    propensity = compute_propensity_scores_lstm(model, dataset, device)
    e_hat = propensity['e_hat']              # (n, K)
    d_logits = propensity['d_logits']        # (n, K, 4)
    x_obs = propensity['x_obs']              # (n, K)
    d_obs_prop = propensity['d_obs']         # (n, K) -- spillover PROPORTION
    decision_steps = propensity['decision_steps']
    K = len(decision_steps)

    n_neighbors = dataset.n_neighbors.numpy().flatten()          # (n,)
    d_count_obs = dataset.d_xs_count.numpy()[:, decision_steps, 0]  # (n, K) -- observed COUNT

    Y = dataset.y.numpy()
    T = dataset.T
    if outcome_time is None:
        outcome_time = T - 1
    M_t = K - 1   # 0-based index of the most recent decision time <= outcome_time
    if M is None or M > M_t + 1:
        m0 = 0
    else:
        m0 = M_t - M + 1
    window = list(range(m0, M_t + 1))
    print(f"Outcome time index: {outcome_time} (0-based); decision window positions: {window} / K={K}")

    # Fitted spillover pmf at the observed exposure, per decision time
    p_d_obs = np.zeros((len(dataset), K))
    for pos in range(K):
        p_d_obs[:, pos] = zoib_binomial_pmf(
            d_logits[:, pos, :], d_count_obs[:, pos], n_neighbors
        )

    ipw_estimates = {}

    for seq in treatment_sequences:
        x_targets, d_targets = _extract_target_path(seq, decision_steps)

        log_weight = np.zeros(len(dataset))
        valid = np.ones(len(dataset), dtype=bool)
        local_support = np.ones(len(dataset), dtype=bool)

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
            estimate = (weights * y_t).sum() / len(dataset)

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
        print(f"  Ê[Y{seq}] = {estimate:.6f} (ESS={ess:.1f})")

    print("="*80)

    return {
        'estimates': ipw_estimates,
        'propensity_scores': propensity,
        'bandwidth': bandwidth,
        'M': M,
        'outcome_time': outcome_time,
    }
