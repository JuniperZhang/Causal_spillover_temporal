"""
Monte Carlo ground truth E[Y_T(x_{1:T}, d_{1:T})] for the simulation DGP
(Supplementary Material).

The spillover channels (Y^S in covariate evolution, U^S in the latent q_S
term, H^XS in the latent dose-response) make a unit's outcome depend on its
neighbours' trajectories. The truth is therefore computed by a two-hop
local structural simulation: each draw is a focal unit, Poisson-many
first-hop neighbours, and Poisson-many second-hop neighbours per first-hop
neighbour.

- Focal unit: own treatment forced to x_tau; at each forced decision time
  round(d_tau · degree) first-hop neighbours are treated, so d_tau is
  rounded to the focal degree lattice (achieved exposures are returned).
- First-hop neighbours: full DGP equations, with their spillover terms
  computed from the focal unit and their second-hop pool.
- Second-hop units: natural treatment assignment and no spillover terms;
  third- and longer-range feedback is truncated.
"""
import numpy as np
from typing import Dict, Tuple

from .generation import (
    get_burden_weights,
    compute_burden,
    assign_baseline_families,
    get_family_multipliers,
    compute_baseline_risk_score,
    compute_population_expected_burden,
    evolve_covariates,
    generate_latent_disease_activity,
    generate_outcome,
    generate_treatment_logistic,
    _get_covariate_params,
)


def _masked_mean_axis1(arr2d: np.ndarray, mask: np.ndarray, denom: np.ndarray) -> np.ndarray:
    """(n, max_deg) -> (n,) masked average over the neighbor axis."""
    return (arr2d * mask).sum(axis=1) / denom


def _assign_family_mult_flat(v_bar_0: np.ndarray, valid_mask: np.ndarray, family_mult_map: Dict) -> np.ndarray:
    """
    Family multipliers m_F from a quantile split over the valid entries of a
    flattened pool of baseline burden scores; padding slots get 0.
    """
    out = np.zeros_like(v_bar_0)
    flat_burden = v_bar_0[valid_mask]
    if flat_burden.size > 0:
        labels = assign_baseline_families(flat_burden)
        out[valid_mask] = [family_mult_map[l] for l in labels]
    return out


def _masked_baseline_risk(
    y0: np.ndarray,
    burden0: np.ndarray,
    ys0: np.ndarray,
    valid_mask: np.ndarray,
    config: Dict,
) -> np.ndarray:
    """Standardized baseline risk score R^0 computed over valid units only.

    Padding slots in the two-hop tensors are excluded so they do not shift
    the mean and sd used for standardization; they get 0.
    """
    out = np.zeros_like(np.asarray(y0, dtype=np.float64))
    valid = np.asarray(valid_mask, dtype=bool)
    if valid.any():
        out[valid] = compute_baseline_risk_score(
            np.asarray(y0)[valid],
            np.asarray(burden0)[valid],
            np.asarray(ys0)[valid],
            config,
        )
    return out


def compute_ground_truth_monte_carlo_arbitrary_T(
    treatment_seq: Tuple,
    config: Dict,
    n_samples: int = None,
    seed: int = 42,
    verbose: bool = True,
    max_degree_cap: int = None,
    window_m0: int = 0,
) -> Dict:
    """
    Compute E[Y_T(x_{1:T}, d_{1:T})] by two-hop local structural Monte Carlo
    (see module docstring).

    Args:
        treatment_seq: (x_1, d_1, x_2, d_2, ..., x_T, d_T), length 2T.
        config: DGP configuration dictionary.
        n_samples: number of Monte Carlo draws (focal units).
        seed: random seed.
        verbose: print progress.
        max_degree_cap: cap on the sampled Poisson degree at both hops,
            bounding the second-hop pool of shape
            (n_samples, max_deg1, max_deg2, v_dim); defaults to 3x the mean
            degree.
        window_m0: 0-based position, among the decision times, at which the
            intervention starts (the window start m_0 of the Section 3.4
            estimator). Earlier decision times follow the natural
            treatment-assignment model for the focal unit and its first-hop
            neighbours; window_m0=0 forces every decision time.

    Returns:
        dict with the Monte Carlo mean, std and standard error of Y_T,
        n_samples, method, mean_degree, achieved_d_mean (mean achieved focal
        exposure at each forced decision time) and an approximation note.
    """
    if n_samples is None:
        n_samples = config.get('n_samples_monte_carlo', 50000)
    np.random.seed(seed)

    seq_values = list(treatment_seq)
    T = len(seq_values) // 2
    assert len(seq_values) == 2 * T, f"Invalid sequence length {len(seq_values)}"
    x_path = [float(seq_values[2 * t]) for t in range(T)]
    d_path = [float(seq_values[2 * t + 1]) for t in range(T)]

    v_dim = config['v_dim']
    update_interval = config['treatment_update_interval']
    decision_times = set(range(1, T + 1, update_interval))
    decision_time_list = sorted(decision_times)
    decision_position = {t: pos for pos, t in enumerate(decision_time_list)}

    mean_degree = max(1, int(round(config.get('ground_truth_star_degree', config.get('network_avg_degree', 6)))))
    if max_degree_cap is None:
        max_degree_cap = max(mean_degree + 1, 3 * mean_degree)

    degrees1 = np.clip(np.maximum(1, np.random.poisson(mean_degree, n_samples)), 1, max_degree_cap)
    max_deg1 = int(degrees1.max())
    mask1 = np.arange(max_deg1)[None, :] < degrees1[:, None]                      # (n, max_deg1)

    # Conditional on being a first-hop neighbor of the focal unit, a unit
    # has the focal unit itself plus an approximately Poisson number of
    # additional neighbors.  `degrees2` is that number of additional
    # neighbors; the first-hop unit's total degree is 1 + degrees2.
    degrees2 = np.clip(
        np.random.poisson(mean_degree, (n_samples, max_deg1)), 0, max_degree_cap
    )
    degrees2[~mask1] = 0
    max_deg2 = max(1, int(degrees2.max()))
    mask2 = ((np.arange(max_deg2)[None, None, :] < degrees2[:, :, None])
             & mask1[:, :, None])
    nei1_total_degree = np.where(mask1, 1 + degrees2, 1).astype(np.float64)

    if verbose:
        print(f"\n{'='*80}")
        print(f"Monte Carlo Ground Truth, two-hop star network (Arbitrary T={T})")
        print(f"{'='*80}")
        print(f"Target: E[Y_T{treatment_seq}]")
        print(f"Samples: {n_samples:,}  mean_degree={mean_degree}  max_deg1={max_deg1}  max_deg2={max_deg2}")

    weights = get_burden_weights(config, v_dim)
    cov_params = _get_covariate_params(config, v_dim)
    baseline_mean = cov_params['baseline_mean'].reshape(1, -1)
    baseline_std = cov_params['baseline_std'].reshape(1, -1)
    v_max_arr = cov_params['v_max'].reshape(1, -1)
    family_mult_map = get_family_multipliers(config)
    rho_x = config.get('history_rho_X', 0.63)

    # ── Baseline (t=0): focal ────────────────────────────────────────────
    V_focal = np.clip(np.random.randn(n_samples, v_dim) * baseline_std + baseline_mean, 0, v_max_arr)
    v_bar_0_focal = compute_burden(V_focal, weights)
    family_mult_focal = _assign_family_mult_flat(
        v_bar_0_focal, np.ones(n_samples, dtype=bool), family_mult_map
    )
    Y_focal = np.clip(
        np.random.randn(n_samples) * config['baseline_y_std'] + config['baseline_y_mean'],
        config.get('y_min', 0.0), config['y_max']
    ).astype(np.float64)

    # ── Baseline (t=0): first-hop neighbors ─────────────────────────────
    V_nei1 = np.clip(
        np.random.randn(n_samples, max_deg1, v_dim) * baseline_std[None] + baseline_mean[None],
        0, v_max_arr[None]
    )
    V_nei1[~mask1] = 0.0
    v_bar_0_nei1 = np.tensordot(V_nei1, weights, axes=([-1], [0]))
    family_mult_nei1 = _assign_family_mult_flat(v_bar_0_nei1.flatten(), mask1.flatten(), family_mult_map).reshape(n_samples, max_deg1)
    Y_nei1 = np.clip(
        np.random.randn(n_samples, max_deg1) * config['baseline_y_std'] + config['baseline_y_mean'],
        config.get('y_min', 0.0), config['y_max']
    ).astype(np.float64)
    Y_nei1[~mask1] = 0.0

    # Focal/first-hop R0, used only for natural treatment draws before the
    # window (window_m0 > 0). YS0 is the simulated neighbour average: the
    # first-hop pool for the focal unit, focal plus second-hop pool for
    # first-hop units (below).
    YS0_focal = _masked_mean_axis1(Y_nei1, mask1, degrees1)
    R0_focal = compute_baseline_risk_score(Y_focal, v_bar_0_focal, YS0_focal, config)

    # ── Baseline (t=0): second-hop pool ─────────────────────────────────
    V_nei2 = np.clip(
        np.random.randn(n_samples, max_deg1, max_deg2, v_dim) * baseline_std[None, None] + baseline_mean[None, None],
        0, v_max_arr[None, None]
    ).astype(np.float32)
    V_nei2[~mask2] = 0.0
    v_bar_0_nei2 = np.tensordot(V_nei2, weights, axes=([-1], [0])).astype(np.float64)
    family_mult_nei2 = _assign_family_mult_flat(
        v_bar_0_nei2.flatten(), mask2.flatten(), family_mult_map
    ).reshape(n_samples, max_deg1, max_deg2)
    Y0_nei2 = np.clip(
        np.random.randn(n_samples, max_deg1, max_deg2) * config['baseline_y_std'] + config['baseline_y_mean'],
        config.get('y_min', 0.0), config['y_max']
    ).astype(np.float64)
    Y0_nei2[~mask2] = 0.0

    # A first-hop neighbor's neighborhood contains the focal unit plus its
    # additional (second-hop) neighbors.
    YS0_nei1 = (
        Y_focal[:, None] + (Y0_nei2 * mask2).sum(axis=2)
    ) / nei1_total_degree
    YS0_nei1[~mask1] = 0.0
    R0_nei1 = _masked_baseline_risk(
        Y_nei1, v_bar_0_nei1, YS0_nei1, mask1, config
    )

    # Second-hop R0 for their natural treatment draws. No third hop is
    # simulated, so the unit's own Y0 stands in for its YS0.
    R0_nei2 = _masked_baseline_risk(
        Y0_nei2, v_bar_0_nei2, Y0_nei2, mask2, config
    )

    Y_nei2 = Y0_nei2
    C_focal = Y_focal.copy()
    C_nei1 = Y_nei1.copy()
    C_nei2 = Y_nei2.copy()

    h_x_focal = np.zeros(n_samples, dtype=np.float64)
    h_xs_focal = np.zeros(n_samples, dtype=np.float64)
    h_x_nei1 = np.zeros((n_samples, max_deg1), dtype=np.float64)
    h_xs_nei1 = np.zeros((n_samples, max_deg1), dtype=np.float64)
    h_x_nei2 = np.zeros((n_samples, max_deg1, max_deg2), dtype=np.float64)

    ys_prev_focal = YS0_focal.copy()
    ys_prev_nei1 = YS0_nei1.copy()
    x_prev_nei2 = np.zeros((n_samples, max_deg1, max_deg2), dtype=np.float64)
    y_2step_prev_nei2 = Y_nei2.copy()

    # Carry-forward treatment and lagged outcomes for focal/first-hop units;
    # used by the pre-window natural treatment draws and between decision times.
    x_prev_focal_carry = np.zeros(n_samples, dtype=np.float64)
    y_2step_prev_focal = Y_focal.copy()
    x_prev_nei1_carry = np.zeros((n_samples, max_deg1), dtype=np.float64)
    y_2step_prev_nei1 = Y_nei1.copy()

    achieved_d_at_decisions = []

    for t in range(1, T + 1):
        x_tau = x_path[t - 1]
        d_tau = d_path[t - 1]
        pos = decision_position.get(t)
        is_decision = pos is not None
        forced = is_decision and (pos >= window_m0)

        # ── Covariate burden this step (t-1 values, before this step's update) ──
        U_focal = compute_burden(V_focal, weights)
        U_nei1 = (V_nei1 * weights.reshape(1, 1, -1)).sum(axis=2)
        U_nei1[~mask1] = 0.0
        U_nei2_prev = (V_nei2 * weights.reshape(1, 1, 1, -1)).sum(axis=3).astype(np.float64)
        US_focal = _masked_mean_axis1(U_nei1, mask1, degrees1)
        US_nei1 = (
            U_focal[:, None] + (U_nei2_prev * mask2).sum(axis=2)
        ) / nei1_total_degree
        US_nei1[~mask1] = 0.0

        if forced:
            # ── Force focal's own treatment, and enough first-hop neighbors to hit d_tau ──
            X_focal = np.full(n_samples, x_tau, dtype=np.float64)
            n_treated_target = np.clip(np.rint(d_tau * degrees1).astype(int), 0, degrees1)

            X_nei1 = np.zeros((n_samples, max_deg1), dtype=np.float64)
            if d_tau > 0:
                rand_vals = np.random.rand(n_samples, max_deg1)
                rand_vals[~mask1] = np.inf
                ranks = np.argsort(np.argsort(rand_vals, axis=1), axis=1)
                X_nei1 = (ranks < n_treated_target[:, None]).astype(np.float64)
                X_nei1[~mask1] = 0.0
            achieved_d_at_decisions.append(
                float(_masked_mean_axis1(X_nei1, mask1, degrees1).mean())
            )
        elif is_decision:
            # ── Pre-window decision time: natural treatment assignment for
            # the focal unit and first-hop neighbours. ──
            X_focal = generate_treatment_logistic(
                U_prev=U_focal, R0=R0_focal, params=config,
                x_prev=x_prev_focal_carry,
                y_prev=Y_focal, y_prev_step=y_2step_prev_focal,
                seed=None,
            ).astype(np.float64)

            X_nei1 = generate_treatment_logistic(
                U_prev=U_nei1.flatten(), R0=R0_nei1.flatten(), params=config,
                x_prev=x_prev_nei1_carry.flatten(),
                y_prev=Y_nei1.flatten(), y_prev_step=y_2step_prev_nei1.flatten(),
                seed=None,
            ).reshape(n_samples, max_deg1).astype(np.float64)
            X_nei1[~mask1] = 0.0
        else:
            # Non-decision time: carry forward.
            X_focal = x_prev_focal_carry
            X_nei1 = x_prev_nei1_carry

        # ── Second-hop units: always natural (unforced) treatment assignment ──
        if is_decision:
            X_nei2 = generate_treatment_logistic(
                U_prev=U_nei2_prev.flatten(), R0=R0_nei2.flatten(), params=config,
                x_prev=x_prev_nei2.flatten(),
                y_prev=Y_nei2.flatten(), y_prev_step=y_2step_prev_nei2.flatten(),
                seed=None,
            ).reshape(n_samples, max_deg1, max_deg2).astype(np.float64)
            X_nei2[~mask2] = 0.0
        else:
            X_nei2 = x_prev_nei2

        m_prev = compute_population_expected_burden(t - 1, config, v_dim)

        h_x_focal_curr = rho_x * h_x_focal + X_focal
        h_xs_focal_curr = rho_x * h_xs_focal + _masked_mean_axis1(X_nei1, mask1, degrees1)
        h_x_nei1_curr = rho_x * h_x_nei1 + X_nei1
        X_nei2_mean_per_nei1 = (
            X_focal[:, None] + (X_nei2 * mask2).sum(axis=2)
        ) / nei1_total_degree
        X_nei2_mean_per_nei1[~mask1] = 0.0
        h_xs_nei1_curr = rho_x * h_xs_nei1 + X_nei2_mean_per_nei1
        h_x_nei2_curr = rho_x * h_x_nei2 + X_nei2

        # ── Focal outcome (uses U^S, H^XS from first-hop pool) ──────────
        C_focal_next = generate_latent_disease_activity(
            U_prev=U_focal, m_prev=m_prev,
            treatment_history_state=h_x_focal_curr,
            family_multiplier=family_mult_focal,
            params=config, latent_prev=C_focal,
            U_S_prev=US_focal, spillover_treatment_history_state=h_xs_focal_curr,
            seed=None,
        )
        Y_focal_next = generate_outcome(C_focal_next, y_prev=Y_focal, params=config, seed=None)

        # ── First-hop neighbors' outcome (uses THEIR OWN U^S, H^XS from second-hop pool) ──
        C_nei1_next = generate_latent_disease_activity(
            U_prev=U_nei1.flatten(), m_prev=m_prev,
            treatment_history_state=h_x_nei1_curr.flatten(),
            family_multiplier=family_mult_nei1.flatten(),
            params=config, latent_prev=C_nei1.flatten(),
            U_S_prev=US_nei1.flatten(), spillover_treatment_history_state=h_xs_nei1_curr.flatten(),
            seed=None,
        ).reshape(n_samples, max_deg1)
        C_nei1_next[~mask1] = 0.0
        Y_nei1_next = generate_outcome(
            C_nei1_next.flatten(), y_prev=Y_nei1.flatten(), params=config, seed=None
        ).reshape(n_samples, max_deg1)
        Y_nei1_next[~mask1] = 0.0

        # ── Second-hop units' outcome (truncated -- no third-order terms) ──
        C_nei2_next = generate_latent_disease_activity(
            U_prev=U_nei2_prev.flatten(), m_prev=m_prev,
            treatment_history_state=h_x_nei2_curr.flatten(),
            family_multiplier=family_mult_nei2.flatten(),
            params=config, latent_prev=C_nei2.flatten(),
            seed=None,
        ).reshape(n_samples, max_deg1, max_deg2).astype(np.float32)
        C_nei2_next[~mask2] = 0.0
        Y_nei2_next = generate_outcome(
            C_nei2_next.flatten(), y_prev=Y_nei2.flatten(), params=config, seed=None
        ).reshape(n_samples, max_deg1, max_deg2).astype(np.float32)
        Y_nei2_next[~mask2] = 0.0

        # ── Covariate evolution ──────────────────────────────────────────
        V_focal_next = evolve_covariates(
            V_focal, config, Y_prev=Y_focal, YS_prev=ys_prev_focal,
            seed=None,
        )
        V_nei1_next = evolve_covariates(
            V_nei1.reshape(-1, v_dim), config, Y_prev=Y_nei1.flatten(), YS_prev=ys_prev_nei1.flatten(),
            seed=None,
        ).reshape(n_samples, max_deg1, v_dim)
        V_nei1_next[~mask1] = 0.0
        V_nei2_next = evolve_covariates(
            V_nei2.reshape(-1, v_dim).astype(np.float64), config, Y_prev=Y_nei2.flatten(),
            seed=None,
        ).reshape(n_samples, max_deg1, max_deg2, v_dim).astype(np.float32)
        V_nei2_next[~mask2] = 0.0

        ys_next_focal = _masked_mean_axis1(Y_nei1_next, mask1, degrees1)
        ys_next_nei1 = (
            Y_focal_next[:, None]
            + (Y_nei2_next.astype(np.float64) * mask2).sum(axis=2)
        ) / nei1_total_degree
        ys_next_nei1[~mask1] = 0.0

        V_focal, V_nei1, V_nei2 = V_focal_next, V_nei1_next, V_nei2_next
        y_2step_prev_focal = Y_focal
        y_2step_prev_nei1 = Y_nei1
        Y_focal, Y_nei1 = Y_focal_next, Y_nei1_next
        y_2step_prev_nei2 = Y_nei2
        Y_nei2 = Y_nei2_next
        C_focal, C_nei1, C_nei2 = C_focal_next, C_nei1_next, C_nei2_next
        h_x_focal, h_xs_focal = h_x_focal_curr, h_xs_focal_curr
        h_x_nei1, h_xs_nei1 = h_x_nei1_curr, h_xs_nei1_curr
        h_x_nei2 = h_x_nei2_curr
        ys_prev_focal, ys_prev_nei1 = ys_next_focal, ys_next_nei1
        x_prev_focal_carry = X_focal
        x_prev_nei1_carry = X_nei1
        x_prev_nei2 = X_nei2

    mean_y = float(np.mean(Y_focal))
    std_y = float(np.std(Y_focal))
    se_y = std_y / np.sqrt(n_samples)

    if verbose:
        print(f"\n{'='*80}")
        print(f"Result: E[Y_T{treatment_seq}] = {mean_y:.6f}")
        print(f"SE: {se_y:.6f}")
        print(f"{'='*80}\n")

    return {
        'mean': mean_y,
        'std': std_y,
        'se': se_y,
        'n_samples': n_samples,
        'method': 'mc_twohop_local_approx',
        'mean_degree': mean_degree,
        'achieved_d_mean': achieved_d_at_decisions,
        'approximation': (
            'Two-hop local structural Monte Carlo; third- and longer-range '
            'feedback is truncated.'
        ),
    }
