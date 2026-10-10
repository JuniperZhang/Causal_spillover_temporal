"""
Building blocks of the simulation data-generating process (Section 5.1;
full equations and parameter values in the Supplementary Material).

- Network: Erdős–Rényi graph; A_norm is the row-normalized adjacency.
- Baseline outcome-effect families F_i: a quantile split of the baseline
  covariate burden. They do not enter treatment assignment or covariate
  evolution; they only scale the accumulated own-treatment benefit in the
  latent equation through a multiplier m_{F_i}.
- Covariate evolution: V_it^(k) = (1-kappa_k) V_{i,t-1}^(k) + mu_k
  + gamma_Y,k (Y_{i,t-1} - c_Y,k) + gamma_YS,k (Y^S_{i,t-1} - c_Y,k) + noise,
  truncated to [0, V_max,k].
- Treatment assignment at decision times: logit(pi) = theta_0 + theta_R R_i^0
  + theta_p X_prev + theta_w W + theta_i I + theta_v U_prev + noise.
- Spillover exposure: D_it = proportion of treated first-order neighbours.
- Latent disease activity: C_it = mu_C + rho_C(C_prev-mu_C)
  + lambda_1 q_1,it + lambda_S q_S,it
  - m_{F_i} beta_X (1 - exp(-a_X H^X_it)) - beta_XS (1 - exp(-a_X H^XS_it))
  + noise,
  q_1,it = sigmoid(U_{i,t-1} - m_{t-1}), q_S,it = sigmoid(U^S_{i,t-1} - m_{t-1}).
- Outcome: Y_it = (1-kappa_Y) Y_{i,t-1} + kappa_Y C_it + noise.

Spillover terms. Each own-unit channel has a neighbour counterpart that
shares its parameters:
  - gamma_YS,k: neighbour-averaged lagged outcome Y^S in covariate
    evolution (same centring c_Y,k as gamma_Y,k);
  - lambda_S q_S,it: neighbour-averaged covariate burden U^S in the latent
    equation (same reference m_{t-1} as q_1,it);
  - beta_XS: saturating effect of the discounted exposure history
    H^XS_it = rho_X H^XS_{i,t-1} + D_it (same rho_X and a_X as the own
    terms). beta_XS is a single population-level effect, not scaled by
    m_{F_i}.
"""
import numpy as np
from scipy.sparse import issparse
from typing import Dict, Tuple, Optional


FAMILY_LABELS = ('B1', 'B2', 'B3', 'B4', 'B5')


def _sigmoid(z):
    """Sigmoid function sigma(z) = 1/(1+exp(-z)), numerically stable."""
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z, dtype=np.float64)

    pos = z >= 0
    neg = ~pos

    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[neg])
    out[neg] = exp_z / (1.0 + exp_z)
    return out


def _to_param_array(value, length: int, default: float) -> np.ndarray:
    """Broadcast scalar/list config values to a fixed-length float array."""
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.array(value, dtype=np.float64)
        if len(arr) >= length:
            return arr[:length]
        return np.pad(arr, (0, length - len(arr)), mode='edge')
    return np.full(length, value if value is not None else default, dtype=np.float64)


def _get_covariate_params(params: Dict, v_dim: int) -> Dict[str, np.ndarray]:
    """Per-coordinate covariate-evolution parameters kappa_k, mu_k, gamma_Y,k,
    gamma_YS,k, c_Y,k, sigma_V,k, V_max,k, and baseline mean/sd, each a
    (v_dim,) array."""
    return {
        'baseline_mean': _to_param_array(params.get('baseline_v_mean', 10.0), v_dim, 10.0),
        'baseline_std': _to_param_array(params.get('baseline_v_std', 1.5), v_dim, 1.5),
        'v_max': _to_param_array(params.get('v_max', 25.0), v_dim, 25.0),
        'kappa': _to_param_array(params.get('cov_kappa', 0.12), v_dim, 0.12),
        'mu': _to_param_array(params.get('cov_mu', 1.05), v_dim, 1.05),
        'gamma_y': _to_param_array(params.get('cov_gamma_y', 0.0), v_dim, 0.0),
        'gamma_ys': _to_param_array(params.get('cov_gamma_ys', 0.0), v_dim, 0.0),
        'c_y': _to_param_array(params.get('cov_c_y', params.get('baseline_y_mean', 72.0)), v_dim, 72.0),
        'sigma_v': _to_param_array(params.get('cov_sigma_v', 0.10), v_dim, 0.10),
    }


def get_burden_weights(params: Dict, v_dim: int) -> np.ndarray:
    """Normalized weights w_k for collapsing vector covariates into the
    scalar burden score U_it = sum_k w_k V_it^(k), sum_k w_k = 1."""
    weights = _to_param_array(params.get('cov_burden_weights', 1.0), v_dim, 1.0)
    weights = np.clip(weights, 0.0, None)
    weight_sum = float(weights.sum())
    if weight_sum <= 0:
        return np.full(v_dim, 1.0 / max(v_dim, 1), dtype=np.float64)
    return (weights / weight_sum).astype(np.float64)


def compute_burden(V: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """U_it = sum_k w_k V_it^(k) (or V_i0-bar at baseline)."""
    return (np.asarray(V, dtype=np.float64) * weights.reshape(1, -1)).sum(axis=1)


def compute_population_expected_burden(t: int, params: Dict, v_dim: int) -> float:
    """
    m_t = E[U_t], the population-expected covariate burden trajectory.

    Propagates only the mean-reversion part of the covariate process in
    expectation, m_t^(k) = (1-kappa_k) m_{t-1}^(k) + mu_k, starting from the
    baseline mean m_0^(k). Deterministic given the DGP parameters; does not
    depend on the realized sample.

    Args:
        t: 0 corresponds to m_0 (baseline), 1 to m_1, etc.
    """
    cov_params = _get_covariate_params(params, v_dim)
    weights = get_burden_weights(params, v_dim)
    m = cov_params['baseline_mean'].astype(np.float64).copy()
    for _ in range(max(t, 0)):
        m = (1.0 - cov_params['kappa']) * m + cov_params['mu']
    return float(np.dot(m, weights))


# ============================================================================
# Network Generation Utilities (treated as part of data generation)
# ============================================================================

def generate_erdos_renyi_network(n_nodes: int, avg_degree: int, seed: int = 42) -> np.ndarray:
    """
    Generate an undirected Erdős–Rényi graph G(n, p) with p = avg_degree/(n-1).

    Returns a dense (n, n) float32 adjacency for n <= 2000 and a scipy CSR
    matrix otherwise.
    """
    p = avg_degree / (n_nodes - 1)
    p = min(p, 1.0)

    if n_nodes > 2000:
        from scipy.sparse import csr_matrix
        print(f"  Using sparse matrix representation (n={n_nodes:,}, p={p:.6f})")

        # Batagelj–Brandes geometric skipping samples G(n,p) in expected
        # O(n + |E|) time.
        rng = np.random.default_rng(seed)
        rows = []
        cols = []
        if p >= 1.0:
            for v in range(1, n_nodes):
                rows.extend([v] * v)
                cols.extend(range(v))
        elif p > 0.0:
            log_q = np.log1p(-p)
            v = 1
            w = -1
            while v < n_nodes:
                w += 1 + int(np.log1p(-rng.random()) / log_q)
                while w >= v and v < n_nodes:
                    w -= v
                    v += 1
                if v < n_nodes:
                    rows.append(v)
                    cols.append(w)

        row_indices = np.asarray(rows, dtype=np.int64)
        col_indices = np.asarray(cols, dtype=np.int64)
        row_all = np.concatenate([row_indices, col_indices])
        col_all = np.concatenate([col_indices, row_indices])

        data = np.ones(len(row_all), dtype=np.float32)
        adj_matrix = csr_matrix((data, (row_all, col_all)), shape=(n_nodes, n_nodes))

        adj_matrix.setdiag(0)
        adj_matrix.eliminate_zeros()
        adj_matrix.data = (adj_matrix.data > 0).astype(np.float32)

        print(f"  Actual edges: {adj_matrix.nnz // 2:,} (sparsity: {100 * adj_matrix.nnz / (n_nodes**2):.4f}%)")

        return adj_matrix

    rng = np.random.default_rng(seed)
    upper = rng.random((n_nodes, n_nodes)) < p
    upper = np.triu(upper, k=1)
    adj_matrix = upper | upper.T
    return adj_matrix.astype(np.float32)


def get_bandwidth_from_network(adj_matrix) -> float:
    """
    Default kernel bandwidth h = 1 / (2 * median degree), clipped to [0.05, 0.15].

    Args:
        adj_matrix: adjacency matrix (scipy sparse or numpy array)

    Returns:
        Bandwidth h; 0.15 when the median degree is 0.
    """
    if issparse(adj_matrix):
        degrees = np.array(adj_matrix.sum(axis=1)).flatten()
    else:
        degrees = adj_matrix.sum(axis=1)

    median_degree = np.median(degrees)

    # Median degree 0: use the upper clip value
    if median_degree == 0:
        bandwidth = 0.15
        print(f"  Network has isolated nodes (degree=0), using bandwidth: {bandwidth:.3f}")
    else:
        bandwidth = 1.0 / (2 * median_degree)
        bandwidth = max(0.05, min(bandwidth, 0.15))
        print(f"  Bandwidth: {bandwidth:.3f}")

    return bandwidth


def normalize_adjacency_matrix(adj_matrix):
    """
    Row-normalize an adjacency matrix (dense or sparse) so that A_norm @ z
    is the neighbour average of z. Isolated nodes get a zero row.
    """
    from scipy.sparse import issparse

    if issparse(adj_matrix):
        degree = np.array(adj_matrix.sum(axis=1)).flatten()
        degree[degree == 0] = 1
        degree_inv = 1.0 / degree
        degree_inv[~np.isfinite(degree_inv)] = 0.0
        from scipy.sparse import diags
        degree_inv_matrix = diags(degree_inv, format='csr')
        normalized = degree_inv_matrix @ adj_matrix
        if normalized.nnz > 0:
            normalized.data = np.nan_to_num(
                normalized.data,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            normalized.eliminate_zeros()
        return normalized.astype(np.float32)

    degree = adj_matrix.sum(axis=1, keepdims=True)
    degree[degree == 0] = 1
    normalized = np.divide(
        adj_matrix,
        degree,
        out=np.zeros_like(adj_matrix, dtype=np.float64),
        where=degree != 0,
    )
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
    return normalized.astype(np.float32)


# ============================================================================
# Baseline generation
# ============================================================================

def generate_baseline_covariates_with_smoothing(
    n_samples: int,
    v_dim: int,
    adj_normalized: np.ndarray,
    params: Dict,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Draw baseline covariates V_0 ~ N(baseline_mean, baseline_std^2),
    truncated to [0, V_max], and their neighbour averages V_0^S = A_norm V_0.

    Returns:
        V0: (n_samples, v_dim) - baseline own covariates
        VS0: (n_samples, v_dim) - baseline spillover covariates (neighbor average)
    """
    np.random.seed(seed)

    cov_params = _get_covariate_params(params, v_dim)
    baseline_mean = cov_params['baseline_mean'].reshape(1, -1)
    baseline_std = cov_params['baseline_std'].reshape(1, -1)
    v_max = cov_params['v_max'].reshape(1, -1)

    V0 = np.random.randn(n_samples, v_dim) * baseline_std + baseline_mean
    V0 = np.clip(V0, 0, v_max)

    VS0_result = adj_normalized @ V0
    from scipy.sparse import issparse
    if issparse(VS0_result):
        VS0 = np.array(VS0_result.todense())
    else:
        VS0 = np.asarray(VS0_result)

    return V0, VS0


def assign_baseline_families(v_bar_0: np.ndarray) -> np.ndarray:
    """
    Assign outcome-effect families F_i by a deterministic quantile split of
    the baseline covariate-burden score v_bar_0: units are ranked and
    partitioned into 5 equal-size bins in ascending order, mapped onto
    FAMILY_LABELS (B1..B5, lowest to highest baseline burden).

    Family membership depends on baseline covariates only; it does not
    enter treatment assignment or covariate evolution.

    Args:
        v_bar_0: (n_samples,) baseline burden score bar V_{i0}

    Returns:
        families: (n_samples,) array of dtype object with values in FAMILY_LABELS
    """
    n_samples = len(v_bar_0)
    ranks = np.argsort(np.argsort(v_bar_0))
    bin_idx = np.minimum((ranks * len(FAMILY_LABELS)) // n_samples, len(FAMILY_LABELS) - 1)
    return np.array([FAMILY_LABELS[i] for i in bin_idx], dtype=object)


def get_family_multipliers(params: Dict) -> Dict[str, float]:
    """Family-specific own-treatment-benefit multipliers m_F in the latent equation."""
    default = {'B1': 0.50, 'B2': 1.16, 'B3': 0.92, 'B4': 1.20, 'B5': 1.10}
    configured = params.get('family_multipliers', None)
    if configured is None:
        return default
    return {label: float(configured.get(label, default[label])) for label in FAMILY_LABELS}


def compute_baseline_risk_score(
    Y0: np.ndarray,
    v_bar_0: np.ndarray,
    YS0: np.ndarray,
    params: Dict,
) -> np.ndarray:
    """
    Standardized baseline risk score R_i^0 = (rho_i - mean(rho)) / sd(rho),
    rho_i = omega_Y0 Y_i0 + omega_V0 bar V_i0 + omega_YS0 Y_i0^S.

    Built once from the baseline outcome, baseline covariate burden and
    baseline neighbour-averaged outcome, and standardized over the realized
    sample. It is fixed for all decision times.
    """
    omega_y0 = params.get('risk_omega_Y0', 0.10)
    omega_v0 = params.get('risk_omega_V0', 0.18)
    omega_ys0 = params.get('risk_omega_YS0', 0.08)

    rho = (
        omega_y0 * np.asarray(Y0, dtype=np.float64)
        + omega_v0 * np.asarray(v_bar_0, dtype=np.float64)
        + omega_ys0 * np.asarray(YS0, dtype=np.float64)
    )
    sd_rho = rho.std()
    if sd_rho < 1e-8:
        return np.zeros_like(rho)
    return (rho - rho.mean()) / sd_rho


def generate_baseline_outcome(
    v: np.ndarray,
    vs: np.ndarray,
    params: Dict,
    seed: int = None
) -> np.ndarray:
    """
    Draw the pre-treatment outcome Y_0 ~ N(mu_Y0, sigma_Y0^2), truncated to
    [y_min, y_max]. `v` sets the sample size only.

    Returns:
        Y0: (n_samples,) - baseline outcome
    """
    if seed is not None:
        np.random.seed(seed)

    n_samples = len(v)
    Y0 = np.random.randn(n_samples) * params['baseline_y_std'] + params['baseline_y_mean']
    Y0 = np.clip(Y0, params.get('y_min', 0.0), params['y_max'])

    return Y0


# ============================================================================
# Temporal evolution
# ============================================================================

def evolve_covariates(
    V_prev: np.ndarray,
    params: Dict,
    Y_prev: Optional[np.ndarray] = None,
    YS_prev: Optional[np.ndarray] = None,
    seed: int = None
) -> np.ndarray:
    """
    One step of covariate evolution:
        V_it^(k) = (1-kappa_k) V_{i,t-1}^(k) + mu_k
                   + gamma_Y,k(Y_{i,t-1}-c_Y,k) + gamma_YS,k(Y^S_{i,t-1}-c_Y,k)
                   + noise
    truncated to [0, V_max,k]. V_prev is (n, v_dim). With YS_prev=None the
    spillover term reduces to the constant -gamma_YS,k c_Y,k, which is zero
    under the default gamma_YS,k = 0.
    """
    if seed is not None:
        np.random.seed(seed)

    n_samples, v_dim = V_prev.shape
    cov_params = _get_covariate_params(params, v_dim)
    kappa = cov_params['kappa'].reshape(1, -1)
    mu = cov_params['mu'].reshape(1, -1)
    gamma_y = cov_params['gamma_y'].reshape(1, -1)
    gamma_ys = cov_params['gamma_ys'].reshape(1, -1)
    c_y = cov_params['c_y'].reshape(1, -1)
    sigma_v = cov_params['sigma_v'].reshape(1, -1)
    v_max = cov_params['v_max'].reshape(1, -1)

    y_prev = np.zeros((n_samples, 1), dtype=np.float64) if Y_prev is None else np.asarray(Y_prev, dtype=np.float64).reshape(-1, 1)
    y_deviation = y_prev - c_y

    ys_prev = np.zeros((n_samples, 1), dtype=np.float64) if YS_prev is None else np.asarray(YS_prev, dtype=np.float64).reshape(-1, 1)
    ys_deviation = ys_prev - c_y

    innovation = np.random.randn(n_samples, v_dim) * sigma_v

    v_next = (
        (1.0 - kappa) * np.asarray(V_prev, dtype=np.float64)
        + mu
        + gamma_y * y_deviation
        + gamma_ys * ys_deviation
        + innovation
    )
    v_next = np.clip(v_next, 0, v_max)
    return v_next


def generate_treatment_logistic(
    U_prev: np.ndarray,
    R0: np.ndarray,
    params: Dict,
    x_prev: Optional[np.ndarray] = None,
    y_prev: Optional[np.ndarray] = None,
    y_prev_step: Optional[np.ndarray] = None,
    seed: int = None
) -> np.ndarray:
    """
    Draw own treatment at a decision time tau_k from the logistic model
        logit P(X_i,tau_k=1) =
            theta_0 + theta_R R_i^0 + theta_p X_{i,tau_k-1}
            + theta_w W_i,tau_k + theta_i I_i,tau_k + theta_v U_i,tau_k-1
            + eps^X_i,tau_k,
    where W = max(Y_{tau_k-1} - Y_{tau_k-2}, 0) is recent worsening,
    I = max(Y_{tau_k-2} - Y_{tau_k-1}, 0) recent improvement, and
    eps^X ~ N(0, sigma_X^2).

    With config 'treatment_nonlinear_variant' = 'worsening_quadratic' the
    logit also contains theta_w2 W_i,tau_k^2.

    Args:
        U_prev: (n_samples,) covariate burden U_{i,tau_k-1}
        R0: (n_samples,) standardized baseline risk score R_i^0 (fixed at baseline)
        x_prev: (n_samples,) previous own treatment X_{i,tau_k-1}
        y_prev, y_prev_step: (n_samples,) Y_{tau_k-1}, Y_{tau_k-2} for W/I
    """
    if seed is not None:
        np.random.seed(seed)

    n_samples = len(U_prev)

    theta_0 = params.get('treatment_theta_0', -3.85)
    theta_r = params.get('treatment_theta_R', 1.00)
    theta_p = params.get('treatment_theta_prev', 0.28)
    theta_w = params.get('treatment_theta_w', 0.19)
    theta_i = params.get('treatment_theta_i', 0.08)
    theta_v = params.get('treatment_theta_v', 0.06)
    sigma_x = params.get('treatment_sigma_X', 0.45)
    treatment_variant = params.get('treatment_nonlinear_variant', 'baseline')
    theta_w2 = params.get('treatment_theta_w2', 0.05)

    if y_prev is not None and y_prev_step is not None:
        delta_y = np.asarray(y_prev, dtype=np.float64).reshape(-1) - np.asarray(y_prev_step, dtype=np.float64).reshape(-1)
        worsening = np.maximum(delta_y, 0.0)
        improvement = np.maximum(-delta_y, 0.0)
    else:
        worsening = np.zeros(n_samples, dtype=np.float64)
        improvement = np.zeros(n_samples, dtype=np.float64)

    x_prev_arr = np.zeros(n_samples, dtype=np.float64)
    if x_prev is not None:
        x_prev_arr = np.asarray(x_prev, dtype=np.float64).reshape(-1)

    logit = np.full(n_samples, theta_0, dtype=np.float64)
    logit += theta_r * np.asarray(R0, dtype=np.float64).reshape(-1)
    logit += theta_p * x_prev_arr
    logit += theta_w * worsening
    logit += theta_i * improvement
    logit += theta_v * np.asarray(U_prev, dtype=np.float64).reshape(-1)
    if treatment_variant not in ('baseline', 'worsening_quadratic'):
        raise ValueError(
            f"Unknown treatment_nonlinear_variant={treatment_variant!r}; "
            "choose from ['baseline', 'worsening_quadratic']"
        )
    if treatment_variant == 'worsening_quadratic':
        logit += theta_w2 * worsening ** 2

    if sigma_x > 0:
        logit += np.random.randn(n_samples) * sigma_x

    prob = _sigmoid(logit)
    X = np.random.binomial(1, prob).astype(np.float32)
    return X


def generate_spillover_treatment_proportion(
    X: np.ndarray,
    adj_normalized: np.ndarray
) -> np.ndarray:
    """
    Spillover exposure D_it = sum_{j!=i} A_ij X_jt / sum_{j!=i} A_ij, the
    proportion of treated first-order neighbours (0 for isolated nodes).

    Returns:
        XS: (n_samples,) exposure in [0, 1]
    """
    XS_result = adj_normalized @ X

    from scipy.sparse import issparse
    if issparse(XS_result):
        XS = np.array(XS_result.todense()).flatten()
    else:
        XS = np.asarray(XS_result).flatten()

    return XS.astype(np.float32)


def generate_latent_disease_activity(
    U_prev: np.ndarray,
    m_prev: float,
    treatment_history_state: np.ndarray,
    family_multiplier: np.ndarray,
    params: Dict,
    latent_prev: Optional[np.ndarray] = None,
    U_S_prev: Optional[np.ndarray] = None,
    spillover_treatment_history_state: Optional[np.ndarray] = None,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    One step of the latent disease activity C_it. With g(u) = 1 - exp(-a_X u):
        q_1,it = sigmoid(U_{i,t-1} - m_{t-1})
        q_S,it = sigmoid(U^S_{i,t-1} - m_{t-1})
        C_it = mu_C + rho_C(C_{i,t-1}-mu_C) + lambda_1 q_1,it + lambda_S q_S,it
               - m_{F_i} beta_X g(H^X_it) - beta_XS g(H^XS_it) + eps^C_it

    Variants (config 'latent_nonlinear_variant'):
      baseline:              the equation above.
      own_spillover_synergy: adds - beta_XD g(H^X_it) g(H^XS_it); the main
                             simulation design.
      burden_modified:       replaces the beta_XS term by
                             - beta_XS g(H^XS_it) [1 + eta_B (2 q_1,it - 1)],
                             a multiplier in [1-eta_B, 1+eta_B].
      burden_synergy:        both modifications.
    The last two are stress-test designs (Supplementary Material).

    U_S_prev=None drops the lambda_S term and spillover_treatment_history_state
    =None drops every H^XS term. beta_XS is a single population-level effect,
    not scaled by m_{F_i}.

    Args:
        U_prev: (n_samples,) covariate burden U_{i,t-1}
        m_prev: scalar, population-expected burden m_{t-1}
        treatment_history_state: (n_samples,) discounted own-treatment history H^X_it
        family_multiplier: (n_samples,) m_{F_i} per unit
        U_S_prev: (n_samples,) neighbor-averaged covariate burden U^S_{i,t-1}
        spillover_treatment_history_state: (n_samples,) discounted
            spillover-exposure history H^XS_it
        latent_prev: (n_samples,) C_{i,t-1}; defaults to mu_C.

    Returns:
        C_t: (n_samples,) float32
    """
    if seed is not None:
        np.random.seed(seed)

    n_samples = len(U_prev)

    if latent_prev is None:
        latent_prev = np.full(n_samples, params.get('latent_mu_C', 72.0))

    mu_c = params.get('latent_mu_C', 72.0)
    rho_c = params.get('latent_rho_C', 0.65)
    lambda_1 = params.get('latent_lambda_1', 5.2)
    lambda_s = params.get('latent_lambda_S', 0.0)
    beta_x = params.get('latent_beta_X', 8.25)
    beta_xs = params.get('latent_beta_XS', 0.0)
    a_x = params.get('latent_a_X', 0.85)
    sigma_c = params.get('latent_sigma_C', 0.80)
    nonlinear_variant = params.get('latent_nonlinear_variant', 'baseline')
    allowed_variants = {
        'baseline',
        'burden_modified',
        'own_spillover_synergy',
        'burden_synergy',
    }
    if nonlinear_variant not in allowed_variants:
        raise ValueError(
            f"Unknown latent_nonlinear_variant={nonlinear_variant!r}; "
            f"choose from {sorted(allowed_variants)}"
        )
    burden_eta = float(params.get('latent_spillover_burden_eta', 0.75))
    beta_xd = float(params.get('latent_beta_XD', 10.0))

    q1 = _sigmoid(np.asarray(U_prev, dtype=np.float64).reshape(-1) - float(m_prev))

    C_t = mu_c + rho_c * (np.asarray(latent_prev, dtype=np.float64).reshape(-1) - mu_c)
    C_t += lambda_1 * q1
    g_x = 1.0 - np.exp(
        -a_x * np.asarray(treatment_history_state, dtype=np.float64).reshape(-1)
    )
    C_t -= np.asarray(family_multiplier, dtype=np.float64).reshape(-1) * beta_x * g_x

    if U_S_prev is not None:
        qS = _sigmoid(np.asarray(U_S_prev, dtype=np.float64).reshape(-1) - float(m_prev))
        C_t += lambda_s * qS

    if spillover_treatment_history_state is not None:
        g_xs = 1.0 - np.exp(
            -a_x
            * np.asarray(
                spillover_treatment_history_state, dtype=np.float64
            ).reshape(-1)
        )
        spillover_modifier = np.ones(n_samples, dtype=np.float64)
        if nonlinear_variant in {'burden_modified', 'burden_synergy'}:
            spillover_modifier += burden_eta * (2.0 * q1 - 1.0)
        C_t -= beta_xs * spillover_modifier * g_xs

        if nonlinear_variant in {'own_spillover_synergy', 'burden_synergy'}:
            C_t -= beta_xd * g_x * g_xs

    if sigma_c > 0:
        C_t += np.random.randn(n_samples) * sigma_c

    return C_t.astype(np.float32)


def generate_outcome(
    C_t: np.ndarray,
    y_prev: np.ndarray,
    params: Dict,
    seed: int = None
) -> np.ndarray:
    """
    Observed outcome:
        Y_it = (1-kappa_Y) Y_{i,t-1} + kappa_Y C_it + eps^Y_it
    truncated to [y_min, Y_max].
    """
    if seed is not None:
        np.random.seed(seed)

    n_samples = len(C_t)
    y_max = params['y_max']
    y_min = params.get('y_min', 0.0)
    sigma_y = params.get('outcome_sigma_Y', 0.45)
    kappa_y = params.get('outcome_kappa_Y', 0.28)

    y_prev_arr = np.asarray(y_prev, dtype=np.float64).reshape(-1)
    c_t_arr = np.asarray(C_t, dtype=np.float64).reshape(-1)

    Y_t = (1.0 - kappa_y) * y_prev_arr + kappa_y * c_t_arr
    if sigma_y > 0:
        Y_t += np.random.randn(n_samples) * sigma_y
    Y_t = np.clip(Y_t, y_min, y_max)

    return Y_t.astype(np.float32)
