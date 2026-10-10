"""
Simulated network panel dataset (Section 5.1; full DGP in the
Supplementary Material).

Runs the generators in generation.py over process times t = 1..T on one
Erdős–Rényi network and stores the realized covariates, own treatment X,
spillover exposure D, latent activity C and outcome Y as torch tensors.
"""
import torch
from torch.utils.data import Dataset
import numpy as np
from scipy.sparse import issparse
from typing import Dict

from .generation import (
    generate_erdos_renyi_network,
    normalize_adjacency_matrix,
    get_bandwidth_from_network,
    generate_baseline_covariates_with_smoothing,
    get_burden_weights,
    compute_burden,
    assign_baseline_families,
    get_family_multipliers,
    compute_baseline_risk_score,
    compute_population_expected_burden,
    generate_treatment_logistic,
    generate_spillover_treatment_proportion,
    evolve_covariates,
    generate_baseline_outcome,
    generate_latent_disease_activity,
    generate_outcome,
    FAMILY_LABELS,
)


# ============================================================================
# Network Temporal Causal Dataset
# ============================================================================

class NetworkTemporalCausalDataset(Dataset):
    """
    One simulated network panel of n units over T process times.

    - Baseline outcome-effect families F_i (quantile split of the baseline
      burden) scale only the own-treatment benefit in the latent equation.
    - Own treatment X is drawn from the logistic assignment model at decision
      times t = 1, 1+L, 1+2L, ... (L = 'treatment_update_interval') and
      carried forward in between.
    - Spillover exposure D = A_norm @ X, the treated-neighbour proportion.

    Per-unit tensors have shape (n, T, ·), with time index t-1 holding
    process time t. Indexing returns a dict of these per-unit sequences.

    Args:
        n_samples: number of units n.
        v_dim: number of time-varying covariates.
        T: number of process times.
        network_avg_degree: expected Erdős–Rényi degree.
        config: DGP parameter dictionary.
        seed: master seed for the network and all DGP draws.
        stratified, min_samples_per_regime: stored, not used by generation.
    """

    def __init__(self,
                 n_samples: int,
                 v_dim: int,
                 T: int,
                 network_avg_degree: int,
                 config: Dict,
                 seed: int = 42,
                 stratified: bool = False,
                 min_samples_per_regime: int = 500):
        """Draw the network and simulate all unit histories (see the class docstring for arguments)."""
        self.seed = int(seed)
        torch.manual_seed(self.seed)

        # Independent, reproducible component seeds derived from the dataset
        # seed: two baseline seeds plus four per process time.
        n_component_seeds = 2 + 4 * T
        seed_children = np.random.SeedSequence(self.seed).spawn(n_component_seeds)
        self._dgp_seeds = [int(s.generate_state(1, dtype=np.uint32)[0])
                           for s in seed_children]

        self.n_samples = n_samples
        self.v_dim = v_dim
        self.T = T
        self.config = config
        self.stratified = stratified
        self.min_samples_per_regime = min_samples_per_regime

        print("\n" + "="*80)
        print("DATA GENERATION")
        print("="*80)
        print(f"Treatment update interval L: {int(config.get('treatment_update_interval', 1))}")
        print(f"Baseline V0 ~ N({config.get('baseline_v_mean', 10.0)}, {config.get('baseline_v_std', 1.5)}^2)")
        print(f"Baseline Y0 ~ N({config['baseline_y_mean']}, {config['baseline_y_std']}^2)")
        print("="*80)

        # Generate network structure
        print(f"\nGenerating Erdos-Renyi network...")
        self.adj_matrix = generate_erdos_renyi_network(n_samples, network_avg_degree, seed)
        self.adj_matrix_normalized = normalize_adjacency_matrix(self.adj_matrix)

        if issparse(self.adj_matrix):
            actual_avg_degree = np.array(self.adj_matrix.sum(axis=1)).flatten().mean()
        else:
            actual_avg_degree = self.adj_matrix.sum(axis=1).mean()
        print(f"Actual average degree: {actual_avg_degree:.2f}")

        if issparse(self.adj_matrix):
            self.adj_matrix_torch = self._sparse_to_torch(self.adj_matrix)
            self.adj_matrix_normalized_torch = self._sparse_to_torch(self.adj_matrix_normalized)
        else:
            self.adj_matrix_torch = torch.from_numpy(self.adj_matrix)
            self.adj_matrix_normalized_torch = torch.from_numpy(self.adj_matrix_normalized)

        if issparse(self.adj_matrix):
            n_neighbors = np.array(self.adj_matrix.sum(axis=1)).flatten()
        else:
            n_neighbors = self.adj_matrix.sum(axis=1)
        self.n_neighbors = torch.from_numpy(n_neighbors).float().unsqueeze(-1)

        # Initialize storage
        self.v = torch.zeros(n_samples, T, v_dim)
        self.x = torch.zeros(n_samples, T, 1)
        self.y = torch.zeros(n_samples, T, 1)
        self.s_vs = torch.zeros(n_samples, T, v_dim)
        self.d_xs = torch.zeros(n_samples, T, 1)  # continuous spillover proportion XS in [0,1]
        self.d_xs_count = torch.zeros(n_samples, T, 1)  # number of treated neighbours
        self.s_dxs = torch.zeros(n_samples, T, 1)  # mean_j D_{j,t}: neighbour-averaged spillover exposure (second hop)
        self.q_ys = torch.zeros(n_samples, T, 1)
        self.c_latent = torch.zeros(n_samples, T, 1)
        self.h_x_state = torch.zeros(n_samples, T, 1)

        self.v_0_baseline = None
        self.y_0_baseline = None
        self.ys_0_baseline = None
        self.z_latents = None
        self.u_zs = None

        # Outcome-effect family (fixed function of baseline covariates only)
        self.family_labels = None            # (n_samples,) object array, values in FAMILY_LABELS
        self.family_multiplier = None        # (n_samples,) m_{F_i}
        self.baseline_risk_score = None      # (n_samples,) standardized R_i^0
        self.decision_steps = []

        # Neighbour-aggregated LSTM recurrent states h^s_tau = A_norm @ h_tau,
        # c^s_tau = A_norm @ c_tau. Updated once per training epoch by
        # train.update_neighbor_hidden_states(); consumed by g_Z.
        self.h_s_seq = None
        self.c_s_seq = None

        print("\nGenerating data...")
        self._generate_data()

        # Default kernel bandwidth; the pipelines override it with --bandwidth.
        self.spillover_bandwidth = get_bandwidth_from_network(self.adj_matrix)

        print(f"\n Spillover proportions D_i:")
        proportions_np = self.d_xs.numpy().flatten()
        unique_props = np.unique(proportions_np)
        print(f"  Unique proportion values: {len(unique_props)}")
        print(f"  Range: [{proportions_np.min():.4f}, {proportions_np.max():.4f}]")
        print(f"  Mean: {proportions_np.mean():.4f}, Std: {proportions_np.std():.4f}")

        self._cache_decision_pattern_summary()

        print("\nDataset generation complete!")

    def _cache_decision_pattern_summary(self):
        """Encode each unit's own-treatment pattern over the decision steps
        as a K-bit integer and cache per-pattern counts."""
        update_interval = int(self.config.get('treatment_update_interval', 1))
        self.decision_steps = list(range(0, self.T, update_interval))
        if not self.decision_steps:
            self.decision_pattern_ids = np.zeros(self.n_samples, dtype=np.int64)
            self.decision_pattern_labels = {0: "0"}
            self.decision_pattern_counts = {0: self.n_samples}
            return

        x_decision = self.x[:, self.decision_steps, 0].cpu().numpy().astype(np.int64)
        K = len(self.decision_steps)
        bit_weights = 1 << np.arange(K - 1, -1, -1, dtype=np.int64)
        pattern_ids = x_decision @ bit_weights
        unique_ids, counts = np.unique(pattern_ids, return_counts=True)

        self.decision_pattern_ids = pattern_ids.astype(np.int64)
        self.decision_pattern_labels = {
            int(pid): format(int(pid), f'0{K}b')
            for pid in range(2 ** K)
        }
        self.decision_pattern_counts = {
            int(pid): int(count)
            for pid, count in zip(unique_ids, counts)
        }

        print("\n Decision-time treatment pattern distribution:")
        for pid in range(2 ** K):
            count = self.decision_pattern_counts.get(pid, 0)
            pct = 100.0 * count / self.n_samples
            print(f"    {self.decision_pattern_labels[pid]}: {count:5d} ({pct:4.1f}%)")

    def _sparse_to_torch(self, sparse_matrix):
        """Convert a scipy sparse matrix to a float32 torch sparse COO tensor."""
        sparse_coo = sparse_matrix.tocoo()
        indices = torch.LongTensor(np.vstack([sparse_coo.row, sparse_coo.col]))
        values = torch.FloatTensor(sparse_coo.data)
        shape = sparse_coo.shape
        return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)

    def _generate_data(self):
        """Simulate baseline (t=0) and process times t = 1..T in DGP order:
        treatment, spillover exposure, latent activity and outcome, then
        covariates for the next step."""

        print("\n" + "="*80)
        print("Step-by-Step Generation")
        print("="*80)

        seed_iter = iter(self._dgp_seeds)
        baseline_v_seed = next(seed_iter)
        baseline_y_seed = next(seed_iter)
        update_interval = int(self.config.get('treatment_update_interval', 1))
        if update_interval < 1:
            raise ValueError(f"treatment_update_interval must be >= 1, got {update_interval}")
        decision_times = set(range(1, self.T + 1, update_interval))

        weights = get_burden_weights(self.config, self.v_dim)

        # ====================================================================
        # BASELINE (t=0)
        # ====================================================================
        print("\n[Baseline t=0]")
        print("  Generating V0 and V0^S...")

        V0, VS0 = generate_baseline_covariates_with_smoothing(
            self.n_samples,
            self.v_dim,
            self.adj_matrix_normalized,
            params=self.config,
            seed=baseline_v_seed,
        )

        self.v[:, 0, :] = torch.from_numpy(V0)
        self.s_vs[:, 0, :] = torch.from_numpy(VS0)

        print(f"   V0: mean={V0.mean():.4f}, std={V0.std():.4f}")
        print(f"   VS0: mean={VS0.mean():.4f}, std={VS0.std():.4f}")

        v_bar_0 = compute_burden(V0, weights)

        # Baseline outcome-effect family: depends on baseline covariates only
        # and does not enter treatment assignment.
        self.family_labels = assign_baseline_families(v_bar_0)
        family_multiplier_map = get_family_multipliers(self.config)
        self.family_multiplier = np.array(
            [family_multiplier_map[label] for label in self.family_labels], dtype=np.float64
        )

        print("  Baseline outcome-effect family distribution:")
        for label in FAMILY_LABELS:
            count = int((self.family_labels == label).sum())
            pct = 100.0 * count / self.n_samples
            print(f"    {label:>12s}: {count:5d} ({pct:4.1f}%), m_F={family_multiplier_map[label]:.2f}")

        # Generate baseline outcome Y0
        print("  Generating Y0 ~ N(mu_Y0, sigma^2)...")
        Y0 = generate_baseline_outcome(V0, VS0, self.config, seed=baseline_y_seed)

        YS0_result = self.adj_matrix_normalized @ Y0
        if issparse(YS0_result):
            YS0 = np.array(YS0_result.todense()).flatten()
        else:
            YS0 = np.asarray(YS0_result).flatten()

        self.v_0_baseline = V0
        self.y_0_baseline = Y0
        self.ys_0_baseline = YS0

        print(f"   Y0: mean={Y0.mean():.4f}, std={Y0.std():.4f}")
        print(f"   YS0: mean={YS0.mean():.4f}, std={YS0.std():.4f}")

        # Standardized baseline risk score R_i^0, fixed at baseline.
        self.baseline_risk_score = compute_baseline_risk_score(Y0, v_bar_0, YS0, self.config)
        print(f"   R_i^0 (standardized): mean={self.baseline_risk_score.mean():+.4f}, std={self.baseline_risk_score.std():.4f}")

        # ====================================================================
        # TEMPORAL LOOP: t = 1, ..., T
        # ====================================================================

        V_history = [V0]
        Y_history = [Y0]

        prev_u = v_bar_0
        x_prev = np.zeros(self.n_samples, dtype=np.float32)
        h_x_prev = np.zeros(self.n_samples, dtype=np.float64)
        h_xs_prev = np.zeros(self.n_samples, dtype=np.float64)
        c_prev = Y0.astype(np.float64)

        rho_x = self.config.get('history_rho_X', 0.63)
        W_norm = self.adj_matrix_normalized

        for t in range(1, self.T + 1):
            treatment_seed = next(seed_iter)
            latent_seed = next(seed_iter)
            outcome_seed = next(seed_iter)
            covariate_seed = next(seed_iter)
            print(f"\n[Time t={t}]")

            V_curr = V_history[t - 1]  # V_{t-1}
            u_curr = prev_u if t == 1 else compute_burden(V_curr, weights)
            m_prev = compute_population_expected_burden(t - 1, self.config, self.v_dim)

            # ----------------------------------------------------------------
            # TREATMENT GENERATION
            # ----------------------------------------------------------------
            is_decision_time = t in decision_times
            if is_decision_time:
                print(f"  Decision time: generating X({t}) via treatment-assignment model...")
                X_t = generate_treatment_logistic(
                    U_prev=u_curr,
                    R0=self.baseline_risk_score,
                    params=self.config,
                    x_prev=x_prev,
                    y_prev=Y_history[t-1],
                    y_prev_step=Y_history[t-2] if t >= 2 else None,
                    seed=treatment_seed
                )
            else:
                print(f"  Non-decision time: carrying forward X({t-1}) into X({t})...")
                X_t = x_prev.copy()

            self.x[:, t-1, 0] = torch.from_numpy(X_t)
            treatment_rate = X_t.mean()
            print(f"   X({t}): treatment rate={treatment_rate:.3f}")

            # ----------------------------------------------------------------
            # SPILLOVER TREATMENT (deterministic proportion)
            # ----------------------------------------------------------------
            treated_counts = self.adj_matrix @ X_t
            if issparse(treated_counts):
                treated_counts = np.array(treated_counts.todense()).flatten()
            else:
                treated_counts = np.asarray(treated_counts).flatten()
            self.d_xs_count[:, t-1, 0] = torch.from_numpy(treated_counts.astype(np.float32))

            XS_t = generate_spillover_treatment_proportion(X_t, W_norm)
            self.d_xs[:, t-1, 0] = torch.from_numpy(XS_t)
            print(f"   XS({t}): mean={XS_t.mean():.4f}, std={XS_t.std():.4f}, range=[{XS_t.min():.4f}, {XS_t.max():.4f}]")

            # Second-hop neighbor-averaged spillover exposure: mean_j D_{j,t}
            SXS_t_result = W_norm @ XS_t
            if issparse(SXS_t_result):
                SXS_t = np.array(SXS_t_result.todense()).flatten()
            else:
                SXS_t = np.asarray(SXS_t_result).flatten()
            self.s_dxs[:, t-1, 0] = torch.from_numpy(SXS_t.astype(np.float32))

            h_x_curr = rho_x * h_x_prev + X_t
            self.h_x_state[:, t-1, 0] = torch.from_numpy(h_x_curr.astype(np.float32))

            # Discounted spillover-exposure history H^XS_it = rho_X H^XS_{i,t-1} + D_it
            h_xs_curr = rho_x * h_xs_prev + XS_t

            # ----------------------------------------------------------------
            # OUTCOME GENERATION (latent disease activity -> observed outcome)
            # ----------------------------------------------------------------
            print(f"  Generating Y({t}) through latent disease activity C({t})...")
            # Neighbour-averaged covariate burden U^S_{t-1}
            vs_curr = self.s_vs[:, t-1, :].numpy()
            u_s_curr = compute_burden(vs_curr, weights)

            C_t = generate_latent_disease_activity(
                U_prev=u_curr,
                m_prev=m_prev,
                treatment_history_state=h_x_curr,
                family_multiplier=self.family_multiplier,
                params=self.config,
                latent_prev=c_prev,
                U_S_prev=u_s_curr,
                spillover_treatment_history_state=h_xs_curr,
                seed=latent_seed,
            )
            Y_t = generate_outcome(
                C_t,
                y_prev=Y_history[t-1],
                params=self.config,
                seed=outcome_seed,
            )

            self.y[:, t-1, 0] = torch.from_numpy(Y_t)
            self.c_latent[:, t-1, 0] = torch.from_numpy(C_t)
            Y_history.append(Y_t)

            print(f"   Y({t}): mean={Y_t.mean():.4f}, std={Y_t.std():.4f}")

            YS_t_result = W_norm @ Y_t
            if issparse(YS_t_result):
                YS_t = np.array(YS_t_result.todense()).flatten()
            else:
                YS_t = np.asarray(YS_t_result).flatten()
            self.q_ys[:, t-1, 0] = torch.from_numpy(YS_t)

            # ----------------------------------------------------------------
            # COVARIATE EVOLUTION
            # ----------------------------------------------------------------
            print(f"  Evolving covariates V({t})...")
            # Neighbor-averaged outcome Y^S_{t-1}: baseline at t=1, else the
            # value stored one iteration back (q_ys[:, t-2, :] holds Y^S at
            # process time t-1 once iteration t-1 has completed).
            ys_prev_for_cov = self.ys_0_baseline if t == 1 else self.q_ys[:, t-2, 0].numpy()
            V_next = evolve_covariates(
                V_curr,
                self.config,
                Y_prev=Y_history[t-1],
                YS_prev=ys_prev_for_cov,
                seed=covariate_seed,
            )

            VS_next_result = W_norm @ V_next
            if issparse(VS_next_result):
                VS_next = np.array(VS_next_result.todense())
            else:
                VS_next = np.asarray(VS_next_result)

            V_history.append(V_next)

            if t < self.T:
                self.v[:, t, :] = torch.from_numpy(V_next)
                self.s_vs[:, t, :] = torch.from_numpy(VS_next)

            print(f"   V({t}): mean={V_next.mean():.4f}, std={V_next.std():.4f}")

            prev_u = u_curr
            x_prev = X_t
            h_x_prev = h_x_curr
            h_xs_prev = h_xs_curr
            c_prev = C_t.astype(np.float64)

        print("\n" + "="*80)
        print(" Data generation complete!")
        print("="*80)

    def update_latents(self, z_latents: torch.Tensor, device: torch.device = None):
        """Store latent representations and their neighbour averages
        u_zs[:, t] = A_norm @ z[:, t].

        Args:
            z_latents: (n_samples, T, z_total_dim) latent representations.
            device: if CUDA, the products run on the GPU; results are always
                stored on CPU.
        """
        self.z_latents = z_latents.cpu()

        z_total_dim = z_latents.shape[2]
        use_gpu = device is not None and device.type == 'cuda'

        if use_gpu:
            if not hasattr(self, '_adj_norm_gpu') or self._adj_norm_gpu is None:
                self._adj_norm_gpu = self.adj_matrix_normalized_torch.to(device)
                print(f"  [GPU] Adjacency matrix cached on {device}")

            adj = self._adj_norm_gpu
            mm_fn = torch.sparse.mm if adj.is_sparse else torch.mm

            z_gpu = self.z_latents.to(device)
            u_zs_gpu = torch.zeros_like(z_gpu)

            for t in range(self.T):
                u_zs_gpu[:, t, :] = mm_fn(adj, z_gpu[:, t, :])

            self.u_zs = u_zs_gpu.cpu()
        else:
            adj = self.adj_matrix_normalized_torch
            mm_fn = torch.sparse.mm if adj.is_sparse else torch.mm

            self.u_zs = torch.zeros(self.n_samples, self.T, z_total_dim)
            for t in range(self.T):
                self.u_zs[:, t, :] = mm_fn(adj, self.z_latents[:, t, :])

    def __len__(self):
        """Number of units."""
        return self.n_samples

    def __getitem__(self, idx):
        """Return the per-unit sequences for an index or slice as a dict,
        plus neighbour aggregates when they have been computed."""
        if isinstance(idx, slice):
            y_0_slice = self.y_0_baseline[idx]
            ys_0_slice = self.ys_0_baseline[idx] if self.ys_0_baseline is not None else np.zeros(len(range(*idx.indices(self.n_samples))))
        else:
            y_0_slice = self.y_0_baseline[idx:idx+1]
            ys_0_slice = self.ys_0_baseline[idx:idx+1] if self.ys_0_baseline is not None else np.zeros(1)

        item = {
            'v': self.v[idx],
            'x': self.x[idx],
            'y': self.y[idx],
            's_vs': self.s_vs[idx],
            'd_xs': self.d_xs[idx],
            'd_xs_count': self.d_xs_count[idx],
            's_dxs': self.s_dxs[idx],
            'n_neighbors': self.n_neighbors[idx],
            'q_ys': self.q_ys[idx],
            'y_0': torch.from_numpy(y_0_slice).float(),
            'ys_0': torch.from_numpy(ys_0_slice).float(),
        }

        if self.u_zs is not None:
            item['u_zs'] = self.u_zs[idx]

        if self.h_s_seq is not None and self.c_s_seq is not None:
            item['h_s'] = self.h_s_seq[idx]
            item['c_s'] = self.c_s_seq[idx]

        if isinstance(idx, slice):
            item['idx'] = torch.arange(*idx.indices(self.n_samples), dtype=torch.long)
        else:
            item['idx'] = torch.tensor(idx, dtype=torch.long)

        return item

    @classmethod
    def from_unit_resample(
        cls,
        source: "NetworkTemporalCausalDataset",
        unit_indices: np.ndarray,
        adj_matrix,
        seed: int = 0,
    ) -> "NetworkTemporalCausalDataset":
        """
        Build a resampled dataset for the network-block bootstrap.

        Each resampled unit keeps its observed record from `source` unchanged,
        including its realized exposure (d_xs, d_xs_count, n_neighbors,
        s_vs, ...). Only the unit set and the adjacency used for neighbour
        aggregation during retraining change.

        Args:
            unit_indices: (n_boot,) indices into `source`, with repeats.
            adj_matrix: (n_boot, n_boot) adjacency among the resampled units,
                built by the caller. It is not source.adj_matrix[idx][:, idx],
                which would connect repeated block copies through the source
                graph's cross-block edges.
            seed: stored as the dataset seed.
        """

        idx = np.asarray(unit_indices, dtype=np.int64)
        n_boot = len(idx)

        obj = cls.__new__(cls)
        obj.seed = int(seed)
        obj.n_samples = n_boot
        obj.v_dim = source.v_dim
        obj.T = source.T
        obj.config = source.config
        obj.stratified = source.stratified
        obj.min_samples_per_regime = source.min_samples_per_regime
        obj._dgp_seeds = None

        obj.adj_matrix = adj_matrix
        obj.adj_matrix_normalized = normalize_adjacency_matrix(adj_matrix)
        if issparse(adj_matrix):
            obj.adj_matrix_torch = obj._sparse_to_torch(adj_matrix)
            obj.adj_matrix_normalized_torch = obj._sparse_to_torch(obj.adj_matrix_normalized)
        else:
            obj.adj_matrix_torch = torch.from_numpy(np.asarray(adj_matrix))
            obj.adj_matrix_normalized_torch = torch.from_numpy(obj.adj_matrix_normalized)

        # n_neighbors is part of the observed record, like d_xs_count, and is
        # not rederived from adj_matrix: block resampling drops cross-block
        # edges, so a rederived degree could fall below d_xs_count. The
        # resampled graph drives neighbour aggregation through
        # adj_matrix_normalized_torch.
        obj.n_neighbors = source.n_neighbors[idx]

        obj.v = source.v[idx]
        obj.x = source.x[idx]
        obj.y = source.y[idx]
        obj.s_vs = source.s_vs[idx]
        obj.d_xs = source.d_xs[idx]
        obj.d_xs_count = source.d_xs_count[idx]
        if n_boot and bool((obj.d_xs_count[:, :, 0] > obj.n_neighbors + 1e-6).any()):
            raise ValueError(
                "resampled d_xs_count exceeds n_neighbors; the ZOIB-Binomial "
                "support would be invalid"
            )
        obj.s_dxs = source.s_dxs[idx]
        obj.q_ys = source.q_ys[idx]
        obj.c_latent = source.c_latent[idx]
        obj.h_x_state = source.h_x_state[idx]

        obj.v_0_baseline = source.v_0_baseline[idx]
        obj.y_0_baseline = source.y_0_baseline[idx]
        obj.ys_0_baseline = (
            source.ys_0_baseline[idx] if source.ys_0_baseline is not None else None
        )
        obj.z_latents = None
        obj.u_zs = None

        obj.family_labels = source.family_labels[idx]
        obj.family_multiplier = source.family_multiplier[idx]
        obj.baseline_risk_score = source.baseline_risk_score[idx]

        obj.h_s_seq = None
        obj.c_s_seq = None

        obj.spillover_bandwidth = get_bandwidth_from_network(obj.adj_matrix)

        obj._cache_decision_pattern_summary()
        return obj

    def get_true_effects(self):
        """Return the main DGP parameters (with defaults) as a dict."""
        return {
            'treatment_theta_0': self.config.get('treatment_theta_0', -3.85),
            'treatment_theta_R': self.config.get('treatment_theta_R', 1.00),
            'treatment_theta_prev': self.config.get('treatment_theta_prev', 0.28),
            'treatment_theta_w': self.config.get('treatment_theta_w', 0.19),
            'treatment_theta_i': self.config.get('treatment_theta_i', 0.08),
            'treatment_theta_v': self.config.get('treatment_theta_v', 0.06),
            'treatment_update_interval': int(self.config.get('treatment_update_interval', 1)),
            'family_multipliers': get_family_multipliers(self.config),
            'latent_mu_C': self.config.get('latent_mu_C', 72.0),
            'latent_rho_C': self.config.get('latent_rho_C', 0.65),
            'latent_lambda_1': self.config.get('latent_lambda_1', 5.2),
            'latent_beta_X': self.config.get('latent_beta_X', 8.25),
            'outcome_kappa_Y': self.config.get('outcome_kappa_Y', 0.28),
        }
