"""
Two-timescale LSTM propensity model with own-treatment and spillover heads
(paper Sections 3.1-3.3).

Time scales
    Process times:   t = 1, ..., T
    Decision times:  every L-th process time (0-based step j with j % L == 0).
    Covariates and outcomes evolve at every process time; treatments are
    assigned only at decision times and carried forward in between.

Recurrent input (Section 3.1), GraphSAGE mean aggregator:
    O_{i,t}       = (V_{i,t-1}, Y_{i,t-1}, X_{i,t-1}, D_{i,t-1})
    mean_j O_{j,t} = (V^s_{i,t-1}, Y^s_{i,t-1}, D_{i,t-1}, D^s_{i,t-1})
    I_{i,t}       = ReLU(W_own O_{i,t} + W_nbr mean_j O_{j,t})

LSTM backbone, shared across units and process times:
    (h_{i,t}, c_{i,t}) = LSTM_θ(I_{i,t}, h_{i,t-1}, c_{i,t-1})
    h_{i,t} uses only information up to t-1; it never contains V_{i,t}.

Reconstruction loss L_I (Section 3.3, every process time):
    Î_{i,t} = reconstruction_map(h_{i,t}, c_{i,t})
    L_I = (1/(nT Σ_r a_r)) Σ_i Σ_t Σ_r a_r ρ((Î_{i,t,r} − I_{i,t,r}) / σ_r)
    ρ is the Huber loss (δ = 1); σ_r is the batch std of component r of I.

Decision-time latents g_Z (Section 3.2):
    (Z^X, Z^D, Z^C)_{i,τ} = g_Z(h_{i,τ}, c_{i,τ}, h^s_{i,τ}, c^s_{i,τ})
    h^s, c^s are the states averaged over first-order neighbours. They are
    recomputed once per epoch (train.update_neighbor_hidden_states) and
    passed in as batch['h_s'], batch['c_s'], so no gradient flows through
    the neighbour average.

Own-treatment head f_X:
    ê^X_{i,τ+1} = σ(f_X(Z^X_{i,τ}, Z^C_{i,τ})),   L_X = mean BCE

Spillover head f_D (teacher forcing on the observed X_{i,τ+1}):
    f_D(Z^D_{i,τ}, Z^C_{i,τ}, X_{i,τ+1}) -> 4 ZOIB-Binomial parameters
    L_D = mean ZOIB-Binomial NLL of D = number of treated neighbours in
    {0, ..., n_i}: point masses at 0 and n_i plus a truncated
    Binomial(n_i, q) on {1, ..., n_i − 1}.
    Z^X is excluded from f_D.

Training loss (Section 3.3):
    L_train = w_X L_X + w_D L_D + w_I L_I + λ ‖Θ‖²

Batch indexing (0-based step j corresponds to process time t = j+1):
    batch['v']     [:, j] = V_j          (V_0 is the baseline covariate)
    batch['s_vs']  [:, j] = V^s_j
    batch['x']     [:, j] = X_{j+1}      (assigned at step j, active from j+1)
    batch['d_xs']  [:, j] = D_{j+1}      (treated-neighbour proportion)
    batch['s_dxs'] [:, j] = D^s_{j+1}    (neighbour average of D)
    batch['y']     [:, j] = Y_{j+1}
    batch['q_ys']  [:, j] = Y^s_{j+1}
    batch['y_0'], batch['ys_0'] = Y_0, Y^s_0   (batch, 1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from .encoder import LSTMBackbone, GraphSAGEMeanAggregator
from .treatment_model import DecisionLatentHead_gZ, OwnTreatmentHead_fX, SpilloverHead_fD
from .reconstruction import ReconstructionHead_I


class ContinuousLSTMCausalModel(nn.Module):
    """
    Two-timescale LSTM propensity model (see module docstring).

    Components: GraphSAGE input I_t, shared LSTM backbone, reconstruction
    head for L_I, decision-time latents g_Z -> (Z^X, Z^D, Z^C), own-treatment
    head f_X(Z^X, Z^C) and spillover head f_D(Z^D, Z^C, X_next).

    config keys used: T, v_dim, lstm_hidden_dim, lstm_num_layers, z_x_dim,
    z_d_dim, z_c_dim, dropout, graphsage_dim, use_reconstruction_head,
    treatment_update_interval (L), loss weights and l2_lambda.
    """

    def __init__(self, config: Dict):
        """Build all sub-networks from the config dict."""
        super().__init__()
        self.config = config
        self.T = config['T']

        v_dim      = config['v_dim']
        hidden_dim = config.get('lstm_hidden_dim', 128)
        num_layers = config.get('lstm_num_layers', 2)
        z_x_dim    = config.get('z_x_dim', 16)
        z_d_dim    = config.get('z_d_dim', 16)
        z_c_dim    = config.get('z_c_dim', 32)
        dropout    = config.get('dropout', 0.1)

        self.v_dim   = v_dim
        own_dim      = v_dim + 3   # O_{i,t} = (V_{t-1}, Y_{t-1}, X_{t-1}, D_{t-1})
        nbr_dim      = v_dim + 3   # mean_j O_{j,t} = (V^s, Y^s, D, D^s)
        graphsage_dim = config.get('graphsage_dim', 128)
        self.own_dim   = own_dim
        self.nbr_dim   = nbr_dim
        self.input_dim = graphsage_dim   # I_{i,t} dimension (post-GraphSAGE)

        # ── Sub-networks ──────────────────────────────────────────────────
        self.graphsage = GraphSAGEMeanAggregator(own_dim, nbr_dim, graphsage_dim)
        self.lstm_backbone = LSTMBackbone(self.input_dim, hidden_dim, num_layers, dropout)

        # f_D outputs the ZOIB-Binomial parameters (logit_0, logit_1, logit_interior, q_logit).
        d_out_dim = 4

        self.gZ_head = DecisionLatentHead_gZ(hidden_dim,
                                             z_x_dim, z_d_dim, z_c_dim, dropout)
        self.fX_head = OwnTreatmentHead_fX(z_x_dim, z_c_dim, dropout)
        self.fD_head = SpilloverHead_fD(z_d_dim, z_c_dim, dropout,
                                        output_dim=d_out_dim)

        self.use_reconstruction_head = config.get('use_reconstruction_head', True)
        if self.use_reconstruction_head:
            self.recon_head = ReconstructionHead_I(hidden_dim, self.input_dim, dropout)

        # Decision-time mask (0-based step j is a decision time iff j % L == 0)
        L = int(config.get('treatment_update_interval', 1))
        self._L = L
        self.decision_mask: List[bool] = [(j % L == 0) for j in range(self.T)]

        self.apply(self._init_weights)

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _init_weights(module):
        """Kaiming-normal init for Linear weights, zero biases."""
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in',
                                    nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def _build_input_sequence(self, batch: Dict) -> torch.Tensor:
        """
        Build the GraphSAGE input sequence I_{i,t} for t = 1..T (Section 3.1):

            O_{i,t}        = (V_{i,t-1}, Y_{i,t-1}, X_{i,t-1}, D_{i,t-1})
            mean_j O_{j,t} = (V^s_{i,t-1}, Y^s_{i,t-1}, D_{i,t-1}, D^s_{i,t-1})
            I_{i,t}        = ReLU(W_own O_{i,t} + W_nbr mean_j O_{j,t})

        The neighbour mean of X is D itself; the neighbour mean of D is the
        precomputed batch['s_dxs']. At t = 1 the treatment entries are zero
        and the outcomes are the baselines Y_0, Y^s_0.

        Returns
        -------
        inputs : (batch, T, input_dim)  -- input_dim = graphsage_dim
        """
        dev        = batch['v'].device
        batch_size = batch['v'].size(0)
        T          = self.T
        inputs_list = []

        for j in range(T):
            v_prev  = batch['v'][:, j, :]
            vs_prev = batch['s_vs'][:, j, :]

            if j == 0:
                y_prev   = batch['y_0']
                ys_prev  = batch['ys_0']
                x_prev   = torch.zeros(batch_size, 1, device=dev)
                d_prev   = torch.zeros(batch_size, 1, device=dev)
                ds_prev  = torch.zeros(batch_size, 1, device=dev)
            else:
                y_prev   = batch['y'][:, j-1, :]
                ys_prev  = batch['q_ys'][:, j-1, :]
                x_prev   = batch['x'][:, j-1, :]
                d_prev   = batch['d_xs'][:, j-1, :]
                ds_prev  = batch['s_dxs'][:, j-1, :]

            o_own = torch.cat([v_prev, y_prev, x_prev, d_prev], dim=1)
            o_nbr = torch.cat([vs_prev, ys_prev, d_prev, ds_prev], dim=1)

            i_t = self.graphsage(o_own, o_nbr)
            inputs_list.append(i_t)

        return torch.stack(inputs_list, dim=1)   # (batch, T, input_dim)

    # ──────────────────────────────────────────────────────────────────────
    # Forward pass
    # ──────────────────────────────────────────────────────────────────────

    def forward(self, batch: Dict, mode: str = 'train') -> Dict:
        """
        Forward pass over all T process times.

        `mode` is accepted for interface compatibility and does not change
        the computation.

        Returns
        -------
        dict with keys:
            'x_logits'     : list[T] — (batch,1) at decision times, else None
            'd_logits'     : list[T] — (batch,4) at decision times, else None
            'i_hat_preds'  : list[T] — (batch, input_dim) or None
            'i_targets'    : (batch, T, input_dim) — I_{i,t}, i.e. `inputs` itself
            'decision_mask': list[T] booleans
        """
        inputs = self._build_input_sequence(batch)
        h_seq, c_seq = self.lstm_backbone(inputs)      # (batch, T, hidden_dim) each

        T           = self.T
        i_hat_preds = []
        x_logits    = []
        d_logits    = []

        for j in range(T):
            h_t = h_seq[:, j, :]

            # ── Reconstruction head for L_I (every step) ────────────────
            c_t = c_seq[:, j, :]
            i_hat_preds.append(
                self.recon_head(h_t, c_t) if self.use_reconstruction_head else None
            )

            # ── Decision-time heads ────────────────────────────────────
            if self.decision_mask[j]:
                # Neighbour-averaged states h^s, c^s from the per-epoch buffer;
                # the unit's own (h, c) is used if no buffer is present.
                if 'h_s' in batch and 'c_s' in batch:
                    h_s_t = batch['h_s'][:, j, :]
                    c_s_t = batch['c_s'][:, j, :]
                else:
                    h_s_t = h_t
                    c_s_t = c_t

                # Three-component decision latent: g_Z(h_tau, c_tau, h^s_tau, c^s_tau)
                z_x, z_d, z_c = self.gZ_head(h_t, c_t, h_s_t, c_s_t)

                # Own-treatment prediction: f_X(Z^X, Z^C)
                x_logit = self.fX_head(z_x, z_c)              # (batch, 1)

                # Spillover prediction: f_D(Z^D, Z^C, X_next)   [Z^X excluded]
                x_next  = batch['x'][:, j, :]                 # observed X_{j+1}
                d_logit = self.fD_head(z_d, z_c, x_next)      # (batch, 4)

                x_logits.append(x_logit)
                d_logits.append(d_logit)
            else:
                x_logits.append(None)
                d_logits.append(None)

        return {
            'i_hat_preds':   i_hat_preds,
            'i_targets':     inputs,
            'x_logits':      x_logits,
            'd_logits':      d_logits,
            'decision_mask': self.decision_mask,
        }

    # ──────────────────────────────────────────────────────────────────────
    # Hidden-state extraction
    # ──────────────────────────────────────────────────────────────────────

    def extract_hidden_states(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run the LSTM without gradients and return all hidden and cell states.

        Used by train.update_neighbor_hidden_states() to build the per-epoch
        neighbour-averaged h^s, c^s consumed by g_Z.

        Returns
        -------
        h_seq, c_seq : each (batch, T, hidden_dim)
        """
        with torch.no_grad():
            inputs = self._build_input_sequence(batch)
            h_seq, c_seq = self.lstm_backbone(inputs)
        return h_seq, c_seq

    @staticmethod
    def _binomial_log_pmf(d: torch.Tensor, n: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """log P(D=d | n, q) for D ~ Binomial(n, q), computed via lgamma."""
        eps = 1e-6
        q_c = q.clamp(eps, 1.0 - eps)
        log_binom_coef = (torch.lgamma(n + 1.0) - torch.lgamma(d + 1.0)
                          - torch.lgamma(n - d + 1.0))
        return log_binom_coef + d * torch.log(q_c) + (n - d) * torch.log(1.0 - q_c)

    @staticmethod
    def _interior_log_normalizer(n: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """
        log Z(n, q) with Z = 1 - Binomial(0; n, q) - Binomial(n; n, q), the
        Binomial mass on {1, ..., n-1}. Normalizes the truncated interior
        component Binomial(d; n, q) / Z of the ZOIB-Binomial.
        """
        eps = 1e-6
        q_c = q.clamp(eps, 1.0 - eps)
        log_p0 = n * torch.log(1.0 - q_c)
        log_pn = n * torch.log(q_c)
        Z = 1.0 - torch.exp(log_p0) - torch.exp(log_pn)
        return torch.log(Z.clamp(min=1e-10))

    def _zoib_binomial_loss(self,
                            d_logit:     torch.Tensor,
                            d_obs_count: torch.Tensor,
                            n_neighbors: torch.Tensor) -> torch.Tensor:
        """
        Zero-one inflated Binomial negative log-likelihood of the
        treated-neighbour count D in {0, ..., n_i}.

        Model:
            pi_0, pi_1, p_cont = softmax(out[:, 0:3])    mixture weights
            q                  = sigmoid(out[:, 3])       Binomial success prob

        NLL = -mean [
            1(D=0)         . log pi_0
          + 1(D=n_i)       . log pi_1
          + 1(0<D<n_i)     . (log p_cont + log Binomial(D; n_i, q) - log Z(n_i, q))
        ]
        where Z(n, q) renormalizes the Binomial to {1, ..., n-1}
        (see _interior_log_normalizer).

        Args
        ----
        d_logit     : (batch, 4)
        d_obs_count : (batch, 1)  D  (treated-neighbour count, float tensor)
        n_neighbors : (batch, 1)  n_i (number of neighbours, float)

        Returns
        -------
        scalar NLL
        """
        d = d_obs_count.squeeze(1)          # (batch,)
        n = n_neighbors.squeeze(1)          # (batch,)

        # ── Mixture weights ───────────────────────────────────────────────
        log_pi  = torch.log_softmax(d_logit[:, :3], dim=1)  # (batch, 3)
        log_pi0 = log_pi[:, 0]
        log_pi1 = log_pi[:, 1]
        log_pc  = log_pi[:, 2]

        # ── Interior Binomial success probability ─────────────────────────
        q = torch.sigmoid(d_logit[:, 3])
        n_safe = n.clamp(min=1.0)
        log_binom_pmf = self._binomial_log_pmf(d, n_safe, q)
        log_z_interior = self._interior_log_normalizer(n_safe, q)

        # ── Component masks on the discrete count ──────────────────────────
        is_zero = (d <= 0.5)
        is_full = (d >= n - 0.5) & (n > 0.5)
        is_interior = ~(is_zero | is_full)

        # ── Log-likelihood per unit ───────────────────────────────────────
        ll = (is_zero.float()     * log_pi0
            + is_full.float()     * log_pi1
            + is_interior.float() * (log_pc + log_binom_pmf - log_z_interior))

        return -ll.mean()

    def _compute_spillover_loss(self,
                                d_logit:     torch.Tensor,
                                d_obs:       torch.Tensor,
                                d_obs_count: Optional[torch.Tensor] = None,
                                n_neighbors: Optional[torch.Tensor] = None,
                                ) -> torch.Tensor:
        """Spillover loss ℓ_D: ZOIB-Binomial NLL of the treated-neighbour count.

        d_obs (the proportion) is not used; d_obs_count and n_neighbors are
        required.
        """
        if d_obs_count is None or n_neighbors is None:
            raise ValueError(
                "ZOIB-Binomial loss requires d_obs_count and n_neighbors. "
                "Pass batch['d_xs_count'] and batch['n_neighbors'].")
        return self._zoib_binomial_loss(d_logit, d_obs_count, n_neighbors)

    # ──────────────────────────────────────────────────────────────────────
    # Loss
    # ──────────────────────────────────────────────────────────────────────

    def _compute_reconstruction_loss(self, outputs: Dict) -> torch.Tensor:
        """
        L_I = (1/nT) Σ_i Σ_t Σ_r a_r · huber_1((Î_{i,t,r} − I_{i,t,r}) / σ_r)

        σ_r : std of component r of I over (batch, T), detached, floored at 1e-2.
        a_r : 1 by default; config 'reconstruction_component_weights'
              (length input_dim) overrides.

        The Huber penalty grows linearly beyond |u| = 1, so near-constant
        components cannot dominate the loss, while every component keeps a
        nonzero gradient.
        """
        i_hat = torch.stack(outputs['i_hat_preds'], dim=1)   # (batch, T, input_dim)
        # The target I is a GraphSAGE output; detach it so L_I cannot be
        # reduced by collapsing I toward a constant.
        i_true = outputs['i_targets'].detach()               # (batch, T, input_dim)

        sigma_r = i_true.reshape(-1, i_true.size(-1)).std(dim=0)
        sigma_r = sigma_r.clamp(min=1e-2)

        a_r_cfg = self.config.get('reconstruction_component_weights', None)
        if a_r_cfg is not None:
            a_r = torch.tensor(a_r_cfg, dtype=i_hat.dtype, device=i_hat.device)
        else:
            a_r = torch.ones(i_true.size(-1), dtype=i_hat.dtype, device=i_hat.device)

        z_hat = i_hat / sigma_r
        z_true = i_true / sigma_r
        huber = F.smooth_l1_loss(z_hat, z_true, reduction='none', beta=1.0)  # (batch, T, input_dim)

        denom = a_r.sum() * i_true.size(0) * i_true.size(1)
        return (a_r * huber).sum() / denom

    def compute_loss(self, outputs: Dict, batch: Dict) -> Dict:
        """
        L_train = w_X·L_X + w_D·L_D + w_I·L_I + λ‖W‖²

        L_X = w_X · (1/K) Σ_{j∈D}  BCE(x_logit_j, X^obs_{j+1})   (D: decision steps)
        L_D = w_D · (1/K) Σ_{j∈D}  ℓ_D(D̂_{j+1}, D^obs_{j+1})
              ℓ_D = ZOIB-Binomial NLL on the treated-neighbour count
        L_I = w_I · (1/(T·Σ_r a_r)) Σ_{t=1}^T Σ_r a_r ρ((Î_{t,r}−I_{t,r})/σ_r)
              ρ = Huber (u²/2 for |u|≤1, |u|−1/2 otherwise)

        All terms are means, so loss scales do not depend on T or K. NaN
        components are reported and replaced by zero.

        Returns a dict with total_loss, treatment_binary_loss (L_X),
        treatment_spillover_loss (L_D), reconstruction_loss (L_I), l2_loss.
        """
        x_actual    = batch['x']          # (batch, T, 1)
        d_actual    = batch['d_xs']       # (batch, T, 1)  proportion XS
        d_count     = batch.get('d_xs_count', None)  # (batch, T, 1)  integer count
        n_nbrs      = batch.get('n_neighbors', None)  # (batch, 1)
        dev         = x_actual.device

        bce = nn.BCEWithLogitsLoss()

        # ── L_X : mean over K decision times ─────────────────────────
        # L_X = w_X · (1/K) Σ_{j∈D} BCEWithLogits(x_logit_j, X^obs_{j+1})
        x_losses = [
            bce(outputs['x_logits'][j], x_actual[:, j, :])
            for j in range(self.T)
            if self.decision_mask[j] and outputs['x_logits'][j] is not None
        ]

        # ── L_D : mean over K decision times ─────────────────────────
        # L_D = w_D · (1/K) Σ_{j∈D} ℓ_D(D̂_{j+1}, D^obs_{j+1})
        # ℓ_D = ZOIB-Binomial NLL on the count D ∈ {0,...,n_i}.
        d_losses = [
            self._compute_spillover_loss(
                outputs['d_logits'][j],
                d_actual[:, j, :],
                d_count[:, j, :] if d_count is not None else None,
                n_nbrs,
            )
            for j in range(self.T)
            if self.decision_mask[j] and outputs['d_logits'][j] is not None
        ]

        # ── L_I : mean over T process times ────────────────────────────
        L_I_raw = (self._compute_reconstruction_loss(outputs)
                   if self.use_reconstruction_head else None)

        # ── Weights ───────────────────────────────────────────────────
        w_X = self.config.get('weight_treatment_binary', 1.0)
        w_D = self.config.get('weight_spillover', 1.0)
        w_I = self.config.get('weight_reconstruction', 1.0)

        zero = torch.tensor(0.0, device=dev)
        L_X = w_X * (torch.stack(x_losses).mean() if x_losses else zero)
        L_D = w_D * (torch.stack(d_losses).mean() if d_losses else zero)
        L_I = w_I * (L_I_raw if L_I_raw is not None else zero)

        # ── NaN guards ────────────────────────────────────────────────
        for name, L in [('treatment', L_X), ('spillover', L_D),
                        ('reconstruction', L_I)]:
            if torch.isnan(L).item():
                print(f"Warning: NaN in {name} loss, zeroed out")
        if torch.isnan(L_X): L_X = zero.clone()
        if torch.isnan(L_D): L_D = zero.clone()
        if torch.isnan(L_I): L_I = zero.clone()

        # ── L2 ────────────────────────────────────────────────────────
        l2 = sum(torch.norm(p, 2) ** 2 for p in self.parameters())
        l2_loss = self.config.get('l2_lambda', 1e-4) * l2

        total_loss = L_X + L_D + L_I + l2_loss
        if torch.isnan(total_loss).item():
            print("Warning: total_loss is NaN, falling back to l2_loss")
            total_loss = l2_loss.clone()

        return {
            'total_loss':               total_loss,
            'treatment_binary_loss':    L_X,
            'treatment_spillover_loss': L_D,
            'reconstruction_loss':      L_I,
            'l2_loss':                  l2_loss,
        }


# Alias used by the training code.
TemporalCausalModelSpillover = ContinuousLSTMCausalModel

