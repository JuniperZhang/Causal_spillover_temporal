"""
Decision-time representation g_Z and assignment models f_X, f_D (paper Section 3.2).

At a decision time τ the unit's recurrent states (h, c) and the
neighbour-averaged states (h^s, c^s) are mapped to three latents:

    (Z^X_{i,τ}, Z^D_{i,τ}, Z^C_{i,τ}) = g_Z(h_{i,τ}, c_{i,τ}, h^s_{i,τ}, c^s_{i,τ})

    Z^X : own-treatment latent    (z_x_dim)
    Z^D : spillover latent        (z_d_dim)
    Z^C : shared latent           (z_c_dim)

h^s, c^s average h_j, c_j over first-order neighbours j; they are computed
once per epoch (train.update_neighbor_hidden_states).

Own-treatment head f_X:
    ê^X_{i,τ+1} = σ(f_X(Z^X_{i,τ}, Z^C_{i,τ})) = P(X_{i,τ+1} = 1 | history)
    L_X = mean BCE(ê^X_{i,τ+1}, X^obs_{i,τ+1})

Spillover head f_D:
    f_D(Z^D_{i,τ}, Z^C_{i,τ}, X_{i,τ+1}) -> 4 ZOIB-Binomial parameters for
    the treated-neighbour count D_{i,τ+1} in {0, ..., n_i}
    L_D = −mean log p_ZOIB(D^obs_{i,τ+1} | n_i)
    Z^X is excluded; X_{i,τ+1} is the observed treatment (teacher forcing).
"""

import torch
import torch.nn as nn


class DecisionLatentHead_gZ(nn.Module):
    """
    g_Z: [h_tau, c_tau, h^s_tau, c^s_tau] -> (Z^X_tau, Z^D_tau, Z^C_tau)

    Maps the unit's recurrent states and the neighbour-averaged states to
    three decision-time latents. Applied only at decision times.

    Architecture:
        [h, c, h^s, c^s]  ->  shared_hidden (ReLU + Dropout)
                          ->  proj_x  ->  Z^X  (ReLU)
                          ->  proj_d  ->  Z^D  (ReLU)
                          ->  proj_c  ->  Z^C  (ReLU)
    """

    def __init__(self, hidden_dim: int,
                 z_x_dim: int, z_d_dim: int, z_c_dim: int,
                 dropout: float = 0.2):
        """hidden_dim: LSTM state size; z_*_dim: latent sizes."""
        super().__init__()
        inp        = 4 * hidden_dim
        shared_dim = max((z_x_dim + z_d_dim + z_c_dim) * 2, 64)

        self.shared = nn.Sequential(
            nn.Linear(inp, shared_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.proj_x = nn.Sequential(nn.Linear(shared_dim, z_x_dim), nn.ReLU())
        self.proj_d = nn.Sequential(nn.Linear(shared_dim, z_d_dim), nn.ReLU())
        self.proj_c = nn.Sequential(nn.Linear(shared_dim, z_c_dim), nn.ReLU())

    def forward(self,
                h_t:    torch.Tensor,
                c_t:    torch.Tensor,
                h_s_t:  torch.Tensor,
                c_s_t:  torch.Tensor):
        """
        Args
        ----
        h_t   : (batch, hidden_dim) — focal-unit hidden state h_tau
        c_t   : (batch, hidden_dim) — focal-unit cell state c_tau
        h_s_t : (batch, hidden_dim) — neighbour-averaged hidden state h^s_tau
        c_s_t : (batch, hidden_dim) — neighbour-averaged cell state c^s_tau

        Returns
        -------
        z_x : (batch, z_x_dim)
        z_d : (batch, z_d_dim)
        z_c : (batch, z_c_dim)
        """
        s = self.shared(torch.cat([h_t, c_t, h_s_t, c_s_t], dim=1))
        return self.proj_x(s), self.proj_d(s), self.proj_c(s)


class OwnTreatmentHead_fX(nn.Module):
    """
    f_X: (Z^X_{i,τ}, Z^C_{i,τ}) → own-treatment logit

    ê^X_{i,τ+1} = σ(f_X(Z^X, Z^C)) = P(X_{i,τ+1}=1 | Z^X_{i,τ}, Z^C_{i,τ})

    Architecture: [Z^X, Z^C] → hidden (ReLU + Dropout) → 1
    """

    def __init__(self, z_x_dim: int, z_c_dim: int, dropout: float = 0.2):
        """z_x_dim, z_c_dim: sizes of Z^X and Z^C."""
        super().__init__()
        inp    = z_x_dim + z_c_dim
        hidden = max(inp, 32)
        self.net = nn.Sequential(
            nn.Linear(inp, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self,
                z_x: torch.Tensor,
                z_c: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        z_x : (batch, z_x_dim)
        z_c : (batch, z_c_dim)

        Returns
        -------
        logit : (batch, 1) — raw logit; apply σ for probability
        """
        return self.net(torch.cat([z_x, z_c], dim=1))


class SpilloverHead_fD(nn.Module):
    """
    f_D: (Z^D_{i,τ}, Z^C_{i,τ}, X_{i,τ+1}) → ZOIB-Binomial parameters for D_{i,τ+1}

    Z^X is excluded by design; X_{i,τ+1} is the observed own treatment
    (teacher forcing). The model uses output_dim = 4:
    (logit_0, logit_1, logit_interior, q_logit), see
    ContinuousLSTMCausalModel._zoib_binomial_loss.

    Architecture: [Z^D, Z^C, X_next] → hidden (ReLU + Dropout) → output_dim
    """

    def __init__(self, z_d_dim: int, z_c_dim: int,
                 dropout: float = 0.2, output_dim: int = 1):
        """z_d_dim, z_c_dim: sizes of Z^D and Z^C; output_dim: 4 in the model."""
        super().__init__()
        self.output_dim = output_dim
        inp    = z_d_dim + z_c_dim + 1   # +1 for X_{t+1}
        hidden = max(inp, 32)
        self.net = nn.Sequential(
            nn.Linear(inp, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, output_dim),
        )

    def forward(self,
                z_d:    torch.Tensor,
                z_c:    torch.Tensor,
                x_next: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        z_d    : (batch, z_d_dim)
        z_c    : (batch, z_c_dim)
        x_next : (batch, 1) — own treatment X_{i,τ+1}

        Returns
        -------
        out    : (batch, output_dim) — raw ZOIB-Binomial parameters
        """
        return self.net(torch.cat([z_d, z_c, x_next], dim=1))

