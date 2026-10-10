"""
Reconstruction map for the auxiliary loss L_I (paper Section 3.3).
"""

import torch
import torch.nn as nn


class ReconstructionHead_I(nn.Module):
    """
    Reconstruction map Î_{i,t} = reconstruction_map(h_{i,t}, c_{i,t}) used in
    the auxiliary loss L_I (Section 3.3; loss computed in
    ContinuousLSTMCausalModel._compute_reconstruction_loss).

    The map reads both h_t and c_t, so L_I measures how much of the input
    I_t the full recurrent state retains.

    Architecture: [h, c] -> hidden (ReLU + Dropout) -> input_dim
    """

    def __init__(self, hidden_dim: int, input_dim: int, dropout: float = 0.2):
        """hidden_dim: LSTM state size; input_dim: size of I_t."""
        super().__init__()
        inp = 2 * hidden_dim
        hidden = max(inp, 32)
        self.net = nn.Sequential(
            nn.Linear(inp, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, input_dim),
        )

    def forward(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (batch, hidden_dim) - LSTM hidden state at processing time t
            c: (batch, hidden_dim) - LSTM cell state at processing time t

        Returns:
            i_hat: (batch, input_dim) - reconstruction of I_{i,t}
        """
        return self.net(torch.cat([h, c], dim=1))

