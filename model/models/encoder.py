"""
Two-scale recurrent encoding (paper Section 3.1): GraphSAGE mean aggregator
producing the input I_t and the LSTM backbone run over process times.
"""

import torch
import torch.nn as nn


class LSTMBackbone(nn.Module):
    """
    Shared LSTM backbone of the two-timescale model.

        (h_{i,t}, c_{i,t}) = LSTM_theta(I_{i,t}, h_{i,t-1}, c_{i,t-1})

    theta is shared across units and across all processing times t=1..T.
    h_{i,t} is built only from I_{i,1..t}, i.e. from process information
    strictly before step t (see _build_input_sequence in combined_model.py),
    so it never contains V_{i,t} or V^s_{i,t}.

    Implemented as a per-layer LSTMCell loop because the reconstruction map
    and g_Z need the cell state c_{i,t} at every t, which nn.LSTM does not
    return. Dropout is applied between layers.
    """

    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.1):
        """Stack of num_layers LSTMCells, input_dim -> hidden_dim."""
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cells = nn.ModuleList([
            nn.LSTMCell(input_dim if layer == 0 else hidden_dim, hidden_dim)
            for layer in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout) if num_layers > 1 else None

    def forward(self, inputs):
        """
        Args:
            inputs: (batch, T, input_dim) - packed I_{i,t} sequence

        Returns:
            h_seq: (batch, T, hidden_dim) - top-layer hidden state at every t
            c_seq: (batch, T, hidden_dim) - top-layer cell state at every t
        """
        batch_size, T, _ = inputs.shape
        dev = inputs.device

        h = [torch.zeros(batch_size, self.hidden_dim, device=dev) for _ in range(self.num_layers)]
        c = [torch.zeros(batch_size, self.hidden_dim, device=dev) for _ in range(self.num_layers)]

        h_seq = []
        c_seq = []
        for t in range(T):
            layer_input = inputs[:, t, :]
            for layer, cell in enumerate(self.cells):
                h[layer], c[layer] = cell(layer_input, (h[layer], c[layer]))
                layer_input = h[layer]
                if self.dropout is not None and layer < self.num_layers - 1:
                    layer_input = self.dropout(layer_input)
            h_seq.append(h[-1])
            c_seq.append(c[-1])

        h_seq = torch.stack(h_seq, dim=1)   # (batch, T, hidden_dim)
        c_seq = torch.stack(c_seq, dim=1)   # (batch, T, hidden_dim)
        return h_seq, c_seq


class GraphSAGEMeanAggregator(nn.Module):
    """
    GraphSAGE mean aggregation layer (Section 3.1):

        I_{i,t} = sigma( W_own O_{i,t} + W_nbr * mean_{j in N^1(i)} O_{j,t} )

    O_{i,t} = (V_{i,t-1}, Y_{i,t-1}, X_{i,t-1}, D_{i,t-1}) is the unit's own
    input; mean_j O_{j,t} is the same block averaged over first-order
    neighbours. sigma is ReLU. W_own and W_nbr are trained jointly with the
    LSTM and the assignment models.

    Each block is standardized with BatchNorm1d before the linear maps.
    O_{i,t} mixes covariates and outcomes on a large positive scale with
    treatment indicators in [0, 1]; standardizing per feature across units
    keeps between-unit level differences (unlike per-sample LayerNorm) and
    avoids dead ReLU units.
    """

    def __init__(self, own_dim: int, nbr_dim: int, out_dim: int):
        """own_dim, nbr_dim: sizes of O_{i,t} and its neighbour mean; out_dim: size of I_t."""
        super().__init__()
        self.norm_own = nn.BatchNorm1d(own_dim)
        self.norm_nbr = nn.BatchNorm1d(nbr_dim)
        self.w_own = nn.Linear(own_dim, out_dim, bias=True)
        self.w_nbr = nn.Linear(nbr_dim, out_dim, bias=False)
        self.relu = nn.ReLU()

    def forward(self, o_own: torch.Tensor, o_nbr_mean: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        o_own      : (batch, own_dim) -- O_{i,t}
        o_nbr_mean : (batch, nbr_dim) -- mean_j O_{j,t} over first-order neighbors

        Returns
        -------
        I_t : (batch, out_dim)
        """
        # BatchNorm1d cannot compute batch statistics from one sample; use
        # running statistics for a batch of size 1.
        if o_own.size(0) == 1:
            self.norm_own.eval()
            self.norm_nbr.eval()
        return self.relu(self.w_own(self.norm_own(o_own)) + self.w_nbr(self.norm_nbr(o_nbr_mean)))

