"""
Training of the LSTM propensity model.

Functions:
    update_neighbor_hidden_states: per-epoch neighbour-averaged LSTM states for g_Z
    train_epoch: one training epoch
    train_model_spillover_distributional: full training with early stopping
    validate: loss on a held-out dataset
"""

from .train import (
    update_neighbor_hidden_states,
    train_epoch,
    train_model_spillover_distributional,
)
from .validation import validate

__all__ = [
    'update_neighbor_hidden_states',
    'train_epoch',
    'train_model_spillover_distributional',
    'validate',
]
