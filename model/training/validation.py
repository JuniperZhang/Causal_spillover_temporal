"""
Validation for the two-timescale LSTM propensity model.

Functions:
    validate: loss components on a held-out dataset
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from typing import Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from ..data import NetworkTemporalCausalDataset

from .train import _make_dataloader, _unwrap_model, update_neighbor_hidden_states


def validate(model: nn.Module,
            dataset: 'NetworkTemporalCausalDataset',
            device: torch.device,
            config: Dict) -> Dict:
    """
    Evaluate the model on a held-out dataset.

    Recomputes the dataset's neighbour-averaged states (h^s, c^s) under the
    current parameters, then computes each loss component without gradients.
    Returns per-sample averages, keyed as in compute_loss.
    """
    update_neighbor_hidden_states(model, dataset, device,
                                  batch_size=config['batch_size'] * 4,
                                  config=config)

    model.eval()

    dataloader = _make_dataloader(dataset, config['batch_size'], shuffle=False, config=config)

    raw_model = _unwrap_model(model)
    use_amp = config.get('use_amp', False) and device.type == 'cuda'

    val_losses = {
        'total_loss': 0.0,
        'treatment_binary_loss': 0.0,
        'treatment_spillover_loss': 0.0,
        'reconstruction_loss': 0.0,
        'l2_loss': 0.0,
    }

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v
                    for k, v in batch.items()}

            if use_amp:
                with autocast():
                    outputs = model(batch, mode='predict')
                    loss_dict = raw_model.compute_loss(outputs, batch)
            else:
                outputs = model(batch, mode='predict')
                loss_dict = raw_model.compute_loss(outputs, batch)

            batch_size = batch['v'].size(0)
            for key in val_losses:
                val_losses[key] += loss_dict[key].item() * batch_size

    n_samples = len(dataset)
    for key in val_losses:
        val_losses[key] /= n_samples

    return val_losses

