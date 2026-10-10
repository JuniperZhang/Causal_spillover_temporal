"""
Training of the two-timescale LSTM propensity model (paper Section 3.3).

Each minibatch forward pass runs the LSTM once over all process times and
produces every head's output. The neighbour-averaged states (h^s, c^s) used
by g_Z (Section 3.2) need every unit's (h, c) sequence, so they are
recomputed once per epoch by update_neighbor_hidden_states.

Functions:
    update_neighbor_hidden_states: per-epoch computation of h^s, c^s
    train_epoch: one training epoch
    train_model_spillover_distributional: full training with early stopping
"""

import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, Optional

from ..models import TemporalCausalModelSpillover
from ..data import NetworkTemporalCausalDataset


# ============================================================================
# Parallelism helpers
# ============================================================================

def _maybe_wrap_dp(model: nn.Module, device: torch.device, config: Dict) -> nn.Module:
    """Wrap the model in DataParallel if config['multi_gpu'] is set and more than one GPU is available."""
    if config.get('multi_gpu', False) and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"  Wrapping model in DataParallel ({torch.cuda.device_count()} GPUs)")
        model = nn.DataParallel(model)
    return model


def _unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying model of a DataParallel wrapper (or the model itself)."""
    if isinstance(model, nn.DataParallel):
        return model.module
    return model


def _make_dataloader(dataset, batch_size: int, shuffle: bool,
                     config: Optional[Dict] = None) -> DataLoader:
    """Create a DataLoader using num_workers, pin_memory, persistent_workers and prefetch_factor from config."""
    if config is None:
        config = {}

    num_workers = config.get('num_workers', 0)
    pin_memory = config.get('pin_memory', True) and torch.cuda.is_available()
    persistent_workers = config.get('persistent_workers', False) and num_workers > 0
    prefetch_factor = config.get('prefetch_factor', 2) if num_workers > 0 else None

    kwargs = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
    }

    if persistent_workers:
        kwargs['persistent_workers'] = True
    if prefetch_factor is not None:
        kwargs['prefetch_factor'] = prefetch_factor

    return DataLoader(dataset, **kwargs)


# ============================================================================
# Neighbour-averaged recurrent state buffer (for g_Z)
# ============================================================================

def update_neighbor_hidden_states(model: nn.Module,
                                  dataset: NetworkTemporalCausalDataset,
                                  device: torch.device,
                                  batch_size: int = 256,
                                  config: Optional[Dict] = None) -> None:
    """
    Set dataset.h_s_seq = A_norm h_seq and dataset.c_s_seq = A_norm c_seq for
    all units under the current parameters, without gradients.

    A_norm is the row-normalized adjacency matrix, so h^s, c^s are
    neighbour means of the LSTM states. The dataset returns them as
    batch['h_s'], batch['c_s'] for g_Z. Called at the start of every
    training epoch and before validation.
    """
    model.eval()
    raw_model = _unwrap_model(model)

    dataloader = _make_dataloader(dataset, batch_size, shuffle=False, config=config)

    all_h, all_c = [], []
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v
                    for k, v in batch.items()}
            h_seq, c_seq = raw_model.extract_hidden_states(batch)
            all_h.append(h_seq.cpu())
            all_c.append(c_seq.cpu())

    h_seq_all = torch.cat(all_h, dim=0)   # (n_samples, T, hidden_dim)
    c_seq_all = torch.cat(all_c, dim=0)

    use_gpu = device.type == 'cuda'
    if use_gpu:
        if not hasattr(dataset, '_adj_norm_gpu') or dataset._adj_norm_gpu is None:
            dataset._adj_norm_gpu = dataset.adj_matrix_normalized_torch.to(device)
        adj = dataset._adj_norm_gpu
        h_seq_all = h_seq_all.to(device)
        c_seq_all = c_seq_all.to(device)
    else:
        adj = dataset.adj_matrix_normalized_torch

    mm_fn = torch.sparse.mm if adj.is_sparse else torch.mm
    T = h_seq_all.shape[1]
    h_s_seq = torch.zeros_like(h_seq_all)
    c_s_seq = torch.zeros_like(c_seq_all)
    for t in range(T):
        h_s_seq[:, t, :] = mm_fn(adj, h_seq_all[:, t, :])
        c_s_seq[:, t, :] = mm_fn(adj, c_seq_all[:, t, :])

    dataset.h_s_seq = h_s_seq.cpu()
    dataset.c_s_seq = c_s_seq.cpu()


# ============================================================================
# Single-pass training
# ============================================================================

def train_epoch(model: nn.Module,
                dataset: NetworkTemporalCausalDataset,
                optimizer: optim.Optimizer,
                device: torch.device,
                config: Dict,
                epoch: int,
                scaler: Optional[GradScaler] = None) -> Dict:
    """
    One training epoch.

    Recomputes h^s, c^s under the current parameters, then runs
    forward/loss/backward over shuffled minibatches with gradient-norm
    clipping at 1.0 (optionally with AMP via `scaler`). `epoch` is unused.

    Returns the per-sample average of each loss component.
    """
    print(f"  Refreshing neighbor hidden-state buffer (h^s, c^s)...")
    update_neighbor_hidden_states(model, dataset, device,
                                  batch_size=config['batch_size'] * 4,
                                  config=config)

    model.train()

    dataloader = _make_dataloader(dataset, config['batch_size'], shuffle=True, config=config)

    epoch_losses = {
        'total_loss': 0.0,
        'treatment_binary_loss': 0.0,
        'treatment_spillover_loss': 0.0,
        'reconstruction_loss': 0.0,
        'l2_loss': 0.0,
    }

    raw_model = _unwrap_model(model)
    use_amp = scaler is not None

    for batch_idx, batch in enumerate(dataloader):
        batch = {k: v.to(device) if torch.is_tensor(v) else v
                for k, v in batch.items()}

        optimizer.zero_grad()

        if use_amp:
            with autocast():
                outputs = model(batch, mode='train')
                loss_dict = raw_model.compute_loss(outputs, batch)
                total_loss = loss_dict['total_loss']

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(batch, mode='train')
            loss_dict = raw_model.compute_loss(outputs, batch)
            total_loss = loss_dict['total_loss']

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=1.0)
            optimizer.step()

        batch_size = batch['v'].size(0)
        for key in epoch_losses:
            epoch_losses[key] += loss_dict[key].item() * batch_size

    n_samples = len(dataset)
    for key in epoch_losses:
        epoch_losses[key] /= n_samples

    return epoch_losses


# ============================================================================
# Complete training pipeline
# ============================================================================

def train_model_spillover_distributional(
    train_dataset: NetworkTemporalCausalDataset,
    val_dataset: NetworkTemporalCausalDataset,
    config: Dict,
    device: torch.device
) -> nn.Module:
    """
    Train the two-timescale LSTM propensity model on pre-built datasets.

    Uses AdamW (weight decay l2_lambda), ReduceLROnPlateau and early stopping
    on the validation loss (training loss if val_dataset is None). Returns
    the unwrapped model with the best parameters.

    Supports:
    - DataParallel (multi-GPU via config['multi_gpu'])
    - Automatic mixed precision (FP16 via config['use_amp'])
    - Multi-worker DataLoader (via config['num_workers'])
    """
    print(f"\nInitializing model...")
    model = TemporalCausalModelSpillover(config).to(device)

    model = _maybe_wrap_dp(model, device, config)
    raw_model = _unwrap_model(model)

    total_params = sum(p.numel() for p in raw_model.parameters())
    print(f"Total parameters: {total_params:,}")

    optimizer = optim.AdamW(
        raw_model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['l2_lambda']
    )

    use_amp = config.get('use_amp', False) and device.type == 'cuda'
    scaler = GradScaler() if use_amp else None
    if use_amp:
        print(f"  Mixed precision training (AMP) enabled")

    nw = config.get('num_workers', 0)
    if nw > 0:
        print(f"  DataLoader: {nw} workers, persistent={config.get('persistent_workers', False)}")

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5,
        patience=config.get('lr_scheduler_patience', 10)
    )

    print("\nStarting training...")
    best_val_loss = float('inf')
    best_state_dict = None
    patience_counter = 0
    patience = config.get('early_stopping_patience', 20)

    for epoch in range(config['epochs']):
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"\nEpoch [{epoch+1}/{config['epochs']}]")

        train_losses = train_epoch(
            model, train_dataset, optimizer, device, config, epoch, scaler=scaler
        )

        if val_dataset is not None:
            from .validation import validate
            val_losses = validate(model, val_dataset, device, config)
            scheduler.step(val_losses['total_loss'])
            current_loss = val_losses['total_loss']
        else:
            scheduler.step(train_losses['total_loss'])
            current_loss = train_losses['total_loss']
            val_losses = None

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Train Loss: {train_losses['total_loss']:.6f}")
            if val_losses is not None:
                print(f"  Val Loss:   {val_losses['total_loss']:.6f}")

        if current_loss < best_val_loss:
            best_val_loss = current_loss
            patience_counter = 0
            best_state_dict = copy.deepcopy(raw_model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    if best_state_dict is not None:
        raw_model.load_state_dict(best_state_dict)

    return raw_model
