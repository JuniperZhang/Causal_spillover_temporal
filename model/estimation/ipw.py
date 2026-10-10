"""
Propensity extraction from the trained two-scale LSTM model.

Runs the trained ContinuousLSTMCausalModel over a dataset without gradients
and collects, at every decision time, the fitted own-treatment propensity
e_hat (from f_X) and the raw f_D output, which together give the fitted
joint propensity of Section 3.2

    p_hat_tau(x, d | H) = e_hat^x (1-e_hat)^(1-x) * p_hat_D(d | H)

used in the K-IPW weights of gaussian_kernel.py.

Functions:
    compute_propensity_scores_lstm: propensity extraction for the LSTM model
"""
import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict


def compute_propensity_scores_lstm(
    model,
    dataset,
    device: torch.device,
    batch_size: int = 256,
) -> Dict:
    """
    Compute fitted own-treatment propensities and raw spillover-head outputs
    at every decision time, for every unit in the dataset.

    Refreshes the dataset's neighbor hidden-state buffer (h^s, c^s) first,
    since g_Z (and hence f_D) depends on it.

    Args:
        model: trained ContinuousLSTMCausalModel (TemporalCausalModelSpillover)
        dataset: NetworkTemporalCausalDataset instance
        device: torch device
        batch_size: batch size for the forward pass

    Returns:
        Dict with:
        - 'e_hat'      : (n_samples, K) fitted P(X_tau=1 | H_tau)
        - 'd_logits'   : (n_samples, K, d_out_dim) raw f_D output at each decision time
        - 'x_obs'      : (n_samples, K) observed own treatment X_tau
        - 'd_obs'      : (n_samples, K) observed spillover proportion D_tau
        - 'decision_steps' : list[K] 0-based processing-time indices of the decision times
    """
    print("\n" + "="*80)
    print("Computing LSTM Propensity Scores")
    print("="*80)

    from ..training.train import update_neighbor_hidden_states
    update_neighbor_hidden_states(model, dataset, device, batch_size=batch_size)

    model.eval()
    n_samples = len(dataset)
    decision_steps = [j for j, is_dec in enumerate(model.decision_mask) if is_dec]
    K = len(decision_steps)
    d_out_dim = model.fD_head.output_dim

    e_hat = np.zeros((n_samples, K), dtype=np.float64)
    d_logits = np.zeros((n_samples, K, d_out_dim), dtype=np.float64)
    x_obs = np.zeros((n_samples, K), dtype=np.float64)
    d_obs = np.zeros((n_samples, K), dtype=np.float64)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    sample_idx = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v
                    for k, v in batch.items()}
            batch_size_actual = batch['v'].size(0)

            outputs = model(batch, mode='predict')

            for pos, j in enumerate(decision_steps):
                x_logit = outputs['x_logits'][j]                    # (batch, 1)
                d_logit = outputs['d_logits'][j]                    # (batch, d_out_dim)

                e_t = torch.sigmoid(x_logit).squeeze(-1).cpu().numpy()
                e_hat[sample_idx:sample_idx + batch_size_actual, pos] = e_t
                d_logits[sample_idx:sample_idx + batch_size_actual, pos, :] = d_logit.cpu().numpy()

                x_obs[sample_idx:sample_idx + batch_size_actual, pos] = batch['x'][:, j, 0].cpu().numpy()
                d_obs[sample_idx:sample_idx + batch_size_actual, pos] = batch['d_xs'][:, j, 0].cpu().numpy()

            sample_idx += batch_size_actual

    print(f"\nComputed propensity scores for {n_samples} samples at {K} decision times")
    print(f"  e_hat (own-treatment propensity): mean={e_hat.mean():.4f}, "
          f"min={e_hat.min():.4f}, max={e_hat.max():.4f}")

    return {
        'e_hat': e_hat,
        'd_logits': d_logits,
        'x_obs': x_obs,
        'd_obs': d_obs,
        'decision_steps': decision_steps,
    }

