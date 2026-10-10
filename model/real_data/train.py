"""
Train the LSTM propensity model (paper Sections 3.1-3.3) on the county panel.

    python -m model.real_data.train --data-dir results/real_data/inputs/Business_Economic_Restrictions \
        --output-dir results/real_data/runs/seed_0042/Business_Economic_Restrictions --seed 42

Decision times are the monthly policy updates in the data, so the model's
decision mask is replaced by the dataset's. Every forward pass uses the full
county graph; the training and early-stopping losses use only the train and
validation counties. Writes trained_model.pt, training_config.json and
data_metadata.json to --output-dir.
"""
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Subset


from model.config import CONFIG
from model.data.real_data_dataset import RealDataDataset
from model.training.train import (
    _make_dataloader, _maybe_wrap_dp, _unwrap_model, update_neighbor_hidden_states,
)


def parse_args() -> argparse.Namespace:
    """Command-line options (defaults are the paper's settings)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", required=True, help="Directory written by build_inputs.py")
    parser.add_argument("--output-dir", required=True, help="Directory to save model and training outputs")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--l2-lambda", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lstm-hidden-dim", type=int, default=128)
    parser.add_argument("--lstm-num-layers", type=int, default=2)
    parser.add_argument("--z-x-dim", type=int, default=16)
    parser.add_argument("--z-d-dim", type=int, default=16)
    parser.add_argument("--z-c-dim", type=int, default=32)
    parser.add_argument("--weight-treatment-binary", type=float, default=1.0)
    parser.add_argument("--weight-spillover", type=float, default=1.0)
    parser.add_argument("--weight-reconstruction", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--lr-scheduler-patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    """Map 'auto' to cuda, then mps, then cpu."""
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def build_config(args: argparse.Namespace, train_dataset: RealDataDataset) -> dict:
    """Model and training configuration: model defaults updated with the data dimensions and options."""
    config = deepcopy(CONFIG)
    config.update({
        "v_dim": train_dataset.v_dim,
        "T": train_dataset.T,
        "dropout": args.dropout,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "l2_lambda": args.l2_lambda,
        "early_stopping_patience": args.early_stopping_patience,
        "lr_scheduler_patience": args.lr_scheduler_patience,
        "lstm_hidden_dim": args.lstm_hidden_dim,
        "lstm_num_layers": args.lstm_num_layers,
        "z_x_dim": args.z_x_dim,
        "z_d_dim": args.z_d_dim,
        "z_c_dim": args.z_c_dim,
        "weight_treatment_binary": args.weight_treatment_binary,
        "weight_spillover": args.weight_spillover,
        "weight_reconstruction": args.weight_reconstruction,
        "num_workers": 0,
        "persistent_workers": False,
        "pin_memory": True,
    })
    return config


def masked_train_epoch(model, full_dataset, train_indices, optimizer, device, config, scaler=None) -> dict:
    """One epoch: refresh neighbour states on the full graph, then update on the training counties only."""
    update_neighbor_hidden_states(model, full_dataset, device,
                                   batch_size=config["batch_size"] * 4, config=config)
    model.train()
    dataloader = _make_dataloader(Subset(full_dataset, train_indices), config["batch_size"],
                                   shuffle=True, config=config)

    epoch_losses = {
        "total_loss": 0.0, "treatment_binary_loss": 0.0, "treatment_spillover_loss": 0.0,
        "reconstruction_loss": 0.0, "l2_loss": 0.0,
    }
    raw_model = _unwrap_model(model)
    use_amp = scaler is not None

    for batch in dataloader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        optimizer.zero_grad()

        if use_amp:
            with autocast():
                outputs = model(batch, mode="train")
                loss_dict = raw_model.compute_loss(outputs, batch)
                total_loss = loss_dict["total_loss"]
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(batch, mode="train")
            loss_dict = raw_model.compute_loss(outputs, batch)
            total_loss = loss_dict["total_loss"]
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=1.0)
            optimizer.step()

        bsz = batch["v"].size(0)
        for key in epoch_losses:
            epoch_losses[key] += loss_dict[key].item() * bsz

    for key in epoch_losses:
        epoch_losses[key] /= len(train_indices)
    return epoch_losses


def masked_validate(model, full_dataset, val_indices, device, config) -> dict:
    """Average loss over the validation counties, with neighbour states from the full graph."""
    update_neighbor_hidden_states(model, full_dataset, device,
                                   batch_size=config["batch_size"] * 4, config=config)
    model.eval()
    dataloader = _make_dataloader(Subset(full_dataset, val_indices), config["batch_size"],
                                   shuffle=False, config=config)
    raw_model = _unwrap_model(model)

    val_losses = {
        "total_loss": 0.0, "treatment_binary_loss": 0.0, "treatment_spillover_loss": 0.0,
        "reconstruction_loss": 0.0, "l2_loss": 0.0,
    }
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            outputs = model(batch, mode="predict")
            loss_dict = raw_model.compute_loss(outputs, batch)
            bsz = batch["v"].size(0)
            for key in val_losses:
                val_losses[key] += loss_dict[key].item() * bsz

    for key in val_losses:
        val_losses[key] /= len(val_indices)
    return val_losses


def main() -> None:
    """Train with early stopping on the validation loss and save the best model."""
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    dataset = RealDataDataset(str(data_dir))
    train_indices = dataset.split_indices["train"]
    val_indices = dataset.split_indices["val"]

    config = build_config(args, dataset)

    print("=" * 80)
    print("REAL-DATA TRAINING")
    print("=" * 80)
    print(f"Data dir: {data_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Device: {device}")
    print(f"Treatment: {dataset.metadata.get('treatment_col')}")
    print(f"Threshold: {dataset.metadata.get('treatment_threshold')}")
    print(f"Outcome: {dataset.metadata.get('outcome_col')}")
    print(f"N total / train / val / test: {dataset.N} / {len(train_indices)} / {len(val_indices)} "
          f"/ {len(dataset.split_indices['test'])}")
    print(f"Weeks T: {dataset.T}")
    print(f"Decision step indices: {dataset.decision_time_indices}")
    print(f"Decision periods: {dataset.metadata.get('decision_periods')}")

    from model.models import TemporalCausalModelSpillover

    model = TemporalCausalModelSpillover(config).to(device)
    model.decision_mask = dataset.decision_mask

    import copy as copy_module

    model = _maybe_wrap_dp(model, device, config)
    raw_model = _unwrap_model(model)
    total_params = sum(p.numel() for p in raw_model.parameters())
    print(f"Total parameters: {total_params:,}")

    optimizer = optim.AdamW(raw_model.parameters(), lr=config["learning_rate"],
                             weight_decay=config["l2_lambda"])
    use_amp = config.get("use_amp", False) and device.type == "cuda"
    scaler = GradScaler() if use_amp else None
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=config.get("lr_scheduler_patience", 10)
    )

    best_val_loss = float("inf")
    best_state_dict = None
    patience_counter = 0
    patience = config.get("early_stopping_patience", 20)

    for epoch in range(config["epochs"]):
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"\nEpoch [{epoch + 1}/{config['epochs']}]")

        train_losses = masked_train_epoch(model, dataset, train_indices, optimizer, device, config, scaler=scaler)
        val_losses = masked_validate(model, dataset, val_indices, device, config)
        scheduler.step(val_losses["total_loss"])
        current_loss = val_losses["total_loss"]

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Train Loss: {train_losses['total_loss']:.6f}  "
                  f"(X={train_losses['treatment_binary_loss']:.4f}, "
                  f"D={train_losses['treatment_spillover_loss']:.4f}, "
                  f"I={train_losses['reconstruction_loss']:.4f})")
            print(f"  Val Loss:   {val_losses['total_loss']:.6f}")

        if current_loss < best_val_loss:
            best_val_loss = current_loss
            patience_counter = 0
            best_state_dict = copy_module.deepcopy(raw_model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

    if best_state_dict is not None:
        raw_model.load_state_dict(best_state_dict)

    checkpoint = {
        "model_state_dict": raw_model.state_dict(),
        "config": config,
        "decision_mask": dataset.decision_mask,
        "decision_time_indices": dataset.decision_time_indices,
        "best_val_loss": best_val_loss,
        "train_metadata": dataset.metadata,
    }
    model_path = output_dir / "trained_model.pt"
    torch.save(checkpoint, model_path)

    with (output_dir / "training_config.json").open("w") as f:
        json.dump(config, f, indent=2, default=str)
    with (output_dir / "data_metadata.json").open("w") as f:
        json.dump(dataset.metadata, f, indent=2, default=str)

    print("\nSaved:")
    print(f"  model: {model_path}")
    print(f"  config: {output_dir / 'training_config.json'}")
    print(f"  metadata: {output_dir / 'data_metadata.json'}")


if __name__ == "__main__":
    main()
