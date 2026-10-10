"""
Estimate policy-path effects on the county panel (paper Section 6).

    python real_data/estimate.py --model-path .../trained_model.pt \
        --data-dir real_data/inputs/Business_Economic_Restrictions \
        --output-dir .../estimates --end-period 2020-10 --outcome-aggregation last_week

For one outcome month (--end-period) and summary (--outcome-aggregation):
- the outcome is the case rate of the month's last week (last_week) or the
  average weekly case rate over the month (final_month_average);
- target paths are the joint baseline x = d = 0 at every decision month plus
  the --top-k most frequent observed (x, d) paths up to the outcome month;
- each mean potential outcome is the self-normalized kernel IPW estimate
  (Section 3.4) with bandwidth h = 1 / (2 * median degree);
- each effect is a path mean minus the baseline mean, with 95% percentile
  intervals from the disjoint network-block bootstrap (randomized k-hop
  blocks, --n-bootstrap draws).

Writes outcome_expectations_<aggregation>_<end-period>.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for `import model`

from model.data.real_data_dataset import RealDataDataset
from model.estimation.bootstrap import joint_network_block_bootstrap
from model.estimation.gaussian_kernel import estimate_ate_ipw_gaussian_kernel
from model.models import TemporalCausalModelSpillover


def parse_args() -> argparse.Namespace:
    """Command-line options (defaults are the paper's settings except --top-k, which the paper sets to 8)."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-path", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--end-period", default=None, help="YYYY-MM; last week within that month is the outcome endpoint")
    p.add_argument("--outcome-aggregation", default="last_week",
                    choices=["last_week", "final_month_average"])
    p.add_argument("--top-k", type=int, default=6)
    p.add_argument("--weight-truncation", type=float, default=99.0)
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--k-hops", type=int, default=1, help="Bootstrap block radius in hops (paper: 1)")
    p.add_argument("--n-partitions", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps", "auto"])
    p.add_argument("--save-draws", action="store_true",
                   help="Store each ATE's bootstrap draws, for pooling across training seeds")
    return p.parse_args()


def resolve_device(name: str) -> torch.device:
    """Map 'auto' to cuda, then mps, then cpu; fall back to cpu if the device is unavailable."""
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if name == "mps" and not torch.backends.mps.is_available():
        return torch.device("cpu")
    return torch.device(name)


def _json_default(obj):
    """JSON encoder for numpy values and paths."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)!r} is not JSON serializable")


def load_model(model_path: Path, device: torch.device) -> tuple:
    """Rebuild the model from a train.py checkpoint; returns (model, config, checkpoint)."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = TemporalCausalModelSpillover(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.decision_mask = checkpoint["decision_mask"]
    model.eval()
    return model, config, checkpoint


def resolve_end_period(dataset: RealDataDataset, end_period: str | None) -> dict:
    """Index and label of the last week of `end_period` (default: last week in the data)."""
    periods = dataset.metadata.get("periods", [])
    week_to_period = dataset.metadata.get("week_to_period", [])
    if not end_period:
        return {
            "resolved_end_period": dataset.metadata.get("resolved_end_period"),
            "resolved_end_week": dataset.metadata.get("resolved_end_week"),
            "end_step_index": dataset.T - 1,
        }
    valid = [i for i, p in enumerate(week_to_period) if p <= end_period]
    if not valid:
        raise ValueError(f"No weekly steps at or before end_period={end_period}")
    end_idx = max(valid)
    return {
        "resolved_end_period": week_to_period[end_idx],
        "resolved_end_week": periods[end_idx],
        "end_step_index": end_idx,
    }


def aggregated_outcome_time(dataset: RealDataDataset, end_info: dict, aggregation: str) -> tuple[torch.Tensor, int]:
    """Outcome tensor and outcome time index; final_month_average writes the month's mean into the last week's slot."""
    end_idx = end_info["end_step_index"]
    if aggregation == "last_week":
        return dataset.y, end_idx

    week_to_period = dataset.metadata.get("week_to_period", [])
    resolved_period = end_info["resolved_end_period"]
    month_indices = [i for i, p in enumerate(week_to_period) if p == resolved_period]
    if not month_indices:
        raise ValueError(f"No weekly steps found for resolved_end_period={resolved_period}")

    y = dataset.y.clone()
    month_vals = y[:, month_indices, 0]
    if aggregation == "final_month_average":
        agg = month_vals.mean(dim=1)
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")
    y[:, end_idx, 0] = agg
    return y, end_idx


def canonical_sequence(T: int, decision_steps: list[int], x_vals: np.ndarray, xs_vals: np.ndarray,
                        round_digits: int = 6) -> tuple:
    """Length-2T (x1, d1, ..., xT, dT) path; only the decision-week slots carry values."""
    seq = [0, 0.0] * T
    for j, x_val, xs_val in zip(decision_steps, x_vals, xs_vals):
        seq[2 * j] = int(round(float(x_val)))
        seq[2 * j + 1] = round(float(xs_val), round_digits)
    return tuple(seq)


def top_observed_sequences(dataset: RealDataDataset, unit_mask: np.ndarray, top_k: int) -> list[tuple[tuple, int]]:
    """The top_k most frequent observed decision-time (x, d) paths with their counts."""
    X = dataset.x.numpy()
    XS = dataset.d_xs.numpy()
    decision_steps = dataset.decision_time_indices
    counter: Counter = Counter()
    for i in np.flatnonzero(unit_mask):
        seq = canonical_sequence(dataset.T, decision_steps, X[i, decision_steps, 0], XS[i, decision_steps, 0])
        counter[seq] += 1
    ranked = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    return ranked[:top_k]


def zero_sequence(T: int) -> tuple:
    """Joint baseline path x = d = 0 at every decision time."""
    return tuple([0, 0.0] * T)


def compact_sequence(full_seq: tuple, decision_steps: list[int]) -> tuple:
    """The K decision-time (x, d) pairs of a length-2T path; used as the JSON key of each path."""
    out = []
    for j in decision_steps:
        out.append(full_seq[2 * j])
        out.append(full_seq[2 * j + 1])
    return tuple(out)


def estimate_with_ci(
    model,
    dataset: RealDataDataset,
    treatment_sequences: list[tuple],
    baseline_seq: tuple,
    device: torch.device,
    outcome_time: int,
    weight_truncation: float,
    n_bootstrap: int,
    k_hops: int,
    n_partitions: int,
    seed: int,
    decision_steps: list[int],
    unit_mask: np.ndarray | None = None,
    save_draws: bool = False,
) -> dict:
    """
    Kernel IPW means of every target path over the counties in unit_mask, their
    effects against baseline_seq, and joint network-block bootstrap intervals.
    With save_draws, each effect also keeps its bootstrap draws (used to pool seeds).
    """
    result = estimate_ate_ipw_gaussian_kernel(
        model, dataset, treatment_sequences, device,
        bandwidth=dataset.spillover_bandwidth,
        weight_truncation_percentile=weight_truncation,
        outcome_time=outcome_time,
        hajek=True,
        return_weights=True,
    )

    if unit_mask is None:
        raise ValueError("unit_mask is required")
    adj_sub = dataset.adj_matrix[np.ix_(unit_mask, unit_mask)]

    target_inputs = {}
    seq_names = {}
    masked_ess = {}
    masked_raw_path_count = {}
    for seq in treatment_sequences:
        est = result["estimates"][seq]
        name = str(compact_sequence(seq, decision_steps))
        seq_names[seq] = name
        weights_full = est.get("weights")
        y_full = est.get("y")
        if weights_full is None or y_full is None:
            continue
        w = weights_full[unit_mask]
        target_inputs[name] = (w, y_full[unit_mask])
        w_sq_sum = float((w ** 2).sum())
        masked_ess[name] = float(w.sum() ** 2 / w_sq_sum) if w_sq_sum > 0 else 0.0
        masked_raw_path_count[name] = int((w > 0).sum())

    contrast_pairs = {}
    baseline_name = seq_names.get(baseline_seq)
    for seq in treatment_sequences:
        if seq == baseline_seq:
            continue
        name = seq_names[seq]
        if name in target_inputs and baseline_name in target_inputs:
            contrast_pairs[f"ATE::{name}"] = (name, baseline_name)

    rng = np.random.default_rng(seed)
    boot = joint_network_block_bootstrap(
        target_inputs, contrast_pairs, adj_sub,
        n_boot=n_bootstrap, ci_level=0.95, k_hops=k_hops,
        n_partitions=n_partitions, rng=rng, return_draws=save_draws,
    )
    summaries = boot.get("summaries", {})
    draws = boot.get("draws", {})

    outcome_expectations = {}
    for seq in treatment_sequences:
        name = seq_names[seq]
        if name not in target_inputs:
            outcome_expectations[name] = {
                "estimate": None, "effective_n": 0.0, "raw_path_count": 0,
                "ci_lower": None, "ci_upper": None, "se": None,
                "n_bootstrap": 0, "bootstrap_reliable": False,
            }
            continue
        s = summaries.get(name, {})
        outcome_expectations[name] = {
            # Point estimate over the same counties as the bootstrap draws.
            "estimate": s.get("point"),
            "effective_n": masked_ess.get(name),
            "raw_path_count": masked_raw_path_count.get(name),
            "ci_lower": s.get("ci_lower"),
            "ci_upper": s.get("ci_upper"),
            "se": s.get("se"),
            "n_bootstrap": s.get("n_valid"),
            "bootstrap_reliable": s.get("reliable"),
        }

    ate_results = {}
    for seq in treatment_sequences:
        if seq == baseline_seq:
            continue
        name = seq_names[seq]
        s = summaries.get(f"ATE::{name}", {})
        base_est = outcome_expectations[baseline_name]["estimate"]
        cur_est = outcome_expectations[name]["estimate"]
        ate_results[name] = {
            "ate_estimate": (cur_est - base_est) if base_est is not None and cur_est is not None else None,
            "ate_se": s.get("se"),
            "ci_lower": s.get("ci_lower"),
            "ci_upper": s.get("ci_upper"),
            "n_bootstrap": s.get("n_valid"),
            "bootstrap_reliable": s.get("reliable"),
        }
        if save_draws:
            ate_results[name]["draws"] = [float(v) for v in draws.get(f"ATE::{name}", [])]

    return {
        "outcome_expectations": outcome_expectations,
        "ate_results": ate_results,
        "bootstrap_metadata": boot.get("metadata", {}),
    }


def main() -> None:
    """Estimate all target paths for one outcome month and summary, and save the JSON."""
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_path = Path(args.model_path).resolve()
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    model, config, checkpoint = load_model(model_path, device)
    dataset = RealDataDataset(str(data_dir))
    assert dataset.decision_mask == model.decision_mask, "dataset/model decision schedule mismatch"

    # Effects are estimated over all counties.
    split_mask = np.ones(dataset.N, dtype=bool)

    end_info = resolve_end_period(dataset, args.end_period)
    dataset.y, outcome_time = aggregated_outcome_time(dataset, end_info, args.outcome_aggregation)

    # Only decisions at or before the outcome month enter the target paths
    # and the weight product.
    decision_steps = [t for t in dataset.decision_time_indices if t <= end_info["end_step_index"]]
    model.decision_mask = [t in set(decision_steps) for t in range(dataset.T)]
    k_decisions = len(decision_steps)
    top_seqs = top_observed_sequences(dataset, split_mask, args.top_k)
    zero_seq = zero_sequence(dataset.T)

    treatment_sequences: list[tuple] = [zero_seq]
    for seq, _ in top_seqs:
        if seq not in treatment_sequences:
            treatment_sequences.append(seq)

    print("=" * 80)
    print("REAL-DATA ESTIMATION")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Counties: {dataset.N}")
    print(f"Outcome end period: {end_info['resolved_end_period']}  aggregation: {args.outcome_aggregation}")
    print(f"Decision steps: {decision_steps}")
    print(f"Top observed sequences ({len(top_seqs)}):")
    for seq, count in top_seqs:
        print(f"  count={count:>3d}  seq={seq}")

    overall = estimate_with_ci(
        model, dataset, treatment_sequences, zero_seq, device,
        outcome_time, args.weight_truncation, args.n_bootstrap,
        args.k_hops, args.n_partitions, args.seed, decision_steps,
        unit_mask=split_mask, save_draws=args.save_draws,
    )

    result = {
        "outcome_expectations": overall["outcome_expectations"],
        "ate_results": overall["ate_results"],
        "bootstrap_metadata": overall["bootstrap_metadata"],
        "treatment_sequences": [list(seq) for seq in treatment_sequences],
        "spillover_info": {
            "method": "ipw_gaussian_kernel",
            "bandwidth": float(dataset.spillover_bandwidth),
            "decision_steps": decision_steps,
            "decision_periods": dataset.metadata.get("decision_periods", []),
            "top_observed_sequences": [{"sequence": list(seq), "count": int(c)} for seq, c in top_seqs],
        },
        "metadata": {
            "model_path": str(model_path),
            "data_dir": str(data_dir),
            "n_units": int(dataset.N),
            "T": int(dataset.T),
            "K": int(k_decisions),
            "decision_steps": decision_steps,
            "decision_periods": dataset.metadata.get("decision_periods", []),
            "n_bootstrap": int(args.n_bootstrap),
            "bandwidth": float(dataset.spillover_bandwidth),
            "seed": int(args.seed),
            "method": "k_ipw_gaussian_kernel_network_block_bootstrap",
            "weight_truncation_percentile": float(args.weight_truncation),
            "treatment_col": dataset.metadata.get("treatment_col"),
            "treatment_threshold": dataset.metadata.get("treatment_threshold"),
            "outcome_col": dataset.metadata.get("outcome_col"),
            "resolved_end_week": end_info["resolved_end_week"],
            "resolved_end_period": end_info["resolved_end_period"],
            "end_step_index": end_info["end_step_index"],
            "outcome_aggregation": args.outcome_aggregation,
            "bootstrap_k_hops": args.k_hops,
        },
    }

    out_name = f"outcome_expectations_{args.outcome_aggregation}"
    if args.end_period:
        out_name += f"_{args.end_period}"
    out_path = output_dir / f"{out_name}.json"
    out_path.write_text(json.dumps(result, indent=2, default=_json_default))
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
