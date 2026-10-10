"""Monte Carlo study of bias, MSE and interval coverage (Section 5).

Each of R replications draws an analysis network of n units on an
Erdős–Rényi graph, fits the propensity model (Section 3) on those n units and
evaluates three estimators on the same network: the proposed LSTM kernel IPW
estimator, kernel IPW with the true (oracle) propensities, and an OLS
baseline. An independent validation network is used only for early stopping;
with --no-validation it is omitted and the LR scheduler and early stopping
use the training loss.

Targets are 15 mean potential outcomes mu(x, d) (5 own-treatment paths x 3
reference spillover paths low/mid/high) and 13 DE/SE/TE contrasts
(Section 2). The reference spillover paths and the Monte Carlo ground truth
are computed once, outside the R replications.

Inference uses a disjoint network-block bootstrap of the final weighted means
(Supplementary Material). It keeps the observed graph and histories fixed,
uses common bootstrap draws for all 28 targets, and conditions on the fitted
nuisance model. An optional retraining bootstrap refits the LSTM on each draw,
and optional sensitivity bounds are computed over a Gamma grid (Section 4).
Results are checkpointed to JSON after each replication, which supports
--resume and sharding by --rep-start/--rep-count.
"""
from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch


def _available_cpus() -> int:
    """Return the number of CPUs this process may use.

    Uses the scheduler affinity mask, which respects cgroup/batch-scheduler
    CPU limits, rather than os.cpu_count(), which reports every core on the
    node; sizing torch's thread pool to the whole node can exhaust memory.
    """
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 4

from ..config import CONFIG
from ..data import NetworkTemporalCausalDataset
from ..data.ground_truth import compute_ground_truth_monte_carlo_arbitrary_T
from ..estimation.bootstrap import (
    joint_network_block_bootstrap,
    k_hop_max_partition,
)
from ..estimation.gaussian_kernel import estimate_ate_ipw_gaussian_kernel
from ..estimation.ols_baseline import fit_predict_ols_baseline
from ..estimation.oracle import estimate_ate_oracle_kipw
from ..estimation.reference_paths import (
    DEFAULT_LEVEL_RANGES,
    find_reference_spillover_paths,
)
from ..training.train import train_model_spillover_distributional


CANONICAL_PATHS = {
    "never": (0, 0, 0, 0),
    "early": (1, 0, 0, 0),
    "late": (0, 0, 0, 1),
    "intermittent": (1, 0, 1, 0),
    "frequent": (1, 1, 1, 1),
}
D_LEVELS = ("low", "mid", "high")
NON_NEVER = tuple(name for name in CANONICAL_PATHS if name != "never")
METHODS = ("lstm_kipw", "oracle_kipw", "ols")


def _jsonable(value):
    """Recursively convert NumPy arrays/scalars and tuples to JSON-native types."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def _write_json(path: Path, payload) -> None:
    """Write payload as JSON atomically (temporary file, then rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as handle:
        json.dump(_jsonable(payload), handle, indent=2, allow_nan=False)
    tmp.replace(path)


def _log(log_path: Optional[str], message: str) -> None:
    """Print a timestamped message and append it to log_path if given."""
    line = f"[{time.strftime('%H:%M:%S')}] {message}"
    print(line, flush=True)
    if log_path:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            handle.write(line + "\n")


def decision_steps_for(config: Mapping) -> list[int]:
    """Return the decision times 0, interval, 2*interval, ... below T."""
    interval = int(config["treatment_update_interval"])
    return list(range(0, int(config["T"]), interval))


def build_sequence(
    x_path: Sequence[int],
    d_path: Sequence[float],
    T: int,
    decision_steps: Sequence[int],
) -> Tuple[float, ...]:
    """Expand decision-time exposure paths to a per-period sequence.

    x_path and d_path give the own treatment and spillover proportion at each
    decision time; each value is held until the next decision. Returns the
    interleaved tuple (x_0, d_0, x_1, d_1, ..., x_{T-1}, d_{T-1}) of length 2T.
    """
    if len(x_path) != len(decision_steps) or len(d_path) != len(decision_steps):
        raise ValueError("x_path and d_path must match the number of decisions")
    by_step = {step: pos for pos, step in enumerate(decision_steps)}
    current_x = float(x_path[0])
    current_d = float(d_path[0])
    out = []
    for t in range(T):
        if t in by_step:
            pos = by_step[t]
            current_x = float(x_path[pos])
            current_d = float(d_path[pos])
        out.extend((current_x, current_d))
    return tuple(out)


def level_name(path_name: str, d_level: str) -> str:
    """Return the target key for mu(own path, spillover level)."""
    return f"mu__{path_name}__{d_level}"


def target_sequences(
    config: Mapping,
    d_paths: Mapping[str, Sequence[float]],
) -> Dict[str, Tuple[float, ...]]:
    """Return the 15 level targets (5 own paths x 3 spillover levels) as
    expanded exposure sequences keyed by level_name."""
    steps =decision_steps_for(config)
    return {
        level_name(path_name, d_level): build_sequence(
            x_path, d_paths[d_level], int(config["T"]), steps
        )
        for path_name, x_path in CANONICAL_PATHS.items()
        for d_level in D_LEVELS
    }


def contrast_pairs() -> Dict[str, Tuple[str, str]]:
    """Return the 13 contrasts (Section 2) as name -> (minuend, subtrahend).

    SE(x) = mu(x, high) - mu(x, low) for each of the 5 own paths;
    DE(x) = mu(x, mid) - mu(never, mid) and
    TE(x) = mu(x, high) - mu(never, low) for the 4 non-never paths.
    """
    pairs = {}
    for path_name in CANONICAL_PATHS:
        pairs[f"SE__{path_name}"] = (
            level_name(path_name, "high"),
            level_name(path_name, "low"),
        )
    for path_name in NON_NEVER:
        pairs[f"DE__{path_name}"] = (
            level_name(path_name, "mid"),
            level_name("never", "mid"),
        )
        pairs[f"TE__{path_name}"] = (
            level_name(path_name, "high"),
            level_name("never", "low"),
        )
    return pairs


def _validate_d_paths(
    d_paths: Mapping[str, Sequence[float]],
    K: int,
) -> Dict[str, list[float]]:
    """Check that each level holds K ordered proportions in [0, 1]."""
    clean = {}
    for level in D_LEVELS:
        if level not in d_paths:
            raise ValueError(f"Missing reference spillover path: {level}")
        path = [float(v) for v in d_paths[level]]
        if len(path) != K:
            raise ValueError(f"{level} must contain {K} values")
        if not all(0.0 <= v <= 1.0 for v in path):
            raise ValueError(f"{level} must contain {K} values in [0,1]")
        clean[level] = path
    for pos in range(K):
        if not clean["low"][pos] <= clean["mid"][pos] <= clean["high"][pos]:
            raise ValueError(f"Reference paths are not ordered at decision {pos + 1}")
    return clean


def load_or_construct_reference_paths(
    config: Mapping,
    reference_paths_json: Optional[str],
    reference_n: int,
    reference_seed: int,
    reference_bandwidth: float,
) -> Tuple[Dict[str, list[float]], Dict]:
    """Return fixed low/mid/high reference spillover paths and their provenance.

    Paths are read from reference_paths_json if given; otherwise they are
    selected on a separate reference dataset (reference_n, reference_seed,
    reference_bandwidth), so they do not depend on the analysis n or h.
    """
    K = len(decision_steps_for(config))
    if reference_paths_json:
        with Path(reference_paths_json).open() as handle:
            payload = json.load(handle)
        candidate = payload.get("d_paths", payload.get("reference_paths", payload))
        if all(isinstance(candidate.get(k), dict) for k in D_LEVELS):
            candidate = {k: candidate[k]["d_path"] for k in D_LEVELS}
        d_paths = _validate_d_paths(candidate, K)
        return d_paths, {
            "source": "json",
            "file": str(Path(reference_paths_json)),
            "reference_bandwidth": float(reference_bandwidth),
        }

    ref_config = copy.deepcopy(dict(config))
    ref_dataset = NetworkTemporalCausalDataset(
        n_samples=int(reference_n),
        v_dim=int(ref_config["v_dim"]),
        T=int(ref_config["T"]),
        network_avg_degree=int(ref_config["network_avg_degree"]),
        config=ref_config,
        seed=int(reference_seed),
    )
    ref_dataset.spillover_bandwidth = float(reference_bandwidth)
    three_ranges = {name: DEFAULT_LEVEL_RANGES[name] for name in D_LEVELS}
    info = find_reference_spillover_paths(
        ref_dataset,
        ref_config,
        CANONICAL_PATHS,
        decision_steps_for(ref_config),
        level_ranges=three_ranges,
        bandwidth=float(reference_bandwidth),
        weight_truncation_percentile=100.0,
    )
    d_paths = _validate_d_paths({k: info[k]["d_path"] for k in D_LEVELS}, K)
    provenance = {
        "source": "fixed_reference_dataset",
        "reference_n": int(reference_n),
        "reference_seed": int(reference_seed),
        "reference_bandwidth": float(reference_bandwidth),
        "selection_diagnostics": info,
    }
    return d_paths, provenance


def compute_ground_truth(
    config: Mapping,
    sequences: Mapping[str, Tuple[float, ...]],
    gt_batches: int,
    gt_inner: int,
    gt_seed: int,
    log_path: Optional[str] = None,
) -> Dict[str, Dict]:
    """Compute Monte Carlo ground truth for each target sequence.

    Uses common random numbers (the same gt_seed) across targets and draws
    gt_batches * gt_inner independent local network samples in one vectorized
    call (Supplementary Material). The two factors are recorded separately in
    the output metadata.
    """
    total = int(gt_batches) * int(gt_inner)
    if total <= 0:
        raise ValueError("gt_batches * gt_inner must be positive")
    results = {}
    for idx, (name, sequence) in enumerate(sequences.items(), 1):
        result = compute_ground_truth_monte_carlo_arbitrary_T(
            sequence,
            dict(config),
            n_samples=total,
            seed=int(gt_seed),
            verbose=False,
        )
        results[name] = _jsonable(result)
        if idx % 5 == 0 or idx == len(sequences):
            _log(log_path, f"ground truth {idx}/{len(sequences)} complete")
    return results


def make_analysis_and_validation(
    config: Mapping,
    n: int,
    seed: int,
    validation_n: Optional[int] = None,
    no_validation: bool = False,
):
    """Generate the analysis network (n units) and the validation network.

    The validation network uses seed + 100000 and, by default,
    max(500, n * n_samples_val_ratio) units. Returns (analysis, None) when
    no_validation is set.
    """
    cfg = dict(config)
    analysis = NetworkTemporalCausalDataset(
        n_samples=int(n),
        v_dim=int(cfg["v_dim"]),
        T=int(cfg["T"]),
        network_avg_degree=int(cfg["network_avg_degree"]),
        config=cfg,
        seed=int(seed),
    )
    if no_validation:
        # All n units are used for training and estimation; with no
        # validation set, training uses the training loss for the LR
        # scheduler and early stopping.
        return analysis, None
    if validation_n is None:
        validation_n = max(500, int(round(n * float(cfg["n_samples_val_ratio"]))))
    validation = NetworkTemporalCausalDataset(
        n_samples=int(validation_n),
        v_dim=int(cfg["v_dim"]),
        T=int(cfg["T"]),
        network_avg_degree=int(cfg["network_avg_degree"]),
        config=cfg,
        seed=int(seed) + 100_000,
    )
    return analysis, validation


def _model_diagnostics(fitted: Mapping, dataset) -> Dict:
    """Summarize fitted own-treatment propensities e_hat (averaged over
    decisions): mean, SD, and correlation with the baseline risk score."""
    e_bar = fitted["e_hat"].mean(axis=1)
    r0 = np.asarray(dataset.baseline_risk_score)
    corr = float(np.corrcoef(e_bar, r0)[0, 1]) if np.std(e_bar) > 0 else None
    if corr is not None and not np.isfinite(corr):
        corr = None
    return {
        "e_hat_mean": float(e_bar.mean()),
        "e_hat_sd": float(e_bar.std()),
        "corr_ehat_baseline_risk": corr,
    }


def _observed_data_diagnostics(dataset, config: Mapping) -> Dict:
    """Summarize the observed data: final outcome, spillover proportions and
    neighbor-treated counts per decision, degree distribution, own-treatment
    rates and decision-pattern counts. Used to tell thin exposure support
    apart from a low-variance outcome."""
    y_t = dataset.y.numpy()[:, -1, 0].astype(np.float64)
    steps = decision_steps_for(config)
    d_decision = dataset.d_xs.numpy()[:, steps, 0].astype(np.float64)
    d_count_decision = np.rint(dataset.d_xs_count.numpy()[:, steps, 0]).astype(int)
    degree = np.rint(dataset.n_neighbors.numpy().flatten()).astype(int)
    x_decision = dataset.x.numpy()[:, steps, 0].astype(np.float64)
    return {
        "outcome_T": {
            "mean": float(y_t.mean()),
            "sd": float(y_t.std(ddof=1)) if len(y_t) > 1 else 0.0,
            "min": float(y_t.min()),
            "max": float(y_t.max()),
        },
        "spillover_by_decision": [
            {
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "q05": float(np.quantile(values, 0.05)),
                "q50": float(np.quantile(values, 0.50)),
                "q95": float(np.quantile(values, 0.95)),
            }
            for values in d_decision.T
        ],
        "spillover_count_freq_by_decision": [
            np.bincount(np.clip(values, 0, 8), minlength=9).tolist()
            for values in d_count_decision.T
        ],
        "degree_freq": np.bincount(np.clip(degree, 0, 20), minlength=21).tolist(),
        "own_treatment_rate_by_decision": [
            float(values.mean()) for values in x_decision.T
        ],
        "decision_pattern_counts": {
            str(dataset.decision_pattern_labels[int(pattern_id)]): int(count)
            for pattern_id, count in dataset.decision_pattern_counts.items()
        },
    }


def _method_inputs(method: str, result, sequences):
    """Extract per-unit (weights, values) for each target from a method's output.

    For the kernel IPW methods these are the IPW weights and observed outcomes;
    for OLS they are unit weights and fitted predictions. Returns
    (inputs, diagnostics), where inputs feeds the bootstrap and sensitivity
    analysis and diagnostics holds the scalar results per target.
    """
    inputs = {}
    diagnostics = {}
    for name, seq in sequences.items():
        if method in ("lstm_kipw", "oracle_kipw"):
            entry = result["estimates"][seq]
            diagnostics[name] = {
                key: _jsonable(value)
                for key, value in entry.items()
                if key not in ("weights", "y")
            }
            if entry.get("weights") is not None:
                inputs[name] = (entry["weights"], entry["y"])
        else:
            entry = result[seq]
            diagnostics[name] = {"estimate": entry.get("estimate")}
            inputs[name] = (
                entry.get(
                    "weights",
                    np.ones(len(entry["predictions"]), dtype=np.float64),
                ),
                entry["predictions"],
            )
    return inputs, diagnostics


def run_replication(task: Mapping) -> Dict:
    """Run one Monte Carlo replication described by the task dict.

    Generates the analysis (and validation) network with seed
    task["seed"] + 10000 * rep_index, trains the propensity model, computes
    the LSTM K-IPW, oracle K-IPW and OLS estimates for all targets, and
    attaches network-block bootstrap intervals, optional retraining-bootstrap
    intervals and optional Gamma sensitivity bounds. Returns a JSON-ready dict.
    """
    rep_index = int(task["rep_index"])
    seed = int(task["seed"]) + rep_index * 10_000
    cfg = copy.deepcopy(CONFIG)
    cfg.update(task.get("config_overrides", {}))
    cfg["num_workers"] = 0
    cfg["pin_memory"] = False
    cfg["persistent_workers"] = False
    cfg["use_amp"] = bool(task.get("use_amp", cfg.get("use_amp", True)))

    threads = int(task.get("threads_per_worker", 1))
    torch.set_num_threads(max(1, threads))
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 2)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + 2)

    device_name = task.get("device", "auto")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda" and task.get("sensitivity_gammas"):
        from ..training.gpu_runtime import configure_cuda_fp32
        configure_cuda_fp32()

    analysis, validation = make_analysis_and_validation(
        cfg,
        int(task["n"]),
        seed,
        task.get("validation_n"),
        no_validation=bool(task.get("no_validation", False)),
    )
    bandwidth = float(task["bandwidth"])
    analysis.spillover_bandwidth = bandwidth

    torch.manual_seed(seed + 3)
    model = train_model_spillover_distributional(analysis, validation, cfg, device)

    save_weights_dir = task.get("save_weights_dir")
    if save_weights_dir:
        rep_dir = Path(save_weights_dir) / f"rep_{rep_index:03d}"
        rep_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), rep_dir / "main_model.pt")

    sequences = {name: tuple(seq) for name, seq in task["sequences"].items()}
    sequence_list = list(sequences.values())
    truncation = float(task.get("weight_truncation_percentile", 100.0))

    lstm = estimate_ate_ipw_gaussian_kernel(
        model,
        analysis,
        sequence_list,
        device,
        bandwidth=bandwidth,
        weight_truncation_percentile=truncation,
        return_weights=True,
    )
    model_diag = _model_diagnostics(lstm["propensity_scores"], analysis)
    oracle = estimate_ate_oracle_kipw(
        analysis,
        cfg,
        sequence_list,
        bandwidth=bandwidth,
        weight_truncation_percentile=truncation,
        return_weights=True,
    )
    ols = fit_predict_ols_baseline(
        analysis,
        sequence_list,
        decision_steps_for(cfg),
        return_predictions=True,
    )

    raw_results = {
        "lstm_kipw": lstm,
        "oracle_kipw": oracle,
        "ols": ols,
    }
    method_diagnostics = {}
    bootstraps = {}
    sensitivity = {}
    pairs = contrast_pairs()
    n_partitions = int(task.get("bootstrap_partitions", 3))
    k_hops = int(task.get("bootstrap_k_hops", 2))
    shared_partitions = None
    if int(task["n_bootstrap"]) > 0:
        partition_rng = np.random.default_rng(seed + 699_999)
        shared_partitions = [
            k_hop_max_partition(analysis.adj_matrix, k_hops=k_hops, rng=partition_rng)
            for _ in range(n_partitions)
        ]
    for method, raw in raw_results.items():
        inputs, diagnostics = _method_inputs(method, raw, sequences)
        method_diagnostics[method] = diagnostics
        if task.get("sensitivity_gammas") and method in ("lstm_kipw", "oracle_kipw"):
            from ..estimation.sensitivity import sensitivity_analysis
            sensitivity_inputs = {
                name: inputs.get(name, (np.zeros(1), np.zeros(1))) for name in sequences
            }
            M = len(decision_steps_for(cfg))
            sensitivity[method] = sensitivity_analysis(
                sensitivity_inputs, pairs, task["sensitivity_gammas"], M=M
            )
            if task.get("save_sensitivity_inputs_dir"):
                directory = Path(task["save_sensitivity_inputs_dir"])
                directory.mkdir(parents=True, exist_ok=True)
                arrays = {}
                for name, (weights, outcomes) in sensitivity_inputs.items():
                    arrays[f"{name}__weights"] = weights
                    arrays[f"{name}__outcomes"] = outcomes
                arrays["__metadata_json"] = np.asarray(json.dumps({
                    "n": int(task["n"]), "rep_index": rep_index, "seed": seed,
                    "method": method, "M": M, "sequences": sequences,
                }))
                destination = directory / f"rep_{rep_index:03d}_{method}.npz"
                temporary = destination.with_suffix(".npz.tmp")
                with temporary.open("wb") as handle:
                    np.savez_compressed(handle, **arrays)
                temporary.replace(destination)
        available_pairs = {
            name: pair
            for name, pair in pairs.items()
            if pair[0] in inputs and pair[1] in inputs
        }
        bootstraps[method] = joint_network_block_bootstrap(
            inputs,
            available_pairs,
            analysis.adj_matrix,
            n_boot=int(task["n_bootstrap"]),
            ci_level=float(task.get("ci_level", 0.95)),
            k_hops=k_hops,
            n_partitions=n_partitions,
            partitions=shared_partitions,
            rng=np.random.default_rng(seed + 700_000),
        )

    point_estimates = {method: {} for method in METHODS}
    for method in METHODS:
        for name in sequences:
            point_estimates[method][name] = method_diagnostics[method][name].get("estimate")
        for name, (a, b) in pairs.items():
            pa = point_estimates[method].get(a)
            pb = point_estimates[method].get(b)
            point_estimates[method][name] = (
                pa - pb if pa is not None and pb is not None else None
            )

    retrain_bootstrap = {}
    n_bootstrap_retrain = int(task.get("n_bootstrap_retrain", 0))
    n_reweight_per_retrain = int(task.get("n_reweight_per_retrain", 1))
    if task.get("bootstrap_retrain") and n_bootstrap_retrain > 0:
        from ..estimation.retrain_bootstrap import retrain_bootstrap_ipw
        print(
            f"[rep {rep_index}] running retrain bootstrap "
            f"({n_bootstrap_retrain} retrainings x {n_reweight_per_retrain} "
            f"reweight{'s' if n_reweight_per_retrain != 1 else ''} = "
            f"{n_bootstrap_retrain * n_reweight_per_retrain} draws)"
        )
        retrain_bootstrap["lstm_kipw"] = retrain_bootstrap_ipw(
            analysis,
            validation,
            cfg,
            device,
            sequence_list,
            sequences,
            pairs,
            bandwidth=bandwidth,
            weight_truncation_percentile=truncation,
            n_boot=n_bootstrap_retrain,
            k_hops=k_hops,
            n_partitions=n_partitions,
            ci_level=float(task.get("ci_level", 0.95)),
            seed=seed + 800_000,
            point_estimates=point_estimates["lstm_kipw"],
            n_reweight_per_retrain=n_reweight_per_retrain,
            log_fn=print,
        )

    return {
        "rep_index": rep_index,
        "seed": seed,
        "n": int(task["n"]),
        "validation_n": len(validation) if validation is not None else 0,
        "bandwidth": bandwidth,
        "device": device.type,
        "weight_truncation_percentile": truncation,
        "observed_data_diagnostics": _observed_data_diagnostics(analysis, cfg),
        "model_diagnostics": model_diag,
        "point_estimates": point_estimates,
        "target_diagnostics": method_diagnostics,
        "bootstrap": bootstraps,
        "retrain_bootstrap": retrain_bootstrap,
        "sensitivity": sensitivity,
    }


def _truth_with_contrasts(gt_levels: Mapping[str, Mapping]) -> Dict[str, float]:
    """Return true values for the 15 level targets and the 13 contrasts."""
    truth = {name: float(result["mean"]) for name, result in gt_levels.items()}
    for name, (a, b) in contrast_pairs().items():
        truth[name] = truth[a] - truth[b]
    return truth


def summarize_replications(replicates: Sequence[Mapping], truth: Mapping[str, float]) -> Dict:
    """Aggregate replications into per-target, per-method performance metrics.

    Reports bias, MSE and empirical SD of the point estimates, and for the
    network-block bootstrap intervals the coverage, mean width, mean bootstrap
    SE, its ratio to the empirical SD, and invalid/unreliable draw fractions.
    """
    output = {}
    for target_name, gt in truth.items():
        output[target_name] = {}
        for method in METHODS:
            estimates = np.asarray([
                rep["point_estimates"][method].get(target_name)
                for rep in replicates
                if rep["point_estimates"][method].get(target_name) is not None
            ], dtype=np.float64)
            errors = estimates - gt

            ci_entries = [
                rep["bootstrap"][method].get("summaries", {}).get(target_name)
                for rep in replicates
            ]
            ci_entries = [entry for entry in ci_entries if entry]
            valid_ci = [
                entry for entry in ci_entries
                if entry.get("ci_lower") is not None and entry.get("ci_upper") is not None
            ]
            covered = [
                entry["ci_lower"] <= gt <= entry["ci_upper"]
                for entry in valid_ci
            ]
            widths = [entry["ci_upper"] - entry["ci_lower"] for entry in valid_ci]
            ses = [entry["se"] for entry in valid_ci if entry.get("se") is not None]
            invalid_fractions = [1.0 - entry.get("valid_fraction", 0.0) for entry in ci_entries]
            empirical_sd = float(estimates.std(ddof=1)) if len(estimates) > 1 else None
            mean_bootstrap_se = float(np.mean(ses)) if ses else None

            output[target_name][method] = {
                "truth": float(gt),
                "n_point_valid": int(len(estimates)),
                "bias": float(errors.mean()) if len(errors) else None,
                "mse": float(np.mean(errors ** 2)) if len(errors) else None,
                "empirical_sd": empirical_sd,
                "n_ci_valid": int(len(valid_ci)),
                "coverage": float(np.mean(covered)) if covered else None,
                "mean_ci_width": float(np.mean(widths)) if widths else None,
                "mean_bootstrap_se": mean_bootstrap_se,
                "bootstrap_se_to_empirical_sd": (
                    mean_bootstrap_se / empirical_sd
                    if mean_bootstrap_se is not None
                    and empirical_sd is not None
                    and empirical_sd > 0
                    else None
                ),
                "mean_invalid_bootstrap_fraction": (
                    float(np.mean(invalid_fractions)) if invalid_fractions else None
                ),
                "reliable_ci_fraction": (
                    float(np.mean([bool(e.get("reliable")) for e in ci_entries]))
                    if ci_entries else None
                ),
            }
    return output


def build_arg_parser() -> argparse.ArgumentParser:
    """Return the command-line parser for the study."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", "--n-train", dest="n", type=int, default=5000)
    parser.add_argument("--R", "--n-rep", dest="R", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--bandwidth", type=float, default=0.03,
                        help="Gaussian kernel bandwidth h on the spillover proportion")
    parser.add_argument("--sensitivity-gammas", type=float, nargs="+",
                        help="Increasing joint-Gamma grid starting at 1; enables hidden-bias bounds")
    parser.add_argument("--save-sensitivity-inputs-dir",
                        help="Save individual IPW weights/outcomes for regridding without refitting")
    parser.add_argument(
        "--dgp-variant",
        choices=(
            "baseline",
            "burden_modified",
            "own_spillover_synergy",
            "burden_synergy",
        ),
        default=CONFIG["latent_nonlinear_variant"],
        help="Latent-outcome DGP variant (default: main design; others: stress tests).",
    )
    parser.add_argument(
        "--latent-spillover-burden-eta",
        type=float,
        default=CONFIG["latent_spillover_burden_eta"],
        help="eta_B in burden_modified/burden_synergy; must lie in [0, 1).",
    )
    parser.add_argument(
        "--latent-beta-xd",
        type=float,
        default=CONFIG["latent_beta_XD"],
        help="beta_XD of the own-by-spillover interaction term.",
    )
    parser.add_argument(
        "--treatment-variant",
        choices=("baseline", "worsening_quadratic"),
        default=CONFIG["treatment_nonlinear_variant"],
        help="worsening_quadratic adds theta_w2 W^2 to the treatment logit (main design).",
    )
    parser.add_argument(
        "--treatment-theta-w2", type=float,
        default=CONFIG["treatment_theta_w2"],
        help="theta_w2 of the squared-worsening term.",
    )
    parser.add_argument("--weight-truncation-percentile", type=float, default=100.0)
    parser.add_argument("--n-bootstrap", "--B", dest="n_bootstrap", type=int, default=100)
    parser.add_argument("--bootstrap-k-hops", type=int, default=2)
    parser.add_argument("--bootstrap-partitions", type=int, default=3)
    parser.add_argument("--ci-level", type=float, default=0.95)
    parser.add_argument(
        "--bootstrap-retrain", action="store_true",
        help="Also run the retraining bootstrap, which refits the LSTM on every draw (expensive).",
    )
    parser.add_argument(
        "--n-bootstrap-retrain", type=int, default=20,
        help="Number of LSTM refits for --bootstrap-retrain.",
    )
    parser.add_argument(
        "--n-reweight-per-retrain", type=int, default=1,
        help="Network-block reweights per refitted model; total draws = refits x reweights.",
    )
    parser.add_argument(
        "--save-weights-dir",
        help="Save each replication's fitted model as <dir>/rep_<NNN>/main_model.pt.",
    )
    parser.add_argument("--reference-paths-json")
    parser.add_argument("--reference-n", type=int, default=5000)
    parser.add_argument("--reference-seed", type=int, default=20260830)
    parser.add_argument("--reference-bandwidth", type=float, default=0.05)
    parser.add_argument("--gt-batches", type=int, default=100)
    parser.add_argument("--gt-inner", type=int, default=100)
    parser.add_argument("--gt-seed", type=int, default=20260829)
    parser.add_argument("--validation-n", type=int)
    parser.add_argument(
        "--no-validation", action="store_true",
        help="Train without a held-out validation network (early stopping uses the training loss).",
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--use-amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--threads-per-worker", type=int, default=1)
    parser.add_argument("--rep-start", type=int, default=0)
    parser.add_argument("--rep-count", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out", required=True)
    parser.add_argument("--log")
    return parser


def run_study(args) -> Dict:
    """Run the full study and return the final JSON payload.

    Validates arguments, checks that a resumed run uses the same design,
    builds reference paths, targets and ground truth, then runs the requested
    replications (args.rep_start to args.rep_start + rep_count), writing a
    checkpoint to args.out after each one.
    """
    if min(args.n, args.R, args.n_workers, args.threads_per_worker) <= 0:
        raise ValueError("n, R, n-workers and threads-per-worker must be positive")
    if args.sensitivity_gammas is not None:
        from ..estimation.sensitivity import validate_gammas
        validate_gammas(args.sensitivity_gammas)
    if args.save_sensitivity_inputs_dir and not args.sensitivity_gammas:
        raise ValueError("--save-sensitivity-inputs-dir requires --sensitivity-gammas")
    if args.sensitivity_gammas and Path(args.out).exists() and not args.resume:
        raise FileExistsError("Sensitivity output already exists; use --resume or a new output path")
    # Limit the main process's torch threads to the available CPUs before
    # reference-path and ground-truth computation.
    main_threads = _available_cpus()
    if args.sensitivity_gammas:
        main_threads = min(main_threads, int(args.threads_per_worker))
    torch.set_num_threads(max(1, main_threads))

    output_path = Path(args.out)
    previous = None
    if args.resume and output_path.exists():
        with output_path.open() as handle:
            previous = json.load(handle)
        old_args = previous.get("arguments", {})
        if old_args.get("sensitivity_gammas") != args.sensitivity_gammas:
            raise ValueError("Cannot resume with a different sensitivity Gamma grid")
        protected = (
            "n", "seed", "bandwidth", "dgp_variant",
            "latent_spillover_burden_eta", "latent_beta_xd",
            "treatment_variant", "treatment_theta_w2",
            "weight_truncation_percentile",
            "n_bootstrap", "bootstrap_k_hops", "bootstrap_partitions",
            "bootstrap_retrain", "n_bootstrap_retrain", "n_reweight_per_retrain",
            "ci_level", "reference_n", "reference_seed",
            "reference_bandwidth", "gt_batches", "gt_inner", "gt_seed",
            "validation_n", "no_validation", "epochs", "patience", "use_amp",
        )
        changed = [
            key for key in protected
            if key in old_args and old_args.get(key) != getattr(args, key)
        ]
        if changed:
            raise ValueError(
                "Cannot resume because the simulation design changed: "
                + ", ".join(changed)
            )

    config = copy.deepcopy(CONFIG)
    config["num_workers"] = 0
    config["latent_nonlinear_variant"] = args.dgp_variant
    if not 0.0 <= args.latent_spillover_burden_eta < 1.0:
        raise ValueError("--latent-spillover-burden-eta must be in [0, 1)")
    if args.latent_beta_xd < 0.0:
        raise ValueError("--latent-beta-xd must be nonnegative")
    config["latent_spillover_burden_eta"] = float(
        args.latent_spillover_burden_eta
    )
    config["latent_beta_XD"] = float(args.latent_beta_xd)
    config["treatment_nonlinear_variant"] = args.treatment_variant
    config["treatment_theta_w2"] = float(args.treatment_theta_w2)
    if args.epochs is not None:
        config["epochs"] = int(args.epochs)
    if args.patience is not None:
        config["early_stopping_patience"] = int(args.patience)

    if previous is not None and args.sensitivity_gammas:
        if previous.get("config_snapshot") != config:
            raise ValueError("Cannot resume: model configuration changed")
        if args.save_sensitivity_inputs_dir:
            from .sensitivity_replay import replay_saved_inputs
            replay_saved_inputs(previous, Path(args.save_sensitivity_inputs_dir),
                                args.sensitivity_gammas, verify_all=True)

    if previous is not None:
        d_paths = _validate_d_paths(
            previous["reference_paths"],
            len(decision_steps_for(config)),
        )
        reference_provenance = previous.get("reference_provenance", {})
    else:
        d_paths, reference_provenance = load_or_construct_reference_paths(
            config,
            args.reference_paths_json,
            args.reference_n,
            args.reference_seed,
            args.reference_bandwidth,
        )
    sequences = target_sequences(config, d_paths)
    _log(args.log, f"fixed reference paths: {d_paths}")

    if previous is not None and previous.get("ground_truth_levels"):
        gt_levels = previous["ground_truth_levels"]
    else:
        gt_levels = compute_ground_truth(
            config,
            sequences,
            args.gt_batches,
            args.gt_inner,
            args.gt_seed,
            args.log,
        )
    truth = _truth_with_contrasts(gt_levels)

    completed = {}
    if previous is not None:
        completed = {
            int(rep["rep_index"]): rep
            for rep in previous.get("replications", [])
        }

    rep_count = args.rep_count if args.rep_count is not None else args.R
    requested_indices = list(range(args.rep_start, args.rep_start + rep_count))
    to_run = [idx for idx in requested_indices if idx not in completed]

    config_overrides = {}
    config_overrides["latent_nonlinear_variant"] = args.dgp_variant
    config_overrides["latent_spillover_burden_eta"] = float(
        args.latent_spillover_burden_eta
    )
    config_overrides["latent_beta_XD"] = float(args.latent_beta_xd)
    config_overrides["treatment_nonlinear_variant"] = args.treatment_variant
    config_overrides["treatment_theta_w2"] = float(args.treatment_theta_w2)
    if args.epochs is not None:
        config_overrides["epochs"] = int(args.epochs)
    if args.patience is not None:
        config_overrides["early_stopping_patience"] = int(args.patience)

    base_task = {
        "n": args.n,
        "seed": args.seed,
        "bandwidth": args.bandwidth,
        "weight_truncation_percentile": args.weight_truncation_percentile,
        "n_bootstrap": args.n_bootstrap,
        "bootstrap_k_hops": args.bootstrap_k_hops,
        "bootstrap_partitions": args.bootstrap_partitions,
        "bootstrap_retrain": args.bootstrap_retrain,
        "n_bootstrap_retrain": args.n_bootstrap_retrain,
        "n_reweight_per_retrain": args.n_reweight_per_retrain,
        "save_weights_dir": args.save_weights_dir,
        "ci_level": args.ci_level,
        "validation_n": args.validation_n,
        "no_validation": args.no_validation,
        "device": args.device,
        "use_amp": args.use_amp,
        "threads_per_worker": args.threads_per_worker,
        "sequences": sequences,
        "config_overrides": config_overrides,
        "sensitivity_gammas": args.sensitivity_gammas,
        "save_sensitivity_inputs_dir": args.save_sensitivity_inputs_dir,
    }

    def checkpoint():
        """Write all completed requested replications, their summaries and the
        design metadata to the output JSON; return the payload."""
        from ..estimation.sensitivity import summarize_sensitivity
        reps = [completed[idx] for idx in sorted(completed) if idx in requested_indices]
        payload = {
            "schema_version": 3,
            "design": {
                "analysis_network_n": int(args.n),
                "outer_replications_R": int(args.R),
                "gt_batches_B_GT": int(args.gt_batches),
                "gt_inner": int(args.gt_inner),
                "bootstrap_replicates_B": int(args.n_bootstrap),
                "inference": "disjoint_k_hop_max_cluster_bootstrap",
                "inference_conditions_on_fitted_nuisance": True,
                "latent_nonlinear_variant": args.dgp_variant,
                "latent_spillover_burden_eta": config["latent_spillover_burden_eta"],
                "latent_beta_XD": config["latent_beta_XD"],
                "treatment_nonlinear_variant": args.treatment_variant,
            },
            "arguments": vars(args),
            "own_treatment_paths": CANONICAL_PATHS,
            "reference_paths": d_paths,
            "expanded_target_sequences": sequences,
            "reference_provenance": reference_provenance,
            "ground_truth_levels": gt_levels,
            "truth_with_contrasts": truth,
            "replications": reps,
            "summary": summarize_replications(reps, truth),
            "sensitivity_summary": summarize_sensitivity(reps, truth),
            "config_snapshot": config,
        }
        _write_json(output_path, payload)
        return payload

    _log(args.log, f"running {len(to_run)} of {len(requested_indices)} requested replications")
    use_cuda = args.device == "cuda" or (
        args.device == "auto" and torch.cuda.is_available()
    )
    # CUDA and sensitivity runs use the one-process-per-replication path
    # below so memory is released after every replication.
    if args.n_workers <= 1 and not args.sensitivity_gammas and not use_cuda:
        for idx in to_run:
            completed[idx] = run_replication({**base_task, "rep_index": idx})
            checkpoint()
            _log(args.log, f"replication {idx} complete")
    else:
        if args.n_workers > 1 and use_cuda:
            raise ValueError("Use --n-workers 1 with a single CUDA device")
        # 'spawn' rather than 'fork': this process has already initialized
        # torch/BLAS thread pools, which forked children can deadlock on.
        # Each replication runs in its own single-worker pool that is shut
        # down when it finishes, so memory is returned to the OS between
        # replications; a new pool starts as soon as any slot frees.
        context = mp.get_context("spawn")
        pending = list(to_run)
        running = {}
        try:
            while pending or running:
                while pending and len(running) < args.n_workers:
                    idx = pending.pop(0)
                    pool = ProcessPoolExecutor(max_workers=1, mp_context=context)
                    future = pool.submit(run_replication, {**base_task, "rep_index": idx})
                    running[future] = (idx, pool)
                done, _ = wait(running, return_when=FIRST_COMPLETED)
                for future in done:
                    idx, pool = running.pop(future)
                    pool.shutdown()
                    completed[idx] = future.result()
                    checkpoint()
                    _log(args.log, f"replication {idx} complete")
        finally:
            for _, pool in running.values():
                pool.shutdown(wait=False, cancel_futures=True)
    return checkpoint()


def main():
    """Command-line entry point."""
    args = build_arg_parser().parse_args()
    if args.R <= 0 or args.n <= 0:
        raise ValueError("R and n must be positive")
    if args.rep_count is not None and args.rep_count <= 0:
        raise ValueError("rep-count must be positive")
    run_study(args)


if __name__ == "__main__":
    main()
