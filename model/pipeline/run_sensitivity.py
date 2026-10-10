"""Run the Gamma sensitivity study of Section 4 on simulated networks.

The design is fixed: M=4 decision times, kernel bandwidth h=0.03, R=100
replications by default, no weight truncation, and the Gamma grid
DEFAULT_GAMMAS. Each replication saves its IPW weights and outcomes so that
bounds can be recomputed without refitting. Runs can be split into
replication shards (--rep-start/--rep-count), resumed (--resume), and checked
by replaying the saved weights (--validate-only). --dry-run prints the
resolved design without computing anything.

From the repository root:
  python -m model.pipeline.run_sensitivity --n 5000 --dry-run
  python -m model.pipeline.run_sensitivity --n 5000 --device cpu --n-workers 8
  python -m model.pipeline.run_sensitivity --n 50000 --device cuda
  python -m model.pipeline.run_sensitivity --n 50000 --device cuda --resume
  python -m model.pipeline.run_sensitivity --n 50000 --rep-start 0 --rep-count 25 --out-dir shard_00
  python -m model.pipeline.run_sensitivity --smoke --out-dir results/smoke

The default output directory is results/sensitivity_n{n}_R{R}.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path

# Thread limits must be set before NumPy/PyTorch are imported.
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

from .bias_mse_study import build_arg_parser, run_study, _write_json
from .sensitivity_replay import replay_saved_inputs
from ..estimation.sensitivity import DEFAULT_GAMMAS, validate_gammas
from ..training.gpu_runtime import runtime_metadata


def source_hashes():
    """SHA-256 of every source file that affects the computation, keyed by relative path."""
    root = Path(__file__).resolve().parents[1]
    files = [root / "config.py", root / "__init__.py", root / "pipeline/bias_mse_study.py",
             root / "pipeline/run_sensitivity.py", root / "pipeline/sensitivity_replay.py"]
    for folder in ("models", "data", "training", "estimation"):
        files.extend((root / folder).glob("*.py"))
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(files)}


def parser():
    """Command-line parser for this script."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=50000, help="Analysis network size (the paper uses 5000 and 50000)")
    p.add_argument("--out-dir", type=Path, help="Default: results/sensitivity_n{n}_R{R}")
    p.add_argument("--R", type=int, default=100)
    p.add_argument("--rep-start", type=int, default=0,
                   help="First replication index of this shard")
    p.add_argument("--rep-count", type=int,
                   help="Replications in this shard; default is all R. Shards need separate --out-dir")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    p.add_argument("--n-workers", type=int, default=1)
    p.add_argument("--threads-per-worker", type=int, default=1)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--validate-only", action="store_true", help="Check complete JSON and saved-weight replay; never fit or plot")
    p.add_argument("--dry-run", action="store_true", help="Print the resolved design without creating files or training")
    p.add_argument("--smoke", action="store_true", help="TEST ONLY: n=128, R=1, one epoch; use a separate output directory")
    return p


def study_arguments(options):
    """Translate the command-line options into bias_mse_study arguments for the paper design.

    Fixes the bandwidth, training schedule, ground-truth Monte Carlo size,
    Gamma grid and the reference potential-outcome paths shared across n,
    and validates the shard range and worker settings.
    """
    if options.out_dir is None:
        if options.smoke:
            raise ValueError("--smoke requires a separate --out-dir; never mix tests and paper results")
        options.out_dir = Path(f"results/sensitivity_n{options.n}_R{options.R}")
    root = options.out_dir.resolve()
    if options.smoke and (options.rep_start != 0 or options.rep_count is not None):
        raise ValueError("--smoke runs one replication; do not shard it")
    reference = Path(__file__).resolve().parents[1] / "sensitivity_reference_paths.json"
    args = build_arg_parser().parse_args([
        "--n", str(options.n), "--R", str(options.R), "--bandwidth", "0.03",
        "--epochs", "300", "--patience", "20", "--n-bootstrap", "0",
        "--gt-batches", "100", "--gt-inner", "100",
        "--weight-truncation-percentile", "100", "--no-use-amp",
        "--reference-paths-json", str(reference), "--device", options.device,
        "--n-workers", str(options.n_workers), "--threads-per-worker", str(options.threads_per_worker),
        "--sensitivity-gammas", *[str(g) for g in DEFAULT_GAMMAS],
        "--save-sensitivity-inputs-dir", str(root / "weights"),
        "--out", str(root / "study.json"), "--log", str(root / "progress.log"),
        "--rep-start", str(options.rep_start),
        *(["--rep-count", str(options.rep_count)] if options.rep_count is not None else []),
        *(["--resume"] if options.resume else []),
    ])
    validate_gammas(args.sensitivity_gammas)
    if options.smoke:
        args.n, args.R, args.validation_n = 128, 1, 64
        args.epochs, args.patience = 1, 1
        args.gt_batches, args.gt_inner = 2, 2
        args.rep_start, args.rep_count = 0, 1
    if options.n <= 0 or options.R <= 0 or options.n_workers <= 0 or options.threads_per_worker <= 0:
        raise ValueError("R, n-workers and threads-per-worker must be positive")
    count = options.R if options.rep_count is None else options.rep_count
    if options.rep_start < 0 or count <= 0 or options.rep_start + count > options.R:
        raise ValueError(
            f"Shard [{options.rep_start}, {options.rep_start + count}) is outside [0, {options.R})")
    if options.device != "cpu" and options.n_workers > 1:
        raise ValueError("Use one worker for auto/CUDA; choose cpu explicitly for multiple workers")
    return args


@contextlib.contextmanager
def _exclusive_lock(path):
    """Hold an exclusive file lock so two jobs cannot write the same output directory (no-op without fcntl)."""
    try:
        import fcntl
    except ImportError:  # Windows
        yield
        return
    with path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another job is using this output directory") from exc
        yield


def shard_range(arguments):
    """Return (first replication index, number of replications) of a run."""
    start = int(arguments.get("rep_start", 0))
    count = int(arguments.get("rep_count") or arguments["R"])
    return start, count


def require_complete(payload):
    """Raise ValueError unless the study contains exactly the replications of its shard range."""
    start, count = shard_range(payload["arguments"])
    indices = sorted(r["rep_index"] for r in payload["replications"])
    if indices != list(range(start, start + count)):
        raise ValueError(
            f"Incomplete study: {len(indices)}/{count} replications in "
            f"[{start}, {start + count}). Resume before final export.")


def execute(options):
    """Dry-run, validate, or run (and on --resume, continue) the study, then verify the saved weights by replay."""
    args = study_arguments(options)
    root = options.out_dir.resolve()
    if options.dry_run:
        print(json.dumps({"action": "validate_only" if options.validate_only else "simulation",
                          "training_started": False, "arguments": vars(args)}, indent=2))
        return
    if options.validate_only:
        payload = json.loads((root / "study.json").read_text())
        require_complete(payload)
        replay_saved_inputs(payload, root / "weights", payload["arguments"]["sensitivity_gammas"], verify_all=True)
        print(f"Complete study and saved-weight replay verified: {root / 'study.json'}")
        return

    # Fail before generating data or computing truth if CUDA was requested but unavailable.
    runtime = runtime_metadata(options.device)
    manifest = {"arguments": vars(args), "source_sha256": source_hashes(),
                "purpose": "smoke_test_not_for_paper" if options.smoke else "paper_simulation",
                "runtime_history": [runtime],
                "reference_paths": json.loads(Path(args.reference_paths_json).read_text())}
    manifest_path = root / "run_manifest.json"
    if options.resume:
        old = json.loads(manifest_path.read_text())
        if old.get("purpose", "paper_simulation") != manifest["purpose"]:
            raise ValueError("Cannot mix smoke tests and paper simulations")
        manifest["runtime_history"] = old.get("runtime_history", []) + [runtime]
        if old["source_sha256"] != manifest["source_sha256"] or old["reference_paths"] != manifest["reference_paths"]:
            raise ValueError("Cannot resume: computation code or reference potential outcomes changed")
        # R and rep-count may grow, so a short run can expand into the full
        # shard; rep-start may not change, since it determines which seeds run.
        ignore = {"resume", "R", "rep_count", "n_workers", "threads_per_worker"}
        changed = [k for k, v in old["arguments"].items() if k not in ignore and vars(args).get(k) != v]
        shrunk = shard_range(vars(args))[1] < shard_range(old["arguments"])[1]
        if changed or args.R < old["arguments"]["R"] or shrunk:
            raise ValueError(
                f"Cannot resume with changed design, smaller R or a shrunken shard: {changed}")
        if not (root / "study.json").exists():
            # No checkpoint means the run stopped during the ground-truth
            # computation; restarting is safe only if no weights were saved.
            if (root / "weights").exists() and any((root / "weights").iterdir()):
                raise ValueError("Weights exist without a checkpoint; use a new output directory")
    else:
        if root.exists() and any(root.iterdir()):
            raise FileExistsError("Output directory is not empty; use --resume or a new --out-dir")
        root.mkdir(parents=True, exist_ok=True)
        # Exclusive creation also rejects a second new job using this directory.
        with manifest_path.open("x") as handle:
            json.dump(manifest, handle, indent=2)

    with _exclusive_lock(root / ".run.lock"):
        _write_json(manifest_path, manifest)
        payload = run_study(args)
        require_complete(payload)
        replay_saved_inputs(payload, root / "weights", args.sensitivity_gammas, verify_all=True)
        _write_json(manifest_path, manifest)
    print(f"Complete JSON results: {root / 'study.json'}; reusable inputs: {root / 'weights'}")


def main(argv=None):
    """Command-line entry point."""
    execute(parser().parse_args(argv))


if __name__ == "__main__":
    main()
