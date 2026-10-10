"""Merge sharded run_sensitivity outputs into one complete study.

Replication seeds depend only on rep_index, so disjoint shards reproduce a
single sequential run. Checks that the shards share one design and cover
[0, R) exactly, collects the saved weight archives (NPZ), and recomputes the
across-replication summaries (JSON).

  python -m model.pipeline.merge_sensitivity_shards \
    --shards results/sens_n50000/shard_* --out-dir results/sensitivity_n50000_R100
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .bias_mse_study import _write_json, summarize_replications
from .run_sensitivity import require_complete, shard_range
from ..estimation.sensitivity import summarize_sensitivity

SENSITIVITY_METHODS = ("lstm_kipw", "oracle_kipw")

# Arguments that legitimately differ between shards.
SHARD_LOCAL_ARGUMENTS = frozenset({
    "rep_start", "rep_count", "out", "log", "save_sensitivity_inputs_dir",
    "save_weights_dir", "resume", "n_workers", "threads_per_worker",
})

# Design, fixed inputs and approximate truth: every shard must agree exactly.
SHARED_BLOCKS = (
    "own_treatment_paths", "reference_paths", "expanded_target_sequences",
    "reference_provenance", "ground_truth_levels", "truth_with_contrasts",
    "config_snapshot",
)


def load_shard(directory):
    """Read one shard and reject it unless its own replication range is filled."""
    study = json.loads((directory / "study.json").read_text())
    manifest = json.loads((directory / "run_manifest.json").read_text())
    require_complete(study)
    return study, manifest


def check_one_design(shards):
    """Raise ValueError unless all shards share source code, design arguments and fixed inputs."""
    (first_dir, first_study, first_manifest) = shards[0]
    for directory, study, manifest in shards[1:]:
        if manifest.get("source_sha256") != first_manifest.get("source_sha256"):
            raise ValueError(f"Computation code differs between {first_dir} and {directory}")
        if manifest.get("reference_paths") != first_manifest.get("reference_paths"):
            raise ValueError(f"Fixed reference potential outcomes differ: {directory}")
        differing = sorted(
            key for key in set(first_study["arguments"]) | set(study["arguments"])
            if key not in SHARD_LOCAL_ARGUMENTS
            and first_study["arguments"].get(key) != study["arguments"].get(key)
        )
        if differing:
            raise ValueError(f"Study design differs in {directory}: {differing}")
        for block in SHARED_BLOCKS:
            if first_study.get(block) != study.get(block):
                raise ValueError(f"{block} differs between {first_dir} and {directory}")


def collect_replications(shards):
    """Map each replication index to (shard directory, record); shards must tile [0, R) without gap or overlap."""
    R = int(shards[0][1]["arguments"]["R"])
    owner = {}
    for directory, study, _ in shards:
        for rep in study["replications"]:
            index = int(rep["rep_index"])
            if index in owner:
                raise ValueError(f"Replication {index} appears in {owner[index][0]} and {directory}")
            owner[index] = (directory, rep)
    missing = [i for i in range(R) if i not in owner]
    if missing:
        raise ValueError(f"Missing {len(missing)} of {R} replications: {missing[:10]}")
    extra = sorted(i for i in owner if i >= R)
    if extra:
        raise ValueError(f"Replication indices outside [0, {R}): {extra[:10]}")
    return R, owner


def gather_weights(owner, destination, symlink=False):
    """Copy (or symlink) each replication's saved IPW weights and outcomes into one directory."""
    destination.mkdir(parents=True, exist_ok=True)
    for index, (directory, _) in sorted(owner.items()):
        for method in SENSITIVITY_METHODS:
            name = f"rep_{index:03d}_{method}.npz"
            source = directory / "weights" / name
            if not source.is_file():
                raise FileNotFoundError(f"Saved weights missing: {source}")
            target = destination / name
            if symlink:
                target.symlink_to(source.resolve())
            else:
                shutil.copy2(source, target)


def merge(shard_dirs, out_dir, symlink=False):
    """Write the merged study.json, run_manifest.json and weights/ to out_dir; return the merged study."""
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {out_dir}")
    if len(shard_dirs) < 2:
        raise ValueError("Merging needs at least two shard directories")

    shards = [(directory, *load_shard(directory)) for directory in shard_dirs]
    check_one_design(shards)
    R, owner = collect_replications(shards)

    out_dir.mkdir(parents=True, exist_ok=True)
    gather_weights(owner, out_dir / "weights", symlink)

    merged = dict(shards[0][1])
    merged["replications"] = [owner[i][1] for i in range(R)]
    truth = merged["truth_with_contrasts"]
    merged["summary"] = summarize_replications(merged["replications"], truth)
    merged["sensitivity_summary"] = summarize_sensitivity(merged["replications"], truth)
    merged["arguments"] = dict(merged["arguments"])
    merged["arguments"].update(
        rep_start=0, rep_count=R, resume=False,
        out=str(out_dir / "study.json"), log=str(out_dir / "progress.log"),
        save_sensitivity_inputs_dir=str(out_dir / "weights"),
    )
    _write_json(out_dir / "study.json", merged)

    manifest = dict(shards[0][2])
    manifest["arguments"] = merged["arguments"]
    manifest["merged_from"] = [
        {"directory": str(directory), "rep_range": list(shard_range(study["arguments"]))}
        for directory, study, _ in shards
    ]
    _write_json(out_dir / "run_manifest.json", manifest)
    return merged


def main(argv=None):
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--symlink", action="store_true",
                        help="Link the NPZ inputs instead of copying; shard directories must stay")
    args = parser.parse_args(argv)
    merged = merge(args.shards, args.out_dir, args.symlink)
    print(f"Merged {len(merged['replications'])} replications from {len(args.shards)} shards: "
          f"{args.out_dir / 'study.json'}")
    print(f"Verify with: python -m model.pipeline.run_sensitivity "
          f"--out-dir {args.out_dir} --validate-only")


if __name__ == "__main__":
    main()
