"""Merge sharded bias_mse_study runs (--rep-start/--rep-count) into one study.json.

Replication seeds depend only on rep_index, so disjoint shards reproduce one
sequential run. Checks that the shards share one design and cover [0, R)
exactly, then recomputes the across-replication summary.

  python -m model.pipeline.merge_bias_mse_shards \
    --shards results/n50000/shard_*/study.json --R 100 \
    --out results/n50000/study.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .bias_mse_study import _write_json, summarize_replications

# Arguments that may differ between shards. Each replication records its own device.
SHARD_LOCAL_ARGUMENTS = frozenset({
    "rep_start", "rep_count", "out", "log", "save_weights_dir", "resume",
    "n_workers", "threads_per_worker", "device",
})
# Design, fixed inputs and approximate truth: every shard must agree exactly.
SHARED_BLOCKS = (
    "own_treatment_paths", "reference_paths", "expanded_target_sequences",
    "reference_provenance", "ground_truth_levels", "truth_with_contrasts",
    "config_snapshot",
)


def merge(shard_paths, R):
    """Check that the shards share one design and tile [0, R); return the merged study dict."""
    shards = [json.loads(Path(p).read_text()) for p in shard_paths]
    if not shards:
        raise ValueError("no shards given")
    first = shards[0]
    design = {k: v for k, v in first["arguments"].items() if k not in SHARD_LOCAL_ARGUMENTS}
    for path, shard in zip(shard_paths, shards):
        for block in SHARED_BLOCKS:
            if shard.get(block) != first.get(block):
                raise ValueError(f"{path}: '{block}' differs from {shard_paths[0]}")
        args = {k: v for k, v in shard["arguments"].items() if k not in SHARD_LOCAL_ARGUMENTS}
        if args != design:
            diff = sorted(k for k in set(args) | set(design) if args.get(k) != design.get(k))
            raise ValueError(f"{path}: arguments differ: {diff}")

    reps = {}
    for path, shard in zip(shard_paths, shards):
        for rep in shard["replications"]:
            idx = int(rep["rep_index"])
            if idx in reps:
                raise ValueError(f"replication {idx} appears in more than one shard ({path})")
            reps[idx] = rep
    missing = sorted(set(range(R)) - set(reps))
    extra = sorted(set(reps) - set(range(R)))
    if missing or extra:
        raise ValueError(f"shards do not cover [0, {R}): missing={missing} extra={extra}")

    merged = dict(first)
    merged["replications"] = [reps[i] for i in range(R)]
    merged["summary"] = summarize_replications(merged["replications"], first["truth_with_contrasts"])
    merged["arguments"] = {**first["arguments"], "rep_start": 0, "rep_count": None, "R": R}
    merged["merged_from"] = [str(p) for p in shard_paths]
    return merged


def main(argv=None):
    """Command-line entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", nargs="+", required=True, help="Shard study.json files")
    parser.add_argument("--R", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    merged = merge(args.shards, args.R)
    _write_json(Path(args.out), merged)
    print(f"merged {len(args.shards)} shards, {args.R} replications -> {args.out}")


if __name__ == "__main__":
    main()
