"""
Pool the application results over training seeds (paper Section 6).

    python real_data/pool_seeds.py OUT_DIR real_data/runs/seed_*.json.gz

For every outcome month, summary and policy path estimated in all seeds:
- point estimate: mean of the per-seed estimates;
- 95% interval: 2.5 and 97.5 percentiles of the pooled per-seed bootstrap draws,
  so it reflects refitting variation as well as block-resampling variation.

Writes OUT_DIR/<domain>/estimates/*.json (read by plot_policy_effects.py) and
OUT_DIR/summary.csv.
"""
import csv
import gzip
import json
import sys
from pathlib import Path

import numpy as np


def main(out_dir: Path, seed_files: list[str]) -> None:
    """Pool the seed files into OUT_DIR; paths missing from any seed are dropped and reported."""
    seeds = [json.load(gzip.open(f, "rt")) for f in seed_files]
    ids = [s["seed"] for s in seeds]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate seeds")
    rows = []
    for rel in seeds[0]["files"]:
        per_seed = [s["files"][rel]["ate_results"] for s in seeds]
        keys = set.intersection(*[set(r) for r in per_seed])
        dropped = set().union(*[set(r) for r in per_seed]) - keys
        ate = {}
        for key in sorted(keys):
            est = np.array([r[key]["ate_estimate"] for r in per_seed], float)
            draws = np.concatenate([np.asarray(r[key]["draws"], float) for r in per_seed])
            draws = draws[np.isfinite(draws)]
            ate[key] = {"ate_estimate": float(est.mean()),
                        "ci_lower": float(np.percentile(draws, 2.5)),
                        "ci_upper": float(np.percentile(draws, 97.5)),
                        "seed_sd": float(est.std(ddof=1)) if len(est) > 1 else None,
                        "n_seeds": len(est), "n_draws": int(draws.size)}
            rows.append({"file": rel, "path": key, **{k: ate[key][k] for k in
                         ("ate_estimate", "ci_lower", "ci_upper", "seed_sd", "n_seeds", "n_draws")}})
        never = [s["files"][rel]["never_effective_n"] for s in seeds]
        out = {"ate_results": ate, "outcome_expectations": {},
               "seed_pooling": {"seeds": ids, "paths_dropped_not_in_all_seeds": sorted(dropped),
                                "never_effective_n_by_seed": never,
                                "point": "mean of per-seed estimates",
                                "interval": "2.5/97.5 percentiles of pooled per-seed bootstrap draws"}}
        path = out_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=1))
        if dropped:
            print(f"{rel}: {len(dropped)} path(s) not estimated in every seed were dropped")
    with (out_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"pooled {len(ids)} seeds into {out_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] in ("-h", "--help"):
        sys.exit(__doc__)
    main(Path(sys.argv[1]), sys.argv[2:])
