"""
Keep what seed pooling needs from one seed's estimate files.

    python -m model.real_data.compact_seed WORK_DIR SEED OUT.json.gz

WORK_DIR holds <domain>/estimates/outcome_expectations_<summary>_<month>.json
written by estimate.py with --save-draws. The output maps each estimate file to
its effect estimates, intervals and bootstrap draws, plus the effective sample
size of the baseline path.
"""
import ast
import gzip
import json
import sys
from pathlib import Path

CASES = [("Business_Economic_Restrictions", "2020-09"), ("Business_Economic_Restrictions", "2020-10"),
         ("Education_Childcare", "2020-07"), ("Education_Childcare", "2020-08")]
AGGS = ["last_week", "final_month_average"]


def never_ess(outcome_expectations):
    """Effective sample size of the baseline path (no treated month)."""
    for key, info in outcome_expectations.items():
        seq = ast.literal_eval(key)
        if not any(seq[0::2]):
            return info.get("effective_n")
    return None


def compact(work: Path, seed: int) -> dict:
    """Collect effect estimates, intervals and draws for every case and summary."""
    files = {}
    for family, end in CASES:
        for agg in AGGS:
            rel = f"{family}/estimates/outcome_expectations_{agg}_{end}.json"
            data = json.loads((work / rel).read_text())
            ate = {}
            for key, row in data["ate_results"].items():
                if row.get("ate_estimate") is None:
                    continue
                if not row.get("draws"):
                    raise ValueError(f"{rel} has no draws; rerun estimate.py with --save-draws")
                ate[key] = {k: row[k] for k in ("ate_estimate", "ci_lower", "ci_upper", "draws")}
            files[rel] = {"ate_results": ate, "never_effective_n": never_ess(data["outcome_expectations"])}
    return {"seed": seed, "files": files}


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    work, seed, out = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    with gzip.open(tmp, "wt") as f:
        json.dump(compact(work, seed), f)
    tmp.replace(out)
