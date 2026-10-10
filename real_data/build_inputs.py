"""
Build the model input tensors from the county-week panel and the county adjacency list.

    python real_data/build_inputs.py --policy-data PANEL.csv --adj-matrix ADJACENCY.csv \
        --treatment-col Business.Economic.Restrictions \
        --output-dir real_data/inputs/Business_Economic_Restrictions

- Process time is weekly (`week_end`); decision times are months. The monthly
  policy intensity is binarized at --treatment-threshold (paper: 0.2) and
  applies to every week of its month.
- Spillover exposure is the proportion of treated neighbouring counties.
- Weekly covariates (death rate, temperature) and baseline county covariates
  are z-scored over the panel; the statistics are stored in the metadata.
- Counties are split at random into train/val/test (60/20/20). The split only
  decides which counties enter the training and early-stopping losses.

Writes {train,val,test}_tensors.pt (X, XS, Y, V, A) and
{train,val,test}_metadata.json, read by model.data.RealDataDataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import torch


POLICY_COLUMNS = [
    "Business.Economic.Restrictions",
    "Education.Childcare",
    "Emergency.Governance",
    "Healthcare.System",
    "Other.Uncategorized",
    "Masking",
    "Gatherings.Venues",
    "Reopening.Phase",
]

DEFAULT_WEEKLY_COVARIATES = [
    "new_deaths_week_rate_10k",
    "avg_temp_f",
]

# Last outcome month per policy domain (paper Section 6).
DEFAULT_OUTCOME_END_MONTH = {
    "Business.Economic.Restrictions": "2020-10",
    "Education.Childcare": "2020-08",
}


def load_adjacency_matrix(adj_file: str) -> nx.Graph:
    """Load adjacency CSV/TXT and return a weighted NetworkX graph."""
    if adj_file.endswith(".csv"):
        adj_df = pd.read_csv(adj_file)
    else:
        adj_df = pd.read_csv(adj_file, sep="\t")

    graph = nx.Graph()
    for _, row in adj_df.iterrows():
        source = int(row["county_geoid"])
        target = int(row["neighbor_geoid"])
        weight = row.get("w_len", 1.0)
        if not np.isfinite(weight):
            weight = 1.0
        graph.add_edge(source, target, weight=float(weight))

    return graph


def infer_end_week(df: pd.DataFrame, treatment_col: str, explicit_end_week: str | None) -> str:
    """Infer default final weekly outcome anchor from treatment family."""
    if explicit_end_week:
        return explicit_end_week

    end_month = DEFAULT_OUTCOME_END_MONTH.get(treatment_col)
    if end_month is None:
        return str(pd.to_datetime(df["week_end"]).max().date())

    subset = df[df["period"] == end_month].copy()
    if subset.empty:
        raise ValueError(f"No weekly rows found for inferred end month {end_month}")

    return str(pd.to_datetime(subset["week_end"]).max().date())


def load_weekly_panel(
    policy_file: str,
    treatment_col: str,
    outcome_col: str,
    start_period: str,
    end_week: str | None,
    fips_col: str = "fips_code",
) -> Tuple[pd.DataFrame, str]:
    """Load and filter the weekly panel."""
    df = pd.read_csv(policy_file)

    required_cols = {
        fips_col,
        "period",
        "week_start",
        "week_end",
        "iso_year_week",
        treatment_col,
        outcome_col,
        "party_ind",
    }
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in weekly panel: {missing}")

    df["week_start"] = pd.to_datetime(df["week_start"])
    df["week_end"] = pd.to_datetime(df["week_end"])

    df = df[df["period"] >= start_period].copy()
    resolved_end_week = infer_end_week(df, treatment_col, end_week)
    end_week_ts = pd.Timestamp(resolved_end_week)
    df = df[df["week_end"] <= end_week_ts].copy()

    if df.empty:
        raise ValueError("No rows remain after filtering weekly panel")

    # Enforce unique county-week rows
    dup_count = int(df.duplicated([fips_col, "week_end"]).sum())
    if dup_count:
        raise ValueError(f"Found {dup_count} duplicate county-week rows")

    # Numeric coercion
    numeric_cols = [c for c in df.columns if c not in {"county", "state", "week_start", "week_end", "period", "iso_year_week"}]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Treatment is monthly policy intensity, binarized at threshold later
    if treatment_col not in df.columns:
        raise ValueError(f"Treatment column not found: {treatment_col}")

    # Forward/back fill within county for numeric columns, then column means
    df = df.sort_values([fips_col, "week_end"]).copy()
    df[numeric_cols] = df.groupby(fips_col)[numeric_cols].ffill()
    df[numeric_cols] = df.groupby(fips_col)[numeric_cols].bfill()
    if df[numeric_cols].isna().sum().sum() > 0:
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    return df, resolved_end_week


def choose_covariates(
    df: pd.DataFrame,
    outcome_col: str,
    user_weekly_covariates: List[str] | None,
) -> Tuple[List[str], List[str]]:
    """Select time-varying and baseline covariates for the weekly pipeline."""
    if user_weekly_covariates:
        time_varying_cols = [c for c in user_weekly_covariates if c in df.columns and c != outcome_col]
    else:
        time_varying_cols = [c for c in DEFAULT_WEEKLY_COVARIATES if c in df.columns and c != outcome_col]

    baseline_start_idx = list(df.columns).index("party_ind")
    baseline_cols = list(df.columns[baseline_start_idx:])
    exclude_cols = {
        "avg_temp_f",
        "treatment_binary",
        "spillover_treatment",
        "new_cases_week",
        "new_deaths_week",
        "new_cases_week_rate_10k",
        "new_deaths_week_rate_10k",
    }
    baseline_cols = [c for c in baseline_cols if c not in time_varying_cols and c not in exclude_cols]

    # Keep only numeric baseline columns
    baseline_cols = [c for c in baseline_cols if pd.api.types.is_numeric_dtype(df[c])]
    time_varying_cols = [c for c in time_varying_cols if pd.api.types.is_numeric_dtype(df[c])]

    return time_varying_cols, baseline_cols


def standardize_covariates(
    df: pd.DataFrame,
    time_varying_cols: List[str],
    baseline_cols: List[str],
) -> Tuple[pd.DataFrame, Dict]:
    """Z-score standardize covariates and store stats."""
    stats: Dict[str, Dict] = {"time_varying": {}, "baseline": {}}

    for col in time_varying_cols:
        mean = df[col].mean()
        std = df[col].std()
        df[col] = (df[col] - mean) / std if std > 0 else df[col] - mean
        stats["time_varying"][col] = {"mean": float(mean), "std": float(std)}

    for col in baseline_cols:
        mean = df[col].mean()
        std = df[col].std()
        df[col] = (df[col] - mean) / std if std > 0 else df[col] - mean
        stats["baseline"][col] = {"mean": float(mean), "std": float(std)}

    return df, stats


def compute_spillover_treatment(
    df: pd.DataFrame,
    graph: nx.Graph,
    binary_treatment_col: str,
    fips_col: str = "fips_code",
) -> pd.DataFrame:
    """Compute equal-weight neighboring treatment exposure on the weekly time grid.

    Mean treatment over a county's graph neighbours observed in the same week;
    zero for counties outside the graph or with no observed neighbour.
    """
    edges = pd.DataFrame(
        [(int(a), int(b)) for a, b in graph.edges()] + [(int(b), int(a)) for a, b in graph.edges()],
        columns=[fips_col, "_neighbor"],
    ).drop_duplicates()
    edges = edges[edges[fips_col] != edges["_neighbor"]]
    neighbor_rows = df[[fips_col, "week_end", binary_treatment_col]].rename(
        columns={fips_col: "_neighbor", binary_treatment_col: "_neighbor_treatment"}
    )
    merged = edges.merge(neighbor_rows, on="_neighbor")
    mean = merged.groupby([fips_col, "week_end"])["_neighbor_treatment"].mean()
    keys = pd.MultiIndex.from_arrays([df[fips_col].astype(int), df["week_end"]])
    spillover_values = mean.reindex(keys).fillna(0.0).astype(float).to_numpy()

    out = df.copy()
    out["spillover_treatment"] = spillover_values
    return out


def reshape_to_tensors(
    df: pd.DataFrame,
    graph: nx.Graph,
    treatment_col_binary: str,
    outcome_col: str,
    time_varying_cols: List[str],
    baseline_cols: List[str],
    fips_col: str = "fips_code",
) -> Dict:
    """Reshape weekly panel to tensors with monthly decision metadata."""
    counties = sorted([int(c) for c in df[fips_col].unique()])
    weeks = sorted(df["week_end"].dt.strftime("%Y-%m-%d").unique().tolist())
    week_period_lookup = (
        df[["week_end", "period"]]
        .drop_duplicates()
        .assign(week_end=lambda x: x["week_end"].dt.strftime("%Y-%m-%d"))
        .sort_values("week_end")
    )
    week_to_period = dict(zip(week_period_lookup["week_end"], week_period_lookup["period"]))
    decision_periods = list(dict.fromkeys(week_period_lookup["period"].tolist()))
    decision_period_to_idx = {period: idx for idx, period in enumerate(decision_periods)}
    decision_index_per_step = [decision_period_to_idx[week_to_period[week]] for week in weeks]
    decision_step_flags = [0] * len(weeks)
    seen_periods = set()
    for step_idx, week in enumerate(weeks):
        period = week_to_period[week]
        if period not in seen_periods:
            decision_step_flags[step_idx] = 1
            seen_periods.add(period)

    n_counties = len(counties)
    n_steps = len(weeks)
    v_time = len(time_varying_cols)
    v_base = len(baseline_cols)
    v_dim = v_time + v_base

    X = np.zeros((n_counties, n_steps, 1), dtype=np.float32)
    XS = np.zeros((n_counties, n_steps, 1), dtype=np.float32)
    Y = np.zeros((n_counties, n_steps, 1), dtype=np.float32)
    V = np.zeros((n_counties, n_steps, v_dim), dtype=np.float32)

    county_groups = {int(fips): group.sort_values("week_end").copy() for fips, group in df.groupby(fips_col)}

    missing_count = 0
    for i, county in enumerate(counties):
        county_df = county_groups[county]
        baseline_values = county_df.iloc[0][baseline_cols].to_numpy(dtype=float) if baseline_cols else np.zeros(0)
        county_rows = {
            row["week_end"].strftime("%Y-%m-%d"): row
            for _, row in county_df.iterrows()
        }

        for t, week in enumerate(weeks):
            row = county_rows.get(week)
            if row is None:
                missing_count += 1
                if t > 0:
                    X[i, t, 0] = X[i, t - 1, 0]
                    XS[i, t, 0] = XS[i, t - 1, 0]
                    Y[i, t, 0] = Y[i, t - 1, 0]
                    V[i, t, :] = V[i, t - 1, :]
                elif v_base:
                    V[i, t, v_time:] = baseline_values
                continue

            X[i, t, 0] = float(row[treatment_col_binary])
            XS[i, t, 0] = float(row["spillover_treatment"])
            Y[i, t, 0] = float(row[outcome_col])
            if v_time:
                V[i, t, :v_time] = row[time_varying_cols].to_numpy(dtype=float)
            if v_base:
                V[i, t, v_time:] = baseline_values

    adj_matrix = np.zeros((n_counties, n_counties), dtype=np.float32)
    county_index = {county: idx for idx, county in enumerate(counties)}
    for county_i in counties:
        if county_i not in graph:
            continue
        for county_j in graph.neighbors(county_i):
            if int(county_j) in county_index:
                adj_matrix[county_index[county_i], county_index[int(county_j)]] = 1.0

    return {
        "X": torch.from_numpy(X),
        "XS": torch.from_numpy(XS),
        "Y": torch.from_numpy(Y),
        "V": torch.from_numpy(V),
        "A": torch.from_numpy(adj_matrix),
        "counties": counties,
        "periods": weeks,
        "decision_periods": decision_periods,
        "week_to_period": [week_to_period[w] for w in weeks],
        "decision_index_per_step": decision_index_per_step,
        "decision_step_flags": decision_step_flags,
        "T": n_steps,
        "N": n_counties,
        "v_dim": v_dim,
        "v_timevarying": v_time,
        "v_baseline": v_base,
        "time_varying_covariate_names": time_varying_cols,
        "baseline_covariate_names": baseline_cols,
    }


def split_data(data: Dict, train_ratio: float, val_ratio: float, seed: int) -> Tuple[Dict, Dict, Dict]:
    """Split by counties to preserve the weekly sequence structure."""
    rng = np.random.default_rng(seed)
    n_counties = data["N"]
    indices = rng.permutation(n_counties)

    n_train = int(n_counties * train_ratio)
    n_val = int(n_counties * val_ratio)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    def subset(idx: np.ndarray) -> Dict:
        """Tensors and metadata restricted to the counties in idx."""
        return {
            "X": data["X"][idx],
            "XS": data["XS"][idx],
            "Y": data["Y"][idx],
            "V": data["V"][idx],
            "A": data["A"][np.ix_(idx, idx)],
            "counties": [data["counties"][int(i)] for i in idx],
            "periods": data["periods"],
            "decision_periods": data["decision_periods"],
            "week_to_period": data["week_to_period"],
            "decision_index_per_step": data["decision_index_per_step"],
            "decision_step_flags": data["decision_step_flags"],
            "T": data["T"],
            "N": len(idx),
            "v_dim": data["v_dim"],
            "v_timevarying": data["v_timevarying"],
            "v_baseline": data["v_baseline"],
            "time_varying_covariate_names": data["time_varying_covariate_names"],
            "baseline_covariate_names": data["baseline_covariate_names"],
        }

    return subset(train_idx), subset(val_idx), subset(test_idx)


def save_processed_data(data: Dict, output_dir: str, split_name: str, metadata_extra: Dict | None = None) -> None:
    """Save tensors and JSON metadata."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "X": data["X"],
            "XS": data["XS"],
            "Y": data["Y"],
            "V": data["V"],
            "A": data["A"],
        },
        out_dir / f"{split_name}_tensors.pt",
    )

    metadata = {
        "counties": data["counties"],
        "periods": data["periods"],
        "decision_periods": data["decision_periods"],
        "week_to_period": data["week_to_period"],
        "decision_index_per_step": data["decision_index_per_step"],
        "decision_step_flags": data["decision_step_flags"],
        "decision_time_indices": [idx for idx, flag in enumerate(data["decision_step_flags"]) if flag],
        "T": data["T"],
        "N": data["N"],
        "v_dim": data["v_dim"],
        "v_timevarying": data["v_timevarying"],
        "v_baseline": data["v_baseline"],
        "time_varying_covariate_names": data["time_varying_covariate_names"],
        "baseline_covariate_names": data["baseline_covariate_names"],
    }
    if metadata_extra:
        metadata.update(metadata_extra)

    with (out_dir / f"{split_name}_metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2, default=str)


def main() -> None:
    """Build and save the tensors and metadata for one policy domain."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy-data", required=True,
                        help="County-week panel CSV; format in real_data/sample_data/county_week_panel.csv")
    parser.add_argument("--adj-matrix", required=True,
                        help="County adjacency CSV; format in real_data/sample_data/county_adjacency.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--treatment-col", default="Business.Economic.Restrictions")
    parser.add_argument("--outcome-col", default="new_cases_week_rate_10k")
    parser.add_argument("--treatment-threshold", type=float, default=0.2)
    parser.add_argument("--start-period", default="2020-04", help="First decision month (YYYY-MM)")
    parser.add_argument("--end-week", default=None,
                        help="Last outcome week YYYY-MM-DD; default: last week of Oct 2020 (Business) or Aug 2020 (Education)")
    parser.add_argument("--weekly-covariates", nargs="*", default=None, help="Optional weekly covariate column names")
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df, resolved_end_week = load_weekly_panel(
        policy_file=args.policy_data,
        treatment_col=args.treatment_col,
        outcome_col=args.outcome_col,
        start_period=args.start_period,
        end_week=args.end_week,
    )

    graph = load_adjacency_matrix(args.adj_matrix)

    if args.treatment_col not in POLICY_COLUMNS:
        raise ValueError(f"Only policy-family treatments are supported. Got: {args.treatment_col}")

    df["treatment_binary"] = (df[args.treatment_col] >= float(args.treatment_threshold)).astype(float)

    time_varying_cols, baseline_cols = choose_covariates(
        df=df,
        outcome_col=args.outcome_col,
        user_weekly_covariates=args.weekly_covariates,
    )
    df, standardization_stats = standardize_covariates(df, time_varying_cols, baseline_cols)
    df = compute_spillover_treatment(df, graph, "treatment_binary")

    data = reshape_to_tensors(
        df=df,
        graph=graph,
        treatment_col_binary="treatment_binary",
        outcome_col=args.outcome_col,
        time_varying_cols=time_varying_cols,
        baseline_cols=baseline_cols,
    )
    train_data, val_data, test_data = split_data(
        data=data,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    metadata_extra = {
        "source_policy_data": str(Path(args.policy_data).resolve()),
        "source_adj_matrix": str(Path(args.adj_matrix).resolve()),
        "treatment_col": args.treatment_col,
        "treatment_threshold": float(args.treatment_threshold),
        "treatment_binary_col": "treatment_binary",
        "outcome_col": args.outcome_col,
        "resolved_end_week": resolved_end_week,
        "resolved_end_period": data["week_to_period"][-1],
        "monthly_decision_policy": True,
        "weekly_outcome_timeline": True,
        "spillover_construction": "equal_weight_neighbor_average",
        "spillover_input_scale": "raw_unstandardized",
        "policy_columns_available": [col for col in POLICY_COLUMNS if col in df.columns],
        "standardization_stats": standardization_stats,
    }

    save_processed_data(train_data, args.output_dir, "train", metadata_extra)
    save_processed_data(val_data, args.output_dir, "val", metadata_extra)
    save_processed_data(test_data, args.output_dir, "test", metadata_extra)

    print("=" * 80)
    print("REAL-DATA INPUTS WRITTEN")
    print("=" * 80)
    print(f"Output dir: {args.output_dir}")
    print(f"Treatment: {args.treatment_col} @ threshold {args.treatment_threshold}")
    print(f"Outcome: {args.outcome_col}")
    print(f"Weeks: {data['T']} ({data['periods'][0]} to {data['periods'][-1]})")
    print(f"Decision periods: {len(data['decision_periods'])} ({data['decision_periods'][0]} to {data['decision_periods'][-1]})")
    print(f"Counties: {data['N']}")
    print(f"v_dim: {data['v_dim']} = {data['v_timevarying']} weekly + {data['v_baseline']} baseline")
    print(f"Resolved end week: {resolved_end_week}")


if __name__ == "__main__":
    main()
