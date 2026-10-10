"""
County-week COVID-19 panel as a dataset for the LSTM propensity model (paper Section 6).

Loads the tensors written by real_data/build_inputs.py and derives the
neighbour aggregates the model expects, so that __getitem__ returns the same
batch schema as NetworkTemporalCausalDataset.

All counties of the train/val/test splits are combined into one dataset over
the full county graph, so every county keeps its true neighbours. The split
memberships are exposed in `split_indices` and used only to restrict the
training and validation losses.

Decision times are the first week of each month (`decision_time_indices` in
the metadata). Callers must set `model.decision_mask = dataset.decision_mask`
after building the model.
"""
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .generation import get_bandwidth_from_network

SPLITS = ("train", "val", "test")


class RealDataDataset(Dataset):
    """
    County panel with weekly process times and monthly decision times.

    Fields match NetworkTemporalCausalDataset: x (own treatment), d_xs
    (proportion of treated neighbouring counties), d_xs_count, n_neighbors,
    y (weekly case rate per 10,000), v (lagged covariates), s_vs and q_ys
    (neighbour averages of v and y), s_dxs (neighbour average of d_xs),
    y_0 / ys_0 (baseline outcome of the week before the first decision).
    """

    def __init__(self, data_dir: str):
        """Load all splits from `data_dir` and rebuild the full county adjacency."""
        data_dir = Path(data_dir)

        tensors_by_split = {}
        metadata_by_split = {}
        for split in SPLITS:
            tensors_by_split[split] = torch.load(
                data_dir / f"{split}_tensors.pt", map_location="cpu", weights_only=False
            )
            with (data_dir / f"{split}_metadata.json").open("r") as f:
                metadata_by_split[split] = json.load(f)

        self.metadata = metadata_by_split["train"]
        counties: List[int] = []
        self.split_indices: Dict[str, np.ndarray] = {}
        offset = 0
        for split in SPLITS:
            split_counties = metadata_by_split[split]["counties"]
            counties.extend(split_counties)
            n_split = len(split_counties)
            self.split_indices[split] = np.arange(offset, offset + n_split)
            offset += n_split
        self.counties = counties

        self.N = len(counties)
        self.T = self.metadata["T"]
        self.v_dim = self.metadata["v_dim"]
        self.n_samples = self.N

        self.x = self._clean(torch.cat([tensors_by_split[s]["X"] for s in SPLITS], dim=0))
        self.y = self._clean(torch.cat([tensors_by_split[s]["Y"] for s in SPLITS], dim=0))
        self.v = self._clean(torch.cat([tensors_by_split[s]["V"] for s in SPLITS], dim=0))

        # Slot j of v holds V_{t-1}: shift by one week and put the pre-period
        # baseline week in slot 0. X and Y stay on the original timeline.
        v_0, y_0 = self._load_pre_period_baseline()
        self.v = torch.cat([v_0.unsqueeze(1), self.v[:, :-1, :]], dim=1)

        A = self._build_full_adjacency(counties, self.metadata["source_adj_matrix"])
        self.adj_matrix = A.numpy()
        degree = A.sum(dim=1, keepdim=True)
        degree_safe = degree.clone()
        degree_safe[degree_safe == 0] = 1.0
        self.adj_matrix_normalized_torch = (A / degree_safe).float()
        self.n_neighbors = degree.float()                                 # (N, 1)

        # Spillover exposure: equal-weight proportion of treated neighbours.
        self.d_xs_count = torch.einsum("ij,jtk->itk", A, self.x)          # (N, T, 1)
        self.d_xs = self.d_xs_count / degree_safe.unsqueeze(1)            # (N, T, 1)

        self.decision_time_indices: List[int] = list(self.metadata["decision_time_indices"])
        self.decision_mask: List[bool] = [t in set(self.decision_time_indices) for t in range(self.T)]

        adj_norm = self.adj_matrix_normalized_torch
        self.s_vs = torch.einsum("ij,jtk->itk", adj_norm, self.v)         # (N, T, v_dim)
        self.q_ys = torch.einsum("ij,jtk->itk", adj_norm, self.y)         # (N, T, 1)
        self.s_dxs = torch.einsum("ij,jtk->itk", adj_norm, self.d_xs)     # (N, T, 1)

        self.y_0_baseline = y_0[:, 0].numpy()
        self.ys_0_baseline = (adj_norm @ y_0)[:, 0].numpy()

        self.h_s_seq = None
        self.c_s_seq = None

        # Kernel bandwidth h = 1 / (2 * median degree), clipped to [0.05, 0.15].
        self.spillover_bandwidth = get_bandwidth_from_network(self.adj_matrix)

    def _load_pre_period_baseline(self):
        """Outcome and standardized weekly covariates of the week before the first decision week."""
        covariates = self.metadata["time_varying_covariate_names"]
        outcome = self.metadata["outcome_col"]
        columns = [outcome] + covariates
        baseline_week = pd.Timestamp(self.metadata["periods"][0]) - pd.Timedelta(weeks=1)
        panel = pd.read_csv(
            self.metadata["source_policy_data"],
            usecols=["fips_code", "week_end"] + columns,
        )
        baseline = panel.loc[pd.to_datetime(panel["week_end"]) == baseline_week]
        if baseline["fips_code"].duplicated().any():
            raise ValueError(f"Duplicate county rows for baseline week {baseline_week.date()}")
        baseline = baseline.set_index("fips_code").reindex(self.counties)
        values = baseline[columns].to_numpy(dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError(
                f"Complete outcomes and covariates are required for the baseline week "
                f"{baseline_week.date()}."
            )

        # Baseline county covariates are kept; weekly covariates use the panel's standardization.
        v_0 = self.v[:, 0, :].clone()
        stats = self.metadata["standardization_stats"]["time_varying"]
        for k, column in enumerate(covariates):
            centered = values[:, k + 1] - stats[column]["mean"]
            std = stats[column]["std"]
            v_0[:, k] = torch.from_numpy(centered / std if std > 0 else centered)
        return v_0, torch.from_numpy(values[:, :1].copy())

    @staticmethod
    def _build_full_adjacency(counties: List[int], edge_list_path: str) -> torch.Tensor:
        """Binary symmetric adjacency over `counties` from the county edge list."""
        county_to_idx = {c: i for i, c in enumerate(counties)}
        n = len(counties)
        A = torch.zeros(n, n, dtype=torch.float32)

        edges = pd.read_csv(edge_list_path)
        county_geoid = edges["county_geoid"].astype("int64").to_numpy()
        neighbor_geoid = edges["neighbor_geoid"].astype("int64").to_numpy()
        for a, b in zip(county_geoid, neighbor_geoid):
            i = county_to_idx.get(int(a))
            j = county_to_idx.get(int(b))
            if i is not None and j is not None:
                A[i, j] = 1.0
                A[j, i] = 1.0
        return A

    @staticmethod
    def _clean(tensor: torch.Tensor) -> torch.Tensor:
        """Replace non-finite values by 0 (with a warning)."""
        if torch.isfinite(tensor).all():
            return tensor
        n_bad = (~torch.isfinite(tensor)).sum().item()
        print(f"Warning: {n_bad} non-finite values in tensor; replacing with 0.0")
        return torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)

    def __len__(self):
        """Number of counties."""
        return self.n_samples

    def __getitem__(self, idx):
        """Batch dictionary for one county (or a slice of counties)."""
        if isinstance(idx, slice):
            y_0_slice = self.y_0_baseline[idx]
            ys_0_slice = self.ys_0_baseline[idx]
        else:
            y_0_slice = self.y_0_baseline[idx:idx + 1]
            ys_0_slice = self.ys_0_baseline[idx:idx + 1]

        item = {
            "v": self.v[idx],
            "x": self.x[idx],
            "y": self.y[idx],
            "s_vs": self.s_vs[idx],
            "d_xs": self.d_xs[idx],
            "d_xs_count": self.d_xs_count[idx],
            "s_dxs": self.s_dxs[idx],
            "n_neighbors": self.n_neighbors[idx],
            "q_ys": self.q_ys[idx],
            "y_0": torch.from_numpy(np.asarray(y_0_slice)).float(),
            "ys_0": torch.from_numpy(np.asarray(ys_0_slice)).float(),
        }

        if self.h_s_seq is not None and self.c_s_seq is not None:
            item["h_s"] = self.h_s_seq[idx]
            item["c_s"] = self.c_s_seq[idx]

        if isinstance(idx, slice):
            item["idx"] = torch.arange(*idx.indices(self.n_samples), dtype=torch.long)
        else:
            item["idx"] = torch.tensor(idx, dtype=torch.long)

        return item
