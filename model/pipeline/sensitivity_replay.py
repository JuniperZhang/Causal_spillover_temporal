"""Recompute sensitivity bounds from saved IPW weights on a new Gamma grid, without refitting."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from .bias_mse_study import contrast_pairs, _write_json
from ..estimation.sensitivity import sensitivity_analysis, summarize_sensitivity, validate_gammas

# Endpoints are sums over sorted observations; floating-point summation order
# can differ across machines by a few ulp. Structure and flags must match exactly.
RTOL = ATOL = 1e-10


def _close(recomputed, stored):
    """Compare two floats within tolerance; None matches only None."""
    if recomputed is None or stored is None:
        return recomputed is None and stored is None
    return bool(np.isclose(recomputed, stored, rtol=RTOL, atol=ATOL))


def _close_all(recomputed, stored):
    """Elementwise _close for equal-length sequences."""
    return len(recomputed) == len(stored) and all(map(_close, recomputed, stored))


def endpoints_match(recomputed, stored):
    """Compare a replayed result with a stored one; floats within tolerance."""
    if (recomputed.keys() != stored.keys() or recomputed["M"] != stored["M"]
            or recomputed["interval_type"] != stored["interval_type"]
            or not _close_all(recomputed["gammas"], stored["gammas"])
            or recomputed["targets"].keys() != stored["targets"].keys()):
        return False
    for name, new in recomputed["targets"].items():
        old = stored["targets"][name]
        if (new["gamma_star_status"] != old["gamma_star_status"]
                or new["contains_zero"] != old["contains_zero"]
                or not _close(new["gamma_star"], old["gamma_star"])
                or not _close_all(new["lower"], old["lower"])
                or not _close_all(new["upper"], old["upper"])):
            return False
    return True


def replay_saved_inputs(payload, inputs_dir, gammas, verify_all=False):
    """Recompute bounds on a new Gamma grid from the saved weights of every replication.

    Args:
        payload: study dict (study.json).
        inputs_dir: directory of rep_XXX_<method>.npz archives.
        gammas: new Gamma grid.
        verify_all: also recompute the stored grid and require all endpoints
            to match the stored results.

    Every archive must reproduce the stored Gamma=1 lower endpoints; archives
    with embedded metadata must also match the study's n, replication, seed,
    method, M and target sequences. Returns a copy of the
    payload with updated per-replication results and summary.
    """
    validate_gammas(gammas)
    result = copy.deepcopy(payload)
    for rep in result["replications"]:
        for method in ("lstm_kipw", "oracle_kipw"):
            filename = Path(inputs_dir) / f"rep_{rep['rep_index']:03d}_{method}.npz"
            old = rep["sensitivity"][method]
            with np.load(filename, allow_pickle=False) as archive:
                if "__metadata_json" in archive:
                    metadata = json.loads(str(archive["__metadata_json"]))
                    expected = {"n": payload["arguments"]["n"], "rep_index": rep["rep_index"],
                                "seed": rep["seed"], "method": method, "M": old["M"],
                                "sequences": json.loads(json.dumps(payload["expanded_target_sequences"]))}
                    if metadata != expected:
                        raise ValueError(f"Saved-weight metadata does not match study: {filename}")
                inputs = {name: (archive[f"{name}__weights"], archive[f"{name}__outcomes"])
                          for name in payload["expanded_target_sequences"]}
            updated = sensitivity_analysis(inputs, contrast_pairs(), gammas, old["M"])
            for target, values in updated["targets"].items():
                if not _close(values["lower"][0], old["targets"][target]["lower"][0]):
                    raise ValueError(f"Saved weights do not match study: {filename}, {target}")
            if verify_all:
                original = sensitivity_analysis(inputs, contrast_pairs(), old["gammas"], old["M"])
                if not endpoints_match(original, old):
                    raise ValueError(f"Saved-weight endpoints do not match study: {filename}")
            rep["sensitivity"][method] = updated
    result["arguments"]["sensitivity_gammas"] = list(gammas)
    result["sensitivity_summary"] = summarize_sensitivity(result["replications"], result["truth_with_contrasts"])
    return result



def main(argv=None):
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("--inputs-dir", type=Path, required=True)
    parser.add_argument("--gammas", type=float, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError("Output already exists; choose a new JSON filename")
    payload = json.loads(args.study.read_text())
    updated = replay_saved_inputs(payload, args.inputs_dir, args.gammas, verify_all=True)
    _write_json(args.out, updated)
    print(f"Recomputed sensitivity values without training: {args.out}")


if __name__ == "__main__":
    main()
