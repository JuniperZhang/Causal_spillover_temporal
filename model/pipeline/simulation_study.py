"""Single-replication entry point for the simulation study.

Reuses the target construction, ground truth and inference code of
bias_mse_study with R=1. Useful for a quick check or for inspecting one
generated dataset before running the full study.
"""

from .bias_mse_study import (
    CANONICAL_PATHS,
    D_LEVELS,
    build_arg_parser,
    build_sequence,
    contrast_pairs,
    decision_steps_for,
    load_or_construct_reference_paths,
    make_analysis_and_validation,
    run_study,
    target_sequences,
)

__all__ = [
    "CANONICAL_PATHS",
    "D_LEVELS",
    "build_sequence",
    "contrast_pairs",
    "decision_steps_for",
    "load_or_construct_reference_paths",
    "make_analysis_and_validation",
    "target_sequences",
]


def main():
    """Command-line entry point; same options as bias_mse_study with R=1."""
    parser = build_arg_parser()
    parser.set_defaults(R=1, rep_count=1)
    args = parser.parse_args()
    run_study(args)


if __name__ == "__main__":
    main()
