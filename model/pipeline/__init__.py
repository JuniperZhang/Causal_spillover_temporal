"""
Simulation pipelines for the numerical experiments (Section 5).

The pipelines are run as scripts via `python -m`, not imported through this package:

    simulation_study.py         -- one-replication diagnostic wrapper.
    bias_mse_study.py           -- R-replication Monte Carlo bias/MSE/
                                    coverage study for 15 level targets
                                    and 13 effects, with joint disjoint
                                    network-block bootstrap intervals.

This package exposes no package-level API.
"""
