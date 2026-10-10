# When treatments spill over: neural spatiotemporal spillover estimation

Code for the paper *When Treatments Spill Over*: an LSTM history encoder with
joint own-treatment and spillover assignment models, feeding a
self-normalized Gaussian-kernel IPW estimator of path-indexed mean potential
outcomes and their direct (DE), spillover (SE) and total (TE) effects, with a
sensitivity analysis for unmeasured confounding, the simulation study, and the
county-level COVID-19 policy application.

## Installation

Python 3.10 or later.

```bash
git clone https://github.com/JuniperZhang/Causal_spillover_temporal.git
```

```bash
cd Causal_spillover_temporal
```

```bash
pip install -r model/requirements.txt
```

Run every command below from the repository root. Outputs go to `results/`.

Check the installation with a one-minute run (tiny network, one epoch; the
numbers are meaningless):

```bash
python -m model.pipeline.run_sensitivity --smoke --out-dir results/smoke
```

## Repository structure

```text
model/
├── config.py                       simulation DGP parameters and model/training hyperparameters
├── sensitivity_reference_paths.json  fixed low/mid/high reference spillover paths (main simulation, sensitivity)
├── stress_test_reference_paths.json  fixed reference spillover paths for the stress-test DGPs
├── requirements.txt
├── data/
│   ├── generation.py               DGP equations: network, covariates, treatment, latent state, outcome
│   ├── dataset.py                  NetworkTemporalCausalDataset: one simulated network and its histories
│   ├── ground_truth.py             two-hop local structural Monte Carlo ground truth
│   └── real_data_dataset.py        RealDataDataset: the county panel for the application
├── models/
│   ├── encoder.py                  GraphSAGE mean aggregator and LSTM backbone
│   ├── treatment_model.py          g_Z, own-treatment model f_X, spillover model f_D
│   ├── reconstruction.py           reconstruction head for the auxiliary loss L_I
│   └── combined_model.py           full model and training loss
├── training/
│   ├── train.py                    joint training with early stopping
│   ├── validation.py               validation loss
│   └── gpu_runtime.py              CUDA settings and runtime metadata
├── estimation/
│   ├── ipw.py                      fitted propensities from the trained model
│   ├── gaussian_kernel.py          self-normalized kernel IPW estimator
│   ├── oracle.py                   kernel IPW with the true propensity
│   ├── ols_baseline.py             outcome-regression benchmark
│   ├── reference_paths.py          construction of the reference spillover paths
│   ├── bootstrap.py                disjoint network-block bootstrap
│   ├── retrain_bootstrap.py        bootstrap that refits the model on every draw
│   └── sensitivity.py              sensitivity bounds over a Γ grid
├── pipeline/
│   ├── bias_mse_study.py           Monte Carlo simulation study (bias, MSE, coverage)
│   ├── simulation_study.py         single-replication run of the same study
│   ├── merge_bias_mse_shards.py    combine sharded simulation runs
│   ├── run_sensitivity.py          sensitivity-analysis study
│   ├── merge_sensitivity_shards.py combine sharded sensitivity runs
│   └── sensitivity_replay.py       recompute and verify bounds from saved weights
└── visualization/
    ├── mean_potential_outcome.py   mean-potential-outcome figure
    ├── simulation_descriptives.py  descriptive figures of the simulated data
    ├── sensitivity_figures.py      sensitivity-bound figures for one study
    └── sensitivity_comparison.py   sensitivity-bound figures across sample sizes

real_data/                          county-level COVID-19 application (Section 6); see real_data/README.md
├── sample_data/                    a few rows of the county-week panel and county adjacency (data format)
├── build_inputs.py                 panel and adjacency → model input tensors
├── train.py                        train the model for one policy domain and seed
├── estimate.py                     path effects with network-block bootstrap intervals
├── run_seed.sh                     train and estimate everything for one seed
├── compact_seed.py                 keep one seed's effects and bootstrap draws
├── pool_seeds.py                   pool the seeds
└── plot_policy_effects.py          Figure 3
```

Imports run one way: `data` and `models` import nothing else from the
package; `training` imports `data` and `models`; `estimation` imports `data`
and `training`; `pipeline` imports `estimation`, `training`, `data` and
`config.py`; `visualization` imports `pipeline` and `config.py`. The
`real_data` scripts import `model`.

## Code map

| Paper | Code |
|---|---|
| Simulation data-generating process (Section 5.1) | `model/config.py`, `model/data/generation.py`, `model/data/dataset.py` |
| LSTM encoder, `g_Z`, assignment models `f_X`, `f_D` (Sections 3.1–3.2) | `model/models/` |
| Joint training (Section 3.3) | `model/training/train.py` |
| Self-normalized kernel IPW (Section 3.4) | `model/estimation/gaussian_kernel.py`, `model/estimation/ipw.py` |
| Oracle-propensity and OLS benchmarks | `model/estimation/oracle.py`, `model/estimation/ols_baseline.py` |
| Reference spillover-exposure paths | `model/estimation/reference_paths.py`, `model/*_reference_paths.json` |
| Monte Carlo ground truth | `model/data/ground_truth.py` |
| Network-block bootstrap | `model/estimation/bootstrap.py` |
| Sensitivity analysis (Section 4) | `model/estimation/sensitivity.py` |
| Simulation and sensitivity runners | `model/pipeline/` |
| Figures | `model/visualization/` |
| County-level COVID-19 application (Section 6) | `real_data/`, `model/data/real_data_dataset.py` |

## Simulation study

The defaults in `model/config.py` are the paper's design: T=20,
decisions at (1, 6, 11, 16), Erdős–Rényi network with mean degree 6, R=100
replications, B=100 network-block bootstrap resamples, and a 100 × 100 Monte
Carlo ground truth. The fixed reference spillover paths are in
`model/sensitivity_reference_paths.json`.

n = 5,000 (CPU; `--n-workers` runs replications in parallel):

```bash
python -m model.pipeline.bias_mse_study --n 5000 --R 100 --bandwidth 0.03 --reference-paths-json model/sensitivity_reference_paths.json --n-workers 8 --out results/sim_n5000_h003.json --log results/sim_n5000_h003.log
```

n = 50,000 (a GPU is recommended):

```bash
python -m model.pipeline.bias_mse_study --n 50000 --R 100 --bandwidth 0.03 --reference-paths-json model/sensitivity_reference_paths.json --device cuda --n-workers 1 --out results/sim_n50000_h003.json --log results/sim_n50000_h003.log
```

Bandwidth sensitivity (Supplement): repeat the n = 5,000 command with
`--bandwidth 0.02`, `0.04` and `0.05`.

Stress tests under alternative data-generating processes (Supplement): DGP A,
B and C are `--dgp-variant burden_modified`, `own_spillover_synergy` and
`burden_synergy`. For example, DGP C:

```bash
python -m model.pipeline.bias_mse_study --n 5000 --R 20 --bandwidth 0.03 --dgp-variant burden_synergy --latent-beta-xd 4.0 --treatment-variant baseline --reference-paths-json model/stress_test_reference_paths.json --n-workers 8 --out results/stress_C.json
```

Long runs can be split into shards with `--rep-start` and `--rep-count`, each
with its own `--out`, and combined with:

```bash
python -m model.pipeline.merge_bias_mse_shards --R 100 --shards results/shard_*/study.json --out results/sim_n50000_h003.json
```

An interrupted run continues with `--resume` and the same arguments.

## Sensitivity analysis

The runner fixes the paper's design (M=4, h=0.03, R=100,
Γ ∈ {1, 1.01, 1.025, 1.05, 1.1, 1.2, 1.3, 1.5, 2, 3, 5}) and writes
`results/sensitivity_n{n}_R100/study.json`. Print the resolved design without
running anything with `--dry-run`.

n = 5,000 (CPU):

```bash
python -m model.pipeline.run_sensitivity --n 5000 --device cpu --n-workers 8
```

n = 50,000 (GPU; about 40 GPU-hours on one A100):

```bash
python -m model.pipeline.run_sensitivity --n 50000 --device cuda
```

To split the n = 50,000 study across GPUs, give each shard its own output
directory, for example `--rep-start 0 --rep-count 25 --out-dir results/sens_shard_00`,
then combine them:

```bash
python -m model.pipeline.merge_sensitivity_shards --shards results/sens_shard_* --out-dir results/sensitivity_n50000_R100
```

## Figures

Mean potential outcomes at two sample sizes (Figure 2):

```bash
python -m model.visualization.mean_potential_outcome results/sim_n5000_h003.json results/sim_n50000_h003.json results/mean_potential_outcome.jpg --fixed-model
```

Descriptive figures of the simulated data (Supplement):

```bash
python -m model.visualization.simulation_descriptives results/sim_n5000_h003.json results/descriptive
```

Sensitivity bounds for one study, and overlaid across the two sample sizes:

```bash
python -m model.visualization.sensitivity_figures --study results/sensitivity_n5000_R100/study.json --out results/sensitivity_figures_n5000
```

```bash
python -m model.visualization.sensitivity_comparison --studies results/sensitivity_n5000_R100/study.json results/sensitivity_n50000_R100/study.json --out results/sensitivity_figures
```

Every runner and figure module accepts `--help`.

## Real-data application

`real_data/` contains the code for the county-level COVID-19 policy analysis
(Section 6) and a few rows of the data that show its format. The
full data are available from the corresponding author on reasonable request.
Data format, pipeline, the paper's settings and commands are in
[`real_data/README.md`](real_data/README.md).

## License

MIT. See [LICENSE](LICENSE).
