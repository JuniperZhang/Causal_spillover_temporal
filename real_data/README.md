# County-level COVID-19 policy application (paper Section 6)

Code for the application in Section 6: effects of state COVID-19 policies on
subsequent county case rates, with spillover across adjacent counties.

The repository contains the analysis code and a few rows of the data in
`sample_data/` to show its format. The full data (3,108 counties in 48 states
and the District of Columbia, March–December 2020) are available from the
corresponding author on reasonable request.

## Data format

### County-week panel (`sample_data/county_week_panel.csv`)

One row per county and week. The sample shows two neighbouring Iowa counties
in May 2020; the full panel covers every county and week from 2020-03-04 to
2020-12-30 with the same columns.

| Column | Description |
|---|---|
| `fips_code`, `county`, `state`, `state_fips` | County identifiers |
| `week_start`, `week_end`, `iso_year_week` | Week of the row |
| `period` | Month `YYYY-MM` of the week; decisions are monthly |
| `new_cases_week`, `new_deaths_week` | Weekly new cases and deaths |
| `new_cases_week_rate_10k` | Weekly new cases per 10,000 residents — **outcome** |
| `new_deaths_week_rate_10k` | Weekly new deaths per 10,000 residents — weekly covariate |
| `Business.Economic.Restrictions`, `Education.Childcare`, `Emergency.Governance`, `Healthcare.System`, `Other.Uncategorized`, `Masking`, `Gatherings.Venues`, `Reopening.Phase` | Monthly state policy intensity in [0, 1] for each policy domain: average daily proportion of the domain's recorded policy types that are active. Repeated for every week of the month. |
| `party_ind` | County partisanship indicator from the 2020 presidential election — baseline covariate |
| `Population.raw.value` … `Long.commute...driving.alone.raw.value` (18 columns) | 2020 county health and socioeconomic measures (population, age, rurality, income, unemployment, poverty, inequality, insurance, providers, health behaviours, housing, air pollution, commuting) — baseline covariates |
| `avg_temp_f` | Monthly average temperature in °F — weekly covariate |

The two policy domains analysed in the paper are
`Business.Economic.Restrictions` and `Education.Childcare`.

### County adjacency (`sample_data/county_adjacency.csv`)

One row per pair of counties sharing a border, within or across state borders.
The sample shows the neighbours of one county.

| Column | Description |
|---|---|
| `County Name`, `Neighbor Name` | County names |
| `county_geoid`, `neighbor_geoid` | FIPS codes of the two counties — **used to build the network** |
| `Length`, `w_len` | Shared border length and its share of the county's border (not used; spillover exposure weights neighbours equally) |

## Pipeline

All commands run from the repository root. Besides
`model/requirements.txt`, this part needs `pandas` and `networkx`
(both listed there).

| Step | Script | Output |
|---|---|---|
| 1. Build model inputs for each policy domain | `build_inputs.py` | `{train,val,test}_tensors.pt`, `_metadata.json` |
| 2. Train the model and estimate all effects for one training seed | `run_seed.sh` (calls `train.py`, `estimate.py`, `compact_seed.py`) | `runs/seed_XXXX.json.gz` |
| 3. Pool the seeds | `pool_seeds.py` | `<domain>/estimates/*.json`, `summary.csv` |
| 4. Draw Figure 3 | `plot_policy_effects.py` | `policy_effects.png` |

`model/data/real_data_dataset.py` loads the step-1 output for the model.

### Settings used in the paper

| Setting | Value |
|---|---|
| Decision months | April 2020 onward (`--start-period 2020-04`) |
| Treatment | state-month policy intensity ≥ 0.2 → X = 1 |
| Spillover exposure | proportion of neighbouring counties with X = 1 |
| Outcome months | Sep and Oct 2020 (Business); Jul and Aug 2020 (Education) |
| Outcome summaries | last-week rate; average weekly rate within the month |
| Target paths | baseline x = d = 0 in every month, plus the 8 most frequent observed paths (`--top-k 8`) |
| Kernel bandwidth | h = 1 / (2 × median degree) = 0.0833 on the full county network |
| Training | 120 epochs, batch size 128, early stopping on the validation counties |
| Inference | 1,000 network-block bootstrap draws per seed, randomized 1-hop blocks |
| Seeds | 100 training seeds (42–141); point estimate = mean over seeds, 95% interval = percentiles of the pooled bootstrap draws |

## Running the analysis

`PANEL.csv` and `ADJACENCY.csv` are the full-data files in the format above.
Build the inputs for the two policy domains:

```bash
python real_data/build_inputs.py --policy-data PANEL.csv --adj-matrix ADJACENCY.csv --treatment-col Business.Economic.Restrictions --output-dir real_data/inputs/Business_Economic_Restrictions
```

```bash
python real_data/build_inputs.py --policy-data PANEL.csv --adj-matrix ADJACENCY.csv --treatment-col Education.Childcare --output-dir real_data/inputs/Education_Childcare
```

Train and estimate for one seed (about 5–10 minutes on a CPU):

```bash
PYTHON=python bash real_data/run_seed.sh 42
```

The paper uses seeds 42–141; each seed runs independently, so they can run
in parallel:

```bash
for s in $(seq 42 141); do PYTHON=python bash real_data/run_seed.sh $s; done
```

Pool the seeds and draw Figure 3 (the paper leaves out four Education paths,
listed in the `--drop` options):

```bash
python real_data/pool_seeds.py real_data/pooled real_data/runs/seed_*.json.gz
```

```bash
python real_data/plot_policy_effects.py --results-root real_data/pooled --out real_data/policy_effects.png --drop "Education: Jul 2020::05-07 | 3 mo" --drop "Education: Jul 2020::07 | 1 mo" --drop "Education: Aug 2020::05-08 | 4 mo" --drop "Education: Aug 2020::07-08 | 2 mo"
```
