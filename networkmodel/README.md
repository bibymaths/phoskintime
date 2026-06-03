# `networkmodel`

`networkmodel` is the integrated PhosKinTime package for fitting a coupled phosphorylation, protein, and RNA ODE model with the current JAX, Diffrax, and JAXopt scalar objective path. It loads configured CSV or Excel inputs, builds kinase and transcription-factor topology, prepares numeric loss arrays, solves the ODE with Diffrax, optimizes raw parameters with `jaxopt.ProjectedGradient`, and writes result tables, diagnostic plots, optional inference summaries, optional sensitivity summaries, and dashboard payloads.

## What this module does not do

- It does not run retired optimizer backends or evolutionary multi-objective solvers.
- It does not document planned configuration keys that are absent from `networkmodel/config.py`.
- It does not treat the Streamlit dashboard as part of the numerical optimizer.
- It does not import pandas inside the differentiable JAX objective functions in `jax_backend.py`; tabular inputs are converted before objective evaluation.

## Current architecture

The command-line workflow starts in `runner.py` and follows this import graph for the current implementation:

```text
runner
├── optproblem ──> jax_backend
├── inference ──> jax_backend
├── sensitivity
├── export
├── analysis
└── dashboard_bundle / dashboard_app
```

Supporting modules provide the data structures and helpers used by that path:

- `config.py` exposes constants loaded from `config.toml`.
- `io.py` loads kinase, transcription-factor, protein, RNA, phospho, and prior-result inputs.
- `network.py` builds the index map, kinase input interpolation, and mutable `System` object.
- `buildmat.py` constructs kinase-site and transcription-factor matrices.
- `params.py` initializes and unpacks optimizer vectors.
- `cache.py` prepares compact numeric arrays for loss evaluation.
- `models.py`, `simulate.py`, and `jax_backend.py` evaluate and solve ODE trajectories.
- `lossfn.py` contains NumPy loss kernels used by non-JAX helper paths.
- `mode_outputs.py`, `export.py`, and `dashboard_bundle.py` write scalar-run outputs.

## Implemented state layouts

The package supports the model names exposed by `config.py` through `AVAILABLE_MODELS` and selected by `MODEL`:

- `distributive`: one mRNA state, one unphosphorylated protein state, and one phosphorylated state per site.
- `sequential`: one mRNA state and ordered phosphorylation states per protein.
- `combinatorial`: one mRNA state and one protein state for each phosphorylation bit pattern.
- `saturation`: a saturating phosphorylation variant selected as internal model code `4`.

For the standard non-combinatorial layout, each protein block contains:

```text
mRNA, unphosphorylated protein, phosphosite 1, phosphosite 2, ...
```

For the combinatorial layout, each protein block contains:

```text
mRNA, phosphorylation state 0, phosphorylation state 1, ..., phosphorylation state 2^S - 1
```

## Optimization path

The current scalar optimizer is `jaxopt.ProjectedGradient`. Raw parameters are projected into configured lower and upper bounds, fixed values are restored when a fixed mask is supplied, and the scalar objective combines available protein, RNA, and phospho losses with the prior penalty configured by `lambda_prior`.

A minimal Python call pattern is:

```python
from networkmodel.jax_backend import detect_data_mode, make_simple_objective, optimize_scalar_objective

mode = detect_data_mode(loss_data=loss_data)
objective = make_simple_objective(
    loss_data=loss_data,
    mode=mode,
    time_grid=time_grid,
    weights={"protein": 1.0, "mrna": 1.0, "phospho": 1.0},
    defaults=defaults,
    prior_weight=0.1,
    networkmodel_layout=True,
    y0=y0,
    sys=sys,
    slices=slices,
)
theta, state, value = optimize_scalar_objective(
    objective,
    theta0,
    lower,
    upper,
    maxiter=30,
    tol=1e-6,
)
```

The example uses only arguments present in the function signatures in `jax_backend.py`.

## Command-line interface

`runner.py` currently defines these command-line flags:

| Flag | Purpose |
| --- | --- |
| `--kinase-net` | Kinase-substrate network path. |
| `--tf-net` | Transcription-factor network path. |
| `--ms` | Protein mass-spectrometry data path. |
| `--rna` | RNA data path. |
| `--phospho` | Phospho data path. |
| `--kinopt` | Kinase prior-result path. |
| `--tfopt` | Transcription-factor prior-result path. |
| `--output-dir` | Output directory. |
| `--cores` | Worker count passed through the workflow. |
| `--n-gen` | Maximum scalar optimizer iterations. |
| `--seed` | Random seed for initialization and inference helpers. |
| `--lambda-prior` | Prior-adherence loss weight. |
| `--lambda-protein` | Protein loss weight. |
| `--lambda-rna` | RNA loss weight. |
| `--lambda-phospho` | Phospho loss weight. |
| `--normalize-fc-steady` | Normalize protein and phospho fold change to the baseline time. |
| `--use-initial-condition-from-data` | Build initial state values from observed baseline data. |
| `--scan` | Run the compatibility hyperparameter-scan entry point. |
| `--sensitivity` | Run sensitivity analysis after optimization. |
| `--solver` | Select the scalar solver option; the documented working value is `jaxopt`. |

Example:

```bash
python -m networkmodel.runner \
  --kinase-net data/input2.csv \
  --tf-net data/input4.csv \
  --ms data/input1.csv \
  --rna data/input3.csv \
  --phospho data/input1.csv \
  --kinopt data/kinopt_results.xlsx \
  --tfopt data/tfopt_results.xlsx \
  --output-dir results_model_global_distributive_jax \
  --cores 8 \
  --n-gen 30 \
  --seed 42 \
  --lambda-prior 0.1 \
  --lambda-protein 1.0 \
  --lambda-rna 1.0 \
  --lambda-phospho 1.0 \
  --solver jaxopt
```

## Configuration fields consumed by `networkmodel/config.py`

The active configuration source is the `[networkmodel]` table in `config.toml`, with nested `timepoints`, `bounds`, `models`, and `solver` tables. The following names are consumed by `config.py` through `load_config_toml`:

| Field | Controls |
| --- | --- |
| `kinase_net` | `KINASE_NET_FILE`, the kinase-substrate input path. |
| `tf_net` | `TF_NET_FILE`, the transcription-factor input path. |
| `ms` | `MS_DATA_FILE`, the protein data path. |
| `rna` | `RNA_DATA_FILE`, the RNA data path. |
| `phospho` | `PHOSPHO_DATA_FILE`, the phospho data path. |
| `kinopt` | `KINOPT_RESULTS_FILE`, the kinase prior-result path. |
| `tfopt` | `TFOPT_RESULTS_FILE`, the transcription-factor prior-result path. |
| `output_dir` | `RESULTS_DIR`, the output directory. |
| `cores` | `CORES`, worker-count setting. |
| `seed` | `SEED`, random seed. |
| `loss` | `LOSS_MODE`, integer robust-loss selector. |
| `lambda_prior` | `REGULARIZATION_LAMBDA`, prior-adherence weight. |
| `lambda_protein` | `REGULARIZATION_PROTEIN`, protein loss weight. |
| `lambda_rna` | `REGULARIZATION_RNA`, RNA loss weight. |
| `lambda_phospho` | `REGULARIZATION_PHOSPHO`, phospho loss weight. |
| `hyperparam_scan` | `HYPERPARAM_SCAN`, compatibility scan toggle. |
| `normalize_fc_steady` | `NORMALIZE_FC_STEADY`, baseline fold-change normalization toggle. |
| `use_initial_condition_from_data` | `USE_INITIAL_CONDITION_FROM_DATA`, data-derived initial-state toggle. |
| `scaling_method` | `SCALING_METHOD`, raw-data scaling method. |
| `weighting_method_protein` | `WEIGHTING_METHOD_PROTEIN`, protein time-weighting method. |
| `weighting_method_rna` | `WEIGHTING_METHOD_RNA`, RNA time-weighting method. |
| `weighting_method_phospho` | `WEIGHTING_METHOD_PHOSPHO`, phospho time-weighting method. |
| `sensitivity_analysis` | `SENSITIVITY_ANALYSIS`, sensitivity toggle. |
| `sensitivity_perturbation` | `SENSITIVITY_PERTURBATION`, relative perturbation size. |
| `sensitivity_trajectories` | `SENSITIVITY_TRAJECTORIES`, perturbation trajectory count. |
| `sensitivity_levels` | `SENSITIVITY_LEVELS`, perturbation-grid level count. |
| `sensitivity_top_curves` | `SENSITIVITY_TOP_CURVES`, number of sensitivity curves to plot. |
| `sensitivity_metric` | `SENSITIVITY_METRIC`, scalar metric for sensitivity scoring. |
| `n_starts` | `N_STARTS`, multistart count. |
| `profile_likelihood` | `PROFILE_LIKELIHOOD`, profile-likelihood toggle. |
| `profile_indices` | `PROFILE_INDICES`, comma-separated parameter indices. |
| `profile_grid_size` | `PROFILE_GRID_SIZE`, number of grid values per profiled parameter. |
| `posterior_sampling` | `POSTERIOR_SAMPLING`, optional NumPyro posterior toggle. |
| `posterior_num_warmup` | `POSTERIOR_NUM_WARMUP`, NumPyro warmup draw count. |
| `posterior_num_samples` | `POSTERIOR_NUM_SAMPLES`, NumPyro posterior draw count. |
| `models.default_model` | `MODEL`, internal integer model selector. |
| `models.available_models` | `AVAILABLE_MODELS`, metadata list of accepted model names. |
| `timepoints.protein` | `TIME_POINTS_PROTEIN`, protein observation times in minutes. |
| `timepoints.rna` | `TIME_POINTS_RNA`, RNA observation times in minutes. |
| `timepoints.phospho_protein` | `TIME_POINTS_PHOSPHO`, phospho observation times in minutes. |
| `bounds.c_k` | Bounds for kinase activity multipliers. |
| `bounds.A_i` | Bounds for basal mRNA production parameters. |
| `bounds.B_i` | Bounds for mRNA degradation parameters. |
| `bounds.C_i` | Bounds for protein production parameters. |
| `bounds.D_i` | Bounds for protein deactivation parameters. |
| `bounds.Dp_i` | Bounds for phosphosite dephosphorylation parameters. |
| `bounds.E_i` | Bounds for transcriptional efficacy parameters. |
| `bounds.tf_scale` | Bounds for transcription-factor scaling. |
| `solver.absolute_tolerance` | `ODE_ABS_TOL`, Diffrax absolute tolerance. |
| `solver.relative_tolerance` | `ODE_REL_TOL`, Diffrax relative tolerance. |
| `solver.max_timesteps` | `ODE_MAX_STEPS`, maximum Diffrax steps. |

## Outputs

A run writes outputs under `--output-dir`. The exact set depends on enabled analyses and available data layers, but the implemented exporters cover:

- optimized parameter summaries,
- convergence history,
- fitted protein, RNA, and phospho trajectories,
- goodness-of-fit plots,
- residual tables,
- kinase activity tables,
- parameter correlation and distribution diagnostics,
- dashboard bundle files,
- optional multistart, profile-likelihood, posterior, and sensitivity outputs.
