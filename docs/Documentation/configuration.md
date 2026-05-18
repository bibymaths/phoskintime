# Configuration Reference

This page documents the `config.toml` configuration file that controls all aspects of the
PhosKinTime pipeline.

---

## Location and Loading

The configuration file is `config.toml` at the project root. It is loaded by two modules:

- **`config_loader.py`** — the root-level loader. Provides `load(mode, section)` (LRU-cached)
  and `load_config_toml()`. Used by the local pipeline and as the backbone for `global_model`.
- **`global_model/config.py`** — loads `config.toml` via `config_loader.load_config_toml()` and
  exports constants for the global model pipeline.

> **Technical debt:** Both modules load overlapping configuration surfaces from `config.toml`.
> They should be unified so `global_model/config.py` entirely delegates to `config_loader.py`.
> See the TODO comment in `config_loader.py`.

---

## Section: `[paths]`

Global project paths. Used by all pipeline stages.

| Key | Type | Default | Description |
|---|---|---|---|
| `data_dir` | string | `"data"` | Input data directory (relative to project root) |
| `results_dir` | string | `"results_model"` | Output results directory |
| `logs_dir` | string | `"results_model/logs"` | Log files directory |
| `ode_data_dir` | string | `"data"` | ODE model data directory |

> **Tip:** Use distinct `results_dir` / `logs_dir` for each optimization mode to avoid overwriting
> results. For example, use `results_kinopt_local` for `kinopt --mode local`.

---

## Section: `[tfopt]`

Controls the TF → mRNA optimization. Consumed by `tfopt.local` and `tfopt.evol`.

| Key | Type | Default | Description |
|---|---|---|---|
| `input1` | string | `"input1.csv"` | mRNA expression data file |
| `input3` | string | `"input3.csv"` | TF activity / Rout-Limma data |
| `input4` | string | `"input4.csv"` | TF-gene network file |
| `out_file` | string | `"tfopt_results.xlsx"` | Output Excel file |
| `time_points` | list[float] | `[4,8,15,30,60,120,240,480,960]` | Time grid (minutes) |
| `lower_bound` | float | `-4.0` | Lower bound for TF coefficients |
| `upper_bound` | float | `4.0` | Upper bound for TF coefficients |
| `loss_type` | int | `5` | Loss function (see below) |

**Loss type codes:**

| Code | Name |
|---|---|
| 0 | MSE |
| 1 | MAE |
| 2 | Soft L1 |
| 3 | Cauchy |
| 4 | Arctan |
| 5 | Elastic Net |
| 6 | Tikhonov |

Mode-specific overrides in `[tfopt.modes.local]` and `[tfopt.modes.evol]` take precedence.

---

## Section: `[kinopt]`

Controls kinase → phosphorylation optimization. Consumed by `kinopt.local` and `kinopt.evol`.

Key fields mirror `[tfopt]` (input files, bounds, loss type, time points). Mode overrides in
`[kinopt.modes.local]` and `[kinopt.modes.evol]`.

---

## Section: `[ode]`

Controls the per-protein ODE fitting pipeline (`models/`, `paramest/`, `sensitivity/`, `steady/`).

### `[ode.bounds]`

Parameter bounds for ODE model fitting.

### `[ode.bootstrap]`

Bootstrap sample count for confidence intervals.

### `[ode.time]`

Time point grid for ODE integration.

### `[ode.fit]`

Fit settings including loss type and composite weighting.

### `[ode.sensitivity.morris]`

Morris sensitivity analysis settings (trajectories, levels).

### `[ode.plot]`

Plotting options.

### `[ode.inputs]` / `[ode.output]`

Input file names and output directory.

---

## Section: `[global_model]`

Controls the global network-scale pipeline (`global_model/`). Consumed by `global_model/config.py`.

### Core metadata

| Key | Type | Example | Description |
|---|---|---|---|
| `app_name` | string | `"Phoskintime-Global"` | Application display name |
| `version` | string | `"0.4.0"` | Package version |
| `output_dir` | string | `"results_model_global_..."` | Output directory |
| `cores` | int | `80` | CPU cores to use (0 = all available) |
| `seed` | int | `42` | Random seed |

### Input files

| Key | Type | Description |
|---|---|---|
| `kinase_net` | string | Kinase-substrate network (`input2.csv`) |
| `tf_net` | string | TF-gene network (`input4.csv`) |
| `ms` | string | Mass-spec protein data (`input1.csv`) |
| `rna` | string | RNA/transcriptomics data (`input3.csv`) |
| `phospho` | string | Phospho data (can be same as `ms`) |
| `kinopt` | string | Previous kinopt results (Excel) |
| `tfopt` | string | Previous tfopt results (Excel) |

### Optimizer settings

| Key | Type | Default | Description |
|---|---|---|---|
| `optimizer` | string | `"pymoo"` | `"optuna"` or `"pymoo"` |
| `n_gen` | int | `1000` | Number of generations (Pymoo) |
| `pop` | int | `300` | Population size |
| `n_trials` | int | `1000` | Number of Optuna trials |
| `refine` | bool | `false` | Enable iterative refinement pass |
| `num_refinements` | int | `0` | Number of refinement iterations |
| `hyperparam_scan` | bool | `false` | Enable Optuna hyperparameter scan |

### Loss and regularization

| Key | Type | Default | Description |
|---|---|---|---|
| `loss` | int | `0` | Loss type (0=MSE, 1=Huber, 2=Pseudo-Huber, ...) |
| `lambda_prior` | float | `0.1` | Weight for prior adherence |
| `lambda_protein` | float | `1.0` | Weight for protein fit error |
| `lambda_rna` | float | `1.0` | Weight for RNA fit error |
| `lambda_phospho` | float | `1.0` | Weight for phospho fit error |

### Data flags

| Key | Type | Default | Description |
|---|---|---|---|
| `normalize_fc_steady` | bool | `false` | Normalize data to t=0 |
| `use_initial_condition_from_data` | bool | `false` | Use data values as t=0 state |
| `scaling_method` | string | `"raw"` | Data scaling: `raw`, `fc_start`, `robust_fc`, `max_scale`, etc. |

### Weighting methods

| Key | Options | Default |
|---|---|---|
| `weighting_method_protein` | `uniform`, `linear_early`, `exp_late`, etc. | `"uniform"` |
| `weighting_method_rna` | same | `"uniform"` |

### Sensitivity analysis

| Key | Type | Default | Description |
|---|---|---|---|
| `sensitivity_analysis` | bool | `true` | Enable post-optimization sensitivity |
| `sensitivity_perturbation` | float | `0.05` | Perturbation fraction |
| `sensitivity_trajectories` | int | `100` | Morris trajectories |
| `sensitivity_levels` | int | `40` | Morris grid levels |
| `sensitivity_metric` | string | `"total_signal"` | Metric: `total_signal`, `mean`, `variance`, `l2_norm` |

### `[global_model.timepoints]`

| Key | Description |
|---|---|
| `protein` | Time grid for protein data (minutes) |
| `phospho_protein` | Time grid for phospho data |
| `rna` | Time grid for RNA data |

### `[global_model.bounds]`

Kinetic parameter bounds for the global ODE. Each entry is `[min, max]`.

| Parameter | Meaning |
|---|---|
| `c_k` | Kinase activity multiplier |
| `A_i` | Basal mRNA transcription rate |
| `B_i` | mRNA degradation rate |
| `C_i` | Translation rate |
| `D_i` | Protein deactivation rate |
| `Dp_i` | Phospho-site dephosphorylation rate |
| `E_i` | Transcriptional efficacy |
| `tf_scale` | TF scaling factor |

### `[global_model.models]`

| Key | Values |
|---|---|
| `available_models` | `["distributive", "sequential", "combinatorial", "saturation"]` |
| `default_model` | `"distributive"` |

### `[global_model.solver]`

| Key | Default | Description |
|---|---|---|
| `absolute_tolerance` | `1e-8` | ODE absolute tolerance |
| `relative_tolerance` | `1e-8` | ODE relative tolerance |
| `max_timesteps` | `200000` | Maximum ODE integration steps |
| `use_custom_solver` | `false` | Use custom Numba-based solver instead of SciPy |

---

## Which module consumes which section

| Section | Consumer |
|---|---|
| `[paths]` | `config_loader.py`, all subpackages |
| `[tfopt]` | `tfopt.local`, `tfopt.evol` |
| `[kinopt]` | `kinopt.local`, `kinopt.evol` |
| `[ode]` | `models/`, `paramest/`, `sensitivity/`, `steady/` |
| `[global_model]` | `global_model/config.py`, `global_model/runner.py` |
