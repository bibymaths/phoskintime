# Configuration Reference

This page documents the `config.toml` configuration file that controls all aspects of the
PhosKinTime pipeline.

---

## Location and Loading

The configuration file is `config.toml` at the project root. It is loaded by two modules:

- **`config_loader.py`** — the root-level loader. Provides `load(mode, section)` (LRU-cached)
  and `load_config_toml()`. Used by the local pipeline and as the backbone for `networkmodel`.
- **`networkmodel/config.py`** — loads `config.toml` via `config_loader.load_config_toml()` and
  exports constants for the global model pipeline.

> **Technical debt:** Both modules load overlapping configuration surfaces from `config.toml`.
> They should be unified so `networkmodel/config.py` entirely delegates to `config_loader.py`.
> See `config_loader.py` for the shared loader behavior.

---

## Section: `[paths]`

Global project paths. Used by all pipeline stages.

| Key | Type | Default | Description |
|---|---|---|---|
| `data_dir` | string | `"data"` | Input data directory (relative to project root) |
| `results_dir` | string | `"results_model"` | Output results directory |
| `logs_dir` | string | `"results_model/logs"` | Log files directory |

> **Tip:** Use distinct `results_dir` / `logs_dir` for each run to avoid overwriting results.

---

## KinOpt and TFOpt settings

`config.toml` has no `[tfopt]` or `[kinopt]` section. KinOpt and TFOpt are separate tools with
their own configuration; PhosKinTime only reads their result workbooks. See
[External tools](external_tools.md).

---

## Section: `[ode]`

Controls the per-protein ODE fitting pipeline (`protwise/models/`, `protwise/paramest/`, `protwise/sensitivity/`, `protwise/steady/`).

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

Input file names and output directory. `psite_excel` is the KinOpt result workbook and
`rna_excel` the TFOpt result workbook; both come from the external tools
([External tools](external_tools.md)).

---

## Section: `[networkmodel]`

Controls the global network-scale pipeline (`networkmodel/`). Consumed by `networkmodel/config.py`.

### Core metadata

| Key | Type | Example | Description |
|---|---|---|---|
| `app_name` | string | `"Phoskintime-Global"` | Application display name |
| `version` | string | `"0.4.0"` | Package version |
| `output_dir` | string | `"results_model_global_distributive_jax"` | Output directory |
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
| `kinopt` | string | KinOpt result workbook (Excel), from the external KinOpt tool |
| `tfopt` | string | TFOpt result workbook (Excel), from the external TFOpt tool |

### Optimizer settings

The current global path uses `jaxopt.ProjectedGradient` with `projection=project_bounds`. The runner exposes `--n-gen` as the command-line name for the maximum JAXopt iteration count; `config_loader.load_config_toml()` maps the `[networkmodel]` key `n_gen` to `MAX_ITERATIONS` in `networkmodel/config.py`.

```bash
python -m networkmodel.runner --solver jaxopt --n-gen 30
```

| Key | Type | Default | Description |
|---|---|---|---|
| `n_gen` | int | `200` in the loader fallback | Maximum JAXopt iterations, exposed as `MAX_ITERATIONS` |
| `hyperparam_scan` | bool | `false` | Enables the compatibility hyperparameter scan entry point |

### Loss and regularization

| Key | Type | Default | Description |
|---|---|---|---|
| `loss` | int | `0` | Loss selector used by the configured loss functions |
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

### `[networkmodel.timepoints]`

| Key | Description |
|---|---|
| `protein` | Time grid for protein data (minutes) |
| `phospho_protein` | Time grid for phospho data |
| `rna` | Time grid for RNA data |

### `[networkmodel.bounds]`

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

### `[networkmodel.models]`

| Key | Values |
|---|---|
| `available_models` | `["distributive", "sequential", "combinatorial", "saturation"]` |
| `default_model` | `"distributive"` |

### `[networkmodel.solver]`

The global path uses Diffrax through `DiffraxSolverConfig`. Current solver names are `Kvaerno4` and `Kvaerno5`; `simulate_diffrax` defaults to `Kvaerno4`.

| Key | Default | Description |
|---|---|---|
| `absolute_tolerance` | `1e-8` | ODE absolute tolerance |
| `relative_tolerance` | `1e-8` | ODE relative tolerance |
| `max_timesteps` | `200000` | Maximum ODE integration steps |


---

## Which module consumes which section

| Section | Consumer |
|---|---|
| `[paths]` | `config_loader.py`, all subpackages |
| `[ode]` | `protwise/models/`, `protwise/paramest/`, `protwise/sensitivity/`, `protwise/steady/` |
| `[networkmodel]` | `networkmodel/config.py`, `networkmodel/runner.py` |
