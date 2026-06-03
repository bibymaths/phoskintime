# API Reference

## Modules

- jax_backend
- optproblem
- inference
- params
- network
- models
- lossfn
- config
- sensitivity
- export
- analysis
- dashboard_app
- dashboard_bundle

## jax_backend

Provide JAX, Diffrax, and JAXopt utilities for scalar networkmodel simulation, multimodal loss evaluation, parameter projection, and ProjectedGradient optimization; it does not describe planned backends or execute unrelated optimization workflows on import, and it depends on networkmodel.config.

| Name | Type | Summary |
| --- | --- | --- |
| `DataMode` | dataclass | Describe which data layers contribute to the scalar loss |
| `ensure_jax_float64` | function | Enable JAX float64 mode |
| `detect_data_mode` | function | Detect which observed data layers are available |
| `validate_loss_data` | function | Validate loss-array presence, shape, and finiteness |
| `DiffraxSolverConfig` | dataclass | Store Diffrax implicit-solver configuration values |
| `make_networkmodel_rhs` | function | Build a JAX right-hand side for the current System topology |
| `solve_diffrax` | function | Solve an ODE trajectory with Diffrax |
| `multimodal_loss_from_trajectory` | function | Compute weighted multimodal loss from a trajectory |
| `project_simplex` | function | Project a vector onto the probability simplex |
| `project_alpha_blocks` | function | Project alpha blocks onto per-block simplexes |
| `project_beta_blocks` | function | Project beta blocks onto bounded per-block simplexes |
| `project_bounds` | function | Project parameters onto bounds and fixed values |
| `JaxoptResult` | dataclass | Store scalar JAXopt optimization outputs |
| `optimize_scalar_objective` | function | Optimize a scalar objective with jaxopt.ProjectedGradient |
| `make_simple_objective` | function | Create the scalar trajectory objective |

### Config keys consumed

None.

### Raises

- RuntimeError: when inputs are invalid or the operation cannot complete.
- ValueError: when inputs are invalid or the operation cannot complete.

## optproblem

Wrap the scalar JAX objective and optimizer used by the runner; it does not describe planned backends or execute unrelated optimization workflows on import, and it depends on networkmodel.jax_backend.

| Name | Type | Summary |
| --- | --- | --- |
| `build_weight_functions` | function | Build placeholder weight functions for scalar optimization |
| `GlobalODEScalarObjective` | class | Evaluate and solve the scalar global ODE objective |
| `GlobalODE_MOO` | class | Reject unsupported multi-objective optimization usage |

### Config keys consumed

None.

### Raises

- ValueError: when class construction or method inputs are unsupported.

## inference

Run multistart optimization, profile-likelihood scans, and optional NumPyro posterior sampling around the scalar JAX objective; it does not describe planned backends or execute unrelated optimization workflows on import, and it depends on networkmodel.jax_backend.

| Name | Type | Summary |
| --- | --- | --- |
| `InferenceContext` | dataclass | Store inputs shared by post-optimization inference routines |
| `configure_jax_parallelism` | function | Configure JAX host parallelism environment variables |
| `generate_multistart_initials` | function | Generate bounded initial vectors for multistart optimization |
| `run_multistart` | function | Run scalar optimization from multiple starting points |
| `run_profile_likelihood` | function | Run profile-likelihood sweeps for selected parameters |
| `run_numpyro_posterior` | function | Run optional NumPyro posterior sampling |

### Config keys consumed

None.

### Raises

- RuntimeError: when inputs are invalid or the operation cannot complete.

## params

Create raw parameter vectors and unpack optimized vectors into named kinetic parameter arrays; it does not describe planned backends or execute unrelated optimization workflows on import, and it depends on networkmodel.config, networkmodel.utils.

| Name | Type | Summary |
| --- | --- | --- |
| `init_raw_params` | function | Initialize raw optimizer parameters, slices, bounds, and defaults |
| `unpack_params` | function | Unpack a raw optimizer vector into physical parameter arrays |

### Config keys consumed

None.

### Raises

- ValueError: when inputs are invalid or the operation cannot complete.

## network

Build index maps, kinase inputs, and mutable system objects for networkmodel ODE evaluation; it does not describe planned backends or execute unrelated optimization workflows on import, and it depends on networkmodel.buildmat, networkmodel.config, networkmodel.models, networkmodel.steadystate.

| Name | Type | Summary |
| --- | --- | --- |
| `Index` | class | Map proteins, sites, kinases, and state-vector offsets |
| `KinaseInput` | class | Interpolate kinase fold-change inputs over time |
| `System` | class | Store topology, parameters, and state for networkmodel ODE evaluation |

### Config keys consumed

None.

### Raises

- ValueError: when class construction or method inputs are unsupported.

## models

Define NumPy right-hand-side kernels for supported phosphorylation topologies; it does not describe planned backends or execute unrelated optimization workflows on import, and it depends on no other networkmodel modules.

| Name | Type | Summary |
| --- | --- | --- |
| `calculate_synthesis_rate` | function | Calculate saturating transcriptional synthesis rate |
| `saturating_rhs` | function | Evaluate the saturating topology right-hand side |
| `distributive_rhs` | function | Evaluate the distributive topology right-hand side |
| `sequential_rhs` | function | Evaluate the sequential topology right-hand side |
| `combinatorial_rhs` | function | Evaluate the combinatorial topology right-hand side |
| `build_random_transitions` | function | Build transition arrays for combinatorial topology |

### Config keys consumed

None.

### Raises

None documented for the public API table.

## lossfn

Compute scalar loss values for protein, RNA, and phospho observations from simulated trajectories; it does not describe planned backends or execute unrelated optimization workflows on import, and it depends on networkmodel.config.

| Name | Type | Summary |
| --- | --- | --- |
| `sq` | function | Compute squared loss |
| `huber` | function | Compute Huber loss |
| `pseudo_huber` | function | Compute pseudo-Huber loss |
| `charbonnier` | function | Compute Charbonnier loss |
| `log_cosh` | function | Compute log-cosh loss |
| `cauchy_loss` | function | Compute Cauchy loss |
| `poisson_scaled_mse` | function | Compute Poisson-scaled mean squared error |
| `geman_mcclure` | function | Compute Geman-McClure loss |
| `loss_function_noncomb` | function | Compute multimodal loss for non-combinatorial state layouts |
| `loss_function_comb` | function | Compute multimodal loss for combinatorial state layouts |

### Config keys consumed

None.

### Raises

None documented for the public API table.

## config

Load networkmodel settings from config.toml and expose typed module constants for paths, time grids, model selection, solver controls, optimization controls, regularization weights, inference options, sensitivity options, and metadata; it does not describe planned backends or execute unrelated optimization workflows on import, and it depends on no other networkmodel modules.

| Name | Type | Summary |
| --- | --- | --- |
| None | function | No public functions or classes |

### Config keys consumed

The module reads these `config.toml` keys through the loaded networkmodel configuration object:

```toml
app_name = "Phoskintime-Global"
version = "0.1.0"
parent_package = "phoskintime"
citation = ""
doi = ""
github_url = ""
docs_url = ""
hyperparam_scan = false
n_starts = 1
profile_likelihood = false
profile_indices = ""
profile_grid_size = 10
posterior_sampling = false
posterior_num_warmup = 20
posterior_num_samples = 30
scaling_method = "none"
weighting_method_protein = "uniform"
weighting_method_rna = "uniform"
weighting_method_phospho = "uniform"
sensitivity_analysis = false
sensitivity_perturbation = 0.2
sensitivity_trajectories = 1000
sensitivity_levels = 400
sensitivity_top_curves = 50
sensitivity_metric = "total_signal"
available_models = []
```

- `app_name`: string, default `"Phoskintime-Global"`.
- `version`: string, default `"0.1.0"`.
- `parent_package`: string, default `"phoskintime"`.
- `citation`: string, default `""`.
- `doi`: string, default `""`.
- `github_url`: string, default `""`.
- `docs_url`: string, default `""`.
- `hyperparam_scan`: boolean, default `false`.
- `n_starts`: integer, default `1`.
- `profile_likelihood`: boolean, default `false`.
- `profile_indices`: string, default `""`.
- `profile_grid_size`: integer, default `10`.
- `posterior_sampling`: boolean, default `false`.
- `posterior_num_warmup`: integer, default `20`.
- `posterior_num_samples`: integer, default `30`.
- `scaling_method`: string, default `"none"`.
- `weighting_method_protein`: string, default `"uniform"`.
- `weighting_method_rna`: string, default `"uniform"`.
- `weighting_method_phospho`: string, default `"uniform"`.
- `sensitivity_analysis`: boolean, default `false`.
- `sensitivity_perturbation`: float, default `0.2`.
- `sensitivity_trajectories`: integer, default `1000`.
- `sensitivity_levels`: integer, default `400`.
- `sensitivity_top_curves`: integer, default `50`.
- `sensitivity_metric`: string, default `"total_signal"`.
- `available_models`: array, default `[]`.

### Raises

None documented for the public API table.

## sensitivity

Run perturbation-based sensitivity analysis and write sensitivity diagnostics; it does not describe planned backends or execute unrelated optimization workflows on import, and it depends on networkmodel.config, networkmodel.simulate.

| Name | Type | Summary |
| --- | --- | --- |
| `compute_bounds` | function | Compute perturbation bounds for sensitivity analysis |
| `run_sensitivity_analysis` | function | Run perturbation sensitivity analysis |

### Config keys consumed

None.

### Raises

None documented for the public API table.

## export

Write scalar optimization outputs, diagnostic plots, residuals, parameter summaries, and fitted activity tables; it does not describe planned backends or execute unrelated optimization workflows on import, and it depends on networkmodel.config, networkmodel.jacspeedup, networkmodel.params, networkmodel.simulate.

| Name | Type | Summary |
| --- | --- | --- |
| `build_site_meta` | function | Build phosphosite metadata rows from an index map |
| `create_convergence_video` | function | Create a convergence-history video |
| `export_pareto_front_to_excel` | function | Export scalar optimization trajectories and summaries to Excel |
| `plot_goodness_of_fit` | function | Plot observed-versus-predicted goodness of fit |
| `plot_gof_from_pareto_excel` | function | Plot goodness of fit from an exported Excel workbook |
| `export_results` | function | Export fitted trajectories and summary outputs |
| `save_gene_timeseries_plots` | function | Save per-gene observed and predicted time-series plots |
| `scan_prior_reg` | function | Scan prior regularization outputs in a result directory |
| `export_S_rates` | function | Export phosphorylation-rate trajectories |
| `plot_s_rates_report` | function | Plot phosphorylation-rate reports from CSV data |
| `process_convergence_history` | function | Export convergence-history tables and plots |
| `export_kinase_activities` | function | Export kinase activity trajectories |
| `export_param_correlations` | function | Export parameter-correlation diagnostics |
| `export_residuals` | function | Export residual tables for fitted observations |
| `export_parameter_distributions` | function | Export optimized parameter distribution plots |

### Config keys consumed

None.

### Raises

- RuntimeError: when inputs are invalid or the operation cannot complete.
- ValueError: when inputs are invalid or the operation cannot complete.

## analysis

Simulate a system to a long time horizon and write steady-state diagnostic plots; it does not describe planned backends or execute unrelated optimization workflows on import, and it depends on networkmodel.config, networkmodel.simulate.

| Name | Type | Summary |
| --- | --- | --- |
| `simulate_until_steady` | function | Simulate the system toward steady state |
| `plot_steady_state_all` | function | Write steady-state diagnostic plots and tables |

### Config keys consumed

None.

### Raises

None documented for the public API table.

## dashboard_app

Render saved networkmodel outputs in a Streamlit dashboard; it does not describe planned backends or execute unrelated optimization workflows on import, and it depends on networkmodel.dashboard_bundle.

| Name | Type | Summary |
| --- | --- | --- |
| `main` | function | Run the networkmodel entry point |

### Config keys consumed

None.

### Raises

None documented for the public API table.

## dashboard_bundle

Save and load compact dashboard payloads for scalar optimization runs; it does not describe planned backends or execute unrelated optimization workflows on import, and it depends on no other networkmodel modules.

| Name | Type | Summary |
| --- | --- | --- |
| `save_dashboard_bundle` | function | Save dashboard input data to disk |
| `load_dashboard_bundle` | function | Load dashboard input data from disk |

### Config keys consumed

None.

### Raises

None documented for the public API table.

## mode_outputs

Write mode-aware metadata, result tables, and simple scalar-run plots; it does not describe planned backends or execute unrelated optimization workflows on import, and it depends on networkmodel.jax_backend.

| Name | Type | Summary |
| --- | --- | --- |
| `write_mode_metadata` | function | Write scalar-run mode metadata |
| `write_scalar_result_tables` | function | Write scalar objective result tables |
| `save_mode_plots` | function | Save scalar-run mode plots |

### Config keys consumed

None.

### Raises

None documented for the public API table.

## simulate

Simulate a System with Diffrax and extract protein, RNA, and phospho measurement tables; it does not describe planned backends or execute unrelated optimization workflows on import, and it depends on networkmodel.config, networkmodel.jax_backend.

| Name | Type | Summary |
| --- | --- | --- |
| `simulate_diffrax` | function | Simulate a System over requested time points with Diffrax |
| `simulate_and_measure` | function | Simulate a System and return measured output tables |

### Config keys consumed

None.

### Raises

None documented for the public API table.

## steadystate

Build initial state vectors from optional observed baseline data; it does not describe planned backends or execute unrelated optimization workflows on import, and it depends on networkmodel.config.

| Name | Type | Summary |
| --- | --- | --- |
| `build_y0_from_data` | function | Build an initial state vector from observed baseline data |

### Config keys consumed

None.

### Raises

None documented for the public API table.

## buildmat

Build transcription-factor and kinase-to-site matrices from interaction tables and index maps; it does not describe planned backends or execute unrelated optimization workflows on import, and it depends on networkmodel.config.

| Name | Type | Summary |
| --- | --- | --- |
| `site_key` | function | Normalize a phosphorylation site label |
| `build_W_parallel` | function | Build kinase-to-site weight matrices in parallel |
| `build_tf_matrix` | function | Build the transcription-factor regulatory matrix |

### Config keys consumed

None.

### Raises

- ValueError: when inputs are invalid or the operation cannot complete.

## cache

Convert observation data frames into compact numeric arrays for fast loss evaluation; it does not describe planned backends or execute unrelated optimization workflows on import, and it depends on networkmodel.config.

| Name | Type | Summary |
| --- | --- | --- |
| `prepare_fast_loss_data` | function | Prepare numeric loss arrays from observation data frames |

### Config keys consumed

None.

### Raises

- ValueError: when inputs are invalid or the operation cannot complete.

## io

Load network, protein, RNA, and phospho input tables from configured paths; it does not describe planned backends or execute unrelated optimization workflows on import, and it depends on networkmodel.config, networkmodel.utils.

| Name | Type | Summary |
| --- | --- | --- |
| `load_data` | function | Load configured networkmodel input tables |

### Config keys consumed

None.

### Raises

None documented for the public API table.

## jacspeedup

Provide small NumPy-compatible helper kernels for phosphorylation-rate cache evaluation; it does not describe planned backends or execute unrelated optimization workflows on import, and it depends on no other networkmodel modules.

| Name | Type | Summary |
| --- | --- | --- |
| `build_S_cache_into` | function | Fill a phosphorylation-rate cache array |
| `kin_eval_step` | function | Evaluate kinase inputs at a time point |

### Config keys consumed

None.

### Raises

None documented for the public API table.

## scan

Expose a compatibility hyperparameter-scan entry point for the scalar JAXopt path; it does not describe planned backends or execute unrelated optimization workflows on import, and it depends on no other networkmodel modules.

| Name | Type | Summary |
| --- | --- | --- |
| `run_hyperparameter_scan` | function | Run the scalar compatibility hyperparameter scan |

### Config keys consumed

None.

### Raises

None documented for the public API table.

## utils

Normalize input data, transform positive parameters, load TOML configuration, and compute optimization bounds; it does not describe planned backends or execute unrelated optimization workflows on import, and it depends on networkmodel.config.

| Name | Type | Summary |
| --- | --- | --- |
| `normcols` | function | Normalize column labels |
| `find_col` | function | Find the first matching column name |
| `slen` | function | Return the length of a sequence-like value |
| `normalize_fc_to_t0` | function | Normalize fold-change values to the baseline time |
| `process_and_scale_raw_data` | function | Convert wide raw data into scaled tidy time-series data |
| `time_bucket` | function | Map a time value to the nearest grid index |
| `softplus` | function | Apply a numerically stable softplus transform |
| `inv_softplus` | function | Apply the inverse softplus transform |
| `PhosKinConfig` | dataclass | Store networkmodel TOML configuration values |
| `load_config_toml` | function | Load networkmodel configuration from TOML |
| `calculate_bio_bounds` | function | Calculate biologically constrained optimizer bounds |
| `get_optimized_sets` | function | Report which parameter groups are optimized |

### Config keys consumed

None.

### Raises

- ValueError: when inputs are invalid or the operation cannot complete.

## runner

Run the command-line networkmodel workflow that loads data, builds topology, optimizes parameters, and writes outputs; it does not describe planned backends or execute unrelated optimization workflows on import, and it depends on networkmodel.analysis, networkmodel.buildmat, networkmodel.cache, networkmodel.config, networkmodel.dashboard_bundle, networkmodel.export, networkmodel.inference, networkmodel.io, networkmodel.jax_backend, networkmodel.mode_outputs, networkmodel.network, networkmodel.optproblem, networkmodel.params, networkmodel.scan, networkmodel.sensitivity, networkmodel.simulate, networkmodel.steadystate, networkmodel.utils.

| Name | Type | Summary |
| --- | --- | --- |
| `main` | function | Run the networkmodel entry point |

### Config keys consumed

None.

### Raises

- ValueError: when inputs are invalid or the operation cannot complete.
