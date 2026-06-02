# PhosKinTime Multi-Modal Input Audit

## 1. Purpose

This is an audit-only planning document for a future implementation of concurrent mRNA, protein, and phosphoprotein/phosphosite data support in PhosKinTime. It documents how PhosKinTime currently loads data, builds model state, constructs losses, optimizes, plots, saves, and logs results, then proposes a safe internal design for mode-aware fitting without changing the user-facing configuration schema.

The intended future behavior is that the same core ODE model continues to expose transcript, protein, and phosphorylation observables, while the objective function includes only the observable layers that are actually available in the input data. The implementation should support all seven non-empty data-layer combinations:

1. mRNA only
2. protein only
3. phospho only
4. mRNA + protein
5. mRNA + phospho
6. protein + phospho
7. mRNA + protein + phospho

The audit focuses on PhosKinTime-relevant source, CLI, configuration, input, model, optimization, plotting, saving, logging, tests, examples, and documentation outside KinOpt and TFOpt.

## 2. Non-goals

- No code changes are proposed or implemented in this audit.
- No Python files, tests, or configuration files should be edited for this audit.
- No configuration schema changes should be required for the future implementation.
- KinOpt and TFOpt are intentionally ignored. This document does not audit, modify, or propose changes for `kinopt`, `tfopt`, or files/modules/configs that specifically belong to KinOpt or TFOpt.
- No unrelated refactor is recommended as a prerequisite.
- No separate unrelated models should be created for different input data types.
- No redesign of the biological model is recommended beyond making observable extraction, loss construction, optimization bookkeeping, outputs, plotting, and logging mode-aware.

## 3. Current Repository Structure Relevant to PhosKinTime

The PhosKinTime-relevant repository structure outside KinOpt and TFOpt appears to have two ODE-oriented workflows:

### Global network model workflow

| File/folder | Apparent role |
|---|---|
| `networkmodel/` | Integrated network-scale ODE pipeline. This is the most relevant target for concurrent mRNA/protein/phospho inputs because it already has RNA, protein, and phospho observables and a three-objective optimizer. |
| `networkmodel/runner.py` | Main orchestration entry point for the global model: parses CLI args, logs configuration, loads data, builds the model index, builds matrices, constructs loss data, runs Pymoo/Optuna, selects a solution, saves outputs, plots, and logs final summaries. |
| `networkmodel/io.py` | Loads kinase and TF networks, optional prior Excel outputs, mass-spec data, RNA data, and reshapes/scales observations into tidy `df_prot`, `df_pho`, and `df_rna` data frames. Currently reads both protein and phospho from `args.ms` and reads RNA from `args.rna`. |
| `networkmodel/network.py` | Defines `Index`, `KinaseInput`, and `System`. These map biology to state-vector indices, interpolate protein/kinase data, attach optional data-driven initial conditions, and expose the ODE RHS. |
| `networkmodel/models.py` | Low-level model-specific RHS kernels for distributive, sequential, combinatorial, and saturating dynamics. Relevant because it determines structural state layout but should not need mode-specific branching. |
| `networkmodel/simulate.py` | Solves the ODE and extracts predicted RNA, protein, and phospho fold-change data frames via `simulate_and_measure`. |
| `networkmodel/lossfn.py` | Numba loss kernels for non-combinatorial and combinatorial models. It computes separate protein, RNA, and phospho losses from state trajectories and indexed observation arrays. |
| `networkmodel/cache.py` | Converts tidy observations into integer arrays for fast loss calculation. Currently processes all three layers and returns count/weight arrays for each. |
| `networkmodel/optproblem.py` | Defines `GlobalODE_MOO`, a Pymoo `ElementwiseProblem` with fixed `n_obj=3` for protein, RNA, and phospho objectives. Also contains time-weighting utilities. |
| `networkmodel/optuna_solver.py` | Alternative Optuna solver path. It has its own loss-index augmentation and currently expects protein, RNA, and phospho target/index keys in a different shape than `prepare_fast_loss_data`. |
| `networkmodel/params.py` | Initializes and unpacks parameter vectors (`c_k`, `A_i`, `B_i`, `C_i`, `D_i`, `Dp_i`, `E_i`, `tf_scale`). These should stay structurally consistent across modes. |
| `networkmodel/buildmat.py` | Builds kinase-substrate and TF regulatory matrices from cleaned network tables. Relevant to model setup, not to data-layer mode selection except when available observations affect the modeled universe. |
| `networkmodel/export.py` | Exports Pareto fronts, predictions, model parameters, residuals, plots, S-rate reports, convergence artifacts, and per-gene time-series plots. Many functions assume all three modalities exist. |
| `networkmodel/dashboard_bundle.py` | Saves a pickle bundle containing args, Pareto data, bounds, defaults, data frames, scores, and selected index for dashboard use. Should include mode metadata later. |
| `networkmodel/dashboard_app.py` and `run_dashboard.py` | Interactive dashboard entry points for saved global-model outputs. Relevant for output schema compatibility. |
| `networkmodel/refine.py`, `networkmodel/scan.py`, `networkmodel/sensitivity.py`, `networkmodel/analysis.py`, `networkmodel/steadystate.py` | Downstream optimization refinement, hyperparameter scan, sensitivity, steady-state simulation, and data-derived initial-condition support. They need only targeted adjustments if they assume three objectives/layers. |
| `networkmodel/config.py` | Imports parsed config values from `config_loader.load_config_toml()` and exposes global constants used by the global model. |
| `networkmodel/README.md`, `networkmodel/INTERPRETATION.md` | Global-model documentation and interpretation guidance. Should be updated only after implementation. |

### Legacy/local PhosKinTime ODE workflow

| File/folder | Apparent role |
|---|---|
| `protwise/` | Legacy/local per-protein ODE workflow and analysis utilities. It fits one gene/protein at a time and already models mRNA, protein, and phosphosite states, but currently requires common genes across mRNA and phosphorylation inputs. |
| `protwise/runner/main.py` | Legacy/local runner. Loads protein CSV, phosphosite Excel, mRNA Excel; validates fixed columns; restricts processing to the intersection of phospho genes and mRNA genes; calls `process_gene_wrapper`; then saves, plots, and reports. |
| `protwise/paramest/core.py` | Per-gene orchestration: extracts protein-only rows, phosphosite rows, and mRNA rows; builds arrays; calls parameter estimation; computes metrics; solves the ODE; generates plots, knockout simulations, sensitivity, and returns result dictionaries. |
| `protwise/paramest/normest.py` | Builds a single flattened target vector as `[mRNA, protein, phospho]`, fits by `scipy.optimize.curve_fit`, and scores the fit. Currently assumes all three layers are present. |
| `protwise/paramest/toggle.py` | Selects/dispatches the parameter-estimation implementation used by the local workflow. |
| `protwise/models/distmod.py`, `protwise/models/succmod.py`, `protwise/models/randmod.py` | Local ODE model implementations. The state layout is `R`, `P`, and one or more phosphosite states. Their output flattening is currently fixed to mRNA + protein + phospho. |
| `protwise/plotting/plotting.py` | Legacy/local plotting. `Plotter.plot_model_fit` hard-codes mRNA, protein, and phosphosite plotting and fixed time slices. |
| `protwise/steady/` | Initial-condition helpers for local ODE models. |
| `protwise/sensitivity/`, `protwise/knockout/` | Local sensitivity and knockout analyses, mostly downstream of a fitted model. |
| `common/utils/display.py` | Shared local workflow output and report helpers. `save_result()` writes Excel sheets for fitted state solutions, site estimates, observations, PCA, t-SNE, and errors; `merge_obs_est()` currently merges only site observed/estimated sheets. |
| `common/utils/tables.py`, `common/utils/latexit.py`, `common/utils/display.py` | Table, LaTeX, and report generation for local outputs. |

### Configuration, CLI, processing, tests, docs, and scripts

| File/folder | Apparent role |
|---|---|
| `config.toml` | Central user-facing configuration. Contains `[paths]`, `[ode]`, and `[networkmodel]` sections relevant here. It also contains KinOpt/TFOpt sections, which are excluded from this audit. |
| `config_loader.py` | Root-level TOML loader. Provides `load()`, `ensure_dirs()`, `PhosKinConfig`, and `load_config_toml()`. `load_config_toml()` maps `[networkmodel]` keys into a dataclass. |
| `config/config.py` | CLI argument parsing and config extraction for the legacy/local ODE workflow. It exposes `--input-excel-protein`, `--input-excel-psite`, and `--input-excel-rna`. |
| `config/constants.py` | Constants derived from the `[ode]` config section, including ODE input paths, bounds, time points, output names, scoring weights, model type, sensitivity flags, and local workflow input defaults. |
| `config/cli.py`, `__main__.py` | Typer CLI shortcuts. `networkmodel` runs `networkmodel.runner`; `model` runs the legacy runner; `all` runs prep and excluded KinOpt/TFOpt/local model sequence. Future multimodal global support should primarily affect `networkmodel`. |
| `config/logconf.py` | Shared logging setup. Future mode messages should use this logger without changing logging infrastructure unless necessary. |
| `processing/cleanup.py`, `processing/map.py` | Preprocessing/mapping helpers that create/transform network and data files. Relevant to examples and input expectations, but should not be required to change for internal mode inference. |
| `tests/test_config.py` | Existing tests cover logging behavior only. Future tests should add global-model data-mode tests outside existing config tests. |
| `docs/Documentation/*.md`, `docs/index.md`, `README.md`, `PYPI_README.md` | User-facing documentation. Current docs describe three-modality global fitting and the legacy/local workflow. Update after implementation only. |
| `scripts/` | Standalone analysis/post-processing scripts that consume outputs. They should generally not be modified unless their assumptions break with mode-aware output files. |

## 4. Current Data Flow

### 4.1 CLI and configuration loading

#### Global network model

1. `config/cli.py` exposes a `networkmodel` command that runs `python -m networkmodel.runner --conf config.toml` through `_run()`/`_python_module()`.
2. `networkmodel/config.py` imports `load_config_toml("config.toml")` from `config_loader.py` at import time and converts fields into module-level constants such as `KINASE_NET_FILE`, `MS_DATA_FILE`, `RNA_DATA_FILE`, `PHOSPHO_DATA_FILE`, `TIME_POINTS_PROTEIN`, `TIME_POINTS_RNA`, `TIME_POINTS_PHOSPHO`, model selection, solver tolerances, optimizer settings, lambdas, scaling, and weighting.
3. `config_loader.load_config_toml(path)` reads the `[networkmodel]` table from `config.toml`. It maps:
   - `kinase_net` to `PhosKinConfig.kinase_net`
   - `tf_net` to `PhosKinConfig.tf_net`
   - `ms` to `PhosKinConfig.ms_data`
   - `rna` to `PhosKinConfig.rna_data`
   - `phospho` to `PhosKinConfig.phospho_data`, defaulting to `ms` when absent or falsey
   - `kinopt` and `tfopt` prior-result paths
   - `[networkmodel.timepoints]` into protein/RNA/phospho time grids
   - `[networkmodel.bounds]`, `[networkmodel.solver]`, optimizer, lambdas, scaling, weighting, sensitivity, and metadata.
4. `networkmodel.runner.main()` separately defines argparse options with defaults imported from `networkmodel.config`, including `--ms`, `--rna`, `--phospho`, and lambda arguments. Important current behavior: `--phospho` is parsed but `networkmodel.io.load_data()` currently reads phospho from `args.ms`, not `args.phospho`.

#### Legacy/local ODE model

1. `config/cli.py` exposes a `model` command that runs `python -m runner.main`, but the repository path inspected here is `protwise/runner/main.py`. Documentation suggests `runner.main`; the actual module layout should be checked before implementation because there may be packaging aliases or historical paths.
2. `protwise/runner/main.py` imports `parse_args()`, `extract_config()`, and `log_config()` from `config/config.py`.
3. `config/config.py` reads defaults from `config/constants.py` and exposes CLI arguments for bounds, bootstraps, `--input-excel-protein`, `--input-excel-psite`, and `--input-excel-rna`.
4. `config/constants.py` derives legacy/local input paths from `[ode.inputs]`: `protein_excel`, `psite_excel`, and `rna_excel`.

### 4.2 Input file reading

#### Global network model (`networkmodel/io.py`)

`load_data(args)` currently performs the following:

1. Loads `args.kinase_net` CSV, normalizes columns with `_normcols()`, finds topology columns by candidate names, expands comma/set kinase notations, and returns `df_kin_clean` with `protein`, `psite`, `kinase`.
2. Optionally loads kinase prior alphas and betas from `args.kinopt` Excel sheets if the file exists; missing/failed loads are logged and filled/defaulted.
3. Loads `args.tf_net` CSV, normalizes columns, finds TF/target columns, and optionally merges TF prior alphas/betas from `args.tfopt` Excel sheets.
4. Loads mass-spec data from `args.ms` into `df_ms_raw`. It uses `process_and_scale_raw_data()` with `TIME_POINTS_PROTEIN`, `id_cols=["GeneID", "Psite"]`, and `SCALING_METHOD`. Then it normalizes columns, finds gene and psite columns, and reshapes either `x1..xN` wide columns or pre-melted `time`/`fc` columns.
5. Splits the reshaped mass-spec tidy table into:
   - `df_prot`: rows where `psite` is blank (`protein`, `time`, `fc`)
   - `df_pho`: rows where `psite` is nonblank (`protein`, `psite`, `time`, `fc`)
6. Loads RNA from `args.rna` CSV, normalizes columns, finds `geneid`/`mrna`/`gene`, renames it to `protein`, and uses `process_and_scale_raw_data()` with `TIME_POINTS_RNA` and `id_cols=["protein"]`.
7. Returns `df_kin_clean`, `df_tf_clean`, `df_prot`, `df_pho`, `df_rna`, `kin_beta_map`, `tf_beta_map`.

Current implication: global mode detection can happen immediately after `load_data()` returns, or inside `load_data()` after each layer is parsed. Since `load_data()` already creates distinct tidy layer data frames, it is the natural place to compute raw layer availability and validation metadata. `networkmodel.runner.main()` is the natural place to finalize the effective mode after filtering observations to the model index and mechanistic phosphosite pairs.

#### Legacy/local ODE (`protwise/runner/main.py` and `protwise/paramest/core.py`)

1. `protwise/runner/main.py` reads:
   - protein data: `pd.read_csv(config['input_excel_protein'])`
   - phosphosite/kinase data: `pd.read_excel(config['input_excel_psite'], sheet_name='Estimated')`
   - mRNA data: `pd.read_excel(config['input_excel_rna'], sheet_name='Estimated')`
2. It validates that both protein and phosphosite data collectively include `Gene`, `Psite`, and `x1..x14`, and that mRNA includes `mRNA` and `x1..x9`.
3. It computes `common_proteins = sorted(set(kinase_data['Gene']).intersection(set(mrna_data['mRNA'])))`, then processes only those common proteins. This currently prevents mRNA-only, protein-only, phospho-only, mRNA+protein without phospho, and protein+phospho without mRNA in the legacy flow.
4. `protwise/paramest/core.py::process_gene()` further extracts:
   - protein rows where `protein_data['Psite'].isna()` and `GeneID == gene`
   - phosphosite rows where `kinase_data['Gene'] == gene`
   - RNA rows where `mrna_data['mRNA'] == gene`
   It then slices values by fixed positional columns and passes all arrays into `estimate_parameters()`.

### 4.3 Preprocessing and normalization

#### Global network model

- `networkmodel.utils._normcols()` lowercases and normalizes column names.
- `networkmodel.utils._find_col()` returns the first matching candidate column but returns `None` if none is found; several callers assume the returned column is valid, so missing columns can fail later with less explicit errors.
- `networkmodel.utils.process_and_scale_raw_data()` expects wide `x1`, `x2`, ... columns. It supports scaling methods `raw`/`none`, `fc_start`, `robust_fc`, `max_scale`, `mean_scale`, and `l2_norm`, then melts to tidy `id_cols + [time, fc]`.
- `networkmodel.utils.normalize_fc_to_t0()` can normalize protein/phospho data by a t=0 baseline. In `runner.main()`, this is currently applied only to `df_prot` and `df_pho`, not RNA.
- `networkmodel.runner.main()` applies a strict mechanistic phospho filter, dropping any `(protein, psite)` in `df_pho` not represented in `df_kin`.
- `networkmodel.runner.main()` restricts all observations to `idx.proteins` after building the model index.
- `networkmodel.optproblem.build_weight_functions()` builds time-dependent weights. `runner.main()` currently applies one protein/phospho weight function to both `df_prot` and `df_pho`, and a separate RNA weight function to `df_rna`. Although config has `weighting_method_phospho`, the current runner imports it but does not use it separately.

#### Legacy/local model

- `protwise/runner/main.py` assumes input tables already contain the expected `x1..x14` and `x1..x9` columns and does not use the global `process_and_scale_raw_data()` utility.
- `protwise/paramest/core.py` uses positional slicing (`iloc[:, 2:]` for protein/phospho, `iloc[:, 1:]` for mRNA), so the order and count of columns are critical.
- `protwise/models/distmod.py::solve_ode()` optionally normalizes model output if `NORMALIZE_MODEL_OUTPUT` is true.

### 4.4 Model setup and ODE solving

#### Global network model

1. `networkmodel.runner.main()` builds `df_tf_model` using TF proxy logic and constructs `idx = Index(df_kin, tf_interactions=df_tf_model, ...)`.
2. `networkmodel.network.Index` defines the state-vector layout. For non-combinatorial models, each protein has `[mRNA_i, protein_i, phosphosite_1, ..., phosphosite_n]`. For combinatorial models, each protein has `[mRNA_i, protein_state_0, ..., protein_state_2^n]`.
3. `networkmodel.network.KinaseInput` builds a piecewise-constant kinase activity matrix from `df_prot`, defaulting to ones for missing kinases.
4. `networkmodel.buildmat.build_W_parallel()` and `build_tf_matrix()` construct sparse kinase-substrate and TF matrices.
5. `networkmodel.network.System` stores parameters, matrices, kinase input, optional data-derived initial conditions, and RHS packing for Numba/ODE solvers.
6. `networkmodel.simulate.simulate_odeint()` uses either the custom solver or `scipy.integrate.odeint` over the union time grid.
7. `networkmodel.simulate.simulate_and_measure()` extracts predicted fold-change tables for all three observable layers from a simulated trajectory.

#### Legacy/local model

1. `protwise.paramest.core.process_gene()` calls `protwise.steady.initial_condition(num_psites)`.
2. `protwise.paramest.toggle.estimate_parameters()` dispatches to the selected parameter-estimation routine.
3. Model solvers in `protwise/models/*.py` solve a per-protein ODE over `TIME_POINTS` and return a flattened output containing mRNA, protein, and phosphosite predictions.
4. `protwise.models.distmod.solve_ode()` returns `sol` and `np.concatenate((R_fitted.flatten(), Pr_fitted.flatten(), P_fitted.flatten()))`, where `R_fitted` currently uses `sol[5:, 0]` and therefore implicitly assumes a fixed offset between protein/phospho and RNA time grids.

### 4.5 Objective/loss creation and optimization

#### Global network model

1. `runner.main()` creates `solver_times = np.unique(np.concatenate([TIME_POINTS_PROTEIN, TIME_POINTS_RNA, TIME_POINTS_PHOSPHO]))`.
2. `networkmodel.cache.prepare_fast_loss_data(idx, df_prot, df_rna, df_pho, solver_times)` maps each tidy data frame into integer arrays:
   - protein: `p_prot`, `t_prot`, `obs_prot`, `w_prot`
   - RNA: `p_rna`, `t_rna`, `obs_rna`, `w_rna`
   - phospho: `p_pho`, `s_pho`, `t_pho`, `obs_pho`, `w_pho`
   - state metadata: `prot_map`, `n_p`, `n_r`, `n_ph`
3. Counts are currently returned as `max(1, len(obs_*))`, which hides empty layers as count `1`. This is risky for mode detection and logging.
4. `networkmodel.lossfn.LOSS_FN` computes raw protein, RNA, and phospho loss sums by looping over the arrays for each modality. Empty arrays naturally produce zero sums, but downstream objective assembly still creates three objectives.
5. `networkmodel.optproblem.GlobalODE_MOO` is fixed to `n_obj=3`. It computes normalizers as inverse sums of `w_prot`, `w_rna`, and `w_pho`, guarded by `max(1e-6, sum(weights))`.
6. It returns `[protein_obj + prior_penalty, rna_obj + prior_penalty, phospho_obj + prior_penalty]` regardless of whether any layer has observations.
7. The Pymoo path then runs `UNSGA3`, saves `pareto_X.npy`, `pareto_F.npy`, `pareto_F.csv`, and exports Pareto front Excel.
8. The Optuna path (`networkmodel/optuna_solver.py`) uses `_augment_loss_data()` and `NativeOptunaObjective`. This path appears inconsistent with `prepare_fast_loss_data()` because `NativeOptunaObjective._build_fast_indices()` expects keys such as `prot_idx`, `prot_target`, `rna_idx`, and `pho_idx`, while `prepare_fast_loss_data()` returns `p_prot`, `obs_prot`, etc. If Optuna is used, it likely needs careful revalidation even before multimodal support.

#### Legacy/local model

1. `protwise.paramest.normest.parameter_estimation()` creates `target = np.concatenate([r_data.flatten(), pr_data.flatten(), p_data.flatten()])`.
2. If regularization is enabled, it appends zeros to form `target_fit`.
3. `model_func()` calls `solve_ode()` and returns the same fixed flattened output, optionally appended with regularization terms.
4. `curve_fit()` is called with that fixed target length. Missing layers would currently produce shape mismatches or empty segments unless explicitly handled.
5. Scoring uses `score_fit()` over the same fixed full target vector.

### 4.6 Result extraction, plotting, saving, reports, logging

#### Global network model

- After optimization, `runner.main()` selects a solution by weighted Fréchet score across protein, RNA, and phospho observations. It currently initializes all three Fréchet components and detailed-score dictionaries and loops over each layer if predictions and observed data are nonempty.
- It saves raw picked predictions to `pred_prot_picked.csv`, `pred_rna_picked.csv`, and `pred_phospho_picked.csv` whenever `simulate_and_measure()` returns non-`None` data frames. Because `simulate_and_measure()` always constructs protein and RNA prediction rows for all modeled proteins, these files are currently written even if no corresponding observed layer was fitted.
- It writes `picked_objectives.json` with `prot_mse`, `rna_mse`, `phospho_mse`, and `scalar_score` for all three layers.
- It calls `plot_goodness_of_fit(df_prot, dfp, df_rna, dfr, df_pho, dfph, ...)`. This function merges all three observed/predicted pairs and then concatenates them. If a layer is empty, empty merges are fine, but if all are empty or a prediction data frame is unexpectedly empty, plotting can fail when calculating min/max/regression.
- It generates per-gene time-series plots through `save_gene_timeseries_plots()` for every `idx.proteins`, with three panels always created for protein, mRNA, and phosphorylation.
- It calls `export_residuals()`, `export_parameter_distributions()`, `export_S_rates()`, `plot_s_rates_report()`, `export_results()`, Pareto plots, convergence video, prior regularization scan, and dashboard bundle save. Several of these assume three objective columns or all layers.
- It logs many details but does not currently log a detected data mode, skipped optional layers, or layer-specific data shapes after filtering.

#### Legacy/local model

- `common.utils.display.save_result()` writes per-gene Excel sheets including parameters, errors, fitted state solution, site estimates, site observations, PCA, t-SNE, and potentially sensitivity/knockout data.
- `common.utils.display.merge_obs_est()` currently merges only `*_site_observed` and `*_site_estimates`, so local goodness-of-fit reporting is phosphosite-centric even though fitting includes mRNA and protein.
- `protwise.plotting.Plotter.plot_model_fit()` creates fixed mRNA/protein/phosphosite plots and hard-codes data slicing for 9 RNA points and 14 protein/phospho points.
- The legacy runner logs common genes, missing required columns, metric definitions, and output report path, but not a mode.

## 5. Current Assumptions About Input Data

### Global network model assumptions

- `args.kinase_net` is a CSV with columns discoverable as gene/protein, psite/site, and kinase/k.
- `args.tf_net` is a CSV with TF/source and target columns discoverable by candidate names.
- `args.ms` is required and contains both protein abundance rows and phosphosite rows in one table, distinguishable by whether `Psite` is blank.
- `args.phospho` exists in config/CLI but is not effectively used by `networkmodel.io.load_data()` today; `phospho` in config defaults to `ms`.
- Protein/phospho mass-spec wide columns are expected as `x1..xN` mapped to `TIME_POINTS_PROTEIN`; alternatively a pre-melted format with `time` and `mean`/`fc` is accepted in part of `load_data()`.
- RNA is required from `args.rna`, with a gene column discoverable as `geneid`, `mrna`, or `gene`, plus `x1..xN` columns mapped to `TIME_POINTS_RNA` by `process_and_scale_raw_data()`.
- Protein identifiers are uppercased in mass-spec processing. RNA identifiers are renamed to `protein` by `process_and_scale_raw_data()`, which uppercases ID columns at the end. Network identifiers are uppercased during kinase-network expansion for `protein` and `kinase`.
- Phosphosite names are stripped but not systematically uppercased; site matching is exact after stripping.
- Missing observation values are coerced to numeric and dropped in mass-spec preprocessing. RNA preprocessing also coerces values during scaling/melting; exact drop behavior is in `process_and_scale_raw_data()`.
- `df_pho` is filtered to `(protein, psite)` pairs present in the kinase network, so phospho data outside the mechanistic network is silently removed except for a count log.
- `df_prot`, `df_rna`, and `df_pho` are filtered to proteins in `idx.proteins` after model-index construction.
- Time values in observations must exactly map to `solver_times`; `prepare_fast_loss_data()` raises if a time is not present in the union grid.
- The state vector always includes an mRNA state and protein/phospho states for each modeled protein. The model therefore already has the structural basis for all data modes.
- The optimizer currently assumes three objectives and names them `prot_mse`, `rna_mse`, and `phospho_mse` regardless of data availability.
- Layer weights are supported by existing lambda values (`lambda_protein`, `lambda_rna`, `lambda_phospho`) and time weighting methods, but no explicit mode-aware disabling exists.
- `TIME_POINTS_PHOSPHO` is available in the config, but current `load_data()` maps both protein and phospho rows from `args.ms` using `TIME_POINTS_PROTEIN`; the future implementation should resolve this ambiguity.

### Legacy/local workflow assumptions

- Protein input is a CSV. Phosphosite and mRNA inputs are Excel files with `Estimated` sheets.
- Protein-only rows are identified by `Psite.isna()` and `GeneID == gene`.
- Phosphosite rows are identified by `Gene == gene`, with one row per psite and fixed `x1..x14` columns.
- mRNA rows are identified by `mRNA == gene`, with fixed `x1..x9` columns.
- Only genes present in both phosphosite and mRNA data are processed. Protein-only availability does not contribute to gene selection.
- The target vector order is fixed: mRNA first, protein second, phosphosite rows third.
- The model-output vector length is fixed to match the full three-layer target. Missing layers are not supported.
- `Plotter.plot_model_fit()` assumes exactly 9 RNA points, 14 protein points, and 14 phospho points per site.
- `save_result()` and `merge_obs_est()` are primarily phosphosite-sheet oriented for observed/estimated output merging.

## 6. Desired Future Data Modes

For all modes, the core model should remain structurally consistent. The internal state can continue to include mRNA, protein, and phosphosite states; mode affects only which observations are validated, fitted, plotted, saved as fitted predictions, and reported as objective terms.

### 6.1 mRNA only

- Required input data: nonempty RNA table after loading, identifier normalization, time mapping, and model-index filtering.
- Unavailable data: no protein observations and no phospho observations.
- Model outputs compared: predicted RNA/mRNA fold-change only (`R(t)`/state offset `start`).
- Loss terms: include RNA loss only; protein and phospho losses must be skipped, not replaced with fake zeros that affect objectives.
- Plotting: mRNA observed vs predicted goodness-of-fit and mRNA time-series plots only. No protein/phospho panels except optional clearly labeled model-simulation diagnostics not counted as fit outputs.
- Tables/files: save `pred_rna_picked.csv`, picked RNA objective, mode metadata, model parameters, and any generic network matrices. Do not create fitted protein/phospho prediction files unless explicitly categorized as unfitted simulation diagnostics.
- Logs: `Detected data mode: mrna_only`; `Fitting layers: mRNA`; `Skipped layers: protein, phospho`; shape of `df_rna` before/after filtering.
- Edge cases: RNA identifiers may not appear in kinase-network-derived `idx.proteins` unless TF edges or topology include them; mode detection should distinguish raw RNA present from effective RNA after model filtering. If all RNA rows are filtered out, fail with an explicit error.

### 6.2 protein only

- Required input data: nonempty protein abundance rows after loading and filtering.
- Unavailable data: no RNA observations and no phospho observations.
- Model outputs compared: predicted total protein fold-change only (`P_unphos + sum(P_phos)` or combinatorial state sum).
- Loss terms: include protein loss only.
- Plotting: protein observed vs predicted goodness-of-fit and protein time-series plots only.
- Tables/files: save `pred_prot_picked.csv`, picked protein objective, mode metadata, and generic model/network outputs.
- Logs: `Detected data mode: protein_only`; `Fitting layers: protein`; skipped RNA/phospho layers.
- Edge cases: protein-only mode still needs a network/model universe. If a protein has no site/network representation, the current `Index` may not include it unless topology includes it; future validation should report unsupported observed proteins rather than silently dropping all rows.

### 6.3 phospho only

- Required input data: nonempty phosphosite observations after mechanistic `(protein, psite)` filtering and model-index filtering.
- Unavailable data: no mRNA or protein abundance observations.
- Model outputs compared: predicted phosphosite fold-change only (`P_site_j(t)` or combinatorial site aggregation).
- Loss terms: include phospho loss only.
- Plotting: phosphosite observed vs predicted plots; per-site time-series panels only.
- Tables/files: save `pred_phospho_picked.csv`, picked phospho objective, S-rate files/reports where meaningful, and mode metadata.
- Logs: `Detected data mode: phospho_only`; dropped phosphosite count from mechanistic filter; skipped RNA/protein layers.
- Edge cases: current `KinaseInput` uses `df_prot` and defaults to flat kinase inputs if protein data is absent. That should be logged because phospho-only fitting may rely on default kinase drivers.

### 6.4 mRNA + protein

- Required input data: nonempty RNA and protein observations after filtering.
- Unavailable data: no phosphosite observations.
- Model outputs compared: RNA and total protein fold-change.
- Loss terms: include RNA and protein losses; skip phospho loss.
- Plotting: separate RNA and protein plots, plus optional combined two-layer goodness-of-fit.
- Tables/files: save `pred_rna_picked.csv`, `pred_prot_picked.csv`, combined summary with RNA/protein objectives, no fitted phospho predictions unless diagnostic-only.
- Logs: detected mode, available layers, skipped phospho, per-layer shapes.
- Edge cases: a topology with no modeled sites still has mRNA/protein states; if `df_kin` is required to build `Index`, mRNA/protein-only use may need a clear rule for representing proteins that appear only in observations/TF targets.

### 6.5 mRNA + phospho

- Required input data: nonempty RNA and phosphosite observations after filtering.
- Unavailable data: no protein abundance observations.
- Model outputs compared: RNA and phosphosite fold-change.
- Loss terms: include RNA and phospho losses; skip protein loss.
- Plotting: RNA and phospho plots; optional combined two-layer diagnostics.
- Tables/files: save `pred_rna_picked.csv`, `pred_phospho_picked.csv`, S-rate reports, combined summary.
- Logs: detected mode, skipped protein, and warning that kinase input trajectories default to flat values for kinases without protein observations.
- Edge cases: data-derived initial conditions currently use protein totals to allocate phospho mass; if protein is unavailable and `use_initial_condition_from_data=true`, y0 construction must use safe defaults for total protein and log that choice.

### 6.6 protein + phospho

- Required input data: nonempty protein and phosphosite observations after filtering.
- Unavailable data: no RNA observations.
- Model outputs compared: total protein and phosphosite fold-change.
- Loss terms: include protein and phospho losses; skip RNA loss.
- Plotting: protein and phospho plots; optional combined two-layer diagnostics.
- Tables/files: save `pred_prot_picked.csv`, `pred_phospho_picked.csv`, S-rate reports, combined summary.
- Logs: detected mode and skipped RNA.
- Edge cases: transcription parameters `A_i`/`B_i` remain part of the structural model even if RNA is not fitted. Regularization/bounds should prevent unconstrained RNA dynamics from destabilizing protein/phospho states.

### 6.7 mRNA + protein + phospho

- Required input data: all three layers nonempty after filtering.
- Unavailable data: none.
- Model outputs compared: RNA, total protein, and phosphosite fold-change.
- Loss terms: current intended full behavior; include all three losses.
- Plotting: current combined goodness-of-fit plus per-layer plots and per-gene three-panel time-series plots.
- Tables/files: preserve current outputs and names where possible.
- Logs: detected mode `mrna_protein_phospho` or `all_layers`; no skipped layers.
- Edge cases: this mode is the backward-compatibility baseline and should be used as the first regression target.

## 7. Recommended Internal Mode Detection Design

### 7.1 Where detection should happen

Use a two-stage detection approach:

1. **Raw layer detection in `networkmodel.io.load_data()` after each layer is parsed.** This detects whether RNA, protein, and phospho files/rows are present before model-specific filtering. It should also capture source path, raw row count, tidy row count, identifiers, time columns/time values, and parsing warnings.
2. **Effective fit-mode finalization in `networkmodel.runner.main()` after mechanistic phospho filtering and filtering observations to `idx.proteins`.** This is the mode that should control loss construction and downstream outputs because it reflects what can actually be fitted.

For the legacy/local workflow, equivalent detection would happen after `protwise/runner/main.py` loads the three input tables and again after `protwise/paramest/core.py` extracts per-gene arrays. However, the safest first implementation target should be the global `networkmodel` pipeline because it already has separate tidy data frames and a separate loss array per layer.

### 7.2 Recommended metadata object

Create a small internal object; do not add config keys. A dataclass is a good fit:

```python
@dataclass(frozen=True)
class DataMode:
    available_layers: tuple[str, ...]          # e.g. ("rna", "protein", "phospho")
    fit_mrna: bool
    fit_protein: bool
    fit_phospho: bool
    mode_name: str                             # e.g. "mrna_protein"
    raw_counts: dict[str, int]
    filtered_counts: dict[str, int]
    identifiers: dict[str, set[str]]
    timepoints: dict[str, np.ndarray]
    skipped_layers: tuple[str, ...]
    warnings: tuple[str, ...]
```

Layer names should be standardized internally as `"mrna"`, `"protein"`, and `"phospho"`. If keeping current variable names, map them carefully:

- `df_rna` -> `mrna`
- `df_prot` -> `protein`
- `df_pho` -> `phospho`

### 7.3 Inference rules without config schema changes

Recommended rules:

1. Use existing configured paths only:
   - RNA source: `[networkmodel].rna` / `args.rna`
   - protein source: `[networkmodel].ms` / `args.ms`, blank `Psite` rows
   - phospho source: `[networkmodel].phospho` / `args.phospho` if it exists and is distinct; otherwise `[networkmodel].ms` / `args.ms` nonblank `Psite` rows
2. Preserve old behavior when `phospho` is absent or equal to `ms`: split the mass-spec table by blank/nonblank `Psite`.
3. If `phospho` is present and distinct from `ms`, read protein abundance from `ms` and phospho observations from `phospho` using the same accepted wide/pre-melted rules. This uses an existing config key that already exists; it is not a schema change.
4. Treat a layer as available only if, after parsing and cleaning, it has at least one finite `fc` value, at least one valid `time`, and at least one identifier.
5. Treat a layer as fit-enabled only if it remains nonempty after all model-index and mechanistic filtering.
6. If raw data are present but all rows are filtered out, log a warning and either fail if no other layer remains, or continue with the remaining layers while recording the layer as skipped due to incompatibility.
7. If no layer remains fit-enabled, fail before optimization with a clear error.

### 7.4 Downstream consumption

Every downstream function should consume mode through one of two patterns:

- Pass `data_mode` explicitly where needed (`prepare_fast_loss_data`, `GlobalODE_MOO`, export/plot functions), or
- Store it in a normalized data bundle object passed around the runner.

Avoid having downstream functions infer mode independently from data-frame emptiness in multiple places. That creates inconsistent behavior. The source of truth should be the finalized `DataMode` after filtering.

### 7.5 Safe failure behavior

- Missing optional layer file: skip if other layers are available and log it as unavailable, not as an error.
- Missing required columns in a file that is intended to supply a layer: error for that layer; continue only if the file is clearly optional and other valid layers exist.
- Ambiguous file content: log the rule used, e.g. `Psite blank rows treated as protein; nonblank Psite rows treated as phospho`.
- Data present but all filtered out: warn with before/after counts and fail if no fit layers remain.
- Duplicate layer sources: allow `ms == phospho` as current behavior; if separate paths contain overlapping phospho rows, deduplicate by `(protein, psite, time)` and log duplicates.

## 8. Recommended Model Output Interface

The model already structurally exposes the needed states:

- mRNA: first state in each protein block (`Y[:, offset_y[i]]`)
- protein: total protein computed from unphosphorylated + phosphosite states or from the sum of combinatorial protein states
- phospho: per-site phosphorylated states or bitwise aggregation for combinatorial states

The future implementation should formalize this in a consistent observable interface rather than scattering extraction logic across `lossfn.py`, `simulate.py`, and `optuna_solver.py`.

Recommended interface:

```python
@dataclass
class ModelObservables:
    rna: pd.DataFrame | None       # columns: protein, time, pred_fc
    protein: pd.DataFrame | None   # columns: protein, time, pred_fc
    phospho: pd.DataFrame | None   # columns: protein, psite, time, pred_fc
```

or, for performance-sensitive code:

```python
extract_observables(Y, idx, time_grid, layers, baseline_times) -> dict[str, np.ndarray | pd.DataFrame]
```

Functions needing adjustment:

- `networkmodel.simulate.simulate_and_measure()` should accept a `layers`/`data_mode` argument and return only requested layers for fitted-output contexts. It may still support all-layer extraction for diagnostics if explicitly requested.
- `networkmodel.lossfn.loss_function_noncomb()` and `loss_function_comb()` can continue to compute separate raw losses from arrays, but the caller should decide which losses become objectives.
- `networkmodel.optuna_solver._augment_loss_data()` should be replaced or aligned with `prepare_fast_loss_data()` and the same observable definitions. Its current direct state-index approach does not match fold-change observable extraction for total protein/combinatorial phospho.
- `protwise/models/*.py::solve_ode()` currently returns a fixed flattened vector. If the legacy workflow is made mode-aware later, it should return structured observables or accept a mask specifying which segments to concatenate.
- `protwise/plotting/plotting.py::plot_model_fit()` should not parse fixed slices from a flattened vector; it should consume a structured observable result.

## 9. Recommended Loss Construction

### 9.1 Global network model

Keep the low-level loss kernels capable of returning three raw loss sums, but make objective assembly mode-aware.

Recommended steps:

1. `prepare_fast_loss_data()` should return true counts, not `max(1, len(obs_*))`, plus an explicit `available_layers` or `fit_layers` list.
2. For each available layer:
   - Validate observation arrays, time indices, weights, and predicted observable shape.
   - Compute residual using the existing layer-specific logic.
   - Apply the existing time weights and layer lambda (`lambda_rna`, `lambda_protein`, `lambda_phospho`).
   - Normalize by the sum of weights for that available layer.
3. For unavailable layers:
   - Pass empty arrays to low-level kernels if convenient, but do not create an optimization objective for that layer unless preserving a fixed 3-objective external shape is explicitly chosen for compatibility.
   - Do not fill targets with zeros.
   - Do not use fake counts.
   - Do not include that layer in scalar score, Fréchet selection, plots, or fitted-output files.
4. Prior regularization should remain available. If objectives are dynamic, either add the same prior penalty to each active objective or include it only in scalar ranking. Preserve current behavior for all-three mode by adding it to all three active objectives.

### 9.2 Objective shape choice

There are two viable designs:

#### Option A: Dynamic objective count

- `GlobalODE_MOO.n_obj = len(data_mode.available_layers)`.
- `F` columns correspond only to active layers.
- Pros: no fake objectives; optimizer focuses only on active data.
- Cons: more output code changes; Pymoo reference directions, Pareto CSV columns, Pareto plots, lambda scans, and dashboards must become dynamic.

#### Option B: Fixed 3 objective columns with inactive objectives masked

- Keep `n_obj=3`, but inactive objectives are excluded from selection/scoring and marked `NaN` or a sentinel in output.
- Pros: less optimizer/export disruption.
- Cons: Pymoo still optimizes inactive dimensions if they are finite constants; `NaN` may break algorithms; constants can distort Pareto ranking/reference directions.

Recommendation: implement dynamic active objectives internally, and write compatibility outputs that include all three named columns with `NaN` for unavailable layers only after optimization. This avoids optimizing fake objectives while keeping old output names where feasible.

### 9.3 Layer-specific weights

Current global code supports:

- `lambda_protein`, `lambda_rna`, `lambda_phospho`
- `weighting_method_protein`, `weighting_method_rna`, and a config value `weighting_method_phospho`

But `runner.main()` currently applies the protein weighting method to both protein and phospho. Future code should use `WEIGHTING_METHOD_PHOSPHO` for `df_pho` if the existing config loader exposes it. This uses an existing schema field, not a new one.

## 10. Recommended Optimization Impact

### Current optimizer expectations

- `GlobalODE_MOO` expects one parameter vector and returns exactly three objectives.
- `UNSGA3` reference directions are built with `problem.n_obj`.
- `pareto_F.csv`, `picked_objectives.json`, `export_pareto_front_to_excel()`, `save_pareto_3d()`, `save_parallel_coordinates()`, `scan_prior_reg()`, and dashboard bundle code assume `F` has columns `[prot_mse, rna_mse, phospho_mse]`.
- Solution selection uses weighted Fréchet distance across all available observed data frames but initializes all three components.
- The parameter vector includes `c_k`, `A_i`, `B_i`, `C_i`, `D_i`, `Dp_i`, `E_i`, and `tf_scale` regardless of layer availability.

### Required future changes

- Add `data_mode` or `active_layers` to `GlobalODE_MOO` and any Optuna objective.
- Change objective assembly to return only active layer objectives.
- Maintain a mapping such as `objective_names = ["protein_mse", "rna_mse"]` for active modes.
- Use dynamic objective names when writing Pareto CSV/Excel and picked objectives.
- For backward compatibility, optionally include a separate `picked_objectives_all_layers.json` with inactive layers as `null`/`NaN`, while keeping existing `picked_objectives.json` schema in all-three mode.
- Ensure Pymoo reference directions use the active objective count.
- Make `save_pareto_3d()` conditional: only call it when exactly three active objectives exist; otherwise call a 2D or 1D plotting function or skip with a log message.
- Make `scan_prior_reg()` dynamic or skip it when not all three objectives exist.
- Optuna solver should be audited/fixed separately because its current indexing appears inconsistent with the global loss-data structure.

### Parameter vector, bounds, constraints, initial guesses

- The core parameter vector should remain structurally unchanged to preserve model consistency.
- Bounds from `calculate_bio_bounds(idx, df_prot, df_rna, tf_mat, kin_in)` may need mode-aware handling if it currently relies on a missing `df_prot` or `df_rna` to estimate ranges. Missing layers should fall back to defaults, not fail silently.
- `KinaseInput` already falls back to flat ones if protein data is empty; this should be logged as a mode-specific assumption.
- Data-derived initial conditions in `build_y0_from_data()` already defaults missing RNA/protein/phospho entries to `1.0`/`0.0`; this should become explicit validation/logging when layers are absent.

## 11. Recommended Plotting Changes

### Global plotting functions needing changes

- `networkmodel.export.plot_goodness_of_fit()` should accept `data_mode` and skip unavailable layers before merging. It should fail only if no active layer has matched observed/predicted rows.
- `networkmodel.export.save_gene_timeseries_plots()` currently creates a fixed three-panel plot. It should create one panel per active fitted layer or clearly gray/label skipped layers if compatibility is desired.
- `networkmodel.export.save_pareto_3d()` should run only for three active objectives.
- `networkmodel.export.save_parallel_coordinates()` should use dynamic objective names.
- `networkmodel.export.plot_gof_from_pareto_excel()` should not require sheets for unavailable trajectory layers.
- Dashboard plotting in `networkmodel/dashboard_app.py` should read mode metadata and hide unavailable layers.
- Legacy `protwise.plotting.Plotter.plot_model_fit()` should be made mode-aware only if the local workflow is targeted later.

### Mode-specific plotting behavior

| Mode | Plotting outputs |
|---|---|
| mRNA only | mRNA observed-vs-predicted scatter and mRNA time-series plots. No fitted protein/phospho panels. |
| protein only | Protein observed-vs-predicted scatter and protein time-series plots. |
| phospho only | Phosphosite observed-vs-predicted scatter and per-site time-series plots. |
| mRNA + protein | Separate mRNA/protein plots plus combined two-layer diagnostic. |
| mRNA + phospho | Separate mRNA/phospho plots plus combined two-layer diagnostic. |
| protein + phospho | Separate protein/phospho plots plus combined two-layer diagnostic. |
| all three | Preserve current combined and per-modality plots. |

## 12. Recommended Saving and Output Changes

### Global saving functions needing changes

- `networkmodel.runner.main()` should write picked prediction files only for fitted layers unless a file is explicitly named as an unfitted simulation diagnostic.
- `picked_objectives.json` should be dynamic or include inactive layers as `null`. In all-three mode, preserve current keys and semantics.
- `pareto_F.csv` should use active objective columns. For compatibility, optionally write `pareto_F_all_layers.csv` with inactive layer columns as `NaN`.
- `networkmodel.export.export_pareto_front_to_excel()` should create trajectory sheets only for active fitted layers. If preserving sheet names, empty sheets should include a note or be omitted with metadata.
- `networkmodel.export.export_results()` currently saves model parameters and layer outputs; it should not require all three predicted/observed tables to be nonempty before export.
- `networkmodel.export.export_residuals()` should export residuals only for active layers.
- `networkmodel.dashboard_bundle.save_dashboard_bundle()` should include `data_mode` metadata, active objective names, and per-layer row counts.
- `common.utils.display.save_result()` and `merge_obs_est()` require analogous changes if local mode-aware support is implemented later.

### Recommended files/metadata

- Save a small `data_mode.json` in the output directory:

```json
{
  "mode_name": "mrna_protein_phospho",
  "available_layers": ["mrna", "protein", "phospho"],
  "fit_layers": ["mrna", "protein", "phospho"],
  "raw_counts": {"mrna": 100, "protein": 100, "phospho": 250},
  "filtered_counts": {"mrna": 90, "protein": 95, "phospho": 220},
  "skipped_layers": [],
  "warnings": []
}
```

- Keep existing output paths unchanged for the current all-three mode.
- Avoid writing misleading empty prediction files for skipped layers.
- If diagnostic simulations for skipped layers are useful, place them under a distinct name such as `diagnostic_pred_protein_unfitted.csv` and log that they were not fitted.

## 13. Recommended Logging and Console Messages

### Current logging hooks

- `config.logconf.setup_logger()` provides the logging infrastructure.
- `networkmodel.runner.main()` logs application metadata, solver/model choice, arguments, network/filter counts, optimization settings, outputs, and final parameters.
- `networkmodel.io.load_data()` logs file loads and prior-load warnings.
- `networkmodel.network.Index` logs model dimensions and proxy rewiring.
- `protwise/runner/main.py` logs local configuration, common genes, per-gene processing, plotting, LaTeX/report generation, and report path.

### Future required messages

At minimum, global mode-aware implementation should log:

- Raw detected layer availability and source paths.
- Effective fit mode after all filtering.
- Available/fitted layers and skipped layers.
- Raw and filtered row counts per layer.
- Unique identifier counts per layer.
- Timepoints detected per layer and configured time grids used.
- Loss terms included and excluded.
- Objective names and optimizer objective count.
- Whether kinase inputs are data-driven or default flat because protein data is absent.
- Whether data-derived initial conditions used defaults for unavailable layers.
- Which output files were generated per layer.
- Warnings for optional missing layers.
- Errors for malformed files, missing required columns in present files, invalid times, and no active fit layers.

Example global log block:

```text
[Mode] Raw layers detected: mRNA=yes (900 rows), protein=no (0 rows), phospho=yes (4200 rows)
[Mode] Effective fit mode after filtering: mrna_phospho
[Mode] Fitting layers: mRNA, phospho
[Mode] Skipped layers: protein (no blank-Psite protein rows in ms input)
[Loss] Active objectives: rna_mse, phospho_mse
[Output] Writing fitted predictions for: rna, phospho
```

## 14. Validation Rules Needed

### File-level validation

- Existing configured path points to a readable file if the layer is expected from that path.
- Empty files should be reported as empty layer sources, not passed downstream to pandas/optimizer without context.
- If `ms` and `phospho` are the same path, split by `Psite` as current behavior.
- If `phospho` is distinct, parse it independently and apply the phospho time grid or documented inference rule.

### Column validation

- Protein/mass-spec input: require a gene/protein identifier column and either wide `x1..xN` columns or pre-melted `time` plus `fc`/`mean`.
- Mixed protein/phospho input: `Psite`/`site` column should be present to split layers. If absent, classify as protein-only unless the file is explicitly the `phospho` source; if explicitly phospho, error.
- RNA input: require gene/mRNA identifier and time/value columns.
- Network input: require protein/gene, psite/site, and kinase columns.
- TF input: require TF/source and target columns if nonempty.

### Time validation

- Observed times must map to configured layer time grids or to the solver union grid after rounding/normalization.
- If wide column count exceeds configured time points, current code truncates columns; future code should log this explicitly.
- If wide column count is less than configured time points, log the mapping used and avoid assuming missing timepoints are zero.
- RNA and phospho baseline times (`4.0` for RNA, `0.0` for protein/phospho currently) should be present or selected by a documented nearest-time rule.

### Identifier validation

- Normalize protein/gene identifiers consistently across RNA, protein, phospho, kinase network, and TF network.
- Report identifiers present in observations but absent from `idx.proteins` before filtering them out.
- For combined modes, do not require all layers to share identical identifier sets; fit each layer for its available identifiers.
- For phosphosite data, validate `(protein, psite)` pairs against the model site map and report dropped sites.
- Duplicated `(layer, protein, time)` or `(phospho, protein, psite, time)` rows should be resolved by a clear rule, preferably average or last with a warning.

### Numeric validation

- Coerce `fc` and `time` columns to numeric and drop or error on missing values according to layer policy.
- Reject nonfinite observations after preprocessing.
- Prevent zero or negative baselines from creating NaNs/infs in fold-change calculations.
- Validate weights are finite, nonnegative, and nonempty for active layers.

### Mode validation

- Valid effective modes are exactly the seven nonempty combinations.
- If zero effective layers remain, fail before model construction or before optimization with a clear error.
- If raw layer is present but effective layer is absent, log the reason and do not include it in the loss.
- If a layer is unavailable, it must not create NaN, wrong array shape, fake zero targets, or misleading fitted output files.

## 15. Backward Compatibility Plan

- The current all-three global workflow should continue to work unchanged when `ms` contains blank-Psite protein rows and nonblank-Psite phospho rows and `rna` contains RNA rows.
- Current config files should still run. No new mandatory config keys should be introduced.
- The existing `[networkmodel].phospho` key should be used if future separate phospho input support is needed; if absent, default to `ms` as `config_loader.load_config_toml()` already does.
- Existing all-three output file names should be preserved in all-three mode:
  - `pred_prot_picked.csv`
  - `pred_rna_picked.csv`
  - `pred_phospho_picked.csv`
  - `picked_objectives.json`
  - `pareto_F.csv`
  - `pareto_front.xlsx`
  - existing plots and dashboard bundle.
- Function signatures should be preserved where possible by adding optional `data_mode=None`/`active_layers=None` parameters with default behavior equivalent to all-three mode.
- Internal changes may introduce data containers and mode metadata, but users should still configure `ms`, `rna`, `phospho`, timepoints, lambdas, scaling, and weighting the same way.
- Inactive layers should not affect optimization or scalar solution selection.
- Legacy/local workflow should not be changed in the first global implementation unless explicitly requested; if changed later, preserve old all-three behavior and fixed output sheets for all-three mode.

## 16. Files That Will Likely Need Future Modification

| File | Current role | Why it needs modification | Type of future change | Risk level | Notes |
|---|---|---|---|---|---|
| `networkmodel/io.py` | Loads networks and RNA/MS observations; splits MS into protein/phospho. | Needs raw layer detection, optional separate phospho file use, clearer validation, row-count metadata. | Add internal data-layer parsing/detection helpers; preserve return compatibility or return a data bundle. | High | Current `args.phospho` is not used for data loading. |
| `networkmodel/runner.py` | Orchestrates global pipeline. | Needs effective mode finalization after filtering, active objective names, mode-aware optimizer calls, selection, plotting, saving, logging. | Add `DataMode` construction/use; condition downstream calls. | High | Most assumptions converge here. |
| `networkmodel/cache.py` | Pre-indexes observations for loss. | Needs true zero counts, layer masks, safe empty arrays, validation. | Add `fit_layers`/`data_mode`, remove fake `max(1, len(...))` counts. | High | Critical for avoiding fake objectives. |
| `networkmodel/optproblem.py` | Pymoo objective with fixed 3 objectives. | Needs dynamic objective count and objective-name mapping. | Add active-layer objective assembly. | High | Affects optimizer behavior and Pareto output shapes. |
| `networkmodel/lossfn.py` | Low-level loss kernels. | May need masks or wrappers to skip unavailable layers cleanly. | Prefer keep kernels and change caller; add tests for empty arrays. | Medium | Numba changes are riskier; minimize edits. |
| `networkmodel/simulate.py` | Extracts predicted RNA/protein/phospho tables. | Needs optional active-layer extraction and shared observable definitions. | Add `layers` parameter or structured return. | Medium | Preserve all-layer default for compatibility. |
| `networkmodel/optuna_solver.py` | Alternative optimizer. | Needs alignment with mode-aware loss data and fold-change observables. | Refactor objective to use same loss arrays/objective assembly as Pymoo. | High | Current keys appear inconsistent with `prepare_fast_loss_data()`. |
| `networkmodel/export.py` | Saves Pareto data, results, plots, residuals, S-rates. | Many functions assume three objectives/layers. | Add `data_mode`/objective names; skip inactive layers; dynamic sheets/plots. | High | Keep all-three output compatibility. |
| `networkmodel/dashboard_bundle.py` | Saves dashboard pickle. | Should include mode metadata and active objective names. | Add metadata fields. | Medium | Backward compatible if new keys are optional. |
| `networkmodel/dashboard_app.py` | Displays saved outputs. | Should hide unavailable layers and handle dynamic objectives. | Conditional UI rendering. | Medium | Only if dashboard is part of release. |
| `networkmodel/scan.py` | Hyperparameter scan over lambdas/objectives. | Assumes protein/RNA/phospho lambdas and score components. | Dynamic active-layer lambdas and scoring. | Medium | Can initially disable scan for non-all modes with warning. |
| `networkmodel/refine.py` | Refines Pymoo results. | May assume scalarization over fixed F columns. | Ensure dynamic objective arrays work. | Medium | Most logic may work if `problem.n_obj` is dynamic. |
| `networkmodel/utils.py` | Column normalization, scaling, bounds. | Needs stronger validation helpers and possibly mode-aware bound fallback. | Add validation utilities; avoid changing scaling semantics. | Medium | Useful shared location for validators. |
| `networkmodel/steadystate.py` | Builds y0 from data. | Needs explicit behavior when protein/RNA/phospho layers are absent. | Add logging/default-source metadata; validate missing data. | Medium | Current defaults may be acceptable but should be transparent. |
| `config_loader.py` | Loads config into `PhosKinConfig`. | Existing schema can be preserved, but ambiguity around `phospho` default should be documented/possibly represented. | Maybe no code change; if changed, only internal default handling. | Low | Do not add mandatory config fields. |
| `networkmodel/config.py` | Exposes config constants. | May need to expose/use `WEIGHTING_METHOD_PHOSPHO` correctly. | Ensure existing value is imported and consumed. | Low | Schema already has the field. |
| `protwise/runner/main.py` | Legacy/local runner. | If local workflow is included later, it must stop requiring mRNA/phospho intersection for all modes. | Add local mode detection and per-gene layer availability. | High | Defer until global support is stable. |
| `protwise/paramest/core.py` | Extracts per-gene arrays and metrics. | Fixed all-layer extraction/metrics. | Structured layer data and mode-aware metrics. | High | Defer if scope is global model first. |
| `protwise/paramest/normest.py` | Local curve-fit target construction. | Fixed concatenated `[mRNA, protein, phospho]` target. | Dynamic target/model-output concatenation. | High | Shape bugs likely. |
| `protwise/models/*.py` | Local ODE solvers and flattening. | Fixed output vector order/length. | Add structured observables or active-layer flattening. | Medium | Core ODE can stay unchanged. |
| `protwise/plotting/plotting.py` | Local plots. | Fixed mRNA/protein/phospho panels and slice lengths. | Mode-aware plotting. | Medium | Defer if local not targeted. |
| `common/utils/display.py` | Local saving/report merging. | Phosphosite-centric observed/estimated merging; fixed sheets. | Mode-aware sheets and report merge. | Medium | Defer if local not targeted. |
| `docs/Documentation/configuration.md`, `docs/Documentation/architecture.md`, `README.md` | Documentation. | Must explain supported data modes after implementation. | Documentation update only after code changes. | Low | Do not alter schema docs prematurely. |
| `tests/` | Existing tests. | Need coverage for seven modes. | Add fixtures/unit/integration tests. | Medium | Do not modify in this audit. |

## 17. Files That Should Probably Not Be Modified

| File | Reason to avoid modifying | Risk if changed |
|---|---|---|
| `networkmodel/models.py` | Core RHS kernels already support the structural states needed for all modes. Mode-specific logic should live in observables/loss, not RHS. | High risk of changing model behavior or numerical stability. |
| `networkmodel/solvers.py`, `networkmodel/model_ivp.py`, `networkmodel/jacspeedup.py` | Solver mechanics should not need data-mode awareness. | High risk of numerical regressions. |
| `networkmodel/buildmat.py` | Matrix construction is topology-driven, not observation-mode-driven. | Medium/high risk of changing model topology. |
| `networkmodel/params.py` | Parameter vector should remain structurally consistent across modes. | High risk of breaking optimizer/export compatibility. |
| `config.toml` | User explicitly requires no schema/config changes. | High risk of breaking existing users and violating requirement. |
| `config/cli.py` | Existing CLI/config interface should remain stable. | Medium risk of user-facing API drift. |
| `config/logconf.py` | Logging infrastructure is adequate; only messages need updates. | Low/medium risk of test breakage or duplicate handlers. |
| `tests/test_config.py` | Existing logging tests are unrelated. | Low risk, but do not touch during feature implementation unless logging setup changes. |
| `processing/cleanup.py`, `processing/map.py` | Preprocessing can remain an upstream source of current files. | Medium risk of altering data-generation behavior. |
| `scripts/*` | Standalone post-processing should not block core support. | Medium risk of broad unrelated changes. |
| `docs/assets/*` | Static assets unrelated to multimodal input support. | Low value, unnecessary churn. |
| `common/frechet/distance.py` | Distance metric can be reused; mode filtering should happen before calls. | Low/medium risk of metric regression. |

## 18. Proposed Implementation Phases for a Later Agent

### Phase 1: Add internal data-layer detection

Expected files:

- `networkmodel/io.py`
- `networkmodel/runner.py`
- possibly a new internal module such as `networkmodel/datamode.py`

Checks:

- Unit-test raw detection for mixed `ms`, separate `phospho`, RNA-only, protein-only, phospho-only, and empty files.
- Verify no config schema change.
- Verify all-three current config detects all layers.

### Phase 2: Normalize data containers

Expected files:

- `networkmodel/io.py`
- `networkmodel/utils.py`
- `networkmodel/runner.py`

Checks:

- Each layer returns a data frame with canonical columns.
- Identifier normalization is consistent.
- Raw and filtered counts are preserved in metadata.
- Separate phospho file uses appropriate time grid or documented fallback.

### Phase 3: Refactor model output interface

Expected files:

- `networkmodel/simulate.py`
- possibly `networkmodel/export.py`

Checks:

- `simulate_and_measure()` all-layer default matches current output in all-three mode.
- Active-layer extraction returns only requested layer data frames.
- Combinatorial and non-combinatorial observable extraction both pass shape checks.

### Phase 4: Dynamic loss construction

Expected files:

- `networkmodel/cache.py`
- `networkmodel/optproblem.py`
- `networkmodel/lossfn.py` only if necessary

Checks:

- Empty unavailable layers produce no objective contribution.
- No fake zeros, NaNs, or fake counts.
- Active-layer objective names and counts are correct for all seven modes.
- All-three mode matches current loss values within tolerance.

### Phase 5: Mode-aware optimization

Expected files:

- `networkmodel/runner.py`
- `networkmodel/optproblem.py`
- `networkmodel/optuna_solver.py`
- `networkmodel/refine.py`
- `networkmodel/scan.py`

Checks:

- Pymoo runs with one, two, and three objectives.
- Reference directions are valid for active objective count.
- Solution selection uses only active layers.
- Optuna is either fixed for mode-aware support or clearly disabled with a warning for unsupported modes until fixed.

### Phase 6: Mode-aware plotting

Expected files:

- `networkmodel/export.py`
- `networkmodel/dashboard_app.py` if dashboard support is included

Checks:

- Goodness-of-fit plots generate for each of seven modes.
- Time-series plots create only active fitted panels.
- 3D Pareto plot is skipped or replaced for one/two objective modes.
- All-three plots remain compatible.

### Phase 7: Mode-aware saving/reporting

Expected files:

- `networkmodel/runner.py`
- `networkmodel/export.py`
- `networkmodel/dashboard_bundle.py`

Checks:

- Prediction files are written only for fitted layers.
- `data_mode.json` is saved.
- Pareto CSV/Excel have correct dynamic columns and compatibility metadata.
- No output path breaks in all-three mode.

### Phase 8: Logging and validation

Expected files:

- `networkmodel/io.py`
- `networkmodel/runner.py`
- `networkmodel/utils.py`
- `networkmodel/steadystate.py`

Checks:

- Logs include detected mode, active/skipped layers, shapes, losses, and outputs.
- Invalid/malformed inputs fail with explicit messages.
- Missing optional layers are warnings, not crashes, when at least one valid layer remains.

### Phase 9: Tests

Expected files:

- New tests under `tests/`, likely `tests/test_networkmodel_datamode.py`, `tests/test_networkmodel_loss_modes.py`, and `tests/test_networkmodel_outputs_modes.py`.

Checks:

- Unit tests for all seven data modes.
- Regression test for all-three current behavior.
- Invalid input tests.
- Plot/output smoke tests with small fixtures.

### Phase 10: Documentation update

Expected files:

- `README.md`
- `docs/Documentation/configuration.md`
- `docs/Documentation/architecture.md`
- `networkmodel/README.md`

Checks:

- Docs state that config schema is unchanged.
- Docs explain layer inference rules and output differences by mode.
- Docs clearly distinguish fitted outputs from optional diagnostic simulations.

## 19. Test Plan for Later Implementation

### Unit tests

- `detect_data_mode()` returns the correct mode for each of the seven combinations.
- Detection distinguishes raw availability from effective availability after filtering.
- Mixed `ms` file splits blank `Psite` rows as protein and nonblank `Psite` rows as phospho.
- Separate `phospho` path is used when provided and distinct from `ms`.
- Missing layer files are handled as optional when other layers exist.
- Malformed present files produce clear layer-specific errors.
- `prepare_fast_loss_data()` returns true zero counts for absent layers and valid arrays for present layers.
- Objective-name mapping is correct for all modes.

### Loss tests

For a tiny synthetic `Index`/`System` and deterministic `Y`:

- mRNA-only objective includes only RNA residuals.
- protein-only objective includes only protein residuals.
- phospho-only objective includes only phospho residuals.
- Combined modes include exactly the expected residual sums.
- Empty inactive arrays do not affect objective values.
- No loss path returns NaN/inf for missing layers.
- All-three mode reproduces previous three-loss behavior.

### Optimization smoke tests

- Pymoo problem can be constructed for one, two, and three objectives.
- Reference directions/algorithm setup works for active objective counts or uses appropriate single-objective/two-objective alternatives.
- A very small optimization run completes for each mode with tiny fixtures.
- Solution selection computes scalar/Fréchet scores using only active layers.

### Output tests

For each mode:

- Correct picked prediction files exist for fitted layers.
- Prediction files for skipped layers do not exist or are clearly diagnostic-only.
- `data_mode.json` exists and matches mode.
- `picked_objectives.json` has active objective values and no misleading inactive values.
- Pareto CSV/Excel objective columns match active layers.
- Goodness-of-fit plots are generated for active layers only.
- Dashboard bundle contains mode metadata.

### Backward compatibility tests

- Existing all-three fixture/config produces the same output file names as before.
- Old config with `phospho` missing or equal to `ms` still works.
- Existing lambdas and weighting settings remain accepted.
- Existing logging tests still pass.

### Invalid/malformed input tests

- No active layers after filtering -> explicit error before optimization.
- Missing required identifier column in a present RNA file -> explicit RNA parse error.
- Missing `Psite` column in an explicit phospho file -> explicit phospho parse error.
- Inconsistent time columns not in configured grid -> explicit time-grid error.
- Duplicate observations -> deterministic deduplication warning or error.
- Phosphosite rows absent from kinase network -> logged filtered count and skipped layer if none remain.
- Protein/RNA identifiers incompatible with model index -> warning and failure if no active rows remain.

## 20. Risks and Failure Modes

- Silent layer misclassification, especially when `Psite` is missing, blank, string `nan`, or present in separate phospho files.
- Existing `[networkmodel].phospho` key is currently ambiguous because it defaults to `ms` and is not used by `load_data()`.
- Wrong observable matched to the wrong data layer, especially protein total vs unphosphorylated protein or phosphosite state vs total phosphoprotein signal.
- Shape mismatch between flattened targets and predictions in the legacy/local workflow.
- NaNs or fake zeros introduced for missing layers.
- Dynamic objective count breaking Pymoo reference directions, Pareto plotting, Excel export, dashboard display, or hyperparameter scan.
- Inactive objective constants distorting Pareto selection if a fixed three-objective design is used incorrectly.
- Overfitting or unstable unconstrained states when fitting sparse layers, e.g. phospho-only without protein/RNA constraints.
- Broken existing all-three/phospho-heavy workflow due to changed file splitting or filtering.
- Data-derived initial conditions using misleading defaults without logging when layers are absent.
- Broken plotting assumptions in fixed three-panel plots and fixed 3D Pareto plots.
- Broken saving assumptions in `picked_objectives.json`, `pareto_F.csv`, `pareto_front.xlsx`, and dashboard bundles.
- Logger not reporting skipped layers, causing users to believe unavailable layers were fitted.
- Downstream scripts expecting old output files for every layer.
- Separate global and legacy/local workflows diverging in mode semantics if both are changed independently.
- Optuna path using different loss-index mechanics than Pymoo path and becoming inconsistent.

## 21. Final Recommendation

Implement multimodal input support first in the global `networkmodel` workflow, not the legacy/local `protwise` workflow. The global workflow already separates observations into `df_rna`, `df_prot`, and `df_pho`, has model states for all three layers, and has separate low-level loss sums. The safest strategy is:

1. Add an internal `DataMode`/`available_layers` object with no config schema changes.
2. Detect raw layer availability in `networkmodel.io.load_data()` and finalize effective fit mode in `networkmodel.runner.main()` after filtering.
3. Preserve the same model structure and parameter vector.
4. Make loss/objective assembly dynamic over active layers.
5. Make plotting, saving, solution selection, and dashboard metadata consume the same `DataMode` object.
6. Preserve all existing file names and semantics for the current all-three mode.
7. Add exhaustive tests for all seven modes before changing documentation.

Avoid modifying RHS kernels, solvers, user-facing config, or KinOpt/TFOpt. Treat missing layers as unavailable observations, not as zero-valued data. Every skipped layer should be visible in logs and output metadata.
