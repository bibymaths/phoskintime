# PhosKinTime JAXopt/Diffrax Migration Audit for networkmodel and protwise

## 1. Purpose

This is an audit-only planning document for a future migration of PhosKinTime's `networkmodel` and `protwise` workflows to a JAX/JAXopt/Diffrax-based optimization and ODE-solving stack.

The target future architecture is:

- JAX float64 enabled globally before numerical computation.
- Differentiable model, solver, observable extraction, and loss paths written with `jax.numpy`/JAX-compatible operations.
- JAXopt-based single-objective optimization.
- Diffrax-based ODE integration with stiff/semi-stiff solver candidates `diffrax.Kvaerno4` and `diffrax.Kvaerno5`.
- Dynamic mRNA/protein/phospho loss inclusion compatible with the previous multi-modal audit.
- Conversion to NumPy/Pandas only outside the differentiable computation path, specifically at data loading, plotting, exporting, dashboard, and reporting boundaries.

This document does not implement the migration. It identifies current SciPy/pymoo/Numba/Optuna/multi-objective/ODE dependencies and describes how later implementation should replace them while preserving the existing configuration interface and biological model structure.

## 2. Non-goals

- No code changes are made in this audit.
- No configuration schema changes are proposed as mandatory.
- No KinOpt or TFOpt work is included.
- No unrelated refactor is proposed as a prerequisite.
- No implementation is included yet.
- No change to the biological model structure is proposed.
- No Python files, tests, configuration files, notebooks, dashboards, or scripts are edited in this audit.
- No dashboard/script/notebook edits should happen until the JAXopt/Diffrax core migration is planned and tested.
- No separate data-type-specific models should be introduced; the existing state structure should remain the source of mRNA, protein, and phospho observables.

## 3. Relationship to Previous Multi-Modal Audit

This audit refines `docs/phoskintime_multimodal_input_audit.md`, which planned future support for all seven combinations of mRNA, protein, and phospho observations. That previous audit recommended an internal data-mode object and dynamic loss assembly. The JAXopt/Diffrax migration changes the optimization target: instead of producing dynamic multi-objective vectors or retaining the current three-objective Pareto front, the future objective must return one scalar JAX value.

The refined rule is:

```text
L_total =
    w_mrna    * L_mrna      if mRNA observations are active
  + w_protein * L_protein   if protein observations are active
  + w_phospho * L_phospho   if phospho observations are active
  + regularization terms
  + constraint penalties only when projection/reparameterization is unsuitable
```

Unavailable data layers must be skipped inside the scalar objective. They must not be represented as fake zeros, NaNs, empty Pandas frames inside JIT, or inactive pymoo objectives. The model remains structurally the same across all modes; only observable selection and scalar loss terms are mode-aware.

The previous audit's `DataMode`/`available_layers` concept should become a static, precomputed metadata object used to construct fixed-shape JAX arrays and layer masks before optimization. JAX functions should receive numeric masks/weights, not strings, sets, Pandas objects, or dynamically changing Python containers.

## 4. Files and Folders Inspected

KinOpt and TFOpt paths, files, tests, configs, and modules were intentionally excluded.

| File/folder | Role in this audit |
|---|---|
| `docs/phoskintime_multimodal_input_audit.md` | Previous audit read first; this document refines its multimodal plan for JAXopt single-objective/Diffrax migration. |
| `networkmodel/` | Main global coupled ODE workflow. Contains data loading, model topology, ODE solving, loss, pymoo/Optuna optimization, exports, dashboard bundle, dashboard app, refinement, hyperparameter scan, and sensitivity. |
| `networkmodel/runner.py` | Main global model orchestration. Imports Pymoo algorithms/operators/termination, constructs `GlobalODE_MOO`, runs Pymoo or Optuna, writes Pareto artifacts, selects picked solution, exports plots/tables, and logs final summaries. |
| `networkmodel/io.py` | Loads kinase/TF networks and RNA/MS observations into `df_prot`, `df_pho`, and `df_rna`. Future migration should keep Pandas loading outside JAX and emit numeric arrays for JAX core. |
| `networkmodel/network.py` | Defines `Index`, `KinaseInput`, and `System`; currently stores mutable NumPy arrays and Numba/Scipy argument buffers. Must be split into static metadata and JAX numeric state/parameter containers. |
| `networkmodel/models.py` | Numba RHS kernels for distributive/sequential/combinatorial/saturating models. Future equivalent RHS functions must be pure JAX. |
| `networkmodel/simulate.py` | Current ODE wrapper using `scipy.integrate.odeint` or custom Numba solver. Must be replaced by centralized Diffrax solve functions. |
| `networkmodel/lossfn.py` | Numba loss kernels returning protein/RNA/phospho loss sums. Future loss must be pure JAX and scalar-composed. |
| `networkmodel/cache.py` | Pre-indexes Pandas observations to numeric arrays. This is a good boundary for converting loaded data to fixed-shape JAX-ready arrays and masks. |
| `networkmodel/optproblem.py` | Pymoo `ElementwiseProblem` wrapper with fixed `n_obj=3`. Must be replaced by a JAXopt scalar objective wrapper. |
| `networkmodel/params.py` | Flat parameter packing/unpacking and softplus transforms. Must become JAX-compatible and projection-aware. |
| `networkmodel/utils.py` | Config dataclass, scaling, bounds, Numba helpers, and objective-selection helpers. Needs JAX-safe separation and config-backward-compat mapping. |
| `networkmodel/optuna_solver.py` | Alternative Optuna solver. Must be removed, deprecated, or converted to a JAXopt-compatible single-objective runner; current Optuna result compatibility should be replaced. |
| `networkmodel/refine.py` | Pymoo-based refinement of Pareto fronts. Must be replaced with JAXopt restart/continuation logic or removed/deprecated. |
| `networkmodel/scan.py` | Optuna + Pymoo hyperparameter scan. Must be replaced with single-objective weight-scan logic or optional non-differentiated outer-loop search. |
| `networkmodel/export.py` | Exports Pareto fronts, trajectories, goodness-of-fit plots, residuals, parameter uncertainty, correlations, convergence video, S-rates. Heavily assumes `res.X`, `res.F`, Pareto sets, and three objective columns. |
| `networkmodel/dashboard_bundle.py` | Serializes Pymoo/Optuna-like result objects and data frames into `dashboard_bundle.pkl`. Must store a JAXopt result summary instead. |
| `networkmodel/dashboard_app.py` | Streamlit dashboard that displays Pareto front, fixed three objective columns, picked objectives, and prediction files. Must be single-objective and mode-aware later; not edited in this audit. |
| `networkmodel/analysis.py` | Uses `simulate_odeint()` for steady-state simulation. Must route through Diffrax solver wrapper. |
| `networkmodel/model_ivp.py` | Contains functions compatible with `scipy.integrate.solve_ivp`. Should be deprecated or ported to JAX RHS functions. |
| `networkmodel/jacspeedup.py`, `networkmodel/solvers.py` | Numba/JIT custom solver and RHS/Jacobian acceleration. Must not be used in JAX-differentiated paths; likely replaced by Diffrax. |
| `networkmodel/steadystate.py` | Builds data-derived initial conditions and analytic steady-state helpers. JAX migration must keep Pandas/dicts outside JAX and produce numeric JAX initial conditions. |
| `protwise/` | Legacy/local per-gene ODE workflow. Currently uses SciPy `curve_fit`, SciPy `minimize` for initial conditions, and SciPy `odeint`. Must also migrate. |
| `protwise/runner/main.py` | Legacy/local orchestration. Loads protein, phosphosite, and mRNA data; validates columns; runs per-gene estimation; saves plots/reports. |
| `protwise/paramest/core.py` | Extracts per-gene arrays, calls `estimate_parameters()`, computes metrics, solves ODE, plots, sensitivity, knockout, and returns result dicts. |
| `protwise/paramest/normest.py` | Main local parameter-estimation objective and `curve_fit` calls. Must be ported to JAX/JAXopt single-objective fitting. |
| `protwise/paramest/toggle.py` | Dispatches parameter estimation; should dispatch to JAXopt implementation later. |
| `protwise/models/distmod.py`, `protwise/models/succmod.py`, `protwise/models/randmod.py` | Local ODE model solvers using `scipy.integrate.odeint` and NumPy/Numba. Must be ported to JAX RHS + Diffrax. |
| `protwise/steady/initdist.py`, `protwise/steady/initsucc.py`, `protwise/steady/initrand.py` | Use `scipy.optimize.minimize` with SLSQP constraints to compute steady-state initial conditions. Must become analytic/JAX root solve/projection-compatible. |
| `protwise/plotting/plotting.py` | Downstream plotting that assumes NumPy arrays and fixed mRNA/protein/phospho slices. Should remain outside JAX but become mode-aware. |
| `common/utils/display.py`, `common/utils/tables.py`, `common/frechet/distance.py` | Local output/report helpers and Frechet metric. Should remain outside differentiable core; any Numba metric use must not be in objective. |
| `config_loader.py`, `config/config.py`, `config/constants.py`, `config/cli.py`, `config/logconf.py` | Existing config/CLI/logging surfaces. Must remain schema-compatible; map old optimizer/solver options internally. |
| `scripts/*.py` | Standalone scripts. Several use Pymoo, `solve_ivp`, or `networkmodel.simulate_odeint`. They should not block core migration but need later reference cleanup or compatibility wrappers. |
| `run_dashboard.py` | Dashboard launcher. No direct solver/optimizer logic, but dashboard content must change later. |
| `README.md`, `PYPI_README.md`, `docs/Documentation/**/*.md`, `networkmodel/README.md` | Documentation contains old SciPy/pymoo/Pareto/multi-objective/evolution references that must be updated after code migration. |
| Notebooks | No `.ipynb` files were found by `rg --files -g '*.ipynb'` outside excluded KinOpt/TFOpt paths during this audit. |

## 5. Current networkmodel Architecture

### Input loading

- `networkmodel.runner.main()` parses CLI/default values for network paths, `--ms`, `--rna`, `--phospho`, prior files, output directory, optimizer, cores, lambdas, scan/refine/sensitivity, and solver choice.
- `networkmodel.config` imports `load_config_toml("config.toml")` from `config_loader.py` and exposes module-level constants.
- `networkmodel.io.load_data(args)` loads and normalizes:
  - kinase network into `df_kin_clean`
  - TF network into `df_tf_clean`
  - optional prior alphas/betas from Excel files
  - MS data from `args.ms`, split into protein rows (`df_prot`) and phosphosite rows (`df_pho`) by blank/nonblank `psite`
  - RNA data from `args.rna` into `df_rna`
- The previous multi-modal audit noted that `args.phospho` is parsed but current loading uses `args.ms` for phospho.

### Preprocessing

- `networkmodel.utils._normcols()` normalizes column names.
- `networkmodel.utils.process_and_scale_raw_data()` handles wide `x1..xN` data and scaling methods.
- `networkmodel.utils.normalize_fc_to_t0()` optionally normalizes protein/phospho to t=0.
- `runner.main()` filters phospho rows to mechanistic `(protein, psite)` pairs in `df_kin`, constructs/filters `df_tf_model`, then filters observations to `idx.proteins`.
- Time weights are created via `networkmodel.optproblem.build_weight_functions()` and added as `w` columns.

### Model construction

- `networkmodel.network.Index` maps protein/site names to state-vector offsets. Non-combinatorial layout is `[mRNA_i, protein_i, phosphosite_1, ...]`; combinatorial layout is `[mRNA_i, protein_state_0, ..., protein_state_2^n]`.
- `networkmodel.network.KinaseInput` creates `Kmat` from protein observations and defaults to ones for missing kinase observations.
- `networkmodel.buildmat.build_W_parallel()` creates `W_global`; `build_tf_matrix()` creates `tf_mat`.
- `networkmodel.network.System` stores mutable NumPy arrays for parameters and packed sparse buffers used by Numba/Scipy solvers.
- `defaults` in `runner.main()` initializes `c_k`, `A_i`, `B_i`, `C_i`, `D_i`, `Dp_i`, `E_i`, and `tf_scale`.

### ODE solving

- `networkmodel.simulate.simulate_odeint()` dispatches either to a custom Numba solver (`solve_custom`) or to SciPy `odeint` with `rhs_odeint` and `fd_jacobian_odeint` from `networkmodel.jacspeedup`.
- `networkmodel.simulate.simulate_and_measure()` solves once over the union of requested protein/RNA/phospho time grids and converts results into Pandas prediction tables.
- `networkmodel.analysis.simulate_until_steady()` also uses `simulate_odeint()`.

### Objective construction

- `networkmodel.cache.prepare_fast_loss_data()` maps observations to integer arrays and weights for protein/RNA/phospho losses.
- `networkmodel.lossfn.LOSS_FN` dispatches to Numba kernels `loss_function_noncomb()` or `loss_function_comb()` based on model type.
- `networkmodel.optproblem.GlobalODE_MOO._evaluate()` unpacks parameters, updates the mutable `System`, solves ODEs, computes three raw losses, normalizes them, applies lambdas, adds prior regularization to each objective, and returns a length-3 objective vector.

### Optimizer setup

- The Pymoo path constructs `GlobalODE_MOO`, reference directions via `get_reference_directions()`, `UNSGA3`, `SBX`, `PM`, `LHS`, and `DefaultMultiObjectiveTermination`, then calls `pymoo.optimize.minimize()`.
- The Optuna path calls `networkmodel.optuna_solver.run_optuna_solver()`, which wraps an Optuna multi-objective result into an `OptunaResult` with `.X` and `.F`.
- Hyperparameter scan (`networkmodel.scan`) combines Optuna outer trials with Pymoo inner UNSGA3 runs.
- Refinement (`networkmodel.refine`) creates refined Pymoo populations and reruns UNSGA3.

### Constraints/bounds

- `networkmodel.params.init_raw_params()` packs parameter arrays into a flat NumPy vector and maps physical bounds into raw softplus space.
- `networkmodel.params.unpack_params()` applies softplus to raw parameters, enforcing positivity.
- `networkmodel.utils.calculate_bio_bounds()` computes custom bounds from topology/data heuristics.
- Pymoo `xl`/`xu` are raw-space box constraints.
- No current networkmodel alpha/beta simplex constraint exists inside the global optimizer. Kinase/TF prior alpha/beta values are loaded from excluded upstream result files and used as priors/weights, not optimized directly in `networkmodel`.

### Result extraction, plotting, saving, reporting

- `runner.main()` saves `pymoo_optimization_result.pkl` or `optuna_optimization_result.pkl`, `pareto_X.npy`, `pareto_F.npy`, `pareto_F.csv`, `pareto_front.xlsx`, `fitted_params_picked.json`, `picked_objectives.json`, prediction CSVs, plots, residuals, parameter distributions, S-rate files, and a dashboard bundle.
- `networkmodel.export.*` functions assume Pymoo-like `.X`, `.F`, Pareto sets, and often fixed columns `prot_mse`, `rna_mse`, `phospho_mse`.
- `networkmodel.dashboard_app.py` displays a 3D Pareto front and fixed three objective columns.

## 6. Current protwise Architecture

### Input loading

- `protwise.runner.main` uses `config.config.parse_args()` and `extract_config()`.
- It reads protein data with `pd.read_csv(config['input_excel_protein'])`.
- It reads phosphosite data from `pd.read_excel(config['input_excel_psite'], sheet_name='Estimated')`.
- It reads mRNA data from `pd.read_excel(config['input_excel_rna'], sheet_name='Estimated')`.
- It validates fixed columns: protein/phospho collectively require `Gene`, `Psite`, `x1..x14`; mRNA requires `mRNA`, `x1..x9`.
- It processes only the intersection of phospho genes and mRNA genes.

### Preprocessing

- `protwise.paramest.core.process_gene()` extracts:
  - protein-only row: `protein_data['Psite'].isna() & GeneID == gene`
  - phosphosite rows: `kinase_data['Gene'] == gene`
  - RNA rows: `mrna_data['mRNA'] == gene`
- It converts fixed positional columns to NumPy arrays:
  - `Pr_data = protein_data.iloc[:, 2:].values`
  - `P_data = gene_data.iloc[:, 2:].values`
  - `R_data = rna_data.iloc[:, 1:].values`
- It computes `num_psites` from phosphosite row count and uses `protwise.steady.initial_condition(num_psites)`.

### Model construction and ODE solving

- `protwise.models.distmod`, `succmod`, and `randmod` define ODE RHS functions and `solve_ode()` wrappers.
- `distmod.py` and `succmod.py` import `scipy.integrate.odeint` and call it over the configured time grid.
- `randmod.py` imports `scipy.integrate.odeint` and solves the random/combinatorial local model.
- The local model state is also structurally mRNA + protein + phospho; the biological structure should remain unchanged.

### Objective construction

- `protwise.paramest.normest.parameter_estimation()` builds `target = np.concatenate([r_data.flatten(), pr_data.flatten(), p_data.flatten()])`.
- `model_func()` calls `solve_ode()` and returns flattened model predictions, optionally appended with regularization terms.
- `score_fit()`/config score logic computes scalar metrics over the flattened target/prediction.
- Regularization is added by appending a regularization vector to the curve-fit target rather than by a clean scalar objective.

### Optimizer setup

- `protwise.paramest.normest.py` imports `scipy.optimize.curve_fit`.
- `worker_find_lambda()` and `_curve_fit_multistart()` call `curve_fit()` repeatedly.
- Bootstrapping also calls `curve_fit()` on noisy targets.
- `protwise.steady.initdist`, `initsucc`, and `initrand` import `scipy.optimize.minimize` and use SLSQP with equality constraints to compute steady-state initial conditions.

### Constraints/bounds

- `config.config.parse_bound_pair()` parses user CLI bounds.
- `config.config.extract_config()` creates bounds for `A`, `B`, `C`, `D`, `S(i)`, and `D(i)`.
- `protwise.paramest.normest.parameter_estimation()` builds lower/upper bound vectors. For `randmod`, it logs/transforms bounds and parameters; for other models, it passes physical bounds directly to `curve_fit()`.
- Initial-condition functions enforce nonnegative variables with SLSQP bounds and equality residual constraints.

### Result extraction, plotting, saving, reporting

- `protwise.paramest.core.process_gene()` computes MSE/MAE against concatenated all-layer targets.
- It solves the fitted ODE, generates PCA/t-SNE/parallel plots, model-fit plots, knockout simulations, sensitivity analysis, and returns a result dictionary.
- `common.utils.display.save_result()` writes Excel sheets, site estimates/observations, PCA, t-SNE, sensitivity, knockout, and errors.
- `protwise.plotting.Plotter.plot_model_fit()` hard-codes mRNA/protein/phospho slices and time grids.
- `common.utils.display.merge_obs_est()` merges site observed/estimated sheets, not a fully multimodal output table.

## 7. Current Optimizer Dependencies

| File | Function/class | Current role | Why it must change | Proposed future replacement |
|---|---|---|---|---|
| `networkmodel/optproblem.py` | `GlobalODE_MOO(ElementwiseProblem)` | Pymoo elementwise multi-objective problem returning 3 objectives. | JAXopt needs a pure scalar objective; Pymoo class/mutation of `System` is incompatible with JAX differentiation/JIT. | Replace with `networkmodel/jax_objective.py` scalar function plus JAXopt solver wrapper. |
| `networkmodel/runner.py` | Pymoo imports: `UNSGA3`, `StarmapParallelization`, `SBX`, `PM`, `LHS`, `DefaultMultiObjectiveTermination`, `get_reference_directions`, `pymoo_minimize` | Main global optimization path. | Evolutionary multi-objective optimization is being removed. | JAXopt projected solver, e.g. `ProjectedGradient`, `LBFGSB` if box-only, or custom projected loop. |
| `networkmodel/runner.py` | `run_optuna_solver()` branch | Alternative Optuna multi-objective backend. | Future stack should be JAXopt single-objective; Optuna can at most be a non-core outer hyperparameter search. | Remove from core path or keep as optional outer hyperparameter tuner that calls JAX scalar objective outside differentiable path. |
| `networkmodel/refine.py` | `UNSGA3`, Pymoo population/operators, `pymoo_minimize` | Refines current Pareto front with tighter bounds. | Pareto-front refinement no longer exists in scalar JAXopt stack. | Replace with restart/multistart JAXopt runs, continuation on bounds, or local polish from previous optimum. |
| `networkmodel/scan.py` | Optuna outer loop + Pymoo inner UNSGA3 | Hyperparameter scan over lambdas using evolutionary runs. | Inner Pymoo and Pareto results incompatible with new scalar objective; Optuna dashboard language still Pareto-heavy. | Optional scalar hyperparameter search that invokes JAXopt; or static lambda config with deprecation warning. |
| `networkmodel/optuna_solver.py` | `NativeOptunaObjective`, `run_optuna_solver`, `OptunaResult` | Alternative MOTPE optimizer returning `.X`/`.F`. | Not JAXopt; result object preserves multi-objective semantics. | Remove/deprecate core use; if retained, make it an external scalar hyperparameter-search wrapper, not model optimizer. |
| `networkmodel/export.py` | functions taking `res.X`, `res.F` | Exports Pareto-front artifacts and parameter distributions. | JAXopt result will not be a Pymoo result and should have a single optimum plus diagnostics. | Export `best_params`, `objective_history`, optional restart table, and scalar diagnostics. |
| `networkmodel/dashboard_bundle.py` | `save_dashboard_bundle()` | Stores Pymoo/Optuna result object and Pareto arrays. | JAXopt result object/diagnostics differ. | Store serializable scalar optimizer summary, objective trace, gradient norm, projection info, data mode. |
| `networkmodel/dashboard_app.py` | `_fig_pareto_3d()`, fixed Pareto load | Displays 3D Pareto front. | No Pareto front in scalar optimization. | Replace with objective trace, residual/loss breakdown, restart comparison, and mode-aware prediction plots. |
| `protwise/paramest/normest.py` | `curve_fit()` in `worker_find_lambda()` | Fits local model for candidate regularization lambdas. | SciPy optimizer and Python/NumPy model callback are not JAX/JAXopt. | JAXopt scalar least-squares/objective with JAX RHS/Diffrax. |
| `protwise/paramest/normest.py` | `_curve_fit_multistart()` | Multi-start SciPy curve fitting. | Should use JAXopt restarts with projected bounds. | JAXopt multistart wrapper around scalar objective. |
| `protwise/paramest/normest.py` | bootstrap `curve_fit()` | Fits noisy targets. | Must use same JAXopt objective to preserve consistency. | JAXopt bootstrap loop outside JIT, with JAX arrays per run. |
| `protwise/steady/initdist.py` | `scipy.optimize.minimize(method='SLSQP')` | Solves equality-constrained steady-state y0. | SciPy constraint dicts are incompatible with JAXopt target. | Analytic y0 where possible; otherwise JAXopt root solve/projection or Diffrax steady-state relaxation. |
| `protwise/steady/initsucc.py` | `scipy.optimize.minimize(method='SLSQP')` | Same for successive model. | Same issue. | Same replacement. |
| `protwise/steady/initrand.py` | `scipy.optimize.minimize(method='SLSQP')` | Same for random model. | Same issue. | Same replacement, likely more complex due combinatorial states. |
| `scripts/compare_estimated_model_simulations_thermal_standard.py` | Pymoo `Problem`, `UNSGA3`, `pymoo_minimize` | Standalone multi-objective thermal comparison. | Script depends on removed Pymoo concepts. | Mark as legacy or port to JAXopt scalar objective if still part of reproducible workflow. |

## 8. Current Multi-Objective Assumptions

| Location | Multi-objective assumption | Future single-objective replacement |
|---|---|---|
| `networkmodel/optproblem.py` module docstring | Describes Pymoo compatibility, three objectives, and Pareto-front prior regularization. | Rename to scalar JAX objective documentation. |
| `GlobalODE_MOO.__init__()` | `n_obj=3`. | Scalar objective function returns one `jax.Array` scalar. |
| `GlobalODE_MOO._evaluate()` | Returns `out["F"] = [obj_protein, obj_rna, obj_phospho]`. | Return `L_total`; also compute optional non-differentiated loss breakdown for reporting. |
| `networkmodel/runner.py` | Constructs `UNSGA3` and reference directions using `problem.n_obj`. | Construct a JAXopt solver and projection function; no reference directions. |
| `networkmodel/runner.py` | Saves `pareto_X.npy`, `pareto_F.npy`, `pareto_F.csv`, `pareto_front.xlsx`. | Save `best_theta.npy`, `best_objective.json`, `objective_history.csv`, optional `restart_results.csv`; keep compatibility aliases only if needed. |
| `networkmodel/runner.py` | Selects picked solution from a set of Pareto candidates using weighted Fréchet score. | The optimizer returns one best solution; optional post-hoc selection only among restarts/checkpoints. |
| `networkmodel/export.py::export_pareto_front_to_excel()` | Excel workbook per Pareto solution with `summary`, `traj_protein`, `traj_rna`, `traj_phospho`. | Replace with `export_optimization_result_to_excel()` containing best solution, loss breakdown, parameters, trajectories, restarts/checkpoints. |
| `networkmodel/export.py::save_pareto_3d()` | 3D plot for `prot_mse`, `rna_mse`, `phospho_mse`. | Objective trace or loss-breakdown bar/line plots. |
| `networkmodel/export.py::save_parallel_coordinates()` | Parallel coordinates over Pareto solutions/objectives. | Optional parameter/restart comparison plot; no Pareto labels. |
| `networkmodel/export.py::create_convergence_video()` | Animation of Pareto front evolution. | Objective/history animation or remove with warning. |
| `networkmodel/export.py::scan_prior_reg()` | Reads `pareto_F.npy` and scans lambda combinations over three objective columns. | Recompute scalar objective breakdown for candidate weights or deprecate. |
| `networkmodel/export.py::export_param_correlations()` | Correlations across entire Pareto front. | Correlations across restarts/checkpoints/bootstrap samples if available; otherwise skip. |
| `networkmodel/export.py::export_parameter_distributions()` | Boxplots over Pareto front. | Use bootstrap/restart distribution; not Pareto. |
| `networkmodel/dashboard_bundle.py` | Stores `pareto_F`, `pareto_X`. | Store `objective_history`, `best_theta`, `loss_breakdown`, `restart_table`. |
| `networkmodel/dashboard_app.py` | Displays “Pareto front” tab/subheader and 3D scatter. | Display “Optimization objective” and “Loss breakdown”. |
| `networkmodel/README.md`, `docs/Documentation/global/README.md` | Mention multi-objective loss aggregation, Pymoo, evolutionary/GA, Pareto. | Update to JAXopt scalar objective and Diffrax solver terminology after migration. |
| `scripts/compare_estimated_model_simulations_thermal_standard.py` | Defines Pymoo multi-objective problem and picks Pareto solution by sum. | Port or mark legacy. |

## 9. Target JAXopt Optimization Design

### Global float64 setup

Add one centralized import/setup module later, for example `networkmodel/jax_config.py` or package-level initialization used by both `networkmodel` and `protwise`:

```python
import jax
jax.config.update("jax_enable_x64", True)
```

This must execute before any JAX arrays/functions are created. Tests should assert `jax.config.read("jax_enable_x64") is True`.

### Pure JAX objective

The future objective path should be:

1. Load and preprocess data with Pandas/NumPy outside JAX.
2. Convert to immutable numeric JAX arrays: indices, time grids, observation values, weights, masks.
3. Pack parameters as either a flat `jax.Array` plus slices or a PyTree with named arrays.
4. Project/reparameterize parameters to biologically valid physical values.
5. Solve ODE with Diffrax using pure JAX RHS and JAX arrays.
6. Extract numeric observables with JAX indexing/reductions.
7. Compute active layer losses with fixed-shape arrays and masks.
8. Return one scalar `L_total`.

No Pandas, Python dictionaries keyed by strings, mutable `System.update()`, Numba, SciPy, or logging should occur inside the JIT/differentiated objective.

### Parameter container choice

Recommended transitional design:

- Keep a flat vector for compatibility with current `theta0`, `slices`, `xl`, `xu` concepts.
- Convert `networkmodel.params.init_raw_params()` and `unpack_params()` to JAX-compatible versions that operate on `jax.numpy` arrays.
- Store slices/static metadata outside JIT, preferably as a small frozen dataclass.
- For `protwise`, use the same flat-vector convention first, then optionally move to PyTrees after parity tests pass.

Longer-term design:

- Use a parameter PyTree for readability: `{"c_k": ..., "A_i": ..., "B_i": ..., ...}`.
- Use `jax.flatten_util.ravel_pytree` if JAXopt solver requires a flat vector.

### Projection strategy

- Box bounds: use JAXopt `projection_box` or a custom `jnp.clip` projection.
- Nonnegative variables: prefer box projection `[0, upper]` in physical space or softplus reparameterization in unconstrained space; avoid both if double-constraining causes poor conditioning.
- Fixed parameters: remove from optimization vector if possible; otherwise set equal lower/upper bounds and projection enforces constant values.
- Simplex constraints: use JAXopt simplex projection where true simplex constraints exist. In `networkmodel`/`protwise`, no active alpha/beta simplex optimizer was found outside excluded KinOpt/TFOpt, so only document compatibility for prior values unless future code brings alpha/beta into these modules.
- Constraint penalties: use only when projection/reparameterization is impractical or non-differentiable shape choices make projection unsuitable.

### Candidate JAXopt solvers

- `jaxopt.ProjectedGradient` for explicit projection over box/simplex constraints.
- `jaxopt.LBFGSB` if only bound constraints are needed and the flat-vector design is kept.
- `jaxopt.GradientDescent` or `jaxopt.LBFGS` on unconstrained raw parameters with softplus transforms if projection is deferred.
- For expensive Diffrax objectives, consider limited iterations, checkpointing, and restarts; measure compile/runtime.

### Diagnostics/result conversion

The future optimizer result should be converted outside JAX to a serializable summary:

- `best_theta`
- `best_params` physical values
- `objective_value`
- `loss_breakdown`: `mrna_loss`, `protein_loss`, `phospho_loss`, regularization
- `gradient_norm`
- `iterations`
- `converged`/status
- `projection_strategy`
- `solver_backend="jaxopt"`
- optional `objective_history`
- optional restart table

## 10. Constraint and Projection Plan

| Existing/future constraint | Current representation | Recommended future representation | Reason |
|---|---|---|---|
| Networkmodel kinetic positivity (`c_k`, `A_i`, `B_i`, `C_i`, `D_i`, `Dp_i`, `E_i`, `tf_scale`) | Softplus transform in `networkmodel.params.unpack_params()` plus raw-space bounds. | Either keep softplus with unconstrained JAXopt or move to physical-space projected box optimization. Prefer one approach consistently. | Avoid negative biological rates while keeping differentiability. |
| Networkmodel upper/lower bounds | `xl`/`xu` raw-space arrays passed to Pymoo. | JAX arrays `lower`, `upper`; use `jaxopt.LBFGSB` or projection box. | Direct replacement for optimizer bounds. |
| Networkmodel fixed parameters | Collapsed lower/upper bounds detected by `get_optimized_sets()`. | Remove fixed parameters from optimization vector or use projection to equal bounds. | Reduces optimization dimension and avoids flat gradients. |
| Protwise kinetic bounds (`A`, `B`, `C`, `D`, `S(i)`, `D(i)`) | `curve_fit(..., bounds=free_bounds)`; random model log-transform. | Shared JAX flat-vector bounds and projection or reparameterization. | Aligns protwise with networkmodel. |
| Protwise nonnegative initial conditions | SLSQP bounds `(1e-6, None)` in `protwise/steady/init*.py`. | Analytic formulas if possible; otherwise JAX root/least-squares solve with nonnegative projection. | Removes SciPy constraint dictionaries. |
| Protwise steady-state equality constraints | SLSQP `constraints={'type': 'eq', 'fun': steady_state_equations}`. | Prefer analytic y0, `jaxopt.ScipyRootFinding` is not acceptable for full migration; use JAXopt root/least-squares or Diffrax relaxation to steady state. | Equality constraints are not simple box constraints. |
| Alpha bounds | In `networkmodel`, alpha values are loaded priors from upstream Excel, not optimized. In legacy docs/common tables they are report values. | Do not add alpha optimization in networkmodel/protwise. If future alpha variables enter these modules, represent bounds with box projection. | Avoid importing excluded KinOpt concerns. |
| Alpha sum-to-one/simplex | Not active in inspected `networkmodel`/`protwise` optimizer paths. | If introduced later, use JAXopt simplex projection and static group indices. | Projection is cleaner than SciPy constraints. |
| Beta bounds | Beta values are loaded priors/maps in `networkmodel.io` and `networkmodel.network`; local tables read beta sheets. | Keep as data/prior constants. If optimized later, use box projection. | Not currently an optimized block in scope. |
| Beta sum-to-one constraint | Not active in inspected `networkmodel`/`protwise` optimizer paths. | If biologically required later, clarify sign domain first; if beta can be negative, simplex is invalid and use affine projection or reparameterization. | Avoid ambiguous simplex with negative bounds. |
| Mode-specific inactive losses | Current Pymoo has separate objectives even when arrays can be empty. | Numeric layer masks in scalar objective; inactive layers multiply by zero and have zero weights/counts outside JIT. | Compatible with JAX static shapes and multimodal audit. |
| Regularization | Added per objective in `GlobalODE_MOO`; appended vector in protwise `curve_fit`. | Add scalar regularization terms directly to `L_total`. | Cleaner and differentiable. |

## 11. Multi-Modal Single Objective Plan

The JAX objective should receive a static `DataMode`-derived numeric bundle:

```text
JaxLossData:
  p_prot, t_prot, obs_prot, w_prot, mask_prot
  p_rna,  t_rna,  obs_rna,  w_rna,  mask_rna
  p_pho,  s_pho,  t_pho,   obs_pho, w_pho, mask_pho
  layer_weights = {mrna, protein, phospho}
  active_layer_mask = [fit_protein, fit_mrna, fit_phospho]
```

Shapes should be fixed before JIT. If variable-length arrays are inconvenient, use arrays sized to actual active observations and compile per mode/run, or padded arrays with boolean masks. Do not use Pandas or Python-side filtering inside the objective.

| Mode | Active outputs | Inactive outputs | Loss terms included/skipped | Expected shapes/checks | Logging/downstream outputs |
|---|---|---|---|---|---|
| mRNA only | `R(t)` fold-change | protein, phospho | Include `L_mrna`; skip `L_protein`, `L_phospho`. | `obs_rna`, `p_rna`, `t_rna`, `w_rna` nonempty and finite. | Log `active_losses=[mrna]`; save RNA fitted predictions and scalar objective/loss breakdown. |
| protein only | total protein fold-change | mRNA, phospho | Include `L_protein`; skip others. | Protein indices/times valid; kinase-input defaults logged if needed. | Save protein predictions only as fitted output. |
| phospho only | site-specific phospho fold-change | mRNA, protein | Include `L_phospho`; skip others. | `(protein, psite)` mapped to model site indices; nonempty after mechanistic filtering. | Save phospho predictions/S-rates; log missing protein-driver assumptions. |
| mRNA + protein | `R(t)`, total protein | phospho | Include `L_mrna + L_protein`. | Both observation sets nonempty. | Save RNA/protein fitted predictions and two-layer loss breakdown. |
| mRNA + phospho | `R(t)`, phosphosites | protein | Include `L_mrna + L_phospho`. | RNA and phospho arrays nonempty; y0 defaults for protein if needed. | Save RNA/phospho predictions; log skipped protein. |
| protein + phospho | total protein, phosphosites | mRNA | Include `L_protein + L_phospho`. | Protein and phospho arrays nonempty. | Save protein/phospho predictions; log skipped mRNA. |
| all three | `R(t)`, total protein, phosphosites | none | Include all three plus regularization. | Preserve all-three current behavior except scalar objective. | Save all prediction files and full loss breakdown. |

### Integration with `networkmodel`

- Replace `GlobalODE_MOO` with a scalar objective builder that accepts `JaxSystemStatic`, `JaxLossData`, `bounds/projection`, and `DataMode` metadata.
- Keep `prepare_fast_loss_data()` concept but return JAX-ready arrays/masks after Pandas preprocessing.
- `simulate_and_measure()` should have a JAX/Diffrax equivalent for predictions, then convert to Pandas at export.

### Integration with `protwise`

- Replace fixed `np.concatenate([r_data, pr_data, p_data])` with mode-aware numeric target containers.
- Preserve the same per-gene model state but compute layer losses separately and combine into one scalar.
- `Plotter.plot_model_fit()` and `save_result()` should consume structured outputs rather than assuming fixed slices.

## 12. Current ODE Solver Dependencies

| File | Function/class | Current role | Solver inputs/outputs | Downstream consumption | Proposed Diffrax replacement |
|---|---|---|---|---|---|
| `networkmodel/simulate.py` | `simulate_odeint()` | Main global solver wrapper. | Inputs: `System`, `t_eval`, tolerances, `mxstep`; output NumPy array `Y`. | Loss evaluation, predictions, analysis. | `simulate_diffrax(system_static, params, y0, save_times, solver_config) -> jax.Array`. |
| `networkmodel/simulate.py` | `simulate_and_measure()` | Solves and returns Pandas prediction tables. | Calls `simulate_odeint()`, extracts observables. | Exports, plots, Frechet selection. | Split into JAX observable extraction plus Pandas conversion wrapper. |
| `networkmodel/optproblem.py` | `_evaluate()` | Calls `simulate_odeint()` inside Pymoo evaluation. | Returns `Y` for Numba loss. | Pymoo objective. | JAX objective calls Diffrax directly. |
| `networkmodel/analysis.py` | `simulate_until_steady()` | Post-optimization dynamics check. | Calls `simulate_odeint()` over long time. | Steady-state plots/logs. | Diffrax wrapper with long-horizon `SaveAt`; optionally separate steady-state solver. |
| `networkmodel/jacspeedup.py` | `rhs_odeint()`, `fd_jacobian_odeint()` | Python wrappers for SciPy `odeint`. | ODEint-compatible `(y, t, *args)`. | Used by `simulate_odeint()`. | Replace with pure JAX RHS `(t, y, args)` for Diffrax. |
| `networkmodel/network.py` | `System.odeint_args()` | Packs mutable NumPy arrays for ODEint/Numba. | Tuple of arrays and buffers. | `simulate_odeint()`, Numba solver. | Replace with frozen JAX static/numeric args PyTree. |
| `networkmodel/model_ivp.py` | `make_solve_ivp_fun_*()` | Creates `solve_ivp`-compatible RHS functions. | `(t, y)` functions. | Potential examples/legacy use. | Deprecate or port to Diffrax RHS builders. |
| `networkmodel/solvers.py` | Custom Numba solver kernels | Avoid SciPy overhead. | Numba arrays, in-place loops. | Used by custom solver path. | Replace with Diffrax; keep only as legacy outside JAX if necessary. |
| `protwise/models/distmod.py` | `solve_ode()` | Local distributive ODE solve. | Parameters, initial condition, time grid; returns `sol`, flattened fitted vector. | protwise objective, plots, saving. | Diffrax local solver returning structured JAX observables. |
| `protwise/models/succmod.py` | `solve_ode()` | Local successive ODE solve. | Same. | Same. | Diffrax local solver. |
| `protwise/models/randmod.py` | `solve_ode()` | Local random/combinatorial ODE solve. | Same with log parameters. | Same. | Diffrax local solver; careful static combinatorial state metadata. |
| `scripts/thermal_distributive_model_protein.py` | top-level `solve_ivp()` calls | Standalone simulation demo. | `t_span`, `y0`, `t_eval`. | Plots in script. | Port if script remains maintained; otherwise mark legacy. |
| `scripts/compare_model_simulations_thermal_standard.py` | `solve_ivp()` | Standalone comparison. | `ode_func`, time span/grid. | Script outputs. | Port/legacy decision. |
| `scripts/compare_estimated_model_simulations_thermal_standard.py` | `solve_ivp()` | Pymoo thermal optimization script. | Local wrapper and temperature parameter. | Script optimization. | Port only if still reproducible workflow. |
| `scripts/compare_mechanisms.py` | `simulate_odeint()` | Uses networkmodel solver under the hood. | `System`, time grid. | Mechanism comparison. | Use new Diffrax wrapper. |

## 13. Target Diffrax Solver Design

### Centralized solver configuration

Create one central solver config object later, shared by `networkmodel` and `protwise`:

```text
DiffraxSolverConfig:
  solver_name: "kvaerno4" | "kvaerno5"
  rtol: existing relative_tolerance
  atol: existing absolute_tolerance
  max_steps: existing max_timesteps / ode_max_steps equivalent
  nonlinear_solver_maxiter: 10 or 20
  saveat_times: jax.Array
  dt0: optional
```

Suggested solver candidates:

- `diffrax.Kvaerno4` for stiff/semi-stiff production use.
- `diffrax.Kvaerno5` for potentially higher accuracy at higher cost.

Solver choice should be centralized, not scattered across `networkmodel/simulate.py`, `networkmodel/analysis.py`, `protwise/models/*.py`, and scripts.

### Diffrax solve design

- RHS signature should be pure JAX: `rhs(t, y, args)`.
- `args` should be a PyTree containing static numeric matrices, parameter arrays, time-grid/driver arrays, model metadata, and mode-independent structures.
- Use `diffrax.ODETerm(rhs)`.
- Use `diffrax.SaveAt(ts=save_times)` where `save_times` is the union of active layer time points and any requested diagnostic times.
- Initial conditions should be precomputed as JAX float64 arrays.
- Use Diffrax adjoint choices deliberately. Start with a stable default and test gradients; choose `RecursiveCheckpointAdjoint` or another appropriate adjoint based on memory/runtime.
- For nonlinear/root solving in implicit Kvaerno methods, configure the nonlinear solver through Diffrax/Optimistix APIs as supported by installed Diffrax versions. Plan for `max_steps`/nonlinear max iterations such as 10 or 20, and log the selected values.

### Error handling

- Solver failure should produce a deterministic high scalar penalty outside JIT or via JAX-safe failure handling if possible.
- Avoid Python exceptions inside JIT objective when possible. Validate inputs before optimization.
- Log solver settings and failure rates outside the JIT loop.

### Conversion to downstream formats

- Diffrax solution `sol.ys` should remain JAX arrays through observable/loss computation.
- For exports/plots, convert with `np.asarray()` at the boundary and build Pandas data frames using existing column names where possible.
- Preserve current prediction file names in all-three/all-active compatibility mode, but store backend metadata (`solver_backend=diffrax`, `solver_name=Kvaerno4/5`).

## 14. JAX Compatibility Risks

| Risk | Affected files/functions | Why it matters | Plan |
|---|---|---|---|
| NumPy inside objective | `networkmodel/params.py`, `networkmodel/lossfn.py`, `networkmodel/models.py`, `protwise/models/*.py`, `protwise/paramest/normest.py` | NumPy breaks JAX tracing/gradients. | Port differentiable functions to `jax.numpy`; keep NumPy only outside core. |
| Pandas inside objective | `networkmodel/io.py`, `networkmodel/simulate_and_measure()`, `protwise/core.py` | Pandas is non-JAX and dynamic. | Precompute numeric arrays before objective; convert to Pandas only after optimization. |
| SciPy calls inside objective | `networkmodel/simulate.py`, `protwise/models/*.py`, `protwise/steady/init*.py` | SciPy solvers/optimizers are not differentiable JAX functions. | Replace with Diffrax/JAXopt. |
| Numba in differentiated path | `networkmodel/lossfn.py`, `networkmodel/models.py`, `networkmodel/jacspeedup.py`, `networkmodel/solvers.py`, `networkmodel/utils.py`, local models | Numba is not JAX-differentiable. | Replace JAX objective path; optionally leave Numba only as legacy non-JAX path during transition. |
| Python-side mutation | `System.update(**params)` mutates arrays; `GlobalODE_MOO._evaluate()` mutates `self.sys`. | JAX functions must be pure. | Use immutable parameter PyTrees and pure RHS functions. |
| Dynamic shapes | Variable number of observations/sites/states by mode/protein. | JIT recompilation or tracing errors. | Precompute static arrays per run; pad/mask if needed. |
| Object arrays/strings | `Index.proteins`, `idx.sites`, Pandas identifiers. | JAX needs numeric arrays. | Map identifiers to integer indices before JAX. |
| NaN handling | Missing values, zero baselines, dropped rows. | NaNs poison gradients. | Validate/preclean data; use eps floors in JAX. |
| float32/float64 mismatch | JAX defaults to float32 unless enabled. | Numerical differences and stiff solver instability. | Enable x64 globally and test dtypes. |
| Non-JAX interpolation | `KinaseInput.eval()` uses Python/NumPy search and mutation. | Not JAX-differentiable. | Use JAX-compatible interpolation/step lookup with static grids. |
| Non-JAX normalization | `normalize_fc_to_t0()`, fold-change extraction with Pandas/NumPy. | Should not happen inside objective. | Do data normalization outside JAX or implement JAX observable normalization for predictions only. |
| Logging inside JIT | Current runner/export logs freely. | Side effects are not allowed in JIT/grad. | Log before/after solver calls, not inside objective. |
| Current Optuna objective direct indexing | `networkmodel/optuna_solver.py` uses raw state indices not fold-change observable aggregation. | Incorrect or inconsistent when porting. | Do not reuse; build one canonical JAX objective. |

## 15. Downstream Output Impact

### Result tables

- Current `pareto_F.csv` and `pareto_X.npy` should be replaced by scalar optimization outputs such as `objective_history.csv`, `best_theta.npy`, `best_params.json`, `loss_breakdown.json`, and optional `restart_results.csv`.
- To preserve compatibility, the implementation may write deprecated aliases with clear metadata, but should avoid pretending there is a Pareto front.
- `picked_objectives.json` should become `best_objective.json` or include `total_objective` and per-layer loss breakdown. Existing filename can be retained for backward compatibility with a schema-version field.

### Plots

- Replace Pareto plots with objective trace, gradient norm trace, loss breakdown bar plots, and restart comparison plots.
- Goodness-of-fit plots should remain but must be mode-aware and should operate on post-optimization Pandas data frames.
- Parameter distribution/correlation plots should use bootstrap/restarts/checkpoints, not Pareto samples.

### Reports/dashboard

- Dashboard should remove “Pareto front” language and show scalar objective convergence and mode-aware layer outputs.
- `dashboard_bundle.pkl` should not store unserializable JAXopt internals. Store NumPy arrays/scalars and metadata only.

### Scripts

- Scripts that consume `pareto_F.npy`, `pareto_F.csv`, or Pymoo result `.F`/`.X` need compatibility wrappers or updates.
- Scripts using `simulate_odeint()` should call a new solver-agnostic `simulate()` wrapper that internally uses Diffrax after migration.

### Logs

- Logs should state backend `JAXopt + Diffrax`, selected solver, selected data mode, scalar objective value, loss breakdown, and any deprecated config mappings.

## 16. Dashboard, Scripts, and Notebook Impact

### Dashboard

| File | Current old-stack reference | Future update |
|---|---|---|
| `networkmodel/dashboard_app.py` | Docstring mentions Pareto frontiers. | Replace with scalar optimization/convergence terminology. |
| `networkmodel/dashboard_app.py::_load_outputs()` | Reads `pareto_F.csv`; otherwise creates DataFrame from `bundle["res"].F` with three fixed columns. | Load `objective_history.csv`, `loss_breakdown.json`, and `restart_results.csv`; no `.F` dependency. |
| `networkmodel/dashboard_app.py::_fig_pareto_3d()` | Plots `prot_mse`, `rna_mse`, `phospho_mse`. | Replace with objective trace/loss breakdown. |
| `networkmodel/dashboard_app.py::main()` | Gallery includes `pareto_front_3d.png`, `pareto_pcp.png`; Overview tab says “Pareto front”. | Replace gallery and labels. |
| `networkmodel/dashboard_app.py` | Time-series sections assume all three observed data frames exist. | Use data-mode metadata to hide skipped layers. |

### Scripts

| File | Reference | Plan |
|---|---|---|
| `scripts/compare_estimated_model_simulations_thermal_standard.py` | Imports Pymoo and `solve_ivp`, defines Pymoo problem and UNSGA3 run. | Port only if still maintained; otherwise mark legacy and remove from reproducible migration path. |
| `scripts/compare_model_simulations_thermal_standard.py` | Uses `solve_ivp`. | Replace with Diffrax or script-level compatibility wrapper. |
| `scripts/thermal_distributive_model_protein.py` | Uses `solve_ivp`. | Replace with Diffrax demo or mark legacy. |
| `scripts/compare_mechanisms.py` | Imports `simulate_odeint`, comments that it uses odeint. | Update to new `simulate_diffrax`/solver-agnostic wrapper and rename comments. |
| Other `scripts/*.py` | Some consume exported results rather than solving/optimizing. | Update only if they expect Pareto columns/files. |

### Notebooks

No `.ipynb` files were found outside excluded paths during this audit. If notebooks are later added or generated, search them for `pymoo`, `Pareto`, `odeint`, `solve_ivp`, `curve_fit`, `Optuna`, `UNSGA`, `scipy.optimize`, and fixed output filenames before migration completion.

## 17. Config Backward Compatibility Plan

### Current config usage

- `[networkmodel].optimizer` accepts values such as `"pymoo"` or `"optuna"`.
- `[networkmodel].n_gen`, `pop`, `refine`, `num_refinements`, `study_name`, `sampler`, `pruner`, `n_trials`, and `hyperparam_scan` are optimizer-related.
- `[networkmodel].loss`, lambdas, scaling, weighting, and regularization values affect objective/loss.
- `[networkmodel.solver]` contains `absolute_tolerance`, `relative_tolerance`, `max_timesteps`, and `use_custom_solver`.
- `[ode]` and `[ode.bounds]` configure the legacy/protwise model and bounds.
- `config/config.py` also parses CLI bounds for the local workflow.

### Backward-compatible mapping

| Existing key/option | Future handling |
|---|---|
| `optimizer="pymoo"` | Accept but log deprecation: internally map to JAXopt default. |
| `optimizer="optuna"` | Accept but log that core optimizer is JAXopt; optional outer hyperparameter search only if explicitly retained. |
| `n_gen` | Map to JAXopt max iterations or accepted but logged as deprecated in favor of internal `maxiter`. No schema change. |
| `pop` | For scalar JAXopt, map to number of restarts only if multistart is enabled; otherwise ignore with warning. |
| `refine`, `num_refinements` | Map to local polish/restart refinement if implemented; otherwise accepted but unused with warning. |
| `study_name`, `sampler`, `pruner`, `n_trials` | Keep accepted; only used if optional outer hyperparameter search remains. Otherwise warn. |
| `loss` | Map to existing robust loss-mode choices inside JAX loss function where feasible. |
| `lambda_protein`, `lambda_rna`, `lambda_phospho`, `lambda_prior` | Continue to weight scalar loss components. |
| `weighting_method_*` | Continue to generate layer weights outside JAX and pass numeric arrays/masks into objective. |
| `absolute_tolerance`, `relative_tolerance`, `max_timesteps` | Map directly to Diffrax `rtol`, `atol`, and `max_steps`. |
| `use_custom_solver` | Accept but log deprecation; Diffrax is the internal solver backend. |
| `[ode.bounds]` and CLI bound args | Preserve for protwise JAXopt projection/bounds. |

Default future backend should be logged as `optimizer_backend=jaxopt`, `solver_backend=diffrax`, `jax_enable_x64=True`.

## 18. Logging and Console Plan

Future logs should include:

- `[Backend] JAX float64 enabled: True`
- `[Optimizer] Backend: JAXopt; solver: ProjectedGradient/LBFGSB; maxiter=...`
- `[Optimizer] Deprecated config optimizer='pymoo' accepted and mapped to JAXopt` when applicable.
- `[Solver] Backend: Diffrax; solver=Kvaerno4/Kvaerno5; rtol=...; atol=...; max_steps=...; nonlinear_maxiter=10/20`
- `[Mode] Detected data mode: ...; active layers: ...; skipped layers: ...`
- `[Loss] Active terms: ...; weights: ...; regularization: ...`
- `[Params] Blocks: c_k=..., A_i=..., B_i=..., ...; optimized_dim=...; fixed_dim=...`
- `[Projection] Strategy: box/simplex/softplus/fixed-mask; bounds validated`
- `[Optimize] Iterations=...; objective=...; grad_norm=...; converged=...`
- `[Output] Saved scalar optimization diagnostics: ...`
- Warnings for stale Pymoo/Optuna/SciPy options and unavailable optional data layers.

Do not log inside JIT-compiled objective functions.

## 19. Files That Must Change Later

| File | Current role | Required future change | Reason | Risk level | Notes |
|---|---|---|---|---|---|
| `networkmodel/runner.py` | Global orchestration with Pymoo/Optuna and Pareto outputs. | Replace optimizer setup/result handling with JAXopt scalar flow; route solver to Diffrax; mode-aware outputs. | Central old-stack dependency. | Very high | Preserve CLI/config surface. |
| `networkmodel/optproblem.py` | Pymoo `GlobalODE_MOO`. | Replace/deprecate with scalar JAX objective module. | Pymoo/multi-objective incompatible. | Very high | Could keep temporarily as legacy behind flag. |
| `networkmodel/simulate.py` | SciPy `odeint` and custom solver wrapper. | Replace with Diffrax centralized solver wrapper. | Required solver migration. | Very high | Preserve `simulate_and_measure` compatibility wrapper. |
| `networkmodel/models.py` | Numba RHS kernels. | Add/replace with pure JAX RHS functions. | Diffrax/JAXopt need JAX RHS. | Very high | Keep model equations unchanged. |
| `networkmodel/lossfn.py` | Numba loss kernels. | Port to pure JAX scalar/masked loss. | Numba cannot be differentiated by JAX. | High | Use previous multimodal layer masks. |
| `networkmodel/cache.py` | Numeric loss pre-indexing. | Emit JAX-ready arrays/masks and no fake counts. | Objective data boundary. | High | Good place for data validation. |
| `networkmodel/params.py` | NumPy packing/unpacking and softplus. | JAX-compatible packing/projection; fixed parameter handling. | Needed for JAXopt. | High | Preserve slice semantics initially. |
| `networkmodel/network.py` | Mutable `System`, Numba buffers, `odeint_args`. | Split into static metadata and immutable JAX numeric containers. | Mutation/Numba incompatible. | Very high | Keep `Index` as preprocessing metadata if outside JAX. |
| `networkmodel/jacspeedup.py` | Numba RHS/Jacobian for odeint. | Remove from differentiable path; deprecate/replace. | Not JAX/Diffrax. | High | Could remain temporarily legacy. |
| `networkmodel/solvers.py` | Custom Numba solver. | Replace with Diffrax or isolate as legacy. | Not JAX/Diffrax. | High | Current comments mention SciPy solvers. |
| `networkmodel/export.py` | Pareto/multi-objective exports. | Convert to scalar objective outputs, loss breakdowns, restart/checkpoint exports. | Downstream assumptions. | High | Keep old filenames only as compatibility aliases if needed. |
| `networkmodel/dashboard_bundle.py` | Stores Pymoo-like result data. | Store JAXopt diagnostics and mode metadata. | Dashboard compatibility. | Medium | Avoid serializing JAX internals. |
| `networkmodel/dashboard_app.py` | Pareto dashboard. | Replace Pareto UI with scalar objective UI. | User-facing dashboard would break. | Medium | Not edited until core complete. |
| `networkmodel/refine.py` | Pymoo refinement. | Replace with JAXopt restarts/local polish or deprecate. | Pymoo dependency. | Medium/high | May be optional. |
| `networkmodel/scan.py` | Optuna+Pymoo scan. | Replace or deprecate for scalar JAXopt. | Pymoo/Optuna core dependency. | Medium/high | Optional outer scan only. |
| `networkmodel/optuna_solver.py` | Optuna alternative solver. | Remove/deprecate or rewrite as optional scalar outer tuner. | Not JAXopt core. | High | Current loss indexing suspect. |
| `networkmodel/analysis.py` | Uses `simulate_odeint()`. | Call Diffrax wrapper. | Solver migration. | Medium | Post-optimization only. |
| `protwise/paramest/normest.py` | SciPy `curve_fit` objective/fitting. | Port to JAXopt scalar objective and multistart/bootstrap wrappers. | Main protwise optimizer. | Very high | Must be mode-aware too. |
| `protwise/models/distmod.py` | Local ODE `odeint`. | Port to Diffrax/JAX RHS. | Solver migration. | High | Preserve output semantics via wrapper. |
| `protwise/models/succmod.py` | Local ODE `odeint`. | Port to Diffrax/JAX RHS. | Solver migration. | High | Same. |
| `protwise/models/randmod.py` | Local random model `odeint`. | Port to Diffrax/JAX RHS. | Solver migration. | High | Static combinatorial metadata needed. |
| `protwise/steady/initdist.py` | SciPy SLSQP y0. | Replace with analytic/JAX root/projection. | SciPy optimize dependency. | Medium/high | Similar for all init files. |
| `protwise/steady/initsucc.py` | SciPy SLSQP y0. | Same. | Same. | Medium/high |  |
| `protwise/steady/initrand.py` | SciPy SLSQP y0. | Same. | Same. | High | More states. |
| `protwise/paramest/core.py` | Calls estimator/solver and builds metrics. | Use JAXopt result and structured mode-aware outputs. | Integrates protwise migration. | High | Keep reporting interface stable. |
| `protwise/plotting/plotting.py` | Fixed output slicing. | Mode-aware plotting from structured outputs. | Multimodal/JAX output changes. | Medium | Outside JAX. |

## 20. Files That May Need Minor Adaptation

| File | Current role | Possible future change | Trigger condition | Risk level |
|---|---|---|---|---|
| `networkmodel/io.py` | Data loading/preprocessing. | Produce JAX-ready numeric data bundle after Pandas loading. | When objective port starts. | Medium |
| `networkmodel/utils.py` | Config dataclass, bounds, helpers. | Remove/replace Numba helpers in core path; add JAX-safe bounds utilities. | When parameter projection is implemented. | Medium |
| `networkmodel/config.py` | Exposes config constants. | Map old optimizer/solver constants to internal JAXopt/Diffrax defaults. | Backend migration. | Low/medium |
| `config_loader.py` | Loads config schema. | Possibly add internal default mapping only, no schema changes. | If new internal defaults need centralization. | Low |
| `config/config.py` | Local CLI bounds/scoring. | Keep bounds but map local optimizer behavior to JAXopt. | protwise migration. | Low/medium |
| `config/constants.py` | Local ODE constants. | Keep current keys; ensure JAX dtype/time grids are converted later. | protwise migration. | Low |
| `common/utils/display.py` | Local saving/reporting. | Mode-aware sheet names and scalar objective fields. | If protwise output schema changes. | Medium |
| `common/utils/tables.py` | Alpha/beta report tables. | Likely unchanged unless output files change. | If prior-result table locations change. | Low |
| `common/frechet/distance.py` | Numba Frechet metric. | Keep outside objective or replace with JAX metric only if used in objective. | If Frechet becomes scalar objective term. | Medium |
| `processing/*.py` | Preprocessing/mapping. | Likely no change. | Only if output schema for upstream files changes. | Low |
| `run_dashboard.py` | Dashboard launcher. | Probably unchanged; app content changes. | If CLI args change, which should be avoided. | Low |
| `scripts/*.py` | Standalone analysis. | Update old solver/optimizer references if maintained. | After core migration. | Medium |
| `docs/**/*.md`, `README.md`, `PYPI_README.md` | Documentation. | Update old optimizer/solver terminology after implementation. | After code migration is tested. | Low |

## 21. Files That Should Not Be Modified Unless Necessary

| File | Reason to preserve | Risk if changed |
|---|---|---|
| `config.toml` | User-facing schema must remain unchanged. | Breaking user configs and violating requirement. |
| `config/cli.py` | CLI should remain stable; backend changes should be internal. | User-facing API break. |
| `config/logconf.py` | Logging infrastructure is sufficient. | Duplicate handler/test regressions. |
| `tests/test_config.py` | Existing logging tests unrelated. | Unnecessary churn. |
| `networkmodel/buildmat.py` | Topology matrix construction can remain preprocessing/outside JAX initially. | Risk of changing biological topology. |
| `networkmodel/io.py` parsing semantics | Data interface should remain stable; only add numeric bundle outputs carefully. | Input regressions. |
| `docs/assets/*` | Static assets unrelated to solver/optimizer. | No value. |
| Excluded `kinopt`/`tfopt` paths | Out of scope by requirement. | Scope violation. |

## 22. Proposed Implementation Phases for a Later Agent

### Phase 1: Add global JAX float64 setup

Files likely involved: new `networkmodel/jax_config.py`, possibly shared utility imported by `networkmodel` and `protwise` JAX modules.

Checks:

- Assert `jax_enable_x64` is true.
- No legacy code behavior changes yet.

### Phase 2: Isolate current objective functions

Files: `networkmodel/optproblem.py`, `networkmodel/lossfn.py`, `protwise/paramest/normest.py`.

Checks:

- Current scalar/layer loss calculations are documented and covered by baseline tests.
- Identify all inputs needed by objective independent of runner state.

### Phase 3: Create JAX-compatible parameter containers

Files: `networkmodel/params.py`, `protwise/paramest/normest.py`, possibly new shared parameter module.

Checks:

- Pack/unpack parity against NumPy current implementation.
- Bounds/projection shapes match current parameter blocks.

### Phase 4: Port networkmodel objective to pure JAX

Files: new JAX objective module, `networkmodel/cache.py`, `networkmodel/models.py` or new JAX RHS module.

Checks:

- Objective returns scalar JAX array.
- Loss breakdown matches old code on small fixed examples without optimization.

### Phase 5: Port protwise objective to pure JAX

Files: `protwise/paramest/normest.py`, local model JAX RHS modules.

Checks:

- One-gene objective returns scalar.
- Existing all-three target parity on small fixture.

### Phase 6: Add projection/bounds/constraint handling

Files: `networkmodel/params.py`, protwise parameter code, new projection utilities.

Checks:

- Box projection, nonnegative projection, fixed parameters, and optional simplex projection tests.

### Phase 7: Replace optimizer calls with JAXopt

Files: `networkmodel/runner.py`, `protwise/paramest/normest.py`, deprecate `optproblem.py` path.

Checks:

- JAXopt run produces best params and scalar diagnostics.
- No Pymoo required in core networkmodel/protwise path.

### Phase 8: Replace ODE solvers with Diffrax

Files: `networkmodel/simulate.py`, `networkmodel/models.py` or JAX RHS module, `protwise/models/*.py`.

Checks:

- `diffrax.Kvaerno4` and `diffrax.Kvaerno5` execute on fixtures.
- Solver output shape matches old solvers.

### Phase 9: Integrate multi-modal single-objective loss handling

Files: `networkmodel/cache.py`, JAX objective modules, `protwise/paramest/core.py`, `protwise/paramest/normest.py`.

Checks:

- All seven modes include/skip correct terms.
- No fake zeros/NaNs for missing layers.

### Phase 10: Update result extraction

Files: `networkmodel/runner.py`, `networkmodel/export.py`, `protwise/paramest/core.py`, `common/utils/display.py`.

Checks:

- Best params/loss breakdown/predictions are exported.
- Existing all-three filenames preserved where feasible.

### Phase 11: Update plotting/saving/reporting

Files: `networkmodel/export.py`, `protwise/plotting/plotting.py`, `common/utils/display.py`.

Checks:

- No Pareto plots in scalar mode.
- Mode-aware plots generate.

### Phase 12: Update dashboard/scripts/notebooks references

Files: `networkmodel/dashboard_app.py`, `networkmodel/dashboard_bundle.py`, scripts, docs.

Checks:

- Dashboard loads scalar diagnostics.
- Maintained scripts no longer import Pymoo/SciPy solvers for core path.

### Phase 13: Add logging and warnings

Files: `runner.py`, `protwise/runner/main.py`, config mapping utilities.

Checks:

- Deprecated config warnings visible.
- Backend/mode/solver/objective logs present.

### Phase 14: Add tests

Files: `tests/` new JAX/Diffrax/mode tests.

Checks:

- See Section 23.

### Phase 15: Remove or deprecate SciPy/pymoo references

Files: old optimizer/solver modules, docs, scripts.

Checks:

- Search confirms no core `networkmodel`/`protwise` optimization path imports `pymoo`, `scipy.optimize`, `odeint`, or `solve_ivp`.
- Excluded KinOpt/TFOpt untouched.

## 23. Test Plan for Later Implementation

- Test JAX float64 is enabled before computation.
- Test `networkmodel` JAX objective returns a rank-0 scalar JAX array.
- Test `protwise` JAX objective returns a rank-0 scalar JAX array.
- Test all seven multimodal modes include exactly the intended loss terms.
- Test alpha bound/simplex projection utilities if alpha variables are introduced in scope; otherwise test alpha priors remain constants.
- Test beta bounds/simplex handling if beta variables are introduced; otherwise test beta priors remain constants.
- Test nonnegative kinetic parameter projection.
- Test lower/upper bound projection.
- Test fixed parameters remain fixed after projection/optimization step.
- Test Diffrax `Kvaerno4` execution for global and local models.
- Test Diffrax `Kvaerno5` configuration/execution for at least one small fixture.
- Test solver output shape and dtype are float64.
- Test optimizer convergence summary contains objective value, iterations, gradient norm/status.
- Test downstream result tables contain scalar objective and per-layer loss breakdown.
- Test plotting compatibility for all-three and at least one single-layer mode.
- Test saving compatibility: prediction files and JSON summaries exist as expected.
- Test config backward compatibility for old `optimizer="pymoo"`, `optimizer="optuna"`, `use_custom_solver`, `n_gen`, and `pop`.
- Test no KinOpt/TFOpt paths changed by migration.
- Test no `pymoo` import in core PhosKinTime `networkmodel`/`protwise` path.
- Test no `scipy.optimize` import in PhosKinTime `networkmodel`/`protwise` optimization path.
- Test no `odeint`/`solve_ivp` use in PhosKinTime `networkmodel`/`protwise` solver path.
- Test dashboard bundle loads without `.F`/`.X` result fields.
- Test scripts either run through compatibility wrappers or are explicitly marked legacy.

## 24. Risks and Failure Modes

- Existing phospho workflow could break if observable aggregation changes from old Numba loss semantics.
- Objective may become non-differentiable if Python/NumPy/Pandas/Numba remains in the path.
- Projection may conflict with existing softplus transforms if both are applied at once.
- Beta simplex constraints are ambiguous if beta values can be negative; do not assume simplex unless sign/domain is explicit.
- Diffrax Kvaerno solvers may expose stiffness or convergence issues hidden by LSODA/odeint.
- Numerical results may change due to solver, tolerances, adjoints, or float64 JAX behavior.
- Runtime may initially be slower due to JAX compilation or Diffrax adjoint overhead.
- Dashboard may break if it expects Pareto files/columns.
- Documentation and notebooks may become stale if old terms remain.
- Config options like `pop`, `n_gen`, `optimizer`, and `use_custom_solver` may become misleading without deprecation logs.
- Downstream files/scripts may expect `pareto_F.csv`, `pareto_X.npy`, and `pareto_front.xlsx`.
- Hidden SciPy dependency may remain in standalone scripts or comments.
- Shape mismatches may appear in multimodal losses if missing layers are not padded/masked consistently.
- JAX dynamic-shape recompilation can be expensive if every gene/mode compiles separately.
- Python-side mutation in `System` may accidentally remain and invalidate gradients.
- Replacing protwise `curve_fit` may alter local model fit behavior and confidence/bootstrap outputs.

## 25. Final Recommendation

Migrate in small, testable layers. First isolate the current `networkmodel` and `protwise` objective/solver paths and create JAX-compatible parameter/data containers while leaving the user-facing config unchanged. Convert the objectives to pure JAX scalar functions before replacing the ODE solver. Then introduce Diffrax with centralized `diffrax.Kvaerno4`/`diffrax.Kvaerno5` configuration and float64 enabled. Replace Pymoo/Optuna/SciPy optimizer calls with JAXopt projected or bound-constrained solvers only after objective and solver parity tests pass.

Handle `networkmodel` and `protwise` separately but consistently: both should use a scalar objective, shared mode-aware loss principles, shared projection/bounds utilities where practical, and the same JAX/Diffrax dtype/config conventions. Preserve downstream output contracts where possible, but replace multi-objective/Pareto language everywhere with scalar objective, loss breakdown, convergence, and restart/checkpoint terminology. Test all seven multimodal modes before removing old code paths, and do not touch KinOpt/TFOpt.
