# Changelog

All notable changes to this project are documented here.

## [Unreleased]



### Documentation


- Add multimodal input audit

- Add jaxopt diffrax migration audit

- Restore documentation context with current networkmodel details

- Fix reviewed documentation issues



### Fixed


- Remove empty orcid field from CITATION.cff

- Fixes in protwise models

- Fixes in protwise models & debugging network model for inflated FC values

- Fixed the dead code error

- Fixed sensitivity analysis, problems in posterior, testing phase, need to pickle objective

- Pickled objective for each posterior chain sampling

- Standalone process for profile likelihood

- Workers in sensitivity simulations

- Equations in notebooks

- Console display, posterior log, workflow testing for frontend

- Fixed the frontend no code dashboard, testing networmodel and protwise done

- Cached rhs in model ode solving, and forward simulation script

- Formatting python code

- Directories path

- Posterior params conversion fix

- Network sweep and forward simulation

- Phosphosite dynamics 4th panel in forward simulation

- Phosphosite dynamics 4th panel in forward simulation & data fit inspector

- Combinatorial model state phase explosion down to O(1) runtime, no memory explosion in for loop, vectorized RHS for combinatorial model.

- Ambiguity in model loading for forward simulation-KO/WT dashboard



### Maintenance


- Add community, governance, citation, and repository-maintenance files

- Update changelog and docs config file

- Chore and fix: cleanup config file, fix mrna modality missing scenario, tested all scenarios, works!



### Other


- Merge pull request #31 from bibymaths/master

Cleanup for README.md

- Merge pull request #32 from bibymaths/master

 Docs Deployment, Cleanup, and README Improvements

- Merge pull request #33

master

- Delete `global_model.py`: remove unused global ODE dual-fit implementation.

- Simplify transformations by removing unnecessary power calculations and update constraints for beta parameter handling

- Refactor optimization workflows: add multi-start support, enhance constraint handling, and improve result export.

- Remove `VECTORIZED_LOSS_FUNCTION`: delete unused constant and related comments from codebase.

- Refactor configuration handling: integrate `config_loader`, remove hardcoded paths, simplify constants, and enhance CLI default handling.

- Update optimizer configuration: switch method to DE, clean up optrun.py, and streamline imports

- Refactor optimization: replace DE with GA, enhance tournament selection, clip negative values, and clean up unused operations

- Refactor optimization problem: accelerate with Numba, improve loss calculations, and simplify constraint handling. Revise `config_loader` to robustly identify project root and adjust directory setup logic.

- Update optimization: switch to NSGA-II, replace GA with DE in optrun.py, adjust termination criteria, and refine algorithm configuration

- Refactor optimization pipeline: replace NSGA-II with UNSGA-III, add Das-Dennis partition heuristic, refactor objective evaluation with Numba, enhance population evaluation, and update loss configuration.

- Refactor multi-objective evaluation: enhance type annotations, accelerate evaluation with threading, optimize parameter handling, and streamline class initialization.

- Update optimization pipeline: replace `DefaultMultiObjectiveTermination` with `get_termination`, adjust config for `results_kinopt` paths, switch optimizer to SMSEMOA, and update ODE model to `distmod`.

- Refactor result directory structure and enhance `plotout` handling of CSV columns for robustness and compatibility.

- Refactor and enhance: standardize input file paths in `weights.py`, improve multiprocessing and logging in `normest.py`, add multistart fitting for robustness, update default directories and optimization settings in `config.toml`, and enhance logging configuration for multiprocessing compatibility.

- Add new scripts for analyzing kinase/TF psites and computing Fréchet distance, update `config.toml` for model switch to `randmod`, and implement a Fréchet distance utility module.

- Refactor and document: enhance type annotations and docstrings across optimization and plotting modules, streamline multistart fitting logic, clarify heuristic for Das-Dennis partitions, and improve readability in initialization and constraint handling functions.

- Merge pull request #36 from bibymaths/logFC

Configuration Enhancement

- Integrate `config.toml` for centralized configuration, refactor constants handling in `tfopt` and `kinopt`, clean unused imports, and add early emphasis weights module under `models/weights.py`.

- Bring main into global; preserve phoskintime_global

- Add `config.toml` for centralized configuration in `phoskintime_global`

- Integrate Fréchet distance for goodness-of-fit evaluation in solution selection, replace weighted sum heuristic, and refactor related selection logic and imports. Enhance modularity and fix minor documentation inconsistencies.

- Add `global_model_evol.py` for global ODE dual-fit optimization model. Includes RNA/protein dynamics, numerical solvers, multi-objective optimization, and utility functions.

- Update runner: add CLI options for `kinopt` and `tfopt`, refactor model indexing to include all TF/kinase interactions, and enhance input sanity checks and debug outputs.

- Refactor runner.py: streamline initialization, enhance parallel evaluation, enable UNSGA3 termination, improve Pareto front export, and update optimization settings in config.toml.

- Enhance simulation workflow: update Jacobian function handling, implement new plotting routines for goodness-of-fit and steady-state dynamics, add functions for activity/export analysis and optimization history, and configure expanded model parameter bounds in config.

- Add residual analysis and parameter uncertainty export functions, update export pipeline, and refine kinase activity/correlation outputs.

- Remove `global_model_evol.py` as it is no longer needed for the optimization pipeline.

- Introduce zoom-in refinement strategy, improve fold-change coupling models, add `reconstruct_global_gof.py`, and update config default model.

- Refine optimization termination: integrate `DefaultMultiObjectiveTermination` with tightened tolerances and update pipeline accordingly.

- Refactor codebase for improved readability by fixing indentation inconsistencies, adhering to PEP 8 style guidelines, and improving code clarity without altering functionality.

- Migrate and refactor `phoskintime_global` to `global_model`, update configuration and file structure, integrate new CLI features, and improve optimization pipeline initialization and modularity.

- Replace `print` statements with `logger.info` across the codebase, integrate `setup_logger` for unified logging, and improve thread management by setting environment variables for parallelized operations.

- Refactor runner and network logic: simplify log formatting, improve initial condition data handling with manual injection for edge cases, update error handling, and modify output directory in config.

- Refactor initial condition handling: simplify logic in `runner.py` by removing redundant error handling, improve configuration parsing in `utils.py`, adjust refinement bounds logging in `refine.py`, update configuration defaults in `config.toml`, and enhance modularity across related files.

- Enhance configuration and logging: add `num_refine` parameter to `config.py` and `config.toml`, update output directory name, replace print statements with logger calls in `network.py` and `runner.py`, and implement iterative refinement in `refine.py`.

- Enhance refinement and CLI functionality: add `get_parameter_labels` in `utils.py`, implement parameter-bound refinement reporting in `refine.py`, introduce `clean` command for cache removal in `cli.py`, and update `runner.py` to utilize new utility.

- Update configuration and logging: change default model to `combinatorial`, adjust output directory name in `config.toml`, enhance logging formats in `refine.py`, and improve `setup_logger` with dynamic module-based naming.

- Add global model simulation UI: introduce `compare_mechanisms.py` for knockout analysis, update `config.toml` to set `distributive` as default model.

- Introduce advanced mechanistic discovery and temporal sensitivity analyses: add `temporal_sensitivity.py` and `mechanistic_insights.py` with Sobol-based GSA, multilayer network analysis, and dynamic sensitivity visualization; update `config.toml` to set `combinatorial` as the default model.

- Add `export_subnetworks.py` script: extract subnetworks based on shared nodes from input data, generate node-specific CSVs and visualizations, and optionally create zipped output archives.

- Update logging, refine configuration handling, and add `find_protein_accumulators.py`: simplify lambda scan logging, fix `lossfn.py` header, adjust loss configuration in `utils.py`, increase evaluation limits in `refine.py`, and introduce script for identifying key protein accumulators.

- Improve network handling, refine optimization behavior, and enhance logging:
- Add topological sensitivity and signaling velocity considerations in `calculate_bio_bounds`.
- Enhance `build_tf_matrix` with orphan TF redirection logic and support for kinase-driven activity.
- Improve `Index` logic with orphan TF proxy mapping and optimized network initialization.
- Introduce detailed Fréchet distance breakdowns and solution logging.
- Optimize logging with dynamic driver mapping and robustness improvements.
- Modify `init_raw_params` to support custom bounds with detailed analysis in `global_model/params.py`.
- Replace outdated debugging blocks and increase default evaluation limits.
- Refine export functions for clearer outputs and logging in `runner.py`.

- - Refine optimization parameters and bounds: enable custom bounds in `init_raw_params`, adjust steady-state simulation duration, and raise degradation rate floors for proteins and mRNA.
- Update config: switch to ABL2-specific inputs, reduce generations, increase refinements, adjust lambda weights, and set `combinatorial` as the default model.
- Enhance stability and logging: clamp tf sensitivity bounds, improve logging formats, and finalize steady-state checks.

- Add hyperparameter tuning and sensitivity analysis modules, refine configuration, and enhance logging:
- Introduce `scan.py` for Optuna-based hyperparameter tuning with visualization and dashboard integration.
- Add `sensitivity.py` for Morris sensitivity analysis, extending parameter exploration.
- Update `config.toml`: adjust default model to `saturation`, revise optimization settings, and streamline lambda handling.
- Enhance logging for model selection, results export, and optimization processes.
- Refine `runner.py` to support hyperparameter scanning and sensitivity analysis.

- Add native Optuna support for multi-objective optimization:
- Introduce `optuna_solver.py` with MOTPE integration, persistent storage, and live dashboard.
- Enhance exception handling and logging in `scan.py` to improve trial robustness and reporting.
- Update `runner.py` to support solver selection (Pymoo or Optuna) and new hyperparameter scan logic.
- Modify `config.toml`: adjust default inputs, output directory, and solver configurations.
- Refactor external dependencies and restructure optimization workflow for clearer modularity and error handling.

- Refactor data preprocessing and runner logic; add new scaling utilities and streamline solver handling:
- Add `process_and_scale_raw_data` function to `utils.py` for preprocessing and scaling time-series data.
- Update `runner.py` to improve argument defaults, enhance solver logic, and conditionally export convergence history.
- Modify RNA and MS data handling in `io.py` to integrate the new scaling utilities and tidy raw datasets.
- Adjust sensitivity analysis visualization and improve its usability.
- Refactor `scan.py` and optimization parameters for enhanced consistency and clarity.

- Increase Optuna trials in `scan.py` and expand data handling in `sensitivity.py` for improved robustness and flexibility.

- Refactor `runner.py` and `utils.py` to streamline configuration, logging, and weight function handling:
- Add `build_weight_functions` for modular weight generation in `optproblem.py`.
- Update `runner.py` to integrate advanced logging, global configurations, and steady-state simulation enhancements.
- Simplify TOML configuration parsing in `load_config_toml` and expand metadata handling.
- Replace redundant debugging blocks with scalable utilities.

- Integrate dashboard visualization, streamline solver logic, and refine optimization configuration:
- Add `dashboard_app.py` for live visualization of results, including Pareto fronts, time-series predictions, and parameter summaries.
- Introduce `dashboard.py` to save and load dashboard-friendly data bundles.
- Refactor `runner.py` to log solver and model configuration details, integrate dashboard bundle export, and improve logging.
- Reduce Optuna trials in `scan.py` for quicker iterations.
- Switch default optimizer to `pymoo` in `config.toml`.

- Improve parameter logging in `runner.py` and enable refinements and sensitivity analysis in `config.toml`:
- Refactor parameter logging by categorizing outputs (kinase activities, protein-specific rates, phosphatase rates, and global scalars) for readability.
- Update `config.toml` to enable second-pass refinements (`refine=True` with 3 refinements) and activate sensitivity analysis (`sensitivity_analysis=True`).

- Enable parallelization in hyperparameter scan, update default configurations, and fix RNA dataframe handling:
- Add multi-core support (`--cores`) and progress bar to Optuna in `scan.py`.
- Refactor parallel runner logic in `runner.py` for enhanced flexibility.
- Update `config.toml` with modified paths, increased cores, and reduced generations.
- Fix incorrect column reference in RNA dataframe export from `sensitivity.py`.

- Increase default core count to 80 in `config.toml` for improved parallel processing.

- Minor formatting fixes across multiple files and adjust optimization settings in `config.toml`:
- Fix indentation in function definitions for consistent style.
- Ensure newlines at EOF for proper file formatting.
- Update `config.toml`: change output directory, modify generations and population size, and disable refinement passes.

- Refactor TF and protein network handling, enhance observability filtering logic in `runner.py`, and streamline configuration:

- Implement advanced TF proxying logic for orphan handling and network matrix construction.
- Refactor filtering logic to restrict observations to model proteins, with detailed diagnostics.
- Add `get_optimized_sets` in `utils.py` for exporting optimized protein, site, and kinase sets.
- Update imports for `dashboard_bundle` and `load_config_toml` across modules.
- Modify `config.toml`: update paths, adjust settings, and disable sensitivity analysis and hyperparameter scanning.

- Add robust sigma calculation, CI bands, and flagged point annotations in `export.py`. Update runner and model logic with stricter filtering, improved initial condition dumping, and new IVP implementations. Adjust optimization weights and configurations in `config.toml`.

- Refactor `export.py` to modularize Goodness-of-Fit plotting with enhanced annotation logic, per-modality handling, and robust CI band visualization. Add steady-state summary generation in `analysis.py` and model-specific parameter adjustments in `utils.py` for enhanced configurability. Adjust plotting styles for improved clarity.

- Refactor logging for consistent use of `RESULTS_DIR` across modules, enable sensitivity analysis in `config.toml`, and enhance kinase activity vs phosphorylation drive analysis with labeled scatter plots and saved summaries.

- Update `config.toml` paths, sensitivity analysis settings, and default model configuration. Refactor sensitivity plotting logic in `sensitivity.py` to streamline trajectory sampling, add top-curve export, and enhance perturbation cloud plots with RNA, protein, and phospho modalities.

- Simplify perturbation cloud title in sensitivity analysis plot.

- Refactor `compare_mechanisms.py` with enhanced TF proxying, phospho filtering, and observation restriction. Modularize analysis functions, update sensitivity calculations, and fix KO simulation logic for phospho dynamics visualization.

- Refactor `compare_mechanisms.py` to improve phospho and RNA visualization, streamline dynamics panels, and modularize graphing logic with network preparation for KO/WT comparisons.

- Remove and refactor:

- Delete `scripts/reconstruct_global_gof.py`, consolidating Goodness-of-Fit plotting workflows into other modules.
- Improve docstrings, exception handling, and argument parsing in `curve_similarity.py` for clarity and robustness.
- Update `temporal_sensitivity.py` with modularized function refactoring, enhanced GSA logic, and parallelization safety improvements.
- Standardize utility function imports in `utils.py` by exposing `_normcols` and `_find_col` as `normcols` and `find_col`.
- Enhance `export_subnetworks.py` with detailed docstrings and improved save/visualization logic for subnetwork approaches.

- Modularize dashboard functionality and improve visualization workflows:

- Add `run_dashboard.py` for streamlined dashboard execution.
- Enhance `dashboard_app.py` with result browsing, quick gallery, and improved protein/phosphosite filtering.
- Refactor `save_dashboard_bundle` to remove unused variables, save Pareto data, and improve robustness.
- Update documentation across dashboard modules for clarity and usability.

- Refactor `compare_mechanisms.py`:

- Enhance transcription factor (TF) handling with improved proxying logic and detailed filtering for matrix construction and observation restriction.
- Update visualization layout by standardizing docstrings, improving line alignment, and enhancing dynamic display logic for RNA, protein, and phospho simulations.
- Modularize functional influence mapping with depth-based edge propagation and cascade visualization for KO/WT comparisons.
- Improve docstring consistency and parameter documentation across helper functions.

- Enhance `export.py` and related modules:

- Add detailed docstrings across functions to improve clarity and explain functionality.
- Improve argument validation for plotting and exporting functions.
- Modularize Goodness-of-Fit logic, residuals plotting, and phosphorylation drive visualization.
- Introduce features like CI bands, flagged points, and robust sigma calculations.
- Simplify and refine parallel coordinate and Pareto front visualizations.
- Update `buildmat.py`, `network.py`, and `lossfn.py` with enhanced docstrings, better exception handling, and modularized logic for matrix construction and loss calculations.

- Update `config.toml` paths, refine model defaults, and introduce enhanced loss options.

- Update data input paths and output directory in `config.toml` for clearer structure and consistent environment setup.
- Change default model to "distributive" and enable new loss settings with extended options for robustness.
- Add and improve docstrings across modules (`refine.py`, `simulate.py`, etc.) for better clarity and usability.
- Modularize `run_iterative_refinement` and `simulate_and_measure` logic for enhanced functionality and flexibility.
- Introduce `scan.py` for hyperparameter tuning with Optuna + Pymoo, including robust Bayesian optimization and live dashboards.
- Add detailed comments and helper functions across all edited modules to improve maintainability and reduce redundant code.

- Add `tfopt_network_viz.py` and `kinopt_network_readout.py`:

- Introduce `tfopt_network_viz.py` for minimal, publication-ready visualizations of TFopt network analysis, including load bars, knockout effects, and network plots.
- Add `kinopt_network_readout.py` to implement mechanistic signal flow analysis for Kinopt data, including latent kinase activities, target decompositions, and in-silico knockout sensitivity.
- Include comprehensive docstrings, configurable parameters, and helper methods for clarity and adaptability.
- Ensure modularity and reproducibility across scripts with efficient file I/O and visualization workflows.

- Add `make_kinopt_diagram.py` for schematic generation:

- Introduce `make_kinopt_diagram.py` to generate publication-ready kinase-optimization and transcription-factor optimization schematics using DOT/Graphviz.
- Include functions for visualizing individual `Kinopt/TFopt` modules and an integrated global diagram.
- Provide comprehensive docstrings, modular configuration options, and robust error handling for visualization workflows.
- Utilize PyGraphviz/PyDot for rendering, with support for customization of layouts, labels, and aesthetics.

- Refactor scripts to improve style, readability, and constraints visualization:

- Standardize inline comments, whitespace alignment, and indentation across `tfopt_network_viz.py`, `kinopt_network_viz.py`, and `kinopt_network_readout.py`.
- Add functions for generating TFopt/Kinopt constraints diagrams with DOT/Graphviz.
- Extend visualization workflows to include global constraints representations and per-module configurations.
- Enhance robust error handling for DOT rendering using PyGraphviz and PyDot.
- Streamline main logic in `make_kinopt_diagram.py` for consistent constraints and non-constraints diagram generation.

- Add `tfopt.py` for streamlit-based network visualization and analysis:

- Introduce a unified Streamlit dashboard for TFopt network readout, visualization, and rendering.
- Incorporate dynamic in-memory analysis, Plotly-based visualizations, and Gravis network rendering.
- Simplify workflows by centralizing data loading and eliminating intermediate files.
- Add detailed docstrings and modular configuration options for enhanced usability and flexibility.

- Reduce the number of transcription factors (TFs) in `make_kinopt_diagram.py` from 4 to 2 for simplified constraints visualization.

- Update `kinopt.py` visualization:

- Rename legend label from "C" to "K".
- Change y-axis title from "Signal (arb.)" to "Fold Changes".

- Add kinase-target heatmap visualization to `kinopt.py`:

- Introduce `build_ko_heatmap_matrix` and `fig_ko_heatmap` functions for kinase-target site effects visualization.
- Add corresponding Streamlit UI for customizing heatmap parameters (e.g., top edges, max targets/kinases).
- Update all Plotly visualizations to use a consistent high-resolution export configuration (`PLOTLY_CONFIG_HIRES`).
- Include CSV download option for heatmap data.

- Convert docstring to raw format in `global_model/models.py` for `sequential_rhs` function.

- Replace old static images, update README with new logo and visuals:

- Removed outdated `.svg` and `.png` files (`dg1.svg`, `goal_2.png`, `logo_1.png`, etc.).
- Added new logo (`phoskintime_logo.svg`) and supporting visuals (`phoskintime_problem.png`, `phoskintime_analysis.png`) for improved branding and user understanding.
- Enhanced `README.md` with a centered header, new logo, problem overview, updated feature highlights, and interactive formatting.

- Update README with Zenodo DOI badge and enhanced feature highlights:

- Added DOI badge for publication reference.
- Introduced detailed feature description, upcoming updates, and improved interactive formatting for better clarity and engagement.

- Add thermal model extension simulation testing.

- Compare baseline with 37 and 42 temperature of protein folding.

- Thermodynamic constraint setup

- Updated thermal estimation and simulation prototype.

- Add `.idea/` to `.gitignore` and include interpretation notes for global phosphorylation-transcription model.

- Apply review comments: fix outlier logic, docstrings, imports, LaTeX, cache_data side effects

Agent-Logs-Url: https://github.com/bibymaths/phoskintime/sessions/901d3738-f641-4127-9b89-83bd2cbbe5b6

Co-authored-by: bibymaths <42838835+bibymaths@users.noreply.github.com>

- Fix audit findings: packaging, docs, changelog, CLI, logconf notes

Agent-Logs-Url: https://github.com/bibymaths/phoskintime/sessions/9d5995ac-7291-443c-830d-3986971c1a8e

Co-authored-by: bibymaths <42838835+bibymaths@users.noreply.github.com>

- Clarify test functions missing test_ prefix in testing docs

Agent-Logs-Url: https://github.com/bibymaths/phoskintime/sessions/9d5995ac-7291-443c-830d-3986971c1a8e

Co-authored-by: bibymaths <42838835+bibymaths@users.noreply.github.com>

- Fix review feedback: docs accuracy and missing dependencies

Agent-Logs-Url: https://github.com/bibymaths/phoskintime/sessions/66aa4994-603c-4976-82a2-44e5c1a04d4e

Co-authored-by: bibymaths <42838835+bibymaths@users.noreply.github.com>

- Merge pull request #45 from bibymaths/copilot/fix-audit-docs-packaging-global

Fix audit findings for global workflow docs and packaging

- Move all modules into the `protwise` namespace [https://github.com/bibymaths/phoskintime/issues/43]

- Refactor imports across files to use the `protwise` package.
- Adjust paths for consistency with the new project structure.
- Fix outdated references (e.g., `frechet` → `common.frechet` and `bin` → `protwise.runner`).

- //github.com/bibymaths/phoskintime/issues/43] Migrate `global_model` to `networkmodel` namespace and update imports and paths project-wide. Adjust asset paths, rename files, and update documentation for consistency.

- //github.com/bibymaths/phoskintime/issues/43] Remove `poetry.lock` file to decouple package dependency management.

- //github.com/bibymaths/phoskintime/issues/43] Remove unused dependencies and add new package entries in `pixi.lock` file.

- Merge pull request #46 from bibymaths/copilot/add-missing-community-files

chore: add community, governance, citation, and repository-maintenance files

- Add CODEOWNERS entry [https://github.com/Normann-BPh] for `tfopt/local/objfn/minfn.py`.

- Add `git-cliff` as a dependency and configure changelog tasks in `pixi.toml`.

- Update `CHANGELOG.md`: Reformat structure and document notable project changes.

- Merge pull request #55 from bibymaths/codex/create-phoskintime-multi-modal-input-audit

docs: add PhosKinTime multimodal input audit

- Refactor PhosKinTime JAXopt Diffrax multimodal fitting

- Strengthen PhosKinTime JAX Diffrax refactor checks

- Add JAX inference diagnostics for PhosKinTime

- Merge pull request #56 from bibymaths/codex/implement-a-to-z-refactor-as-per-audits

Refactor PhosKinTime networkmodel/protwise to JAX/JAXopt + Diffrax with multimodal support

- Fix local ODE parameter optimization setup

- Fix local ODE parameter optimization setup

### Motivation
- Prevent optimizer collapse to near-zero / flat trajectories by removing mixed transformed/physical parameter semantics and silent bound resizing which misaligned params and bounds.
- Ensure a single source of truth for parameter counts/names so the optimizer, ODE RHS, and I/O agree on vector length and ordering.
- Stabilize loss scaling and regularization so the data fit (not the regularizer or unit mismatches) drives the solution.
- Add diagnostics and a minimal reproducible example to detect and debug lower‑bound parking and flat fits early.

### Description
- Replaced ad-hoc mixed transforms with an explicit physical-space optimizer mapping and helpers in `protwise/paramest/normest.py` by adding `to_opt_space` / `from_opt_space`, removing the special-case softplus/log branching, and using the same coordinates the ODE receives (addresses parameter transformations and optimizer-space mismatch). (protwise/paramest/normest.py)
- Removed all uses of silent `np.resize` bound expansion and now expand dict bounds deterministically in parameter-name order using `get_param_names()` and `get_num_params()` so bounds length and param ordering always match the ODE parameterization; added validation to raise on mismatches. (protwise/paramest/normest.py, config/constants.py)
- Reworked local-model solving to use model-specific Diffrax RHS functions via `protwise/models/diffrax_solver.py` (implemented `_dist_rhs`, `_succ_rhs`, `_rand_rhs`, and `make_local_model_rhs`) and made `solve_protwise_ode()` validate expected parameter count before calling Diffrax; `protwise.models.__init__` now normalizes model aliases when importing modules. This ensures the ODE uses the same parameter layout as `get_num_params`. (protwise/models/diffrax_solver.py, protwise/models/__init__.py)
- Normalized layer losses by a data-scale heuristic `_loss_scale(...)` so each observed layer (mRNA/protein/phospho) contributes proportionally to the objective, and reduced regularization weight to `1e-6` to discourage extreme rates without making low-magnitude/flat dynamics attractive. (protwise/paramest/normest.py)
- Added optimizer diagnostics: initial objective, objective at a scaled initial guess, final objective, optimizer iteration count, and fraction of parameters at their lower bounds to make collapse-to-lower-bound behavior visible in logs. Also added gradient-norm logging in `networkmodel/jax_backend.py` so convergence diagnostics include gradient magnitude. (protwise/paramest/normest.py, networkmodel/jax_backend.py)
- Added a small reproducible example `examples/protwise_optimization_sanity.py` which simulates data from known positive parameters and runs `normest` to validate final loss < initial loss, parameters not collapsed to zero, and non-zero trajectory variance. (examples/protwise_optimization_sanity.py)
- Added focused tests `tests/test_protwise_parameterization.py` asserting parameter name/count synchronization, dict-bound expansion order, and that optimizer-space mapping preserves physical-space values. (tests/test_protwise_parameterization.py)
- Minor README update documenting single-source parameter metadata and that JAXopt now optimizes in physical space. (protwise/paramest/README.md)

Files changed (high-level):
- protwise/paramest/normest.py — unify opt/physical spaces, deterministic bounds expansion, loss scaling, logging, reduced regularization.
- protwise/models/diffrax_solver.py — implement model-specific Diffrax RHSs and parameter-count validation.
- protwise/models/__init__.py — normalize model aliases when importing.
- networkmodel/jax_backend.py — log gradient norm at optimizer exit for diagnostics.
- examples/protwise_optimization_sanity.py — synthetic single-gene sanity script.
- tests/test_protwise_parameterization.py — new unit tests for parameterization and bounds expansion.
- protwise/paramest/README.md — short doc update.

### Testing
- Compiled the modified modules successfully: `python -m py_compile config/constants.py networkmodel/jax_backend.py protwise/paramest/normest.py protwise/models/diffrax_solver.py protwise/models/__init__.py examples/protwise_optimization_sanity.py tests/test_protwise_parameterization.py` (succeeded in this environment).
- Ran unit test collection locally for the new focused tests: `PYTHONPATH=. pytest -q tests/test_protwise_parameterization.py` (test collection/execution failed in this environment because required runtime deps such as `numpy` are not installed here; the tests themselves assert parameter-count/name consistency and bounds expansion order and should pass in a fully provisioned environment).
- Performed local smoke checks during development (import/compile and basic function calls) and added explicit runtime logs for initial objective, scaled initial objective, final value, iteration count, and gradient norm to help reproduce and debug remaining cases.

Notes on reproducing the sanity check (after installing dependencies):
- Run the synthetic example for protwise:
  - `PYTHONPATH=. python examples/protwise_optimization_sanity.py --model protwise`
  - Expected qualitative outputs: `not_collapsed=True`, `trajectory_variance` > 0, final residual smaller than initial, estimated parameters positive and not all at lower bounds.
- Run the same for randmod to validate the random-model parameterization:
  - `PYTHONPATH=. python examples/protwise_optimization_sanity.py --model randmod`

These changes synchronize parameter counts/names, ensure bounds map to parameters in the correct order (no silent resizing), run the optimizer in physical space (avoiding half-transformed semantics that compressed midpoints toward zero), normalize losses by data scale, and reduce the regularizer — collectively preventing the optimizer from being rewarded for trivially small/flat trajectories.

- Centralize Diffrax protwise solver, canonicalize randmod subsets, and refactor optimization/bounds

### Motivation
- Centralize and modernize the local-protwise ODE solver interface to share a single Diffrax backend across all local mechanisms and avoid inconsistent behavior when modules are called directly.
- Canonicalize randmod subset ordering and provide bitmask utilities so parameter/ state ordering is consistent with labels and legacy behavior.
- Simplify and harden parameter bounds and optimizer space handling so projected-gradient (`jaxopt.ProjectedGradient`) operates directly in physical parameter space with robust bound normalization.
- Improve diagnostics and add tests/examples to validate parameterization, aggregation, and wrapper dispatch.

### Description
- Added `config.helpers.generate_randmod_subsets` and `randmod_subset_masks` and updated randmod-related naming/label/bounds generation to use these canonical subsets.
- Introduced a centralized Diffrax local-model solver in `protwise/models/diffrax_solver.py` with per-mechanism RHS implementations (`_dist_rhs`, `_succ_rhs`, `_rand_rhs`) and `make_local_model_rhs`/`solve_protwise_ode` dispatching by canonical model name.
- Converted mechanism modules to lightweight wrappers (`protwise/models/protwise.py`, and updated `distmod.py`, `randmod.py`, `succmod.py`) that call the centralized solver with an explicit model name.
- Refactored `protwise.paramest.normest` to: normalize bounds by parameter names, map optimizer space to physical space (`to_opt_space`/`from_opt_space`), aggregate randmod subset states into site-level observations, scale loss terms by data magnitude, and apply a small regularization weight; added defensive checks and richer logging.
- Minor improvements in `networkmodel.jax_backend` logging to include gradient norm diagnostics and a small example script `examples/protwise_optimization_sanity.py` plus unit tests `tests/test_protwise_parameterization.py` validating naming, bounds expansion, aggregation, opt-space identity, subset masks, and wrapper dispatch.

### Testing
- Ran unit tests in `tests/test_protwise_parameterization.py` with `pytest` and all tests passed.
- Executed the new example scenario for local fitting (`examples/protwise_optimization_sanity.py`) locally as a sanity check (simulates and fits synthetic trajectories) and observed expected diagnostics.

- Refactor protwise local models: randmod subset helpers, Diffrax integration, bounds/optimization overhaul, and tests

### Motivation
- Centralize and canonicalize randmod subset ordering and masks so parameter/state naming, bounds, and aggregation are consistent across code paths.
- Unify local ODE RHS implementations and expose a single Diffrax-backed solver entrypoint that validates parameter counts and supports multiple mechanism names.
- Simplify the optimizer parameterization to operate in physical space with projected box constraints and make bounds handling explicit and safe.
- Improve diagnostics and add an integration-style example plus unit tests to lock down the new behavior.

### Description
- Introduced `generate_randmod_subsets` and `randmod_subset_masks` in `config.helpers` and updated `get_param_names_rand`, `generate_labels_rand`, and `get_bounds_rand` to use the canonical subset ordering.
- Reworked `protwise.models.diffrax_solver` to provide mechanism-specific RHS implementations (`_dist_rhs`, `_succ_rhs`, `_rand_rhs`), `make_local_model_rhs`, `aggregate_randmod_site_phospho`, and a validated `solve_protwise_ode` that enforces expected parameter counts and returns site-aggregated phospho for `randmod`.
- Added per-mechanism wrappers (`protwise`, `distmod`, `succmod`, `randmod`) to pass an explicit `model_name` into the centralized solver so direct module calls are unambiguous.
- Overhauled `protwise.paramest.normest` to normalize model names, replace the previous softplus/log mix with identity optimizer space via `to_opt_space`/`from_opt_space`, implement robust `_normalize_bounds` that expands dict bounds in parameter order, add loss scaling (`_loss_scale`), and small regularization via `REGULARIZATION_WEIGHT`.
- Improved optimizer diagnostics in `networkmodel.jax_backend.optimize_scalar_objective` by logging a gradient norm (best-effort) alongside iterations and final objective.
- Updated randmod steady-state initializer to use canonical subsets and added an example script `examples/protwise_optimization_sanity.py` exercising local-model fitting.
- Added comprehensive unit tests in `tests/test_protwise_parameterization.py` covering naming/counts, bounds expansion, optimizer-space identity, randmod subset/mask canonical order, aggregation consistency, and wrapper dispatch behavior.

### Testing
- Ran the new unit tests with `pytest tests/test_protwise_parameterization.py` and they passed.
- Ran the full test suite with `pytest -q` after the changes and all tests, including the new protwise parameterization tests, succeeded.
- Executed the example sanity script `python examples/protwise_optimization_sanity.py --model randmod` locally to validate end-to-end fit behavior and observed expected regression-style checks (used as a manual integration check).

- Refactor local protwise solvers, randmod subset ordering, bounds/optimization mapping, and add unit tests

### Motivation

- Provide a single, centralized Diffrax-backed solver for local protein-wise mechanisms and ensure each mechanism uses the correct RHS and naming conventions.
- Make randmod subset ordering explicit and reproducible so parameter/label ordering and bitmask mappings match the JAX objective and legacy expectations.
- Simplify and harden parameter bounds handling and optimizer coordinate mapping by running projected optimization directly in physical parameter space with stronger validation.
- Improve diagnostics (optimizer gradient norm logging) and add tests/examples to validate the new behavior.

### Description

- Add canonical randmod helpers `generate_randmod_subsets` and `randmod_subset_masks` and use them in parameter name/label generation and bounds expansion (`config/helpers/__init__.py`).
- Implement a unified local solver in `protwise/models/diffrax_solver.py` with `_dist_rhs`, `_succ_rhs`, `_rand_rhs`, `make_local_model_rhs`, `solve_protwise_ode`, and aggregation helpers `aggregate_randmod_site_phospho`, plus canonical model name normalisation.
- Update per-mechanism modules and imports so direct calls pass an explicit `model_name` to the centralized solver and dynamic imports use a normalized model name (`protwise/models/*`, `protwise/models/__init__.py`, `protwise/steady/__init__.py`).
- Refactor parameter estimation in `protwise/paramest/normest.py` to: use a canonical model name, operate optimizer in physical parameter space with `to_opt_space`/`from_opt_space`, provide `_normalize_bounds` that expands dict bounds in parameter-name order and validates infinities, add loss scaling and a small regularization weight, and wire the local RHS into the JAX objective.
- Add JAXopt diagnostic logging of gradient norm in `networkmodel/jax_backend.py`.
- Add an example sanity script `examples/protwise_optimization_sanity.py` and comprehensive unit tests `tests/test_protwise_parameterization.py` validating param-name/count parity, bounds expansion, randmod aggregation/masking, normalization behavior, and wrapper dispatch.
- Minor docs update in `protwise/paramest/README.md` to reflect the changed optimizer/bounds behavior.

### Testing

- Ran the new unit tests with `pytest tests/test_protwise_parameterization.py`, and all tests passed.
- Exercised the randmod aggregation and normalization behavior via the tests that simulate raw solver output and verify site-level aggregation and normalization succeeded.

- Merge pull request #57 from bibymaths/codex/fix-optimization-issues-causing-parameter-collapse

Fix local ODE parameter optimization setup

- Fix global networkmodel phospho objective mapping

- Merge pull request #58 from bibymaths/codex/fix-global-model-phospho-mapping-and-loss

Fix global networkmodel phospho objective mapping

- Fix networkmodel state offsets and parameter packing

- Restore JAX combinatorial phosphorylation transitions

- Fix networkmodel prior for raw theta objective

- Merge pull request #59 from bibymaths/codex/fix-multiple-bugs-in-networkmodel-path

Fix per-protein state indexing and optimizer parameter packing in networkmodel

- Network model optimization fixed. correctly working.

- Remove deadcode - scipy/pymoo custom and odeint functions

- Restore few functions to not break anything in network model from jacspeedup.py and steadystate.py and utils.py

- Wire networkmodel inference config

- Wire networkmodel inference config

### Motivation
- Remove dead evolutionary/protwise/legacy ODE backend configuration and expose inference/post-optimization controls so the JAXopt+Diffrax networkmodel path can use `inference.py` features via `config.toml` and the existing config loader.
- Integrate multistart/profile-likelihood/posterior-sampling flows into `runner.py` while keeping `jax_backend.py` and `inference.py` unchanged and without adding new CLI arguments.

### Description
- Removed legacy solver/evolutionary/protwise config keys from `config.toml` and added a clearly labelled `# INFERENCE / POST-OPTIMIZATION ANALYSES` section with `n_starts`, `profile_likelihood`, `profile_indices`, `profile_grid_size`, `posterior_sampling`, `posterior_num_warmup`, and `posterior_num_samples`.
- Updated the config loader (`config_loader.py`)/`PhosKinConfig` to stop exporting legacy fields (e.g. `population_size`, `use_custom_solver`, `optimizer`, Optuna/Pymoo knobs and refinement fields) and to parse the new inference/profile/posterior fields using the existing `getattr`/conversion patterns.
- Modified `networkmodel/config.py` to remove constants tied only to legacy backends and to export the new constants `N_STARTS`, `PROFILE_LIKELIHOOD`, `PROFILE_INDICES`, `PROFILE_GRID_SIZE`, `POSTERIOR_SAMPLING`, `POSTERIOR_NUM_WARMUP`, and `POSTERIOR_NUM_SAMPLES`.
- Wired `networkmodel/runner.py` to import `InferenceContext`, `run_multistart`, `run_profile_likelihood`, `run_numpyro_posterior`, and `configure_jax_parallelism` from `networkmodel.inference`; replaced the single-call `problem.solve(...)` with an `InferenceContext`-driven multistart branch (when `N_STARTS>1`) and added optional post-optimization blocks that call profile-likelihood and NumPyro posterior sampling; removed legacy CLI knobs `--pop` and `--refine` and set the solver default to `"jaxopt"` so default behavior remains the same when the new fields are at defaults.

### Testing
- Compiled modified modules with `python -m py_compile config_loader.py networkmodel/config.py networkmodel/runner.py` and the compilation succeeded.
- Verified `load_config_toml('config.toml')` returns the new inference defaults and that the removed legacy fields are not present, and asserted the new constants import from `networkmodel.config` (smoke tests succeeded).
- Performed a smoke import of `networkmodel.runner` to ensure there are no import-time errors (succeeded).
- Added and ran `PYTHONPATH=. pytest -q tests/test_networkmodel_inference_config.py` which passed (tests confirming defaults, removed legacy exports, and runner wiring).
- Ran a subset of the existing integration tests (`tests/test_phoskintime_jax_multimodal.py` and `tests/test_protwise_parameterization.py`) with `PYTHONPATH=.`, which executed but reported 4 existing test failures unrelated to the config wiring (one `GlobalODEScalarObjective` slice/bounds mismatch and three `pytest.approx` nested-list compatibility issues) while many other tests passed; these failures pre-existed and were not introduced by the inference wiring.

- Merge pull request #60 from bibymaths/codex/wire-inference.py-into-runner.py-with-config-audit

Wire networkmodel inference config

- Remove dead configurations

- Formatting the code

- Merge branch 'missing-data-scenarios' into codex/audit-and-rewrite-documentation-in-networkmodel-jz2t7v

- Merge pull request #62 from bibymaths/codex/audit-and-rewrite-documentation-in-networkmodel-jz2t7v

Migrate networkmodel scalar path to JAX/Diffrax/JAXopt and update docs/CLI

- Merge remote-tracking branch 'origin/missing-data-scenarios' into missing-data-scenarios

# Conflicts:
#	networkmodel/README.md
#	networkmodel/inference.py

- Formatting the code

- Modules names changed in networkmodel

- Added param names to posterior inference, increased image quality, fixed non negative parameters

- Added speedup for posterior via numpyro - testing TBD

- Renamed module names

- Renamed module names

- Figures dpi set to 300

- Update changelog and documentation

- Update README.md

- Add notebook readiness tests across modules

- Revert KinOpt and TFOpt notebook refactor

- Validate networkmodel slice layouts exactly

- Merge pull request #63 from bibymaths/codex/fix-failing-tests-and-increase-coverage

Introduce notebook-safe projected finite-difference optimizers, lightweight LinearConstraint, tests and docs updates

- Added coverage badge, fixed slice bounds tests, removed old pypi README.md

- Dead scripts for thermal extension

- Add executable educational notebooks

- Add SALib to CI test dependencies

- Merge pull request #64 from bibymaths/codex/create-educational-jupyter-notebooks-for-modules

Add executable educational notebooks and CI/tests for KinOpt, TFOpt, protwise, and networkmodel

- Pixi and README.md

- Merge pull request #65 from bibymaths/missing-data-scenarios

Missing data scenarios

- Frontend plan

- Frontend plan renamed

- Standardize dashboard-ready output contract

- Add dashboard result browser

- Add dashboard CLI launcher

- Add dashboard upload and configuration UI

- Integrate workflow-specific dashboard panels

- Add dashboard tests and documentation

- Fix networkmodel custom config handling

- Fix dashboard input extension validation

- Fix ProtWise custom config handling

- Add dashboard user, developer, and troubleshooting guides

- Fix dashboard result discovery for reports and child runs

- Restrict dashboard config inputs to TOML

- Serialize run metadata with JSON-safe arrays

- Fix advanced analysis dashboard commands

- Merge pull request #67 from bibymaths/codex/standardize-dashboard-ready-output-contract

Standardize dashboard-ready output contract

- Add backward alias for old pymooo results to display in legacy mode and added imageio and gravis libs, module levelling for scripts

- Density plots for posterior with gaussian smoothing

- Fix forward phosphosite ODE panel

- Merge pull request #68 from bibymaths/codex/fix-phosphosite-state-dynamics-panel

Fix forward phosphosite ODE panel

- Memory issue fixing

- Make combinatorial model memory safe

- Fix combinatorial S-rate export cache shape

- Merge pull request #69 from bibymaths/codex/implement-memory-safe-fixes-for-combinatorial-model

Make combinatorial model memory safe

- Made a project presentation using the sample results.



### Tests


- Testing network model - distributive - fixing errors, and bounds args passing issue

- Testing network model with posterior sampling, added logger and progress bar to MCMC NUTS

- Testing network model with posterior sampling, removed history video

- Notebooks are working


## [0.4.0] - 2025-05-06



### Other


- Merge pull request #29 from bibymaths/master

Archive the devleopment in master - updated readme

- Merge pull request #30 from bibymaths/master

publish 0.4.0


## [0.4.0-alpha] - 2025-05-03



### Fixed


- Fixes

- Fixes - python path in toml

- Fixes - python path in toml

- Fixes - python path in toml

- Fixes - file names change in workflow yaml

- Fixes - püath problem

- Fixes and integration from corpus

- Fixes

- Fixes and readme updates

- Fixed warnings in curve_fit, added noramlization functionality.

- Fixes in formulas in readme

- Fixed toml file

- Fixed toml file

- Fixed io test

- Fix in logger

- Fixes for data structure names, mmino  fixes in plotting - general  fixes

- Fixed preprocessing to load all mRNAs for phospho- network TFs from collectTRI - except COMPLEX formations

- Fix water fall plot

- Fixed dir creation in processing

- Fixed dir moving

- Fixed typo in plotout.py

- Fixed docs/reference.md

- Fixed docs/index.md

- Fixed .toml

- Fixes and adding support for network via cytoscape



### Other


- Add my files

- Removed logs form repo

- Updated imports, corrected logging and its working.

- Github workflow corrected, lets see

plots refined.

- Corrected python version in poetry toml file

- Corrected python version and deleted env loader

- Corrected dependecnies name

- Adding pytest

- Adding pytest fixture for coverage

- Debugging python path

- Debugging python path

- Debugging python path

- Debugging python path

- Debugging python path

- Debugging python path

- Adding status on readme

- Added coverage badge

- Removed data and updates

- Chnaged modules names

- Profiles estimation with interpolation added functionality

- Removed logger from utils

- Working on random model integration
successive and distributive done
init done
class of plotting done

- Random model integrated in seqest.py but remainnig in adapest.py

- Added optimization prior to modelling and estimation
random ODE model works fine

- Chnaged file name due to windows warning

- Linked optimization results ot ODE input file

- Added nsga 2 to optimization step

- Chnaged testing and cached data

- Corrected tables and printing for optimality of solution

- Created two objecive modules within evol for nsga and diffevol

- Worked on diffevol, only linking with plotting of opritmzation results is remainning

- Diffevol done
plotting analysis and fit analysis added- not linked.

- Added powells methods in julia

- Normal estimation added

- Added readme to every module, added plotting bar for params in plotter class, created logo, added the goal

- Added the heading in goal

- Edited top README.md

- Minor fixes

- Added readme for abopt and inside as well

- Minor spelling fix in toml

- Added normalization to compare with FC data

- Added new unit tests

- Created pytest.ini

- Added test 10 models, and 6 of them failed. fixed major warnings in paramest

- Added test model

- Minor fixes

- Added model illustrative diagram, and SE calculation along with confidence intervals using linearization approach

- Added experimental background

- Added processing script for raw files - including TF, collectTRI

- Worked on TF - optimization, looks okay.

- Corrections in equation mapping. works well.

- TF-mRNA optimization done. Proceeding to modularize into subpackage. Filtered mRNA based on interaction data from phosphorylation.

- Work on bootstrap and error landscape

- Integrated tfopt into package for SLSQP, pymoo remaining

- Updated instructions on installing system wide graph viz for "dot" binary

- Merge remote-tracking branch 'origin/master'

- Labels corrected - mRNA is being estimated via TF (Prtoein Group/ GeneID) and its Psites - okay

- Labels in main corrected

- Integrated global (nsga, etc) to tfopt - looks good

- Minor fixes.

- Minor fixes -worked on post optimization analysis

- Added fitanalysis to tfopt - minor fixes

- Preprocessing steps cleared - minor fixes and organizing

- Minor fix - path

- Deleted logo1 - unnecessary

- Updated .gitignore file

- Result file neames fixes for tfopt and kinopt and minor fixes in io utils

- Minor changes in plotting display.

- Added mapping for TF - mRNA - Psite _ Kinase.

minor change in .gitignore

- Added data processing workflow README.md

- Added a brief dependency graph.

- Update README.md

- Merge remote-tracking branch 'origin/master'

- Minor fix - priting and mapping in tfopt

- Vectorized residuals over time in tfopt now added.

- Switched to trust-constr in tfopt

- Minor edits - trying out Tikhonov regularization in tfopt loss function

- Silly mistake in residuals in kinopt - FIXED

- Matching plots to tfopt for kinopt and change in plotting colors, lines and markers

- No optimization for synthetic placeholder in kinase and no imputation - FIXED

- Added optimization gif

- Added optimization gif - Fix perma path

- Added optimization gif - Fix perma path

- Minor fix in path

- Minor fix - plot params, objfunc, method

- Workon - tfopt

- Minor fixes - testing in plotly object

- Moved starting code to legacy/

- Removed legacy code and fixed plotly graph object

- Preprocessing and data reading fixed - silly processing mistake

- Added summary stats and data scaling

- Added docstrings - config, kinopt.evol, kinopt.fitanalysis

- Work on package

- Added docstrings - kinopt.local, models and kinopt.powell

- Added docstrings - paramest

- Added docstrings - steady, sensitivity, processing

- Added docstrings - tfopt.evol, utils

- Added docstrings - tfopt.local - Completed v.0.1.0

- Added bsd v3 license, dockerfile, .toml file, Acknowldgements, code acknowledgements

- Added installing instructions for four scenarios

uv, venv, conda, etc

- Typo fix

- Added & updated unit tests

- Added & updated unit tests

- Removed two unit tests

- Update pytests files

- Update pytests files

- Added mkdocs for reference manual

- Minor updates and mapping added, file would be in data folder of root

- Added nodes, edges with labels and attributions from mapping.

- Updated gitignore

- Updated metadata, README's of tfopt and its modules

- Minor fixes

- Added CLI wrappers for entry point

- Added direct link to open file from CLI

- Display the missing kinases in output before optimization in kinopt && working on some fixes - tfopt

- Added CHANGELOG.md

- Added light grid in plotting of model

- Removed clipping of predicted expression and added deployment configuration file

- Update CHANGELOG.md with recent additions and fixes

- Add vectorized loss function flag and update default loss type

- Introduced a boolean flag `VECTORIZED_LOSS_FUNCTION` to toggle between vectorized and standard loss function implementations for performance optimization. (ToDo)
- Updated the default value of `loss_type` in command line arguments to 3 (Cauchy).
- Enhanced plotting functions for better visualization and labeling of data points.

- Skip psite values that don't start with S_, Y_, or T_ in iodata.py

- Add configuration file for PhosKinTime settings and update population size in optrun.py

- Update parameter bounds and model settings in configuration files

- Refactor logging statements and improve data filtering in main processing files

- Enhance analysis and plotting functions: add upper bound parameter, update loss type defaults, and improve legend formatting

- Delete abopt directory

- Update CHANGELOG.md: add new features, enhancements, changes, and removals

- Update mkdocs.yml: add new documentation sections and changelog link

- Update documentation paths in mkdocs.yml and CHANGELOG.md; enhance plotting function aesthetics in plotout.py

- Update .gitignore: add documentation files to ignore list

- Refactor optimization methods: change default method to DE, enhance time series data handling, and improve parallel processing

- Refactor plotting functions in postfit.py: enhance aesthetics, adjust text labels, and save figures with improved dimensions; update sheetutils.py to change Excel sheet name for estimated values

- Update cleanup.py and README.md: clarify comments, enhance data processing descriptions, and improve module structure

- Update mkdocs.yml and README.md: enhance documentation structure, add markdown extensions, and clarify usage instructions

- Update mkdocs.yml and README.md: correct comments and improve mathematical notation for clarity

- Update mkdocs.yml and add publish.yml: improve site configuration, enhance documentation features, and automate PyPI publishing

- Refactor ODE system in randmod.py: improve parameter handling, enhance clarity of equations, and optimize state transitions; update ci.py, config.py, logconf.py, and main.py for consistency and improved logging

- Update project configuration and versioning: enhance .gitignore to exclude bash scripts and environment files; update version to 0.2.0 in __init__.py and pyproject.toml; improve scatter plot edge color in plotting.py.

- Update project metadata in pyproject.toml: remove changelog entry and clarify project description

- Update logging format and configuration settings: correct CI header formatting, adjust max_workers for parallel processing, limit gene processing for testing, and enable parallel execution in ODE system

- Refactor output logging in main.py and related files: replace print statements with logger.info for better logging consistency; adjust njit decorator in minfn.py and randmod.py to disable parallel execution.

- Enable parallel execution in njit decorator for objective function in minfn.py

- Add non-psite time series to K_array and clean up commented code in construct.py; adjust docstring formatting in filter.py

- Fix LaTeX formatting in README.md: replace parentheses with dollar signs for inline math expressions

- Update configuration and data handling: modify default loss function, adjust beta value bounds, and enhance data loading and result saving processes; add new results directory for output files.

- Comment out output file copy in main.py and clean up package import in powell.jl

- Refactor package imports in powell.jl and remove commented-out code in __main__.py; add Project.toml for dependency management

- Refactor function names and improve data handling in powell.jl; update README.md to remove outdated requirements and dependencies

- Enhance threading support and update parameter handling in powell.jl: set JULIA_NUM_THREADS, change default constraint type to nonlinear, and improve beta counts data structure; add detailed documentation for calculate_estimated_series function.

- Update threading configuration and improve residuals calculation in powell.jl; enhance README.md with instructions for running the Julia script directly.

- Remove powell module references from README.md and update usage instructions

- Update .gitignore to ignore lock files and change README reference in pyproject.toml; add new PYPI_README.md for package documentation

- Refactor plotting functions in plotting.py to generate separate scatter and density plots for parameters A, B, C, and D; update estimation mode in constants.py to 'sequential' and adjust import statement in __main__.py.

- Update constants and enhance data merging and plotting functionality: change ODE_MODEL to 'distmod', update ESTIMATION_MODE to 'normal', implement merge_obs_est function in display.py, and improve plotting methods in plotting.py for better visualization of observed and estimated data.

- Update confidence interval logging, adjust parameter estimation description, and enhance plotting functions: modify header format in ci.py, update description in config.py, enable parallel processing, and improve Kullback-Divergence visualization in plotting.py.

- Update confidence interval logging format, adjust regularization parameter, and modify color palette in plotting: change header format in ci.py, update LAMBDA_REG in constants.py, and switch color palette to 'husl' in plotting.py.

- Update image path handling in display.py to use URI format for PNG files

- Update confidence interval logging format for improved readability in ci.py

- Update CHANGELOG, modify ODE_MODEL, and enhance display styles: set ODE_MODEL to 'randmod', update header format in ci.py, and improve CSS for h2 elements in display.py.

- Update log configuration width to improve log formatting in logconf.py

- Update CLI usage to remove module flag for phoskintime commands

- Refactor logging format and update model parameters: improve log readability, change ODE_MODEL to 'distmod', and switch to sequential estimation mode.

- Update estimation mode and enhance parameter plotting: switch to 'normal' estimation mode, adjust regularization parameters, and add confidence interval bar plotting functionality.

- Update estimation mode and enhance parameter plotting: switch to 'sequential' estimation mode, adjust regularization parameter, and improve confidence interval logging and plotting functionality.

- Update parameter bounds and estimation settings: adjust bounds for parameters A, B, C, D, Ssite, and Dsite to 0-20, change ODE_MODEL to 'randmod', switch ESTIMATION_MODE to 'normal', and modify regularization parameters.

- Update parameter bounds for A, B, C, D, and Dsite: change bounds from 0-20 to 0-2 for improved estimation accuracy.

- Update ODE model and regularization parameter: change ODE_MODEL to 'distmod' and increase LAMBDA_REG from 1e-3 to 1e-2 for improved estimation performance.

- Update sensitivity_analysis function signature: change parameters to accept 'data' and 'popt' for improved analysis flexibility.

- Enhance logging and code clarity: integrate logger for sensitivity analysis and streamline comments in core.py

- Implement sensitivity analysis integration: update functions to accept bounds and enable parallel execution for improved performance.

- Refactor sensitivity analysis simulation: reduce sample size and levels, streamline parameter handling, and enhance logging for improved clarity and performance.

- Update analysis function to use sampled parameters: modify parameter unpacking to utilize sampled values from Morris method for improved flexibility in sensitivity analysis.

- Refactor sensitivity analysis function: update parameter handling to include psite_labels, reduce sample size, and enhance plotting for improved clarity and visualization of results.

- Refactor sensitivity analysis: increase sample size and levels, implement tolerance bands, and select closest simulations for improved accuracy and visualization.

- Refactor sensitivity analysis: increase sample size and levels, update parameter handling, and streamline simulation for improved accuracy and performance.

- Refactor simulation selection: reduce the number of closest simulations from 10 to 5 for improved performance and clarity.

- Refactor analysis parameters: increase closest simulations from 5 to 50, adjust plot size and alpha for clarity, and update parameter bounds for improved flexibility.

- Refactor gene loading: remove restriction to the first gene for improved data processing.

- Refactor sensitivity analysis: reduce sample size and levels, adjust closest simulations from 50 to 5, and enhance plotting for improved clarity and visualization.

- Refactor plotting parameters: reduce alpha value from 0.2 to 0.1 for improved visualization clarity.

- Refactor simulation framework: disable sensitivity analysis, adjust regularization parameter, and add knockout simulation plotting functionality.

- Refactor knockout simulation: add support for multiple knockout targets and enhance plotting for wild-type and knockout comparisons.

- Refactor knockout settings and plotting: add knockout configuration, update simulation logic, and enhance visualization for wild-type and knockout comparisons.

- Refactor knockout settings and plotting: update knockout configuration for phosphorylation and enhance plotting clarity with consistent line widths.

- Refactor knockout settings and gene loading: update knockout configuration for translation and transcription, and modify gene loading to focus on a specific gene for testing.

- Refactor knockout settings and plotting: update knockout configuration for phosphorylation, adjust regularization parameter, and enhance model fit plotting with psite values.

- Refactor knockout settings and plotting: remove hardcoded knockout settings, implement dynamic knockout combinations generation, and enhance plotting with distinct markers for better visualization.

- Refactor knockout settings and analysis: update helper functions for knockout application and combination generation, enhance sensitivity analysis integration, and improve plotting with knockout results.

- Refactor analysis and display: update sensitivity analysis handling, improve knockout results formatting, and rename plotting functions for clarity.

- Refactor analysis and display: update sensitivity analysis parameters, enhance trajectory selection logic, and improve x-axis label formatting for better readability.

- Refactor analysis and plotting: update parameter space for sensitivity analysis, enhance plotting aesthetics with improved line widths and markers, and streamline x-axis label formatting for clarity.

- Refactor analysis and plotting: disable console output for sensitivity analysis, adjust DataFrame export settings, and enhance plotting aesthetics with consistent line widths and improved layout.

- Refactor constants: disable sensitivity analysis to optimize computation time during development

- Refactor configuration and main logic: enable parallel processing by restoring max_workers setting and clean up commented-out gene filtering code.

- Update utils/display.py

Co-authored-by: Copilot <175728472+Copilot@users.noreply.github.com>

- Merge remote-tracking branch 'origin/master'

- Add development mode flag and modify gene loading logic for testing

- Refactor analysis and configuration files: improve code formatting, adjust parameter bounds for testing, and add confidence interval saving functionality.

- Update regularization parameter and improve model fitting calculations: adjust LAMBDA_REG value, refine error calculation, and enhance logging for best weight selection.

- Update parameter bounds and improve parameter handling: adjust default bounds for A, B, C, D, Ssite, and Dsite, and enhance parameter vector conversion in model predictions.

- Refactor scoring and confidence interval calculations: simplify score_fit function by removing unnecessary parameters, enhance confidence interval computation with model predictions, and improve logging of metrics.

- Refactor analysis and confidence interval logging: update sensitivity analysis completion message, improve output file naming for sensitivity plots, and enhance confidence interval logging with gene context.

- Add LaTeX table and figure generation utility: implement functions to convert Excel data and PNG images into LaTeX format, and integrate into main processing workflow.

- Update LaTeX image path generation: modify image path format to organize figures by gene symbol and improve caption labeling.

- Update weight handling and sensitivity analysis configuration: disable sensitivity analysis, adjust regularization parameter, and implement custom weight extraction for gene-specific parameter estimation.

- Enhance scoring function and regularization: add L2 norm parameter to score_fit, update weight handling, and implement parallel processing for lambda optimization.

- Refactor scoring and regularization: update L2 norm calculation, adjust regularization parameter, and improve weight handling in model fitting.

- Update configuration and output handling: modify .gitignore to ignore all result directories, enhance constants documentation, and adjust output file naming conventions based on ODE model.

- Add LateX result generation: integrate latexit utility into main processing flow and update function calls to include output directory.

- Add RNA data handling: introduce input for mRNA data, update processing functions, and enhance logging for common proteins between datasets.

- Update model fitting and error metrics: modify MSE and MAE calculations to include RNA data, adjust model fit plotting to incorporate RNA levels, and enhance weight handling in custom weights function.

- Update ODE model and data handling: change ODE model to 'distmod', adjust RNA data handling in core functions, and refine error metric calculations for improved model fitting.

- Update sensitivity analysis and model fitting: modify sensitivity analysis function to include RNA data and adjust model fit reshaping for consistency in plotting.

- Refactor sensitivity analysis: adjust parameter sampling and restore normalization calculation for model output.

- Update sensitivity analysis and model fitting: incorporate mRNA data handling, adjust parameter sampling, and refine output metrics for improved analysis.

- Update sensitivity analysis and model fitting: adjust RNA data handling, increase parameter sampling, and refine model output normalization for improved accuracy.

- Update sensitivity analysis: enhance trajectory storage with RMSE calculations and switch ODE model to 'succmod' for improved analysis.

- Update sensitivity analysis: normalize RNA and psite differences for improved RMSE calculations and enhance perturbation results export to Excel.

- Update sensitivity analysis: adjust RNA predictions to use dynamic length and sort perturbation results by RMSE for improved clarity.

- Update sensitivity analysis and model fitting: switch ODE model to 'randmod', enhance parameter relationship plotting, and add top parameter pairs visualization for improved analysis.

- Update dependencies: add jinja2 to pyproject.toml and requirements.txt for enhanced templating support.

- Refactor sensitivity analysis: rename 'bounds' to 'optimal_params' for clarity and update problem definition to use optimal parameters in ODE model.

- Refactor sensitivity analysis: replace bounds with computed perturbation values for parameter sensitivity, update function signatures, and enhance plotting for top parameter pairs.

- Update analysis parameters: adjust K selection criteria to use 5% of NUM_TRAJECTORIES, increase NUM_TRAJECTORIES to 10000, and refine plotting aesthetics for better visualization.

- Refactor cleanup script: replace print statements with logger for better logging, enhance file handling by copying and removing files conditionally, and improve output file reporting.

- Update model configuration and enhance weight handling: change ODE_MODEL to 'distmod', enable custom weights, and improve weight calculation methods for better parameter estimation.

- Remove combined data time weight calculation for clarity and simplification.

- Refactor analysis parameters: adjust K selection criteria for top simulations, reduce NUM_TRAJECTORIES to 1000, and increase PERTURBATIONS_VALUE to 0.5 for improved sensitivity analysis.

- Update analysis parameters and enhance plotting: adjust K selection criteria to top 5% of RMSE values, improve logging output formatting, and refine plot aesthetics for better clarity and visualization.

- Refactor estimation process: remove deprecated estimation mode handling, simplify parameter estimation function, and enhance clarity by eliminating unused constants and code.

- Update documentation and dependencies: enhance README clarity on time points, adjust ODE_MODEL description, and update package versions in poetry.lock for improved compatibility.

- Refactor estimation tests: remove unused imports and tests, simplify normest test parameters, and enhance clarity by focusing on relevant functionality.

- Update documentation structure: move Confidence Intervals section to identifiability directory for improved organization.

- Bump version to 0.3.0

- Format YAML and Python files: standardize spacing and remove unnecessary newlines for improved readability

- Improve logging format: align log messages for better readability by adding indentation

- Update ODE_MODEL constant: change from 'distmod' to 'randmod' for improved model accuracy

- Refactor analysis and plotting: streamline data handling and enhance visualization for model perturbations

- Enhance sensitivity analysis: update return values, adjust trajectory parameters, and improve plotting clarity

- Enhance sensitivity analysis: add state labels, improve plotting functions, and update regularization setting

- Refactor mapping functions: streamline parameter handling, enhance data merging, and improve visualization in plotting

- Enhance plotting functionality: add strip plots for state distributions, implement phase space plots, and improve visualization layout

- Enhance plotting functionality: replace time-wise changes plot with time-state grid, add phase space plots, and streamline configuration logging

- Enhance plotting functionality: improve time labeling in state distribution plots, streamline phase space plotting, and adjust visualization aesthetics

- Enhance plotting functionality: increase number of trajectories for sensitivity analysis, improve strip plot aesthetics, and adjust phase space plot dimensions

- Enhance sensitivity analysis: increase number of trajectories to 10,000, improve logging format for parameter bounds, and update site parameter labels

- Refactor configuration parameters: update site parameter labels, adjust development mode flag, and refine sensitivity analysis settings

- Fix formatting in README and PYPI_README: correct markdown syntax for author acknowledgments

- Update CHANGELOG for version 0.4.0: document new features, changes, fixes, and removals

- Refactor configuration and logging: update development mode flag, adjust trajectory and parameter space settings, and enhance logging for plotting processes

- Enhance sensitivity analysis and logging: update perturbation value to 50%, improve parameter logging format, and integrate model fit into plotting functions

- Refactor constants and enhance logging: reduce NUM_TRAJECTORIES and PARAMETER_SPACE, adjust PERTURBATIONS_VALUE, and implement TqdmToLogger for progress tracking

- Refactor development mode flag and update progress logging: set DEV_TEST to False and modify progress bar description for clarity

- Add mapping file handling and kinetic strength enrichment: create mapping directory, update ODE model, and implement function to add kinetic strength columns to mapping files

- Update plotting/plotting.py

Co-authored-by: Copilot <175728472+Copilot@users.noreply.github.com>

- Update config/constants.py

Co-authored-by: Copilot <175728472+Copilot@users.noreply.github.com>

- Merge remote-tracking branch 'origin/master'

- Update constants and refactor ODE functions: enable development mode, change ODE model to distributive, and optimize parameter unpacking

- Refactor comments in randmod.py for clarity and consistency

- Refactor analysis and constants: update trajectory calculations and model parameters

- Enhance sensitivity analysis: implement parallel processing, add Y_METRIC options, and improve logging

- Enhance sensitivity analysis: improve progress logging and add future time points functionality

- Refactor analysis and logging: remove progress bar implementation and update opacity constant

- Refactor plotting functions: update opacity constant and streamline simulation curve plotting

- Enhance sensitivity analysis: add regularization term to outputs and improve logging format

- Enhance sensitivity analysis: add regularization term to outputs and improve logging format

- Refactor ODE system: optimize state transitions and improve parameter unpacking

- Refactor constants and improve parameter handling: reduce NUM_TRAJECTORIES and PARAMETER_SPACE, update plotting dimensions, and enhance state filtering in analysis functions

- Refactor parameter handling and improve plotting dimensions: adjust lambda range in normest.py and enhance figure sizing and label management in plotting.py

- Refactor confidence interval logging and update regularization term calculation: improve log formatting in ci.py, adjust ODE model in constants.py, and enhance regularization term computation in normest.py

- Refactor sheet processing in plotting: improve handling of parameter sheets and regularization checks in Excel data import

- Refactor constants and improve plotting file naming: update ODE model and perturbation value, enhance file naming conventions in plotting functions

- Refactor color assignment in plotting: update regex for parameter label matching to support multiple digits

- Refactor configuration and enhance plotting functionality: update upper bounds for parameters, adjust default values, and add model error plotting

- Refactor constants and enhance output file organization: update bootstrap parameter, improve file naming conventions, and refine output file organization in display functions

- Refactor output file organization: update output directory parameter to list format for improved compatibility

- Refactor display functions and update constants: disable development mode, enhance HTML report structure, and improve log file handling

- Refactor constants: update ODE model from 'randmod' to 'succmod' for improved simulation accuracy

- Enhance metric descriptions: add detailed descriptions for Y_METRIC options and log additional configuration parameters in main.py

- Refactor configuration handling and logging: streamline argument parsing, enhance logging details, and update trajectory count for improved clarity and functionality

- Update CHANGELOG for version 0.4.0 alpha: add new features, changes, and improvements in visualization, logging, and configuration handling

- Refactor constants and enable development mode: set DEV_TEST to True and remove unused early emphasis function for cleaner code

- Update README.md: refine input format, enhance knockout module documentation, and clarify parameter descriptions

- Refactor score_fit function: optimize parameters, remove unused weight argument, and enhance performance with Numba JIT compilation

- Update README.md and PYPI_README.md: enhance command-line usage instructions, clarify module descriptions, and add new plotting module documentation

- Optimize Numba JIT compilation: enable caching for objective function and set matplotlib backend to 'Agg' for plotting

- Enhance documentation: add argument and return descriptions for functions across multiple files

- Enhance documentation: add argument and return descriptions for functions across multiple files

- Enhance documentation: update parameter descriptions to use 'Args' and 'Returns' format across multiple files

- Enhance documentation: update function return descriptions and add missing 'Args' sections across multiple files

- Add initial documentation and API reference for PhosKinTime package

- Add initial CHANGELOG.md and README.md files for PhosKinTime package

- Bump version to 0.4.0 and update CHANGELOG for release date

- Bump version to 0.4.0 and update CHANGELOG for release date

- Update README.md: remove unnecessary badges and images for a cleaner presentation

- Update index.md: remove DOI badge for a cleaner documentation layout

- Update PYPI_README.md: remove unnecessary badges for a cleaner presentation

- Update .gitignore: add rules to ignore CSV and Excel files in the project root

- Update .gitignore: add rules to ignore __pycache__ directories and .pyc files

- Update .gitignore and cleanup.py: add .venv to ignore list and import os in cleanup.py

- Update mkdocs.yml and add deploy-docs.yml: configure MkDocs deployment to GitHub Pages

- Remove CI/CD workflow configuration from GitHub Actions

- Refactor deploy-docs.yml: simplify deployment steps and remove unused environment variable

- Update deploy-docs.yml: remove branch specification from push event

- Update deploy-docs.yml: install pymdown-extensions alongside MkDocs

- Update deploy-docs.yml and index.md: enhance installation step and add DOI badge to documentation

- Remove ODE Estimation entry point documentation from reference.md

- Update acknowledgments in README and index.md: correct Klipp-Linding Lab URL and improve formatting

- Remove Entry Backend from navigation in mkdocs.yml

- Update deploy-docs.yml and config files: add protein and psite input Excel paths, enhance data loading in main.py

- Update function signatures and data handling: add protein data support in core, normest, and toggle modules

- Update PYPI_README.md: streamline package description, enhance features section, and improve installation instructions

- Enhance analysis and plotting functions: add protein data support, improve weight calculations, and update model fitting visualizations

- Enhance analysis and plotting: add protein and phosphorylation data support, update sensitivity analysis, and improve RMSE calculations

- Tested also protein only data included estimation - done.

- Add global phosphorylation ODE simulator and update development mode flag

- Update .gitignore and rename global_phospho_model.py to global_model.py for clarity

- Add global ODE dual-fit implementation with effects from Alpha and Beta values

- Add poetry.toml to set virtual environments in-project

- Remove `global_model.py`: deprecate unused ODE dual-fit model implementation

- Refactor `global_model_v2.py`: simplify imports, optimize parallelization logic, and replace pyrecorder-based video rendering with Matplotlib animations.

- Remove deprecated `global_model_v2.py` implementation.

- Add E_i parameter, normalize RNA/protein baselines, and implement lambda weight scan script

- Add export functions for Pareto front solutions: Excel export, Goodness of Fit plotting, and animation rendering. Update weight parameters and solver configurations for improved optimization.

- Add `global_model_v5.py`: multi-objective ODE optimization with UNSGA3, Pareto front visualization, export utilities, and Numba-optimized kernels.

- Optimize `global_model_v5.py`: add time-bucketing, cache phosphorylation rates, refactor fast_rhs_loop, and reduce default genetic algorithm parameters.

- Add time-series plotting for gene-specific RNA/protein fold changes and update default configuration parameters.

- Update `BOUNDS_CONFIG` in global_model_v3/v4/v5: expand `c_k` bounds, adjust `tf_scale` lower limit

- Adjust `BOUNDS_CONFIG` parameter floors and reduce `lambda-prior` default value in `global_model_v3.py`.

- Update `BOUNDS_CONFIG` in `global_model_v4/v5`: raise lower bounds for stability and meaningful dynamics

- Introduce `phoskintime_global`: modularized multi-objective ODE optimization framework with UNSGA3, Numba-optimized loss functions, simulation utilities, and export tools. Remove outdated tests and graph files.

- Remove outdated files: `CHANGELOG.md`, `Dockerfile`, `PYPI_README.md`, `requirements.txt`, `pytest.ini`, and old scripts (`global_model/scan_lambdas.py`). Modularize and refactor `phoskintime_global` pipeline: utilize centralized configuration (`config.toml`), add prior regularization scanning, suppress warnings, streamline TIME_POINTS handling, and improve bounds validation.

- Refactor and modularize: Add combinatorial and sequential ODE models with dynamic switching, centralized tolerances, and optimized loss functions. Consolidate protein state handling, cache phosphorylation rates, and improve simulation accuracy.

- Remove deprecated files and modules, including CLI utilities, helpers, parameter estimation, configuration presets, and diagram generators, to streamline the codebase and eliminate unused functionality.

- Remove redundant code in `network.py`: eliminate unnecessary blank lines and unused variables for cleaner and more maintainable function definitions.

- Add `file_prefix` parameter to `plot_goodness_of_fit` and adjust output directory handling to streamline solution plot naming.

- Add optimization parameters (`max_iterations`, `population_size`, `seed`, `regularization_lambda`, `regularization_rna`, `results_dir`) to configuration and streamline their integration into the pipeline.

- Refactor ODE and combinatorial model handling: streamline caching, preallocate buffers, update dynamic state handling, and simplify `S_cache` integration. Adjust default model and optimization parameters in `config.toml`.

- Introduce `solvers.py` with Numba-optimized solvers for ODE models and add `USE_CUSTOM_SOLVER` configuration flag.

- Replace KKTPM analysis with hypervolume metrics, update UNSGA3 sampling to LHS, adjust generation intervals for ref. directions, and increase video and plot resolution. Update `population_size` in `config.toml`.

- Remove hypervolume calculation and plotting functionality, along with HV imports, dependencies, and related code for cleaner optimization workflow in `runner.py` and `export.py`.

- Integrate PI control logic with history term into adaptive RK45 solvers for improved step size selection and stability. Simplify error estimation logic and ensure consistency across solver implementations.

- Remove `global_model_v3.py` and associated functionality to streamline the codebase and eliminate unused legacy components.

- Refactor loss function signatures to include base indices, update phospho site data parsing for better handling, and enhance `optproblem.py` integration with new loss parameters. Adjust optimization and regularization settings in `config.toml` for improved convergence.

- Update weight schemes with parameterized early boost in `optproblem` and integrate piecewise early weighting into `runner`.

- Update README with DOI badge, add LOG.md, and switch default model to "combinatorial" in config.

- Add `Dp_i` parameter to Numba solvers and update all solver function signatures and logic accordingly.

- Remove `export_S_rates_with_times` function and related functionality from `runner.py` and `export.py`. Update `Dp_i` handling in phosphorylation rate logic and adjust plot dimensions. Switch default model to "sequential" and increase optimization parameters in `config.toml`.

- Switch default model to "combinatorial" in `config.toml`, fix tuple unpacking alignment in `solvers.py`, and correct argument slicing in `jacspeedup.py`.

- Add advanced loss functions (Huber, Pseudo-Huber, Charbonnier) with mode selection, integrate preprocessing options (`normalize_fc_steady`, `use_initial_condition_from_data`), and implement comprehensive steady-state computations for all models (`distributive`, `sequential`, `combinatorial`).

- Integrate SBX crossover and polynomial mutation into NSGA-III, improve reference direction generation, and add phosphorylation rates report plotting functionality. Update `config.toml` with new optimization and regularization settings.

- Switch to `output_dir` argument in `runner.py` for improved path handling, update default model to "sequential" in `config.toml`.

- Update solver tolerances and timesteps in `config.toml`, disable custom solver usage.

- Implement TF repression dynamics, adjust TF scaling, add protein degradation to phospho states, and update bounds, tolerances, and default model in `config.toml`. Include TODO file for addressing TF regulation gaps.

- Update parameter bounds in `config.toml`, adjust solver settings, and disable custom solver usage.

- Update scaling approach in `models.py`, enable steady-state normalization in `config.toml`, adjust parameter bounds and regularization settings, and switch loss function to Huber.

- Adjust regularization settings in `config.toml`, update protein and RNA lambda values, and add notes on the distributive model in `LOG.md`.

- Merge pull request #13 from bibymaths/master

Master

- Merge pull request #14 from bibymaths/master

Master

- Merge pull request #15 from bibymaths/master

Master

- Merge pull request #21 from bibymaths/master

Added functionalities

- Merge pull request #22 from bibymaths/master

Added sensitivity analysis, knockouts, and better visualization - saving results

- Merge pull request #23 from bibymaths/master

https://github.com/bibymaths/phoskintime/pull/23#pullrequestreview-2801331424

- Merge pull request #24 from bibymaths/master

Master

- Merge pull request #25 from bibymaths/master

Master

- 'added finaliyzed sensitivity analsis and interpretation'Merge branch 'master'

- 'fixed fatal config'Merge branch 'master'

- Merge pull request #26 from bibymaths/master

Master

- Merge pull request #27 from bibymaths/master

Master

- Merge pull request #28 from bibymaths/master

Final version before editing readme



### Tests


- Testing workflow - series of tests

- Testing workflow - include in root for poetry

- Testing workflow - include in root for poetry

- Testing workflow

- Testing

- Testing

- Testing with coverage

- Testing and fixes

- Testing and fixes

- Tests correction

- Testing random model - minor fixes

- Tested on server for SLSQP

- Tested on server for SLSQP - 100 iterations - 30 minutes

- Testing mcmc - failed



