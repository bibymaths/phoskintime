# Known issues

This file lists the scientific and correctness issues found in the November 2026 audit of this repository. Issues fixed since then say so in the State column, with the fixing commits. Fixing any of them changes model results, so each fix needs a separate, reviewed change with a before/after comparison.

The reorganisation that followed the audit moved files but did not change any equation, objective, bound, tolerance, seed or parameter meaning.

**Status markers**:

- **✔** verified: the cited lines were read, or the command was run, during the audit.
- **◐** reviewer-verified: a read-only sub-reviewer read the code, in some cases with an isolated reproduction. These have not been re-checked line by line.
- **?** suspected: plausible from reading, not demonstrated.

**Severity** is the expected impact on results or usability: H (high), M (medium), L (low).

File:line references point to the tree at the time of writing unless stated otherwise. IDs are the audit's finding numbers; gaps are numbers not listed here.

## networkmodel

| ID | Sev | Status | Location | Description | State |
|---|---|---|---|---|---|
| N1 | H | ✔ | `networkmodel/backend.py:368-436` | The JAX RHS has branches only for `model_id == 2` and `model_id == 4`; MODEL 1 (sequential) falls through to the distributive equations. | Fixed in ca5becf (JAX RHS sequential branch; parity with the numba kernel ≤ 1e-12). Sequential fits change; the sequential phospho observation operator is a separate open issue (N20) |
| N2 | H | ✔ | `networkmodel/runner.py:934` → `networkmodel/params.py:41-44`; `networkmodel/utils.py:305-314` | `calculate_bio_bounds` returns all 8 parameter groups, and these take priority over every entry in `[networkmodel.bounds]`. | Fixed in ddb9be1: bounds only from calculate_bio_bounds (all 8 groups logged); [networkmodel.bounds] removed and rejected |
| N3 | H | ✔ | `networkmodel/backend.py:605`; `networkmodel/LossFunction.py` | The active loss is weighted MSE; `LossFunction.py` (robust losses, `LOSS_MODE`) has no importers, so the configured `loss` is not used. `networkmodel/README.md:411` refers to an adaptive RK45 solver. | Fixed: weighted MSE is the only loss (config loss = 0; other values rejected); LossFunction.py deleted |
| N4 | H | ◐ | `networkmodel/runner.py:799` | Observation weights are set to 1.0 (protein; RNA likewise); `weighting_method_*` is only logged. | Fixed in cd09713: uniform weighting only; other settings rejected |
| N5 | H | ✔ | `networkmodel/backend.py:900` (defaults `:151-152`); `networkmodel/simulate.py:178`; `config.toml:978-979`; `networkmodel/InitialConditions.py:224` | The configured ODE tolerances (1e-8) are not used: the fit uses rtol = atol = 1e-6 and the export uses rtol 1e-5, atol 1e-7, `max_steps=5000`. Phospho initial conditions can be 1e-9 ◐, below atol. | Fixed in 0c7c920: one solver config ([networkmodel.solver] 1e-8/1e-8/200000) for fit, export, steady state, sensitivity and posterior/profile rebuild. Changed results: all networkmodel fits (slightly); PINN paths out of scope (quarantined) |
| N6 | H | ✔ | `networkmodel/BayesianInference.py:774-776` | The posterior likelihood is `-0.5 * objective / (sigma² + 1e-6)` on a sum of per-modality MSEs, with no N·log σ term and one σ shared across modalities. | Mitigated in 617bfc4: posterior sampling disabled (requests stop with this reference). Proper likelihood deferred |
| N7 | H | ◐ | `networkmodel/runner.py:920`; `networkmodel/simulate.py:84`; `networkmodel/PosteriorObjective.py:281` | Predicted fold change is `Y[t]/Y[base]` with base t = 0 (protein/phospho) and t = 4 (RNA, hard-coded `4.0`), while the data are not 1 at the baseline. | Fixed in 689e891: scaling_method = fc_start; baseline times defined once; one model fold change; bad baselines skipped and counted. Changed results: all networkmodel data fits |
| N8 | H | ◐ | `networkmodel/backend.py:433-434`; `networkmodel/runner.py:1641,1654`; `networkmodel/INTERPRETATION.md:75-76`; `config.toml` comments | In the RHS, E_i acts as protein-level dephosphorylation and Dp_i as a per-site decay; the labels in the runner, `INTERPRETATION.md` and `config.toml` describe them differently ("initial condition multipliers", "transcriptional efficacy", "dephosphorylation"). | Fixed in 38db895 (labels and docs; a test pins the E/Dp/D roles). Not a math change |
| N9 | M | ◐ | `networkmodel/models.py:409`; `networkmodel/backend.py:405` | Combinatorial decay adds `D_i` once per phosphorylated bit, giving k·D_i decay for a state with k bits. | Fixed in f68906d, 77c526b: decay of combinatorial state m = D_i + sum Dp over its sites (JAX RHS and numba oracle; docs generator). Changed results: combinatorial (B6) |
| N10 | M | ◐ | `networkmodel/network.py:437`; `networkmodel/SteadyStateAnalysis.py:190` | The legacy numba `System.rhs` differs from the JAX RHS (TF input squashed twice; MODEL 4 returns dy = 0). It is used for the steady-state `\|dy/dt\|` diagnostic and for SBML export. | Fixed in 6464741: System.rhs retired; steady-state diagnostic and SBML check use the JAX RHS (numba kernels kept as test oracles) |
| N12 | M | ◐ | `networkmodel/simulate.py:271,275`; `networkmodel/pinn/objective.py:56`; `networkmodel/pinn/neuralsolver.py:105` | The PINN path uses `throw=False` with `nan_to_num(nan=1e6)`, and zeroes NaN gradients. | Fixed in 0b5a32e, 3a6ffb5, 0a3d1d4: PINN/NeuralODE quarantined under experimental/; the runner refuses enable_pinn |
| N13 | M | ◐ | `networkmodel/BayesianInference.py:527` | The profile-likelihood grid is a linspace in raw softplus space (for E_i the raw lower bound is about −27.6). Profile deltas are MSE differences relative to the grid minimum, not −2 log L. | Mitigated in 6200cac: profiles labelled as Δ training objective (not −2 log L), with the fitted optimum and per-point convergence. Likelihood-scale grid deferred with N6 |
| N17 | M | ◐ | `networkmodel/export.py:1465`; `networkmodel/backend.py:291`; `networkmodel/runner.py:1463` | Exported kinase activity is linearly interpolated, while the RHS uses a piecewise-constant step; the export is truncated at t = 120. No `jump_ts` is passed to Diffrax at step discontinuities (?). | Fixed in c4feda1, da55a48: PCHIP kinase drive in fit and exports (one function); extrapolation = hold end values (pending owner decision); SBML export blocked until the drive is encoded (b0c379b). Changed results: all networkmodel fits |
| N18 | L | ◐ | `networkmodel/InitialConditions.py:161`; `networkmodel/io.py:249`; `networkmodel/cache.py:96`; `networkmodel/BuildMatrix.py:57` | The RNA initial condition is the t = 4 observation placed at t = 0. `TIME_POINTS_PHOSPHO` is not used for mapping. Observations of pruned or unknown sites are dropped without a count. Duplicate (site, kinase) rows are summed by `csr_matrix`. | Not fixed |
| N19 | L | ◐ | `networkmodel/sensitivity.py:385` | Seeds: sensitivity analysis uses the config `SEED`, not the `--seed` CLI argument. | Not fixed |
| X8 | M | ✔ | `config.toml:811-853` (after the header at `config.toml:752`); `config_loader.py:241,316-322` | The inference keys `n_starts`, `profile_likelihood`, `profile_indices`, `profile_grid_size`, `posterior_sampling`, `posterior_num_warmup` and `posterior_num_samples` sit under `[networkmodel.hyperedge_preprocessing]`, but are read from `[networkmodel]`, so the code defaults apply. Today the only effective difference is `profile_grid_size` (configured 5, used 10). | Fixed in b5c3ad3: keys moved under [networkmodel] (profile_grid_size = 10, the value in effect); misplaced keys warn |
| N11 | L | ✔ | `networkmodel/runner.py` (post-fit steady-state log) | The steady-state log read `Y_check[i, -window:]` with `Y_check` shaped (time, state): time row i over the last states, not protein i over time. Logs only. | Fixed in a131695 |
| N16 | L | ✔ | `networkmodel/export.py` (`export_param_correlations`, `export_parameter_distributions`) | The c_k correlation heatmap, c_k boxplot and tf_scale violin plotted raw optimizer coordinates (inverse softplus) under physical labels. PNGs only. | Fixed in 883fdfb |
| N20 | M | ◐ | `networkmodel/backend.py` (`_global_networkmodel_observable`); `networkmodel/simulate.py:141-144`; `networkmodel/InitialConditions.py:219-224` | For MODEL 1 (sequential) state j+1 means "exactly j+1 sites phosphorylated", but the loss, export and initial conditions map observed site j to that single state; the site occupancy would be the sum of states j+1…n. Site order (residue order) is taken as the reaction order, undocumented. | Fixed in 6799b10: site occupancy = sum of sequential states j..n-1 in loss, export and initial conditions (combinatorial already summed masks). Changed results: sequential (B5) |
| N21 | M | ? | `networkmodel/network.py:85-88`; `networkmodel/backend.py` (`driver_map`) | Orphan redirection overwrites `p2i[orphan]` with the proxy's index; the JAX RHS then drives the proxy's row, while `System.rhs` drives the orphan's own row by name. Unverified; B4/B5 rewire no orphans. | Not a bug (f380f54): the TF matrix never reads or writes the orphan's own row, so derivatives are identical; the orphan's own state rows are inert |

## protwise

| ID | Sev | Status | Location | Description | State |
|---|---|---|---|---|---|
| P1 | M→H | ✔ | `protwise/paramest/normest.py:157`; `protwise/models/diffrax_solver.py:176` | RNA predictions are read from the protein time grid (`r[-n_rna_times:]` in the objective, `sol_out[5:, 0]` in the solver output). With the shipped grids the model is read at [4, 8, 16, 30, …] and the data are at [4, 8, 15, 30, …]; the two code paths slice differently. | Fixed in aecc428, 591f7f7, d99e6d0: union grid with RNA read at the RNA times for any grid, in objective, final fit, export and sensitivity |
| P3 | H (when on) | ✔ | `protwise/models/diffrax_solver.py:156-159`; `protwise/paramest/normest.py:161-182` | `normalize_model_output` divides the reported solution by the initial conditions; the objective does not normalise. | Fixed in 9c20618: normalize_model_output removed |
| P4 | M-H | ◐ | `protwise/steady/initdist.py:12-15` | Initial conditions are fixed constants, not steady states, while the objective includes t = 0 fold-change data of about 1. | Fixed in 77ddcee: initial conditions from the t0 data (R(0) = 1 and the randmod allocation rule pending owner confirmation). Changed results: protwise (B3) |
| P7 | M | ◐ | `protwise/sensitivity/analysis.py:186,224` | Seeds: Morris sensitivity sampling is unseeded. | Fixed in be80c54: one seed (`--seed`) for Morris sampling and the SALib confidence intervals in protwise and networkmodel |
| P8 | M | ◐ | `protwise/paramest/normest.py:175-179`; `protwise/paramest/core.py:75,78,81` | Targets are not NaN-masked, so one NaN aborts the run. Positional `iloc[:, 2:]` column slicing misreads any extra column. | Fixed in ded650c: NaN targets masked (counts in metadata), failing genes isolated (metadata failed_genes); column selection by position unchanged |
| P9 | M | ◐ | `protwise/models/diffrax_solver.py:28,45,78`; `config.toml:86-87` | A dephosphorylation rate of 1.0 min⁻¹ is hard-coded in all three RHSs, while `config.toml` describes `phospho_deg` as the dephosphorylation rate. | Relabelled in 38db895; experiment (.sci): fitting k does not improve the B3 fits and is not identifiable; k = 1.0 kept |
| P2 | L | ✔ | `protwise/plotting/plotting.py`; `protwise/models/weights.py`; `protwise/paramest/core.py` | The flat fit vector was split with hard-coded 9/14/23 offsets (9 RNA + 14 protein), wrong for other grids and for genes without an RNA or protein layer. | Fixed in 16725f8, a34adec |

## Cross-cutting and data

| ID | Sev | Status | Location | Description | State |
|---|---|---|---|---|---|
| X6 | M | ✔ | `.gitignore:27,30`; `processing/cleanup.py:229` | `data/` and `processing/raw/` (CollecTRI, MS_Gaussian_updated_09032023, Rout_LimmaTable) are git-ignored, so a fresh clone cannot reproduce a real result. The raw files have no checksum, source citation or version. Gene-symbol mapping queries mygene live, so it is not reproducible over time. | Not fixed |
| X11 | L | ✔ | `config/constants.py:148`; `protwise/models/diagram/helpers.py:120,166,223`; `protwise/sensitivity/analysis.py:277`; `config/logconf.py:89-115`; `networkmodel/*.py` module loggers (`setup_logger(log_dir=RESULTS_DIR)`) | protwise writes diagrams, sensitivity plots and logs to the hard-coded `results/<Model>_results` and `results/logs`, ignoring `--outdir`; networkmodel module loggers write to the configured `output_dir` (`results_ABL2_network_hybrid`) regardless of `--output-dir`. Two concurrent protwise runs collide there (graphviz `FileNotFoundError`, seen 2026-11-26). | Fixed in f4340d8, 5cb76b6 (lazy, redirectable file logs; run-specific diagram/sensitivity folders; no default folders created at import). Verified with two concurrent B3 and B4 runs |
| X12 | M | ✔ | `protwise/paramest/normest.py:185`; `config.toml` `[ode.bootstrap] n` | `normest(..., bootstraps)` accepts the bootstrap count but never uses it, so `[ode.bootstrap] n` has no effect. | Fixed in 9bb7497: [ode.bootstrap] and the unused parameter removed; setting it is an error |
| X13 | M | ✔ | `protwise/models/diffrax_solver.py` (`make_local_model_rhs`); `networkmodel/backend.py` (`optimize_scalar_objective`); diffrax `throw=True` | Runtime is dominated by JAX re-tracing and recompilation: a new RHS object per solve defeated diffrax's jit cache (116 compilations per protwise gene), eager objective evaluations recompiled, and host callbacks (jaxopt printing, diffrax's error check) block the persistent compilation cache. | Partly fixed in ffacf79, a293070, f874f2e (bit-identical on B1–B5). acb2c80 (jit of the repeated objective evaluations) reverted in 1707991, c85bc46: warm gain below 5 % (0.2 % B4, 1.0 % B5). diffrax `throw` not changed (owner decision) |

## Scripts

| ID | Sev | Status | Location | Description | State |
|---|---|---|---|---|---|
| X9 | M | ✔ | `scripts/export/export_model_documentation.py:44-48` | The script imports 14 helpers from `export_model_to_sbml` that were removed in `247c877`, so it exits at start-up. The `except Exception` reports a "must be placed next to export_model_to_sbml.py" error instead of the `ImportError`. | Not fixed |

## Applies to KinOpt/TFOpt

The `kinopt/` and `tfopt/` packages were removed from this repository. They now live in separate repositories:

- KinOpt: https://github.com/bibymaths/KinOpt
- TFOpt: repository not yet public.

The findings below were made on the code as it was in this repository. **Their file:line references refer to commit `6eb04fa` of this repository**, the last commit that contains `kinopt/` and `tfopt/`. They have not been checked against the standalone repositories.

| ID | Sev | Status | Location (at `6eb04fa`) | Description | State |
|---|---|---|---|---|---|
| K1 | H | ✔ | `kinopt/local/optcon/construct.py:77,128,150,191-197`; `kinopt/local/objfn/minfn.py:46,139` | The K rows are built in first-appearance kinase order, while the β offsets assume sorted kinase order. When `input2` is not already in sorted kinase order, β is applied to a different kinase's time series. | Not fixed |
| K3 | H | ✔ | `kinopt/local/optcon/construct.py:95-104`; evol `kinopt/evol/optcon/construct.py:193-197` ◐ | With `estimate_missing_kinases = true`, the first kinase missing from `input1` raises `UnboundLocalError` (`synthetic_ts` is never assigned). The evol variant substitutes zeros. | Not fixed |
| K4 | M | ◐ | `kinopt/local/optcon/construct.py:42-45`; evol `kinopt/evol/optcon/construct.py:124-127` | An `input2` site missing from `input1` reuses the previous site's time series. Duplicate sites may make numba index out of bounds (?). | Not fixed |
| K5 | M | ◐ | `kinopt/local/opt/optrun.py:116-122`; `tfopt/local/opt/optrun.py:68-87` | SciPy 1.17 SLSQP results have no constraint-violation field, so multistart reports cv = 0 for every start and selects by `fun` alone. | Not fixed |
| K6 | M | ◐ | `kinopt/local/optcon/construct.py:227-230`; `tfopt/local/utils/params.py:47,59`; `kinopt/local/__main__.py:118`; `kinopt/evol/opt/optrun.py:61-66` | The multistart base point uses the unseeded global `np.random` despite `base_seed`; `kinopt.evol` passes no pymoo seed. | Not fixed |
| K7 | M | ◐ | `kinopt/local/objfn/minfn.py:72` vs `:104-167`; `tfopt/local/objfn/minfn.py:66` vs `:97-134` | The objectives clip predictions at 0; the exported "Estimated", residual and R² outputs are unclipped. | Not fixed |
| K10 | L | ◐ | `kinopt/local/utils/iodata.py:46`; evol `kinopt/evol/optcon/construct.py:71` | `set(full_hgnc_df['GeneID'][1:])` skips the first `input1` row. | Not fixed |
| K11 | L | ◐ (?) | `kinopt/local/optcon/construct.py:254` | The weighted loss uses 1/(Var_t + 1e-8), so a near-constant time column (e.g. fold change at t0) gets a weight of about 1e8. | Not fixed |
| K12 | L | ◐ | `kinopt/local/objfn/minfn.py:5,103` | `fastmath=True` on the objectives makes NaN handling unreliable. | Not fixed |
| T1 | H | ✔ | `tfopt/local/utils/iodata.py:89-94`; evol `tfopt/evol/utils/iodata.py:45-50` | TF protein-level rows are dropped and the first phosphosite is used as the protein series. | Not fixed |
| T2 | H | ✔ | `tfopt/local/optcon/construct.py:146-149`; evol `tfopt/evol/objfn/minfn.py:35` ◐ | The α sum-to-1 constraint includes padded regulator slots, so genes with fewer than `n_reg` regulators are effectively constrained by α ≤ 1. | Not fixed |
| T3 | H | ◐ | `tfopt/evol/utils/iodata.py:47`; `tfopt/evol/optcon/construct.py:49,69` | tfopt.evol slices `[:T]` and fits MS times 0–16 min against RNA times 4–960 min; the local variant slices `[5:]`. | Not fixed |
| T4 | M | ◐ | `tfopt/evol/objfn/minfn.py:473,512` | `pool.apply_async` results are never `.get()`; if a worker raises, fitness stays as uninitialised `np.empty` memory. | Not fixed |
| T5 | M | ◐ | kinopt NSGA and tfopt evol variants (several files) | The evol variants optimise the constraints as objectives and select by weighted sum, so Σα and Σβ need not equal 1. kinopt "DE" is a GA with ±1e-10 inequality pairs. | Not fixed |

Related documentation statements (at `6eb04fa`): `kinopt/README.md:88` and `tfopt/README.md:70` give β ∈ [−2, 2] while code and config use [−4, 4] ◐; `tfopt/README.md:82` documents a multiplicative form while the code is additive ◐; `tfopt/README.md:64` states Σα = 1 per gene (see T2) ◐.
