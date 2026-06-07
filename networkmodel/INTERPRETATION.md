# Interpreting `networkmodel` Outputs

This document describes how to read outputs that are currently produced by the integrated `networkmodel` scalar JAXopt workflow. It covers fitted trajectories, scalar objective diagnostics, parameter summaries, sensitivity outputs, inference outputs, and dashboard files written by the implemented modules in this package.

## What this module does not do

- It does not infer causal certainty from a fitted edge or parameter.
- It does not implement node-deletion knockout simulation as part of the documented runner path.
- It does not report retired multi-objective fronts from the current scalar optimizer.
- It does not guarantee that every optional analysis exists unless the corresponding flag or configuration toggle was enabled and its dependencies are installed.

## Core interpretation principle

The fitted outputs are trajectories from a parameterized ODE system. A predicted change should be interpreted as the model's propagated response under the fitted parameters, observed time grid, topology, and loss weights, not as a standalone correlation.

The three fitted layers are:

- protein observations from the mass-spectrometry input,
- RNA observations from the RNA input,
- phospho observations from the phospho input after the mechanistic kinase-site filter in `runner.py`.

## Fitted trajectory tables and plots

Protein, RNA, and phospho trajectory outputs compare observed fold-change values with model-predicted values on the configured time grids:

```text
TIME_POINTS_PROTEIN -> protein fitted trajectories
TIME_POINTS_RNA -> RNA fitted trajectories
TIME_POINTS_PHOSPHO -> phospho fitted trajectories
```

Use these plots to check whether the optimized scalar objective is fitting the available data layers with comparable quality. Large residuals at early time points can indicate a mismatch in initial conditions or kinase input scaling; large late residuals can indicate turnover or transcriptional regulation parameters that do not match the observed trajectory.

## Scalar objective and convergence diagnostics

The scalar objective combines available layer losses and the prior penalty. The active layers are detected by `detect_data_mode` from non-empty protein, RNA, and phospho arrays.

Interpret objective outputs as follows:

| Output | Meaning |
| --- | --- |
| `scalar_total` | Total scalar value minimized by `jaxopt.ProjectedGradient`. |
| protein loss | Contribution from protein observations when protein data are available. |
| RNA loss | Contribution from RNA observations when RNA data are available. |
| phospho loss | Contribution from phospho observations when phospho data are available. |
| prior | Penalty for movement away from default physical parameters when `lambda_prior` is nonzero. |

A lower scalar objective is better for the exact same input data, weights, bounds, model, and seed. Do not compare objective values across different scaling methods, layer weights, or data subsets without accounting for those changes.

## Parameter summaries

The optimizer works on raw parameters and converts them to positive physical parameters with softplus transforms before simulation. Bounds are applied through projection.

The main parameter groups are:

| Group | Interpretation |
| --- | --- |
| `c_k` | Kinase activity multipliers applied to kinase input trajectories. |
| `A_i` | Basal mRNA production parameters. |
| `B_i` | mRNA degradation parameters. |
| `C_i` | Protein production parameters. |
| `D_i` | Protein deactivation or turnover parameters. |
| `Dp_i` | Phosphosite dephosphorylation parameters. |
| `E_i` | Transcriptional efficacy parameters. |
| `tf_scale` | Transcription-factor scaling parameter. |

A parameter near a configured lower or upper bound should be read as a constrained optimum for the current objective, not as a measured biochemical constant.

## Kinase activity outputs

`export_kinase_activities` writes kinase activity trajectories over a generated time grid. These values are model inputs or fitted multipliers applied to observed kinase proxies, so they should be interpreted as effective activity signals in the fitted model.

High kinase activity can produce a small fitted phospho effect when the kinase has few represented substrates, when the affected substrates are weakly weighted in the loss, or when dephosphorylation parameters counteract the forward drive.

## Phosphorylation-rate reports

`export_S_rates` and `plot_s_rates_report` summarize phosphorylation-rate trajectories by protein and site. These reports are useful for ranking model-implied phospho fluxes within the represented kinase-site topology.

Do not read a high rate as experimental validation of a kinase-site relationship. The relationship must already exist in the input topology to be represented in the model.

## Residual outputs

Residual tables compare observed and predicted values for matched entities and time points. Use residuals to identify:

- proteins or sites that consistently miss across time,
- time windows with systematic underprediction or overprediction,
- layer-specific mismatch after changing `lambda_protein`, `lambda_rna`, or `lambda_phospho`.

Residuals are conditional on preprocessing. Changing `scaling_method` or fold-change normalization changes the residual scale.

## Sensitivity outputs

When sensitivity analysis is enabled, the package perturbs fitted parameters within computed bounds and summarizes how the selected scalar metric changes. The implemented metrics include names accepted by `SENSITIVITY_METRIC`, such as `total_signal`, `mean`, `variance`, and `l2_norm`.

Interpret sensitivity scores as local model-response summaries around the fitted parameter set. They do not prove biological necessity, and they depend on the chosen perturbation size, trajectory count, level count, and metric.

## Inference outputs

Optional inference helpers are controlled by these configuration constants:

```text
N_STARTS
PROFILE_LIKELIHOOD
PROFILE_INDICES
PROFILE_GRID_SIZE
POSTERIOR_SAMPLING
POSTERIOR_NUM_WARMUP
POSTERIOR_NUM_SAMPLES
```

Multistart summaries compare solutions from multiple bounded starting vectors. Profile-likelihood outputs sweep selected raw parameter indices and re-evaluate objective behavior across a grid. Posterior sampling uses the optional NumPyro path in `inference.py` when that dependency is available.

## Dashboard outputs

The dashboard utilities load saved output files from the result directory and render static tables, figures, and scalar-run summaries. The dashboard does not recompute the ODE solution; it presents files that were already exported by the run.

## Recommended checks after a run

1. Confirm that the detected data mode includes the layers you expected.
2. Inspect convergence and the final scalar objective.
3. Check goodness-of-fit plots for each available data layer.
4. Review residuals for systematic layer or time-window bias.
5. Inspect parameter distributions and bound-adjacent values.
6. Use sensitivity outputs only as model-response diagnostics, not as independent validation.
