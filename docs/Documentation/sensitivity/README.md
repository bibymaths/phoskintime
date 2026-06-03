# Sensitivity Analysis

## Current public API

Sensitivity analysis is implemented by `compute_bounds` and `run_sensitivity_analysis`. The analysis perturbs fitted parameters and summarizes output changes under the selected metric.

## Configuration

```toml
n_starts = 1
profile_likelihood = false
profile_grid_size = 10
posterior_sampling = false
posterior_num_warmup = 20
posterior_num_samples = 30
sensitivity_analysis = false
sensitivity_perturbation = 0.2
sensitivity_trajectories = 1000
sensitivity_levels = 400
sensitivity_top_curves = 50
sensitivity_metric = "total_signal"
```

## What this module does not do

This page does not claim biological causality from sensitivity scores. It documents the implemented perturbation-analysis helpers only.
