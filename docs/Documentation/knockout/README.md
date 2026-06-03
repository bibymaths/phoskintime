# Perturbation Notes

## Current scope

The current networkmodel source contains sensitivity analysis, fitted trajectory export, and dashboard display utilities. It does not expose a public knockout runner in the networkmodel public API inventory.

## Current analysis path

Sensitivity analysis uses `run_sensitivity_analysis` with perturbation settings from `config.py`.

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

This page does not document node deletion or knockout simulation as an implemented networkmodel workflow.
