# Config Package Notes

## Current networkmodel configuration source

The active networkmodel constants are loaded in `networkmodel/config.py`. The source inventory exposes the following fallback keys through `getattr` calls.

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

## What this module does not do

This page does not document configuration fields that are absent from the current networkmodel source inventory.
