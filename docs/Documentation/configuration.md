# Networkmodel Configuration

## Current keys

The current networkmodel documentation uses only keys read by `networkmodel/config.py`. The defaults below match the fallback values in that module.

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

## Command-line flags

The runner exposes these flags in `runner.py`.

```bash
python -m networkmodel.runner --solver jaxopt --seed 42 --cores 1
```

The command uses the implemented scalar optimizer path. File path flags such as `--kinase-net`, `--tf-net`, `--ms`, `--rna`, `--phospho`, `--kinopt`, `--tfopt`, and `--output-dir` are also defined by the runner.

## What this module does not do

This page does not document configuration keys that are absent from the source inventory. It does not document non-JAXopt solver choices as supported workflows.
