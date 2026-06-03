# Identifiability Notes

## Current scope

The implemented inference helpers are in `inference.py`. The module provides `run_multistart`, `run_profile_likelihood`, and `run_numpyro_posterior` for post-optimization analysis.

## Multistart behavior

`run_multistart` creates bounded starting vectors and executes starts with `ThreadPoolExecutor`. The relevant configuration key exposed by `config.py` is:

```toml
n_starts = 1
```

## What this module does not do

This page does not claim structural identifiability guarantees. It documents only the implemented post-optimization helpers in the current source.
