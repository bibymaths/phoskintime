# Utilities

## Current networkmodel utilities

The public networkmodel utility API includes `normcols`, `find_col`, `slen`, `normalize_fc_to_t0`, `process_and_scale_raw_data`, `time_bucket`, `softplus`, `inv_softplus`, `load_config_toml`, `calculate_bio_bounds`, and `get_optimized_sets`.

## Configuration loading

`load_config_toml` returns `PhosKinConfig` from a TOML path. The networkmodel configuration page lists the keys exposed by `config.py`.

## What this module does not do

This page does not document utility functions outside the current networkmodel public API inventory.
