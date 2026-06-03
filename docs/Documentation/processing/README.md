# Data Processing

## Current networkmodel processing

The networkmodel processing path uses `load_data`, `prepare_fast_loss_data`, `process_and_scale_raw_data`, `normalize_fc_to_t0`, and `build_y0_from_data`. These functions load source tables, normalize time-series values, prepare loss arrays, and build initial conditions.

## Scaling configuration

The source inventory exposes this default scaling key:

```toml
scaling_method = "none"
```

## What this module does not do

This page does not document processing functions outside the current networkmodel source inventory.
