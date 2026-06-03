# Dashboard Documentation

## Current scope

The dashboard source lives in `dashboard_app.py` and `dashboard_bundle.py`. It reads saved result files and dashboard bundles from an output directory, then renders tables and plots for inspection.

## Public API

The dashboard bundle API exposes `save_dashboard_bundle` and `load_dashboard_bundle`. The Streamlit entry point exposes `main` in `dashboard_app.py`.

## What this module does not do

The dashboard does not optimize parameters. It does not recompute ODE trajectories. It presents files already written by the runner and exporter modules.
