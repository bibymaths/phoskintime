# Plotting and Export

## Current scope

Plotting functions live in `export.py`, `analysis.py`, `mode_outputs.py`, and `dashboard_app.py`. These modules write convergence plots, goodness-of-fit plots, parameter diagnostics, sensitivity plots, and dashboard views from saved outputs.

## Public API

The export API includes `plot_goodness_of_fit`, `plot_gof_from_pareto_excel`, `save_gene_timeseries_plots`, `plot_s_rates_report`, `process_convergence_history`, `export_param_correlations`, and `export_parameter_distributions`.

## What this module does not do

This page does not describe plotting outputs that are not produced by the current source files.
