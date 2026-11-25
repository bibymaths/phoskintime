# scripts/figures/egfr_cbc2026/

These scripts were used for the EGFR-focused inference, ensemble and phase-space analyses and for the CBC 2026 poster figures. They are kept unchanged as provenance. Run them from the repository root (see [`../../README.md`](../../README.md)).

All of them work from an existing fitted **distributive** networkmodel run (default folder `results_network_distributive/`, not in the repository).

## Scripts

| Script | What it does | Main inputs | Default outputs | Poster figure(s) produced |
|---|---|---|---|---|
| `run_egfr_inference.py` | EGFR-specific profile likelihoods and full-network NUTS posterior from a saved fit. `--stage check\|profile\|posterior\|both\|summarize` (default `check`). Changes into the repository root itself. | `--source` (default `results_network_distributive`) | `--output` (default `results_network_distributive/egfr_inference/`): e.g. `profile_likelihood_summary.csv`, `egfr_posterior_samples.csv`, `egfr_posterior_summary.csv`, `profile_*.png`, `posterior_*.png` | TODO(owner) |
| `reproduce_poster_plots.py` | Redraws the poster figures from numerical result exports. Needs only numpy, pandas, scipy and matplotlib. | positional `results` (an extracted `results_network_distributive` folder) | `--output` (default `poster_figures/`): `<name>.pdf` (plus `.png` with `--preview`), `reproduction_manifest.json` | TODO(owner) |
| `egfr_forward_ensemble_72h.py` | Non-MCMC EGFR kinetic-rate ensemble with 72 h forward ODE curves. Unweighted quantiles are prior-predictive. | `--source` (repo root, default `.`), `--results` | `--out` (default `results_network_distributive/egfr_forward_72h/`): `forward_ensemble_10000_points.csv`, `parameter_ensemble.csv`, `observed_data.csv`, `methods_and_diagnostics.json` (`all_forward_draws.npz` with `--save-raw-draws`) | TODO(owner) |
| `phoskintime_phase_portraits_72h.py` | 72 h phase-space analysis of the full network for the fitted model and EGFR parameter draws. | `--source`, `--results`, optional `--parameter-csv` (the `parameter_ensemble.csv` from `egfr_forward_ensemble_72h.py`) | `--out` (default `results_network_distributive/egfr_phase_72h/`): e.g. `egfr_nominal_phase_coordinates.csv`, `egfr_phase_draws.npz`, `egfr_phase_ensemble_quantiles.csv`, `network_PCA_loadings.csv`, figures as `.pdf`/`.png` | TODO(owner) |
| `phoskintime_upstream_flux_from_phase.py` | EGFR site/kinase forward-flux (or drive) ensembles for sites `--sites` (default `S991 S1166`). | `--source`, `--results`, `--phase-dir` (required: output of `phoskintime_phase_portraits_72h.py`) | `--out` (default `results_network_distributive/egfr_upstream_flux_poster/`): `egfr_upstream_flux_or_drive_10000_points.csv`, `methods_and_diagnostics.json`, figures | TODO(owner) |
| `report_fitted_network_odes.py` | Exports the fitted distributive network ODE equations (LaTeX, and PDF unless `--no-compile`), plus the module's rate PDF and time-series plots (skipped with `--skip-native-reports`). | `--results` (default `results_network_distributive`), `--gene` (repeatable) | `--output` (default `<results>/fitted_ode_report/`): e.g. `fitted_network_equations.tex`, `fitted_rate_parameters.csv`, `kinase_to_site_coefficients.csv`, `README.md` | TODO(owner) |
| `phoskintime_dynestyx_smoothing.py` | Dynestyx ODE posterior credible intervals for two target EGFR rates, plus discrete-time residual smoothing. **Also the reference helper** (`load_run`, reference RHS) that the three 72 h scripts load by file path. | `--results`, `--source` (repo root) | `--out` (default `dynestyx_ode_smoothing/`): e.g. `dense_ode_credible_intervals.csv`, `discrete_smoothing_at_observed_times.csv`, `methods_and_diagnostics.json`, `<protein>_dynestyx_ODE_CI_discrete_smoothing.png` | TODO(owner) |
| `phoskintime_dynestyx_smoothing_all_egfr_stiff.py` | Variant of the script above that infers **all** EGFR-specific rates and adds a `--solver` option (`tsit5`/`kvaerno5`) and prior options. | as above | as above: **the same default `--out` and the same file names** | TODO(owner) |

## Dependencies between scripts

- `egfr_forward_ensemble_72h.py`, `phoskintime_phase_portraits_72h.py` and `phoskintime_upstream_flux_from_phase.py` load `phoskintime_dynestyx_smoothing.py` from `--source`. They look for `<source>/scripts/figures/egfr_cbc2026/phoskintime_dynestyx_smoothing.py` first, then for `<source>/scripts/phoskintime_dynestyx_smoothing.py` (the layout before the reorganisation).
- Typical order: `egfr_forward_ensemble_72h.py` → `phoskintime_phase_portraits_72h.py --parameter-csv …/parameter_ensemble.csv` → `phoskintime_upstream_flux_from_phase.py --phase-dir …`.

## Notes (current behaviour, not changed)

- The two dynestyx scripts import `dynestyx` inside functions. `dynestyx` is only installed in the separate `dynestyx` pixi environment (`pixi.toml`, `[feature.dynestyx]`). No pixi task uses that environment, and running these scripts there was not verified during the audit.
- The two dynestyx scripts share the default `--out dynestyx_ode_smoothing` and the same output file names, so running one after the other overwrites the first run's outputs unless `--out` is set.
- `reproduce_poster_plots.py` reads posterior files from `<results>/egfr_smoke_posterior/…`, while `run_egfr_inference.py` writes to `<source>/egfr_inference/` by default. The folder names do not match.
- `dynestyx_ode_smoothing/` and `poster_figures/` at the repository root are git-ignored.
