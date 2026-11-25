# scripts/export/

Model export. The two scripts must stay in the same folder, because `export_model_documentation.py` imports its sibling by module name.

## `export_model_to_sbml.py`

Exports a fitted **distributive** networkmodel run (networkmodel MODEL=0 only) as reaction-based SBML Level 3 Version 2.
- It does not fit and does not integrate the ODE.
- It rebuilds topology, TF weights, kinase inputs, fitted parameters and initial states from the run's own saved exports.
- Before writing, it compares the reaction-form right-hand side against the JAX fitting right-hand side.

- **Inputs:**
  - `--results`: a networkmodel results folder (required). It must contain the saved tables, e.g. `network_W_global.csv`, `network_tf_mat.csv`, `network_kinase_inputs.csv`, `model_parameters_*.csv`, `initial_conditions_y0.csv`, `fitted_params_picked.json`.
  - `--conf`: the original TOML (default `config.toml`).
  - `--run-config`: an optional `posterior_run_config.json`.
- **Outputs:** in `--out` (default `<results>/sbml_reactions/`):
  - `networkmodel_reactions.xml` (SBML)
  - `networkmodel_reaction_graph_edges.csv`
  - `networkmodel_sbml_manifest.json`
- **Other flags:** `--check-only` validates without writing. `--overwrite` allows replacing an existing output.
- **Requires** `python-libsbml` to write SBML. It is in the pixi environment.
- **Tests:** `tests/test_export_model_to_sbml.py`. Two of these tests need an untracked fixture directory, `tests/fitted_ode_test/`, and are skipped without it.

```bash
pixi run python scripts/export/export_model_to_sbml.py \
  --results results_network_distributive --conf config.toml --check-only
```

## `export_model_documentation.py`

Intended to export model documentation (Markdown, LaTeX, PDF) for a results folder (`--results-dir`, `--output-dir`, `--model-family`).

**Status: this script currently cannot start. It exits with status 1 for every invocation, including `--help`.**

- **Cause:** it imports 14 helper functions from `export_model_to_sbml.py`, for example `build_network_genes`, `find_col` and `parse_kinopt_xlsx`. Commit `247c877` (2026-11-15) replaced `export_model_to_sbml.py`, and the new version no longer defines any of them.
- **Why the error message is misleading:** the import sits inside `try … except Exception` (`export_model_documentation.py:44-48`), which hides the `ImportError`. The script then raises `RuntimeError: export_model_documentation.py must be placed next to export_model_to_sbml.py or run from the scripts directory.` That message is wrong: the two files are next to each other.
- **Status:** recorded as finding X9 in the audit. Not fixed.
