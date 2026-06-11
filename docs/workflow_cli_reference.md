# Workflow CLI reference

This reference lists actual Pixi tasks and Typer wrapper commands present in the repository. The CLI remains the source of truth for workflow execution; the dashboard builds and previews these commands.

## Config precedence

For workflows that support `--conf`, the dashboard currently validates assigned config files as TOML because the target runners parse TOML:

```text
Explicit CLI flags override values from --conf.
Values from --conf override repository defaults.
If --conf is not supplied, repository config.toml is used.
```

Do not assume a custom config was used unless `metadata.json` and `config_resolved.*` confirm it.

## Pixi tasks

| Task | Purpose | Backend command/module | Example |
| --- | --- | --- | --- |
| `prep` | Preprocessing cleanup. | `python -m processing.cleanup` | `pixi run prep` |
| `kinopt-local` | Local KinOpt. | `python -m kinopt.local` | `pixi run kinopt-local --outdir results/example_kinopt` |
| `kinopt-evol` | Evolutionary KinOpt. | `python -m kinopt.evol` | `pixi run kinopt-evol` |
| `kinopt-fitanalysis` | KinOpt fit analysis. | `python -m kinopt.fitanalysis` | `pixi run kinopt-fitanalysis` |
| `tfopt-local` | Local TFOpt. | `python -m tfopt.local` | `pixi run tfopt-local --outdir results/example_tfopt` |
| `tfopt-evol` | Evolutionary TFOpt. | `python -m tfopt.evol` | `pixi run tfopt-evol` |
| `model` | ProtWise model runner. | `python -m protwise.runner.main` | `pixi run model --conf config.toml --outdir results/example_protwise` |
| `networkmodel` | Global network model runner. | `python -m networkmodel.runner` | `pixi run networkmodel --conf config.toml --output-dir results/example_networkmodel` |
| `phoskintime` | Typer workflow wrapper. | `python -m phoskintime.config.cli` | `pixi run phoskintime --help` |
| `phoskintime-all` | Typer wrapper `all` command. | `python -m phoskintime.config.cli all` | `pixi run phoskintime-all` |
| `network-dashboard` | Legacy Networkmodel dashboard launcher. | `python run_dashboard.py` | `pixi run network-dashboard` |
| `dashboard` | Unified no-code dashboard. | `streamlit run dashboard/app.py` | `pixi run dashboard` |
| `dashboard-dev` | Unified dashboard with reload-on-save. | `streamlit run dashboard/app.py --server.runOnSave=true` | `pixi run dashboard-dev` |
| `test` | Run pytest with project path. | `PYTHONPATH=.:.. pytest` | `pixi run -e dev test` |
| `test-notebooks` | Execute notebooks as tests. | `pytest --nbmake notebooks/*.ipynb` | `pixi run -e dev test-notebooks` |
| `test-cov` | Coverage test run. | `pytest --cov=...` | `pixi run -e dev test-cov` |
| `check-import` | Import smoke check. | Python one-liner | `pixi run check-import` |
| `docs-serve` | Serve documentation. | `zensical serve` | `pixi run -e docs docs-serve` |
| `docs-build` | Build documentation. | `zensical build` | `pixi run -e docs docs-build` |

## Typer wrapper commands (`config/cli.py`)

These commands are available through the package CLI wrapper when invoked as the configured module.

### `prep`

- Purpose: run preprocessing cleanup.
- Backend module: `processing.cleanup`.
- Required inputs: project preprocessing inputs.
- Config support: no explicit `--conf` on wrapper command.
- Output directory flag: none.
- Example:

```bash
pixi run phoskintime prep
```

Dashboard integration: registered as `prep`.

### `kinopt`

- Purpose: run KinOpt local or evolutionary mode.
- Backend module: `kinopt.<mode>`.
- Required inputs: KinOpt inputs configured in `config.toml`/backend constants.
- Config support: `--conf`.
- Output directory flag: `--outdir` / `--output-dir` for local mode.
- Optional arguments: `--mode local|evol`.
- Example:

```bash
pixi run phoskintime kinopt --mode local --conf config.toml --outdir results/example_kinopt
```

Dashboard integration: local mode registered as `kinopt-local`.

### `tfopt`

- Purpose: run TFOpt local or evolutionary mode.
- Backend module: `tfopt.<mode>`.
- Required inputs: TFOpt inputs configured in `config.toml`/backend constants.
- Config support: `--conf`.
- Output directory flag: `--outdir` / `--output-dir` for local mode.
- Optional arguments: `--mode local|evol`.
- Example:

```bash
pixi run phoskintime tfopt --mode local --conf config.toml --outdir results/example_tfopt
```

Dashboard integration: local mode registered as `tfopt-local`.

### `model`

- Purpose: run ProtWise protein-wise ODE fitting.
- Backend module: `protwise.runner.main`.
- Required inputs: protein CSV, phosphosite Excel workbook, RNA/TFOpt Excel workbook.
- Config support: `--conf`; custom config values are used unless overridden by explicit CLI fields.
- Output directory flag: `--outdir` / `--output-dir`.
- Example:

```bash
pixi run phoskintime model --conf config.toml --outdir results/example_protwise
```

Dashboard integration: registered as `protwise-model` / Pixi task `model`.

### `networkmodel`

- Purpose: run the integrated global network model.
- Backend module: `networkmodel.runner`.
- Required inputs: kinase network CSV, TF network CSV, MS/protein CSV, RNA CSV, optional phosphoproteomics CSV, optional KinOpt/TFOpt Excel priors.
- Config support: `--conf`; custom config values are used unless overridden by explicit CLI fields.
- Output directory flag: `--outdir` / `--output-dir`; dashboard uses `--output-dir`.
- Example:

```bash
pixi run phoskintime networkmodel --conf config.toml --outdir results/example_networkmodel
```

Dashboard integration: registered as `networkmodel`.

### `all`

- Purpose: run preprocessing, TFOpt, KinOpt, and ProtWise model stages in sequence.
- Backend modules: `processing.cleanup`, `tfopt.<mode>`, `kinopt.<mode>`, `protwise.runner.main`.
- Config support: `--tf-conf`, `--kin-conf`, `--model-conf`.
- Output directory flag: `--outdir` / `--output-dir` as a base directory for stage outputs.
- Example:

```bash
pixi run phoskintime-all --outdir results/example_all
```

Dashboard integration: registered as `phoskintime-all`.

### `clean`

- Purpose: remove bytecode, cache, and build artifacts.
- Backend: filesystem cleanup in `config/cli.py`.
- Config support: none.
- Output directory flag: none.
- Example:

```bash
pixi run phoskintime clean
```

Dashboard integration: not a dashboard workflow.

## Workflow details

### Preprocessing / `prep`

- Command name: `prep`.
- Purpose: cleanup/preprocessing.
- Backend module: `processing.cleanup`.
- Required inputs: repository raw/processing inputs.
- Optional arguments: none in the Pixi task.
- Config file support: not exposed by task.
- Output directory flag: none.
- Expected outputs: preprocessing outputs defined by `processing.cleanup`.
- Dashboard status: executable from dashboard registry; result browsing is limited unless outputs follow the result contract.

### KinOpt local

- Command name: `kinopt-local`.
- Purpose: local kinase/phosphosite optimization.
- Backend module: `kinopt.local`.
- Required inputs: configured KinOpt input files, commonly CSV files under `data/`.
- Optional arguments include `--conf`, `--lower_bound`, `--upper_bound`, `--loss_type`, and `--method` where supported.
- Output directory flag: `--outdir` / `--output-dir`.
- Expected outputs: `kinopt_results.xlsx`, tables, plots, reports, logs, provenance.
- Dashboard status: executable and result-browsable.

### KinOpt evol

- Command name: `kinopt-evol`.
- Purpose: evolutionary KinOpt optimization.
- Backend module: `kinopt.evol`.
- Inputs/config: backend-defined.
- Output directory flag: not dashboard-standardized in the Pixi task.
- Dashboard status: not directly registered for dashboard execution; produced folders can be browsed if layout is recognized.

### TFOpt local

- Command name: `tfopt-local`.
- Purpose: local TF/mRNA optimization.
- Backend module: `tfopt.local`.
- Required inputs: configured TFOpt files, commonly CSV files under `data/`.
- Optional arguments include `--conf`, `--lower_bound`, `--upper_bound`, and `--loss_type` where supported.
- Output directory flag: `--outdir` / `--output-dir`.
- Expected outputs: `tfopt_results.xlsx`, tables, plots, reports, logs, provenance.
- Dashboard status: executable and result-browsable.

### TFOpt evol

- Command name: `tfopt-evol`.
- Purpose: evolutionary TFOpt optimization.
- Backend module: `tfopt.evol`.
- Inputs/config: backend-defined.
- Output directory flag: not dashboard-standardized in the Pixi task.
- Dashboard status: not directly registered for dashboard execution; produced folders can be browsed if layout is recognized.

### ProtWise / model

- Command name: `model`.
- Purpose: protein-wise ODE fitting.
- Backend module: `protwise.runner.main`.
- Required inputs: protein CSV, phosphosite Excel workbook, RNA/TFOpt Excel workbook.
- Config support: `--conf`.
- Output directory flag: `--outdir` / `--output-dir`.
- Example:

```bash
pixi run model --conf config.toml --outdir results/example_protwise
```

- Expected outputs: result workbook, plots, report, logs, provenance, standard folders.
- Dashboard status: executable and result-browsable.

### Networkmodel

- Command name: `networkmodel`.
- Purpose: global coupled signaling/GRN model.
- Backend module: `networkmodel.runner`.
- Required inputs: kinase network CSV, TF network CSV, MS/protein CSV, RNA CSV, optional phosphoproteomics CSV; optional prior workbooks from KinOpt/TFOpt.
- Config support: `--conf`.
- Output directory flag: `--output-dir` / `--outdir`.
- Example:

```bash
pixi run networkmodel --conf config.toml --output-dir results/example_networkmodel
```

- Expected outputs: scalar objective/convergence CSVs, predictions, `dashboard_bundle.pkl`, plots, optimization/profile/posterior outputs, logs, provenance.
- Dashboard status: executable and result-browsable.

### Dashboard

- Commands: `dashboard`, `dashboard-dev`, `network-dashboard`.
- Purpose: `dashboard` and `dashboard-dev` run the unified dashboard; `network-dashboard` runs the legacy Networkmodel dashboard.
- Backend modules: `dashboard/app.py` or `run_dashboard.py`.
- Example:

```bash
pixi run dashboard
pixi run dashboard-dev
```

### Tests and docs

- Test command:

```bash
pixi run -e dev test
```

- Focused dashboard tests:

```bash
pixi run -e dev pytest tests/dashboard -v
```

- Docs build:

```bash
pixi run -e docs docs-build
```

## Standalone analysis scripts

The `scripts/` directory contains standalone utilities, including:

- `analyze_tf_kin_counts.py`
- `compare_mechanisms.py`
- `curve_similarity.py`
- `export_subnetworks.py`
- `find_protein_accumulators.py`
- `kinopt_network_readout.py`
- `kinopt_network_viz.py`
- `make_kinopt_diagram.py`
- `mechanistic_insights.py`
- `temporal_sensitivity.py`
- `tfopt_network_readout.py`
- `tfopt_network_viz.py`

These are not all Pixi tasks. Run them directly only after reading each script's arguments and expected inputs. When integrating a script into the dashboard, parameterize input/output paths and write outputs into the selected run directory under `tables/`, `plots/`, `reports/`, or `artifacts/`.

## File-format constraints

- Do not document YAML/JSON execution config support unless the target runner implements YAML/JSON loading; dashboard `--conf` assignments are restricted to `.toml`.
- Do not document TSV/XLSX execution support for Networkmodel network/data inputs unless the backend reader is extended beyond default `pandas.read_csv`.
- Do not document Excel support for ProtWise protein input unless the backend reader changes from `pd.read_csv`.
- Dashboard upload preview can read more formats than a specific workflow can execute.
