# Repository layout

A folder-level map of the repository. Each package or folder has its own README with details.

## Top-level directories

| Directory | Contents |
|---|---|
| `.github/` | CI workflows (tests, docs deployment, publishing), issue and pull-request templates, `CODEOWNERS`. |
| `common/` | Shared helpers: Fréchet distance (`common/frechet/`), result-directory provenance (`results.py`), the KinOpt/TFOpt result-file contract (`external_results.py`), utilities (`utils/`). |
| `config/` | Typer CLI (`cli.py`, run with `pixi run phoskintime`), constants, logging, configuration helpers. |
| `dashboard/` | Unified Streamlit dashboard (`app.py`, run with `pixi run dashboard`): workflow registry, command builder, runner, result parser, components and workflow panels. |
| `docs/` | Documentation sources for the documentation site (`mkdocs.yml`). |
| `examples/` | Small example scripts. |
| `global_model/` | Compatibility package so that old pickled dashboard bundles still load; the code lives in `networkmodel/`. |
| `networkmodel/` | Global network-scale ODE model: data loading, topology, JAX/Diffrax backend, optimization, inference, sensitivity, export, legacy Streamlit dashboard. Run with `pixi run networkmodel`. |
| `networkpruning/` | Network preprocessing and pruning helpers used by `networkmodel`. |
| `notebooks/` | Tutorial notebooks (ProtWise, Network Model). |
| `processing/` | Preprocessing (`cleanup.py`, run with `pixi run prep`) and result mapping (`map.py`). Raw inputs go in `processing/raw/`, which is not in the repository. |
| `protwise/` | Per-protein ODE workflow: models, parameter estimation, sensitivity, steady state, knockout, plotting, runner. Run with `pixi run model`. |
| `scripts/` | Stand-alone scripts on top of the library packages (see below). |
| `tests/` | Test suite (`pixi run -e dev test`). |

Not in the repository (git-ignored): `data/` (processed inputs written by `pixi run prep`, and the
KinOpt/TFOpt result workbooks) and result folders such as `results/` and `results_*/`.

Root files include `config.toml` (project configuration), `config_loader.py` (shared
`config.toml` loader), `pixi.toml` / `pixi.lock` (environments and tasks), `pyproject.toml`
(metadata, pytest, coverage and ruff settings), `mkdocs.yml`
(documentation site) and `cliff.toml` (changelog generation).

## `scripts/` subgroups

| Folder | Contents |
|---|---|
| `scripts/analysis/` | Analysis backends behind the dashboard's "Advanced analysis" panel; each takes CLI arguments. |
| `scripts/apps/` | Launcher for the legacy networkmodel dashboard (`pixi run network-dashboard`). |
| `scripts/export/` | Model export: SBML (reaction form) and model documentation. |
| `scripts/figures/egfr_cbc2026/` | EGFR inference, ensembles and CBC 2026 poster figures, kept as provenance. |
| `scripts/legacy/` | Older interactive tools kept for reference. |

## Not in this repository

KinOpt and TFOpt are separate tools. PhosKinTime reads only their result workbooks; see
[External tools](../Documentation/external_tools.md).
