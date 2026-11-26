# CLAUDE.md

Guidance for contributors and coding agents working in this repository.

## Layout

Library packages live at the repository root (no `src/`).

- `common/` – shared code: result-directory contract (`results.py`), KinOpt/TFOpt file contract (`external_results.py`), Fréchet distance, table/LaTeX utilities.
- `config/` – CLI (`config/cli.py`, typer), protwise config parsing, constants, logging setup.
- `config_loader.py`, `config.toml` – networkmodel config loader and the single config file for all workflows.
- `processing/` – preprocessing and mapping (`pixi run prep`).
- `protwise/` – protein-wise ODE models and fitting (`pixi run model`).
- `networkmodel/` – global network model, JAX backend, Bayesian inference (`pixi run networkmodel`).
- `networkpruning/` – network pruning.
- `global_model/` – backward-compatibility shim so old pickled dashboard bundles still load; do not add code here.
- `dashboard/` – unified Streamlit dashboard (`pixi run dashboard`).
- `scripts/` – stand-alone scripts, run as `pixi run python scripts/<group>/<name>.py`:
  - `analysis/` – analysis backends behind the dashboard's "Advanced analysis" panel;
  - `export/` – SBML and model-documentation export (keep the two files together);
  - `figures/egfr_cbc2026/` – EGFR inference and CBC 2026 poster scripts, kept unchanged as provenance;
  - `apps/` – stand-alone Streamlit launchers;
  - `legacy/` – older tools kept for reference.
- `notebooks/` – educational notebooks (03 protwise, 04 networkmodel).
- `examples/` – small runnable examples.
- `tests/` – pytest suite.
- `docs/` – documentation sources (zensical, configured by `mkdocs.yml`).
- `__init__.py`, `__main__.py` at the root keep the legacy `python phoskintime <cmd>` entry working.

`data/` and `results*/` are git-ignored; raw data is not in the repository.

## Import convention

- Import library code by its top-level name: `protwise`, `networkmodel`, `networkpruning`,
  `config`, `config_loader`, `common`, `processing`, `dashboard`.
- Never `import phoskintime.*`, never set `PYTHONPATH=.:..`, and never depend on the
  checkout folder's name.
- pixi puts the repository root on the path: `[activation.env] PYTHONPATH = "$PIXI_PROJECT_ROOT"`
  in `pixi.toml`. This applies to `pixi run`, `pixi shell` and `.envrc` (direnv).
- Scripts do not manipulate `sys.path`; run them through pixi.
- Notebooks are the exception: their setup cell keeps `sys.path.insert(0, str(REPO_ROOT))`,
  because a Jupyter kernel started outside pixi does not get the activation environment.

## Commands

| Task | Command |
|---|---|
| Unit tests | `pixi run -e dev test` |
| Tests with coverage | `pixi run -e dev test-cov` |
| Notebook tests (executes notebooks) | `pixi run -e dev test-notebooks` |
| Import smoke test | `pixi run check-import` |
| Build docs | `pixi run -e docs docs-build` |
| Serve docs locally | `pixi run -e docs docs-serve` |
| CLI | `pixi run phoskintime --help` |
| Dashboard | `pixi run dashboard` |

pytest configuration lives in `pyproject.toml`. `pixi.toml` is the environment manifest;
`pyproject.toml` holds project metadata and tool configuration only.

`pixi run phoskintime all` runs preprocessing, checks the KinOpt/TFOpt result files, then
the protwise model. It does not run networkmodel, KinOpt or TFOpt.

## Hard rules

1. **Scientific code is owner-only.** Do not change numerical or scientific code, ODE
   right-hand sides, objectives/loss functions, likelihoods or priors, bounds, state or
   parameter order, units, tolerances, dtypes, random seeds, or any value in `config.toml`
   without the owner's explicit approval. Report the issue instead (see `KNOWN_ISSUES.md`).
2. **Never edit `CITATION.cff` or `LICENSE`.**
3. **KinOpt and TFOpt are external.** Do not add their code back, import them, or install
   them here. PhosKinTime's only interface to them is the result-workbook file contract in
   `common/external_results.py`; change it only together with the External tools page.
4. **No history rewrites**, force-pushes or tag changes.
5. **Do not run full workflows on real data in automated sessions** (prep, model,
   networkmodel, the scripts, or notebook execution). Unit tests and static checks are fine.
6. Do not move a file and edit its content in the same commit.
7. Do not strip or re-execute notebook outputs unless asked.

CI (`.github/workflows/tests.yml`) runs on pushes to `main` and on pull requests. Pushes to
the `global` branch do not trigger it, so run the tests locally before pushing there.

## Pointers

- `KNOWN_ISSUES.md` (repository root) – known issues, including scientific ones that need
  the owner's decision, and a section "Applies to KinOpt/TFOpt".
- `docs/Documentation/external_tools.md` – the External tools page: KinOpt/TFOpt file
  contract, config keys, errors.
- `docs/result_directory_contract.md` – layout of workflow result directories.
- `CONTRIBUTING.md` – contribution workflow.
