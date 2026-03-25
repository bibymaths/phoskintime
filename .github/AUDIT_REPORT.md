# Codebase Audit Report

**Repository:** `bibymaths/phoskintime`  
**Branch:** `global` (main)  
**Date:** 2026-03-25  
**Scope:** Architecture, documentation (README.md, `/docs`, inline docstrings), core engines, and utility scripts.

---

## 1. Missing Elements

### 1.1 Completely Undocumented Modules and Files

| Item | Location | Notes |
|---|---|---|
| `background/` | `background/` (root) | Directory containing images referenced by the package; no `__init__.py`, and its own `README.md` is not referenced from the main project README or docs, so its purpose may be unclear to users. |
| `scripts/` | `scripts/` (root) | Contains seven standalone analysis scripts (`analyze_tf_kin_counts.py`, `compare_mechanisms.py`, `curve_similarity.py`, `export_subnetworks.py`, `find_protein_accumulators.py`, `mechanistic_insights.py`, `temporal_sensitivity.py`). None are mentioned in the README, docs, or any module index. |
| `run_dashboard.py` | `run_dashboard.py` (root) | Root-level entry point that launches the interactive `global_model` Dash dashboard. Not mentioned in README, CLI docs, or the docs portal. |
| `config_loader.py` | `config_loader.py` (root) | Central TOML configuration loader for the global pipeline. Exports the `PhosKinConfig` frozen dataclass and `load_config_toml()`. Critical to `global_model` operation, but absent from all documentation. |
| `frechet/` | `frechet/` (root) | Implements discrete Fréchet distance (`frechet_distance`) via Numba JIT. Used directly in `global_model/runner.py` as a trajectory-similarity metric. No README, no docs entry, no mention in any documentation page. |
| `kinopt/optimality/` | `kinopt/optimality/` | Contains `KKT.py` (post-optimization KKT feasibility, sensitivity reporting, LaTeX tables) and its own `README.md`. Listed in `docs/reference.md` API section but has no dedicated page under `docs/Documentation/`. |
| `global_model/dashboard_app.py` & `dashboard_bundle.py` | `global_model/` | Interactive Plotly/Dash dashboard for global model results. Exposed via `run_dashboard.py` and the `phoskintime-global` CLI, but absent from all narrative documentation. |
| `global_model/scan.py` | `global_model/` | Implements hyperparameter scanning (`run_hyperparameter_scan`). Referenced in `runner.py` when `HYPERPARAM_SCAN=True`. No documentation beyond a function-level docstring. |
| `global_model/refine.py` | `global_model/` | Implements iterative refinement (`run_iterative_refinement`). Called in `runner.py` when `REFINE=True`. No documentation beyond a function-level docstring. |
| `global_model/jacspeedup.py` | `global_model/` | Provides Jacobian acceleration utilities for the ODE solver. Referenced in `docs/reference.md` API list but otherwise undocumented in narrative form. |
| `Dockerfile` | `Dockerfile` (root) | A container definition exists but is never mentioned in the README, `docs/index.md`, or any installation guide section. |

### 1.2 Packaging Gaps (`pyproject.toml`)

The `[tool.poetry.packages]` list omits several packages that are part of the runtime execution path. Although `phoskintime-global` is registered as a CLI entry point pointing to `global_model.runner:main`, the `global_model` package itself is not listed under `packages`, meaning a PyPI installation would silently fail to include it. The same applies to the following:

| Omitted Package | Runtime Dependency? |
|---|---|
| `global_model` | Yes — `phoskintime-global` entry point requires it |
| `frechet` | Yes — imported by `global_model.runner` |
| `processing` | Yes — `prep` CLI command targets `processing.cleanup` |
| `knockout` | Yes — imported by `paramest.core` |
| `config_loader` (module) | Yes — imported by `global_model.runner` and `global_model.config` |

### 1.3 Undocumented Engine Behavior

- **ODE convergence failure handling:** What happens when `scipy.integrate.solve_ivp` or the custom solver fails to converge is not documented. There is no user-facing explanation of error codes, fallback strategies, or minimum required data density.
- **Parallel execution (`ProcessPoolExecutor`):** The README mentions parallel gene processing, but the interaction between the `CORES` environment variable (set in `global_model/runner.py`), the OpenBLAS/MKL thread-pinning environment variables, and `pymoo`'s `StarmapParallelization` is undocumented.
- **Log-space parameter optimization for `randmod`:** The `paramest` README mentions log-space bounds for the random model, but the back-transformation step and its effect on confidence intervals from `paramest/identifiability/ci.py` are not explained.
- **Proxy logic for orphan TFs:** The `global_model/README.md` describes the concept, but the implementation in `global_model/network.py` (`Index.__init__` proxy redirection logic) is not cross-referenced, and the conditions that trigger proxy assignment are not stated.

---

## 2. Areas Requiring Fixes (Documentation/Sync Issues)

### 2.1 `paramest/seqest.py` Referenced but Does Not Exist

**Location:** `README.md` (under "Example" → "Parameter Estimation")  
**Current text:**
> Depending on the chosen estimation mode (sequential or normal), functions from `paramest/seqest.py` or `paramest/normest.py` are used.

**Reality:** `paramest/seqest.py` does not exist in the repository. The `paramest/toggle.py` function `estimate_parameters()` only calls `normest()`. There is no sequential estimation mode in the current codebase. The docstring for `toggle.py` also incorrectly implies mode-switching logic:
> "This function allows for the selection of the estimation mode"

Both the README and `toggle.py`'s docstring must be corrected to reflect that only `normest` is currently available.

### 2.2 Python Compatibility Discrepancy

| Location | Stated Requirement |
|---|---|
| `docs/index.md` (line 34) | "Python 3.8+" |
| `README.md` (Installation section) | "python 3.10 or higher" |
| `pyproject.toml` | `python = ">=3.10,<3.14"` |

The `docs/index.md` claim of Python 3.8 compatibility is incorrect and contradicts both the README and the enforced build requirement. Users on Python 3.8 or 3.9 will encounter installation failures.

### 2.3 Duplicate and Mislabelled "models" Entry in `docs/index.md`

**Location:** `docs/index.md`, "Core Modules" section (lines 93–99)  
The section lists `**models**` twice. The second entry reads:
> Parameter estimation routines for ODE models.

This description belongs to `paramest`, not `models`. The `models` module is already correctly described in the first entry. The second entry should either be corrected to reference `paramest` with the appropriate header, or removed.

### 2.4 `all` CLI Command Excludes `global_model` Stage

**Location:** `config/cli.py` (`all` command), `README.md`, `docs/index.md`  
The `all` command chains `prep → tfopt → kinopt → model` but never invokes `global_model`. Because the `global_model` command is a peer CLI command (not a sub-stage of `model`), running `python phoskintime all` silently skips the global network simulation. The documentation does not warn users about this omission, creating a false expectation that `all` runs the complete pipeline.

### 2.5 `CHANGELOG.md` Version 0.4.0 Date Discrepancy

| File | Version 0.4.0 Header |
|---|---|
| `CHANGELOG.md` (root) | `## [0.4.0] – 2025-06-05` |
| `docs/Documentation/CHANGELOG.md` | `## [0.4.0] – Unreleased` |

The two files are otherwise identical. The root file marks 0.4.0 as released on 2025-06-05 while the documentation copy still shows it as unreleased. One copy should be the source of truth.

### 2.6 `phoskintime-global` Entry Point vs. Packaging Inconsistency

**Location:** `pyproject.toml`  
The file registers:
```toml
phoskintime-global = "global_model.runner:main"
```
Yet `global_model` is absent from `packages`. Any user installing the package from PyPI and running `phoskintime-global` will receive a `ModuleNotFoundError` at runtime.

### 2.7 `docs/index.md` Incorrectly Describes `global_model/README.md` Link

**Location:** `docs/index.md` line 72  
> For the full mathematical specification of the coupled global system (all equations), see: `global_model/README.md`

This is a bare file-system path, not a hyperlink. In the MkDocs-rendered portal it renders as plain text, not a navigable link, and users cannot follow it from the documentation site.

---

## 3. Completeness Gaps

### 3.1 No Architecture or Data-Flow Diagram

Despite the system's complexity (three coupled biological layers, two distinct optimization stacks, global vs. local model separation), no architecture diagram or end-to-end data-flow illustration exists in any documentation. Even a simple Mermaid or ASCII diagram showing how data moves from raw input → `processing` → `kinopt`/`tfopt` → `global_model` → outputs would substantially reduce onboarding friction.

### 3.2 Input Data Format Specification

The documentation for `processing/README.md` lists the expected output files but does not specify the required input column schema for each file (`CollecTRI.csv`, `MS_Gaussian_updated_09032023.csv`, `Rout_LimmaTable.csv`, `input2.csv`). Users supplying their own data have no reference for what columns must be present, what types they must be, or what naming conventions must be followed.

### 3.3 `config.toml` Schema Not Documented

The `config.toml` file at the repository root is the central configuration for the global model, parsed by `config_loader.py`. Its full schema (all valid top-level sections, sub-keys, types, defaults, and constraints) is not documented anywhere. The only way to discover valid keys is by reading `config_loader.py` directly.

### 3.4 Relationship Between `model` and `global_model` CLI Commands Is Unclear

The README and docs describe the `model` command (which runs `bin/main.py`, the local per-protein ODE pipeline) and the `global_model` command (which runs `global_model/runner.py`, the network-scale system). However, no documentation explains:
- When to use one versus the other.
- Whether `global_model` depends on outputs from `kinopt`/`tfopt`.
- Whether both can be run independently or sequentially.

### 3.5 No Documentation for `scripts/` Utility Scripts

The seven scripts in `scripts/` perform non-trivial post-processing analyses (mechanism comparison, subnetwork export, temporal sensitivity, curve similarity). Users would benefit from even brief per-script descriptions, their expected inputs/outputs, and when in the workflow to invoke them.

### 3.6 `Dockerfile` Not Covered in Installation Guide

The Dockerfile provides a containerized execution path (useful for reproducibility), but the installation guide covers only local environment scenarios (pip, Poetry, uv, Conda). A "Scenario 5: Docker" section is absent.

### 3.7 `PYPI_README.md` vs `README.md` Relationship Unexplained

Two top-level README files exist. `pyproject.toml` references `PYPI_README.md` as the PyPI readme, while `README.md` is the GitHub-facing file. There is no comment or note explaining this split, potentially causing contributors to update one and not the other.

### 3.8 Interactive Dashboard Usage Not Documented

The `run_dashboard.py` script launches a Plotly/Dash dashboard for exploring global model results. No documentation describes:
- How to start the dashboard.
- What results files it requires.
- What port it serves on.
- What browser-based interactions are available.

### 3.9 Sensitivity Analysis: Two Different Methods With the Same Name

The package contains two distinct sensitivity analysis implementations:
- `sensitivity/analysis.py`: Morris method (local ODE models, per-protein).
- `global_model/sensitivity.py`: Trajectory-based perturbation analysis (global coupled model).

Both are referred to as "sensitivity analysis" in documentation without distinguishing which applies to which workflow. The `docs/Documentation/sensitivity/README.md` exclusively covers the Morris method and makes no mention of the global model sensitivity routine.

### 3.10 Fréchet Distance as a Fit Metric Is Not Explained

`frechet/distance.py` computes a discrete Fréchet distance and is imported in `global_model/runner.py`. The rationale for using this metric (as opposed to RMSE or other residual-based metrics), its role in the loss aggregation pipeline, and any normalization or thresholding applied are not explained anywhere.

### 3.11 Test Coverage Scope Not Documented

`pytest.ini` and a `tests/` directory exist (with `test_config.py` as the only test file). There is no documentation of what is and is not covered by tests, how to run the test suite, or what the testing philosophy is (unit vs. integration). Contributors adding new code have no guidance on what test patterns to follow.

---

## 4. Duplication

### 4.1 Logging Configuration Repeated Across `config` and Four Subpackages

The logging setup pattern (colored console formatter + rotating file handler) is independently re-implemented in:
- `config/logconf.py`
- `kinopt/local/config/logconf.py`
- `kinopt/evol/config/logconf.py`
- `tfopt/local/config/logconf.py`
- `tfopt/evol/config/logconf.py`

All five files define nearly identical `ColoredFormatter` classes and `setup_logger` functions with minor parameter differences. A single shared `config/logconf.py` already exists; the subpackage copies create maintenance risk (e.g., a log format change must be applied in five places).

### 4.2 Parallel Mirror Directory Structures in `kinopt` and `tfopt`

`kinopt/evol/` and `kinopt/local/` are structurally identical (both contain `config/`, `exporter/`, `objfn/`, `opt/`, `optcon/`, `utils/`, `__main__.py`). The same pattern is repeated in `tfopt/evol/` and `tfopt/local/`. While the design intent (shared interface, different solver backends) is acknowledged in the docs, there is no shared base layer or strategy abstraction to prevent logic drift between the four parallel implementations over time.

### 4.3 Two Overlapping Configuration Systems

`config_loader.py` contains the comment:
```python
# sync with global_model/config.py
```
This acknowledges that `config_loader.py` (`PhosKinConfig` dataclass + `load_config_toml()`) and `global_model/config.py` (module-level constants loaded from the same `config.toml`) define overlapping configuration surfaces. Constants such as `TIME_POINTS_PROTEIN`, `RESULTS_DIR`, `SEED`, and `REGULARIZATION_LAMBDA` exist in both places, creating a risk that one is updated while the other is not.

### 4.4 README Content Duplicated Verbatim Between `README.md` and `docs/index.md`

The following sections are duplicated word-for-word between the two files:
- Complete CLI command reference (all `python phoskintime` examples).
- Full installation guide (all four scenarios with all prerequisite steps).

Any update to one file must be manually mirrored to the other.

### 4.5 `processing/README.md` and `docs/Documentation/processing/README.md` Are Identical

Both files contain exactly the same content. One is the canonical source; the other should either be a symlink or removed in favor of a single authoritative file.

### 4.6 `CHANGELOG.md` Maintained in Two Locations

`CHANGELOG.md` at the repository root and `docs/Documentation/CHANGELOG.md` are nearly identical (diverging only on the version 0.4.0 release date). Changes to one must be manually applied to the other.

### 4.7 `global_model/README.md` and `docs/Documentation/global/README.md` Are Identical

Both files contain the same mathematical framework, notation, and parameter dictionary for the global model. As with the other duplicates above, this creates a split-brain maintenance problem.

### 4.8 Two Independently Named Sensitivity Analysis Routines

- `sensitivity/` (package) → Morris-based, local ODE models.
- `global_model/sensitivity.py` (module) → Trajectory perturbation, global model.

Both are called "sensitivity analysis" in their respective documentation sections without a clear namespace or naming convention distinguishing them. Users new to the codebase have no guidance on which to invoke, when, or why.
