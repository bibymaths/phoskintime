# evol — Evolutionary Optimization Framework

The `evol` subpackage within **PhosKinTime** provides a modular and extensible framework for evolutionary optimization of transcriptional regulatory models. It supports multi-objective optimization via `pymoo` and integrates tightly with the broader modeling and estimation infrastructure of the project.

---

## 📁 Module Structure

```
evol/
├── config/      # Global constants and logging setup
├── exporter/    # Output sheet and plotting utilities
├── objfn/       # Objective function and optimization problem
├── opt/         # Optimization algorithm runners (NSGA2, AGEMOEA, SMSEMOA)
├── optcon/      # Data filtering and input tensor construction
├── utils/       # Utility functions for parallelism and parameter organization
├── __main__.py  # Main CLI entry for global TF-mRNA optimization
```

---

## Features

### Global Optimization Engine

- Multi-objective optimization for TF–mRNA modeling
- Supports:
  - NSGA2
  - SMSEMOA
  - AGEMOEA
- Parallel execution using `StarmapParallelization`

### Modular Inputs & Preprocessing

- Preprocessing pipelines to load, clean, and format:
  - mRNA time-series data
  - TF protein signals
  - Phosphorylation sites (PSites)
  - Regulation networks
- Handles inconsistent and missing regulatory mappings

### Objective Functions

- Minimizes 3 objective terms:
  - Prediction error (MSE, MAE, Huber, etc.)
  - Alpha constraint violation (regulatory weights)
  - Beta constraint violation (TF + PSite effects)
- Loss types include MSE, MAE, soft L1, Cauchy, Arctan, Elastic Net, Tikhonov

### Output & Reporting

- Automatically plots model fits and residuals
- Produces Excel reports with:
  - α (regulator strength)
  - β (TF/PSite effect)
  - Residuals, metrics, diagnostics
- Organizes output by gene folders
- Generates summary HTML report

---

## Key Modules

### `__main__.py`

Orchestrates the full optimization pipeline:
- Argument parsing
- Data loading and filtering
- Optimization execution
- Result extraction
- Reporting and plot generation

### `config/`
- `constants.py`: Paths, time points, input/output file setup
- `logconf.py`: Logging configuration

### `exporter/`
- `sheetutils.py`: Save results to multi-sheet Excel
- `plotout.py`: Matplotlib + Plotly plots for expression fits

### `objfn/`
- `minfn.py`: Numba-accelerated multi-objective loss + pymoo `Problem` subclass

### `opt/`
- `optrun.py`: Configures and launches NSGA2, AGEMOEA, SMSEMOA runs

### `optcon/`
- `construct.py`: Builds tensors and fixed-shape matrices for optimization
- `filter.py`: Pre-filters genes and TFs, syncs time axes

### `utils/`
- `params.py`: Parameter layout, initial guess, bounds, index mapping
- `iodata.py`: File parsing, output file organization, and HTML report generation

---

## Usage

You can run the global optimization by executing:

```bash
python -m tfopt.evol
```

Or directly call the main:

```bash
python tfopt/evol/__main__.py --loss_type 0 --optimizer 0
```

With options:
- `--loss_type`: 0 (MSE), 1 (MAE), ..., 6 (Tikhonov)
- `--optimizer`: 0 (NSGA2), 1 (SMSEMOA), 2 (AGEMOEA)

---

## Output Files

- `tfopt_results.xlsx`: Summary of estimated parameters and metrics
- `*.png`, `*.html`: Static + interactive visualizations of mRNA-TF dynamics
- `report.html`: Auto-generated report of all plots and tables
- Subfolder structure: One folder per mRNA or TF

---

## Dependencies

- `pymoo`
- `scikit-learn`
- `matplotlib`
- `plotly`
- `numba`
- `pandas`
- `openpyxl` / `xlsxwriter`

---

## Acknowledgment

Original structure by **Abhinav Mishra**.  
The base implementation of multi-objective optimization was contributed by **Julius Normann** and adapted here with performance improvements.