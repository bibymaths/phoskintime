# local — Local Constrained Optimization Framework

The `local` subpackage provides a constrained optimization backend for fitting transcriptional regulatory models to gene expression time-series data using local solvers like `SLSQP`. It is designed for precision optimization when global heuristics (e.g., genetic algorithms) are not necessary or for refinement after global search.

---

## Module Structure

```
local/
├── config/      # Global paths, constants, logging
├── exporter/    # Sheet output and plots
├── objfn/       # Objective function and model predictions
├── opt/         # Optimization runners using SciPy solvers
├── optcon/      # Data construction and constraints
├── utils/       # I/O, param setup, report generation
├── __main__.py  # CLI entrypoint
```

---

## Features

### Constrained Local Optimization

- Uses **SLSQP** to fit transcriptional model parameters
- Minimizes a multi-part objective:
  - Fit error (MSE, MAE, etc.)
  - α parameter constraints (sum to 1 across TFs)
  - β parameter constraints (sum to 1 across protein + PSites)
- Supports regularization (Elastic Net, Tikhonov)

### Parameter Estimation

- Per-mRNA TF weighting (`α`)
- Per-TF protein & PSite influence (`β`)
- Handles missing PSite data
- Bounds and constraints built dynamically from data

### Output & Visualization

- Saves α and β values to Excel
- Computes and logs fit metrics (MSE, MAE, MAPE, R²)
- Generates plots:
  - Model fit curves (static & interactive)
  - Residuals
  - PCA, KLD, boxplots
  - CDFs, heatmaps

- HTML report builder auto-organizes all output per mRNA

---

## Key Modules

### `__main__.py`

The orchestrator script that:
- Parses CLI args
- Loads and filters input data
- Builds optimization arrays and constraints
- Runs optimization
- Saves results
- Generates visualizations and report

Run directly from one top level of the root directory via:

```bash
python -m phoskintime tfopt --mode local
```

### `config/`

- `constants.py`: Data file paths, default inputs, output folders, time points
- `logconf.py`: Centralized logging
- `parse_args()`: Handles bounds and loss type from CLI

### `objfn/`

- `minfn.py`: Numba-accelerated loss and prediction functions
- Supports loss types:
  - 0: MSE
  - 1: MAE
  - 2: soft L1
  - 3: Cauchy
  - 4: Arctan
  - 5: Elastic Net
  - 6: Tikhonov

### `opt/`

- `optrun.py`: Runs `scipy.optimize.minimize` with method SLSQP

### `optcon/`

- `construct.py`: Builds TF/mRNA arrays, phosphorylation tensors, constraints
- `filter.py`: Loads raw data and filters gene-TF mappings

### `exporter/`

- `plotout.py`: Generates fit plots (matplotlib + Plotly)
- `sheetutils.py`: Exports all data to Excel with multiple formatted sheets

### `utils/`

- `iodata.py`: Loads CSV input data, normalization, HTML report generation
- `params.py`: Sets up bounds, constraints, initial guess, and postprocessing logic

---

## Output Artifacts

- `tfopt_results.xlsx`: All α/β parameters, metrics, and errors
- One folder per gene, containing:
  - Fit plots (`.png`, `.html`)
  - Residual plots
  - Intermediate CSV/XLSX files
- Global HTML report aggregating all outputs

---

## Acknowledgments 

This module was developed by **Abhinav Mishra** as part of the PhosKinTime project, with base optimization ideas adapted  
from early global frameworks. The local variant enables refined and faster convergence for small- to medium-scale systems.
The base implementation of single-objective optimization was contributed by **Julius Normann** and adapted here with performance  
improvements tailored for local constrained optimization.