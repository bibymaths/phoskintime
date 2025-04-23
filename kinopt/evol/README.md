# evol: PhosKinTime Evolutionary Optimization Module

evol is a submodule of the **abopt** framework designed for optimizing gene–phosphorylation site time-series data. It formulates and solves custom optimization problems to estimate phosphorylation levels by tuning model parameters (alphas and betas) under biological constraints. The module supports multiple loss functions and optimization methods, making it flexible for various experimental setups.

---

## Directory Structure

```
evol/
├── config/
│   ├── constants.py         # Project paths, time points, and argument parsing.
│   ├── logconf.py           # Logging configuration.
│   └── helpers/             # Additional configuration helpers.
├── exporter/
│   ├── plotout.py           # Functions for plotting residuals, objective spaces, and convergence.
│   └── sheetutils.py        # Exports results to Excel sheets.
├── objfn/
│   ├── minfndiffevo.py      # Single-objective optimization problem formulation.
│   └── minfnnsgaii.py       # Multi-objective optimization problem formulation.
├── opt/
│   └── optrun.py            # Runs the optimization using DE or NSGA-II (via pymoo).
├── optcon/
│   └── construct.py         # Constructs input arrays and problem data (P_initial, K_array, etc.).
├── utils/
│   ├── iodata.py            # Handles data input/output and file organization.
│   └── params.py            # Extracts and organizes optimization parameters.
└── __main__.py              # Entry point that ties all components together.
```

---

## Features

- **Custom Optimization Problems:**  
  Implements both single- and multi-objective formulations tailored for phosphorylation analysis.
  
- **Flexible Loss Functions:**  
  Supports various loss types including mean squared error, Huber, MAPE, and autocorrelation-based metrics—with optional L1/L2 regularization.
  
- **Multiple Optimization Methods:**  
  Choose between Differential Evolution (DE) or NSGA-II for solving the optimization problem.
  
- **Data Construction and Scaling:**  
  Pre-processes experimental CSV data, applies scaling, and builds the necessary input structures for optimization.
  
- **Comprehensive Exporting:**  
  Outputs results as Excel sheets and generates plots (residuals, objective space, convergence, etc.) for in-depth analysis.

---

## Usage

1. **Prepare Input Data:**  
   Place your experimental data CSV files (e.g., `input1.csv` and `input2.csv`) in the designated data directory as defined in `config/constants.py`.

2. **Configure Parameters:**  
   Adjust settings such as bounds, loss function, regularization, and scaling method by either editing `constants.py` or passing command-line arguments.

3. **Run Optimization:**  
   Execute the module from one top level up in the terminal from root:

   ```bash
   python -m phoskintime kinopt --mode evol 
   ``` 
   
   This will:
   - Load and scale the input data.
   - Build the optimization problem.
   - Run the optimization using the chosen algorithm.
   - Export results (Excel files and plots) to the output directory.

---

## Advanced Options

- **Optimization Methods:**  
  Set the `METHOD` parameter in `config/constants.py` (or via command-line) to either `DE` for Differential Evolution or `NSGA-II` for a multi-objective approach.

- **Loss Functions & Regularization:**  
  Customize the objective via the `--loss_type` argument and enable regularization with `--regularization yes`.

- **Output & Reporting:**  
  Results including optimized parameters, residuals, and diagnostic plots are saved in the output directory. Additional post-processing functions generate convergence and waterfall plots for performance evaluation.

---
