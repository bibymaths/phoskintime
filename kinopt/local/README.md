# local: PhosKinTime Local Optimization Module

local is a submodule of the **abopt** framework that implements a local optimization strategy for gene–phosphorylation time-series data. It is tailored for preparing optimized parameter estimates prior to subsequent ODE modelling. The module leverages SciPy’s optimization algorithms (SLSQP or TRUST-CONSTR) combined with Numba-accelerated objective evaluations for efficient computation.

---

## Directory Structure

```
local/
├── config/
│   ├── constants.py         # Defines project paths, time points, and command-line argument parsing. citeturn1file0
│   ├── helpers/             # Auxiliary configuration helpers.
│   ├── __init__.py
│   └── logconf.py           # Logging configuration for clear console and file outputs. citeturn1file1
├── exporter/
│   ├── helpers/             # Additional export helpers.
│   ├── __init__.py
│   ├── plotout.py           # Functions for generating diagnostic plots (fits, residuals, autocorrelation, etc.). citeturn1file2
│   └── sheetutils.py        # Exports results and error metrics to Excel and triggers plotting. citeturn1file3
├── objfn/
│   ├── __init__.py
│   ├── minfn.py             # Numba-accelerated objective and estimated series functions. citeturn1file4
├── opt/
│   ├── __init__.py
│   ├── optrun.py            # Runs the optimization using SciPy’s minimize (SLSQP/TRUST-CONSTR). citeturn1file5
├── optcon/
│   ├── construct.py         # Constructs input matrices, sparse data structures, constraints, and precomputes mappings. citeturn1file6
├── utils/
│   ├── __init__.py
│   ├── iodata.py            # Loads and scales input CSV data; organizes output and generates global HTML reports. citeturn1file7
│   ├── params.py            # Extracts optimized parameters and computes error metrics (MSE, RMSE, MAE, MAPE, R²). citeturn1file8
└── __main__.py              # Entry point that orchestrates data loading, optimization, result export, and reporting. citeturn1file9
```

---

## Features

- **Local Optimization Framework:**  
  Implements the PhosKinTime optimization problem using local solvers (SLSQP or TRUST-CONSTR) to estimate alpha (mixing) and beta (scaling) parameters.

- **Numba-Accelerated Objectives:**  
  Uses Numba to accelerate the evaluation of the objective function and estimated series calculations for enhanced performance. citeturn1file4

- **Flexible Loss Functions & Constraints:**  
  Offers multiple loss functions (e.g., base, weighted, softl1, cauchy, arctan) with built-in constraint handling to ensure parameter normalization across genes and kinases. Constraints are built using both linear and nonlinear approaches depending on the selected optimization method. citeturn1file6

- **Comprehensive Data Handling:**  
  Loads and scales input data from CSV files using several scaling methods (min-max, log, temporal, segmented, etc.), and organizes outputs for further analysis. citeturn1file7

- **Result Export & Reporting:**  
  Generates detailed Excel reports with optimized parameters, error metrics, and diagnostic plots (fit curves, residual histograms, autocorrelation, QQ-plots) to aid in the assessment of model performance. citeturn1file3, citeturn1file2

- **User-Friendly Logging:**  
  Enhanced logging with colored console output provides clear insights into the optimization process and performance statistics. citeturn1file1

---

## Usage

1. **Prepare Input Data:**  
   Place your experimental CSV files (e.g., `input1.csv` and `input2.csv`) in the designated data directory as defined in `config/constants.py`. These files should contain the gene-phosphorylation time-series data and kinase interactions.

2. **Configure Parameters:**  
   Adjust settings such as lower and upper bounds, loss function, estimation of missing kinases, and scaling method through command-line arguments.
   The command-line parsing is handled in `constants.py`. 

3. **Run the Optimization:**  
   Execute the module from top level in the terminal from the root directory: 

   ```bash
   python -m phoskintime kinopt --mode local
   ``` 
   
   This will:
   - Load and scale the input data.
   - Build the necessary matrices and constraint structures.
   - Initialize and run the optimization problem.
   - Compute error metrics (MSE, RMSE, MAE, MAPE, R²).
   - Export results (Excel files and plots) and generate a global HTML report.

---

## Advanced Options

- **Optimization Method:**  
  Choose between "slsqp" and "trust-constr" via the `--method` argument to select the appropriate SciPy optimizer. citeturn1file0

- **Scaling and Preprocessing:**  
  Multiple scaling methods (min-max, log, temporal, segmented, slope, cumulative) are available to normalize your data before optimization. Customize these via command-line options.

- **Reporting & File Organization:**  
  After optimization, the module organizes output files into gene-specific folders and generates a comprehensive HTML report summarizing plots and data tables for further analysis. citeturn1file7

---