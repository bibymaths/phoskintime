# kinopt: A Comprehensive Optimization Framework for PhosKinTime

**kinopt** is a modular framework designed for the analysis and optimization of gene–phosphorylation time-series data. It integrates several specialized submodules that cater to different optimization strategies and post-processing analyses. Whether you need a global evolutionary approach, a local constrained optimization, or a Julia-based Powell optimization routine, **kinopt** offers the tools to process your experimental data and generate in-depth reports on model performance.

---

## Directory Structure

```
kinopt/
├── data
│   ├── input1.csv             # Primary input data file with gene-phosphorylation data.
│   └── input2.csv             # Interaction data file containing kinase information.
├── evol
│   ├── config                 # Configuration files (constants, logging, etc.) for the evolutionary approach.
│   ├── exporter               # Plotting and Excel sheet export functions.
│   ├── __init__.py
│   ├── __main__.py            # Entry point for global optimization using evolutionary algorithms.
│   ├── objfn                  # Objective function implementations (single- and multi-objective).
│   ├── opt                    # Optimization routines (integration with pymoo).
│   ├── optcon                 # Functions to construct input data, constraints, and precomputed mappings.
│   ├── __pycache__
│   ├── README.md              # Detailed readme for the evol module.
│   └── utils                  # Utility functions for data I/O and parameter extraction.
├── fitanalysis
│   ├── helpers                # Auxiliary scripts for additional performance evaluation.
│   ├── __init__.py
│   ├── __main__.py            # Entry point for fit analysis.
│   └── __pycache__
├── __init__.py
├── local
│   ├── config                 # Configuration files specific to local optimization.
│   ├── exporter               # Functions for exporting local optimization results and diagnostic plots.
│   ├── __init__.py
│   ├── __main__.py            # Entry point for local optimization (SLSQP/TRUST-CONSTR based).
│   ├── objfn                  # Local objective function implementations with Numba acceleration.
│   ├── opt                    # Local optimization routines using SciPy.
│   ├── optcon                 # Construction of local constraints and precomputation of mappings.
│   ├── __pycache__
│   ├── README.md              # Detailed readme for the local module.
│   └── utils                  # Utilities for data scaling, file organization, and report generation.
├── optimality
│   ├── __init__.py
│   ├── KKT.py                 # Post-optimization analysis: feasibility, sensitivity, and reporting.
│   ├── __pycache__
│   └── README.md              # Detailed readme for the optimality module.
├── powell
│   ├── __init__.py
│   ├── __main__.py            # Entry point for Julia-based Powell optimization.
│   ├── powell.jl              # Julia script implementing Powell’s method.
│   ├── __pycache__
│   ├── README.md              # Detailed readme for the powell module.
│   └── runpowell.py           # Python script to run the Powell optimization and log output.
├── __pycache__
│   ├── __init__.cpython-313.pyc
│   └── __main__.cpython-313.pyc
└── README.md                  # This file.
```

---

## Overview

**kinopt** provides an end-to-end solution for:

- **Data Preparation:**  
  Preprocess and scale input CSV files containing time-series data and kinase interactions.

- **Global Optimization (evol):**  
  Uses evolutionary algorithms (via pymoo) to search the global parameter space for optimal α (mixing) and β (scaling) values.

- **Local Optimization (local):**  
  Implements local constrained optimization using SciPy's solvers (SLSQP or TRUST-CONSTR) with efficient objective evaluation via Numba.

- **Optimality Analysis (optimality):**  
  Post-processes optimization results to check constraint feasibility, perform sensitivity analysis, generate LaTeX summary tables, and produce diagnostic plots.

- **Powell Optimization (powell):**  
  Integrates Julia-based Powell optimization routines with automatic multi-threading to leverage high-performance computing resources.

- **Fit Analysis (fitanalysis):**  
  Provides additional tools to evaluate the fit and performance of the optimized model.

---

## Features

- **Modular Architecture:**  
  Each submodule (evol, local, optimality, powell, fitanalysis) is designed to operate independently while integrating seamlessly into the overall workflow.

- **Flexible Optimization Strategies:**  
  Choose between global evolutionary algorithms, local constrained solvers, or Powell-based methods depending on your specific needs.

- **Robust Post-Processing:**  
  Comprehensive post-optimization analysis includes constraint validation, sensitivity analysis, detailed reporting (both in LaTeX and Excel), and extensive plotting of diagnostic metrics.

- **Automated Reporting:**  
  After running optimization routines, the framework organizes outputs into structured directories and generates a global HTML report summarizing key results and diagnostic plots.

- **User-Friendly Logging:**  
  Custom logging configurations provide real-time feedback during execution, ensuring transparency in the optimization process.

---

## Installation

### Prerequisites

- **Python 3.x**
- **Julia:** Required for the Powell module.
- **System Utilities:** `lscpu` for thread detection on Linux systems.

### Python Dependencies

Install the necessary Python packages using pip:

```bash
pip install numpy scipy pandas scikit-learn matplotlib seaborn statsmodels numba openpyxl
```

Ensure that Julia is installed and accessible via the command line.

---

## Usage

### Running Global Optimization (evol)

Navigate to the `evol` directory and run:

```bash
python -m evol
```

### Running Local Optimization (local)

Navigate to the `local` directory and run:

```bash
python -m local
```

### Running Powell Optimization (powell)

To execute the Powell module, run:

```bash
python -m kinopt.powell
```

This will automatically run the Julia script, process the results, and generate the necessary reports.

### Fit Analysis

For additional model fit evaluation, navigate to the `fitanalysis` directory and run:

```bash
python -m fitanalysis
```

### Post-Optimization Processing

After any optimization run, the **optimality** module is invoked (either directly or as part of the workflow) to analyze the results, validate constraints, and generate comprehensive reports.

---

## Contributing & License

Contributions are welcome! Please refer to the main repository documentation for guidelines on contributing, reporting issues, and feature requests.

**kinopt** is distributed under the Apache 2.0 license. See the LICENSE file in the repository for more details.

---

This README provides an overview of the **kinopt** framework, outlining its structure, features, and usage instructions. For detailed documentation on each submodule, please refer to the individual README.md files within the respective directories.