# kinopt: A Comprehensive Optimization Framework for PhosKinTime

**kinopt** is a modular framework designed for the analysis and optimization of gene–phosphorylation time-series data. It integrates several specialized submodules that cater to different optimization strategies and post-processing analyses. Whether you need a global evolutionary approach, a local constrained optimization, or a Julia-based Powell optimization routine, **kinopt** offers the tools to process your experimental data and generate in-depth reports on model performance.

---

## Directory Structure

```
kinopt/
├── data
│   ├── input1.csv             # Primary input data file with phosphorylation time series data.
│   └── input2.csv             # Interaction data file containing protein-phosphorylation-kinase information.
├── evol
│   ├── config                 # Configuration files (constants, logging, etc.) for the evolutionary approach.
│   ├── exporter               # Plotting and Excel sheet export functions.
│   ├── __init__.py
│   ├── __main__.py            # Entry point for global optimization using evolutionary algorithms.
│   ├── objfn                  # Objective function implementations (single- and multi-objective).
│   ├── opt                    # Optimization routines (integration with pymoo).
│   ├── optcon                 # Functions to construct input data, constraints, and precomputed mappings.
│   ├── README.md              # Detailed readme for the evol module.
│   └── utils                  # Utility functions for data I/O and parameter extraction.
├── fitanalysis
│   ├── helpers                # Auxiliary scripts for additional performance evaluation.
│   ├── __init__.py
│   ├── __main__.py            # Entry point for fit analysis.
├── local
│   ├── config                 # Configuration files specific to local optimization.
│   ├── exporter               # Functions for exporting local optimization results and diagnostic plots.
│   ├── __init__.py
│   ├── __main__.py            # Entry point for local optimization (SLSQP/TRUST-CONSTR based).
│   ├── objfn                  # Local objective function implementations with Numba acceleration.
│   ├── opt                    # Local optimization routines using SciPy.
│   ├── optcon                 # Construction of local constraints and precomputation of mappings.
│   ├── README.md              # Detailed readme for the local module.
│   └── utils                  # Utilities for data scaling, file organization, and report generation.
├── optimality
│   ├── __init__.py
│   ├── KKT.py                 # Post-optimization analysis: feasibility, sensitivity, and reporting.
│   └── README.md              # Detailed readme for the optimality module.
├── __init__.py
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

- **Fit Analysis (fitanalysis):**  
  Provides additional tools to evaluate the fit and performance of the optimized model.

---

## Features

- **Modular Architecture:**  
  Each submodule (evol, local, optimality, fitanalysis) is designed to operate independently while integrating seamlessly into the overall workflow.

- **Flexible Optimization Strategies:**  
  Choose between global evolutionary algorithms, local constrained solvers depending on your specific needs.

- **Robust Post-Processing:**  
  Comprehensive post-optimization analysis includes constraint validation, sensitivity analysis, detailed reporting (both in LaTeX and Excel), and extensive plotting of diagnostic metrics.

- **Automated Reporting:**  
  After running optimization routines, the framework organizes outputs into structured directories and generates a global HTML report summarizing key results and diagnostic plots.

- **User-Friendly Logging:**  
  Custom logging configurations provide real-time feedback during execution, ensuring transparency in the optimization process.

---

## Usage 
 
Go to the one top level up in the terminal from root and run:

### Running Global Optimization (evol)

```bash
python -m phoskintime kinopt --mode evol
```

### Running Local Optimization (local)

```bash
python -m phoskintime kinopt --mode local
```

### Post-Optimization Processing

After any optimization run, the **optimality** module is invoked (either directly or as part of the workflow) to analyze the results, validate constraints, and generate comprehensive reports.

---

## Contributing & License

Contributions are welcome! Please refer to the main repository documentation for guidelines on contributing, reporting issues, and feature requests.

**kinopt** is distributed under the BSD Clause 3 license. See the LICENSE file in the repository for more details.

---

This README provides an overview of the **kinopt** framework, outlining its structure, features, and usage instructions. For detailed documentation on each submodule, please refer to the individual README.md files within the respective directories.