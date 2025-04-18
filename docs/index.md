# PhosKinTime Documentation
 
<img src="../static/images/logo_3.png" alt="Package Logo" width="200"/>    

Welcome to the official documentation for **PhosKinTime**, an ODE-based modeling toolkit for phosphorylation kinetics and transcriptional time-series analysis. This index page provides an overview of each package and submodule in the project.

---

## Overview

PhosKinTime integrates:

- Parameter estimation (normal and sequential modes)
- Mechanistic ODE models (distributive, successive, random)
- Steady-state computation
- Morris sensitivity analysis
- Static and interactive visualization
- Modular design for extensibility

PhosKinTime uses ordinary differential equations (ODEs) to model phosphorylation kinetics and supports multiple mechanistic hypotheses, including:
- **Distributive Model:** Phosphorylation events occur independently.
- **Successive Model:** Phosphorylation events occur sequentially.
- **Random Model:** Phosphorylation events occur in a random manner.

The package is designed with modularity in mind. It consists of several key components:
- **Configuration:** Centralized settings (paths, parameter bounds, logging, etc.) are defined in the config module.
- **Models:** Different ODE models (distributive, successive, random) are implemented to simulate phosphorylation.
- **Parameter Estimation:** Multiple routines (sequential and normal estimation) estimate kinetic parameters from experimental data.
- **Sensitivity Analysis:** Morris sensitivity analysis is used to evaluate the influence of each parameter on the model output.
- **Steady-State Calculation:** Functions compute steady-state initial conditions for ODE simulation.
- **Utilities:** Helper functions support file handling, data formatting, report generation, and more.
- **Visualization:** A comprehensive plotting module generates static and interactive plots to visualize model fits, parameter profiles, PCA, t-SNE, and sensitivity indices.
- **Exporting:** The package can export results to Excel and generate HTML reports for easy sharing and documentation. 
- **CLI Interface:** A command-line interface allows users to run the package without needing to modify the code directly. 
- **Documentation:** The package includes extensive documentation to help users understand the functionality and usage of each module.
- **Testing:** The package includes unit tests to ensure the reliability and correctness of the implemented algorithms.
- **Logging:** A logging system is integrated to track the execution flow and capture important events during the analysis.
- **Error Handling:** The package includes error handling mechanisms to manage exceptions and provide informative error messages.
- **Cross-Platform Compatibility:** The package is designed to work on various operating systems, including Windows, macOS, and Linux.
- **Version Control:** The package is version-controlled using Git, allowing users to track changes and collaborate effectively.
- **Continuous Integration:** The package is set up with continuous integration (CI) tools to automate testing and ensure code quality.
- **Documentation Generation:** The package uses tools like Sphinx to generate documentation from docstrings, making it easy to maintain and update the documentation as the code evolves.
- **Code Quality:** The package follows best practices for code quality, including PEP 8 style guidelines, type hints, and docstrings for functions and classes.
---

## Core Packages

### bin/
Entry point for the pipeline. Contains `main.py`, which orchestrates configuration, data loading, parameter estimation, ODE simulation, visualization, and report generation.

### config/
Holds global constants, CLI parsing, and logging setup:

- `constants.py`: model settings, time points, directories, scoring weights
- `config.py`: argument parsing and configuration extraction
- `logconf.py`: colored console and rotating file logging
- `helpers/`: utilities for parameter names, state labels, bounds, and clickable paths

### models/
Implements ODE systems for different phosphorylation hypotheses:

- `distmod.py`: distributive model
- `succmod.py`: successive model
- `randmod.py`: random model with JIT optimization
- `weights.py`: weighting schemes for parameter estimation

### paramest/
Parameter estimation routines:

- `seqest.py`: sequential (time-point–wise) fitting
- `normest.py`: global fit across all time points
- `adapest.py`: adaptive profile estimation
- `toggle.py`: selects estimation mode
- `core.py`: integrates estimation, ODE solve, error metrics, and plotting

### steady/
Computes steady-state initial conditions for each model:

- `initdist.py`, `initsucc.py`, `initrand.py`

### sensitivity/
Morris sensitivity analysis:

- `analysis.py`: defines problem, sampling, analysis, and sensitivity plots

### plotting/
Visualization tools:

- `Plotter` class with methods for parallel coordinates, PCA, t-SNE, parameter bar and series plots, model fit, GoF diagnostics, Kullback–Leibler divergence, clusters, and heatmaps

### utils/
Helper functions:

- `display.py`: file and directory management, data loading, result saving, report generation
- `tables.py`: table creation and export (LaTeX and CSV)

---

## Optimization Framework (kinopt)

The **kinopt** package provides advanced optimization and post-processing:

### kinopt/evol
Global evolutionary optimization using pymoo (DE, NSGA-II):

- Problem formulation, data construction, exporter for Excel and plots

### kinopt/local
Local constrained optimization using SciPy solvers (SLSQP, TRUST-CONSTR) with Numba-accelerated objectives

### kinopt/optimality
Post-optimization analysis: feasibility checks, sensitivity reporting, LaTeX table generation, diagnostic plots

### kinopt/powell
Julia-based Powell optimization bridge: runs `powell.jl`, configures threads, integrates results into post-processing

### kinopt/fitanalysis
Additional fit-evaluation utilities for residual and performance analysis

---

## Optimization Framework (tfopt) 

###### Originally implemented by Julius Normann.

###### This version has been modified and optimized by Abhinav Mishra. 

The **tfopt** package estimates transcriptional regulation using mRNA and TF time-series data through constrained optimization.

### tfopt/evol  
Global evolutionary optimization using pymoo (NSGA-II, AGEMOEA, SMSEMOA):

- Multi-objective loss (fit error, α and β constraint violations)  
- Parallel evaluation, Excel export, and HTML/plot reports

### tfopt/local  
Local constrained optimization using SciPy solvers (SLSQP):

- Fast deterministic optimization under linear constraints  
- Numba-accelerated objectives, identical output and reports as `evol`

### tfopt/objfn  
Shared objective logic and prediction functions for both backends

### tfopt/optcon  
Data construction and constraint generation from TF–mRNA interaction files

### tfopt/utils  
Input parsing, Excel + plot output, and HTML report generation

---
## Features at a Glance

- 🧬 **Mechanistic ODE Models**: Distributive, successive, and random phosphorylation models.
- 🧪 **Parameter Estimation**: Both normal and sequential fitting modes.
- 🧠 **Sensitivity Analysis**: Morris method to analyze model response to parameters.
- 🧰 **Steady-State Calculations**: Compute initial conditions for all model types.
- 📊 **Visualization Tools**: Model fit plots, PCA/t-SNE visualizations, and HTML reports.
- 🔁 **Modular Design**: Easy to extend and customize each component.

---

## Quick Start

### Option 1: pip + virtualenv (Debian/Ubuntu/Fedora)

#### For **Debian/Ubuntu**
```bash
sudo apt update && sudo apt install -y python3 python3-pip python3-venv git
```

#### For **Fedora**
```bash
sudo dnf install -y python3 python3-pip python3-virtualenv git
```

### Setup
```bash
git clone git@github.com:bibymaths/phoskintime.git
cd phoskintime

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

### Option 2: Poetry + `pyproject.toml`

### Install Poetry (all platforms)
```bash
curl -sSL https://install.python-poetry.org | python3 -# Or: pip install poetry
```

---

## Modules

- `bin/`: Entry point (`main.py`) to run the full pipeline.
- `config/`: Constants, CLI parsing, and logging setup.
- `models/`: Distributive, successive, and random ODE systems.
- `paramest/`: Estimation logic (normal, sequential, adaptive).
- `steady/`: Model-specific steady-state calculators.
- `sensitivity/`: Morris sensitivity analysis.
- `utils/`: IO, table generation, and result handling.
- `plotting/`: Visualizations for model fits, sensitivity, PCA, and more.

---

## Acknowledgments

This project originated as part of my master's thesis work at Theoretical Biophysics group (now, [Klipp-Linding Lab](https://www.klipp-linding.science/tbp/index.php/en/)), Humboldt Universität zu Berlin.

- **Conceptual framework and mathematical modeling** were developed under the supervision of **[Prof. Dr. Dr. H.C. Edda Klipp](https://www.klipp-linding.science/tbp/index.php/en/people/51-people/head/52-klipp)**.
- **Experimental datasets** were provided by the **[(Retd. Prof.) Dr. Rune Linding](https://www.klipp-linding.science/tbp/index.php/en/people/51-people/head/278-rune-linding)**.
- The subpackage `tfopt` is an optimized and efficient derivative of [original work](https://github.com/Normann-BPh/Transcription-Optimization) by my colleague **[Julius Normann](https://github.com/Normann-BPh)**, adapted with permission.

I am especially grateful to [Ivo Maintz](https://rumo.biologie.hu-berlin.de/tbp/index.php/en/people/54-people/6-staff/60-maintz) for his generous technical support, enabling seamless experimentation with packages and server setups.
 
- The package is built on the shoulders of giants, leveraging the power of [NumPy](https://numpy.org/), [SciPy](https://www.scipy.org/), [Matplotlib](https://matplotlib.org/), and [Pandas](https://pandas.pydata.org/) for numerical computations and data handling. 
- The package also utilizes [Numba](https://numba.pydata.org/) for JIT compilation, enhancing performance for computationally intensive tasks.
- The package is designed to be compatible with [Python 3.8+](https://www.python.org/downloads/) and is tested on various platforms, including Windows, macOS, and Linux. 
 
---