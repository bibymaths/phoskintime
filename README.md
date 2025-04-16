# PhosKinTime
  
<img src="static/images/logo_3.png" alt="Package Logo" width="200"/> 
![NSGA-2 Optimization Run on Kinases](static/gif/optimization_run.gif)

![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)
![NumPy](https://img.shields.io/badge/Numpy-1.24.3-red.svg)
![SciPy](https://img.shields.io/badge/SciPy-1.10.1-orange.svg)
![Pandas](https://img.shields.io/badge/Pandas-1.5.3-yellowgreen.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0.2-lightgrey.svg)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7.1-blueviolet.svg)
![Seaborn](https://img.shields.io/badge/Seaborn-0.12.2-orange.svg)
![Plotly](https://img.shields.io/badge/Plotly-5.15.0-brightgreen.svg)
![SALib](https://img.shields.io/badge/SALib-1.4.5-lightgrey.svg)
![Numba](https://img.shields.io/badge/Numba-0.56.4-yellowgreen.svg)

[![CI/CD](https://github.com/bibymaths/phoskintime/actions/workflows/test.yml/badge.svg)](https://github.com/bibymaths/phoskintime/actions/workflows/test.yml)   

<!-- [![codecov](https://codecov.io/gh/bibymaths/phoskintime/branch/master/graph/badge.svg?token=JVCFNL8VLZ)](https://codecov.io/gh/bibymaths/phoskintime) --> 

**PhosKinTime** is an ODE-based modeling package for analyzing phosphorylation dynamics over time. It integrates parameter estimation, sensitivity analysis, steady-state computation, and visualization tools to help researchers explore kinase-substrate interactions in a temporal context.

## Overview

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

## Installation

To install PhosKinTime, clone the repository and install the required dependencies via pip:

```bash
git clone https://github.com/bibymaths/phoskintime.git
cd phoskintime
pip install -r requirements.txt
```
For making illustration diagrams, you need to install Graphviz. You can do this via conda or apt-get:
 
```bash 
conda install graphviz
``` 
or 

```bash 
apt-get install graphviz
``` 
or download it from the [Graphviz website](https://graphviz.gitlab.io/download/). 
For macusers, you can use Homebrew:

```bash  
brew install graphviz
``` 

Ensure you have Python 3.7+ and that packages such as NumPy, SciPy, Pandas, scikit-learn, Matplotlib, Seaborn, Plotly, SALib, and Numba are installed.

## Usage
 
```bash
usage: main.py [-h] [--A-bound A_BOUND] [--B-bound B_BOUND] [--C-bound C_BOUND] [--D-bound D_BOUND] [--Ssite-bound SSITE_BOUND] [--Dsite-bound DSITE_BOUND] [--fix-A FIX_A] [--fix-B FIX_B] [--fix-C FIX_C] [--fix-D FIX_D] [--fix-Ssite FIX_SSITE] [--fix-Dsite FIX_DSITE] [--fix-t FIX_T]
               [--bootstraps BOOTSTRAPS] [--profile-start PROFILE_START] [--profile-end PROFILE_END] [--profile-step PROFILE_STEP] [--input-excel INPUT_EXCEL]

PhosKinTime - ODE Parameter Estimation of Phosphorylation Events in Temporal Space

options:
  -h, --help            show this help message and exit
  --A-bound A_BOUND
  --B-bound B_BOUND
  --C-bound C_BOUND
  --D-bound D_BOUND
  --Ssite-bound SSITE_BOUND
  --Dsite-bound DSITE_BOUND
  --fix-A FIX_A
  --fix-B FIX_B
  --fix-C FIX_C
  --fix-D FIX_D
  --fix-Ssite FIX_SSITE
  --fix-Dsite FIX_DSITE
  --fix-t FIX_T         JSON string mapping time points to fixed param values
  --bootstraps BOOTSTRAPS
  --profile-start PROFILE_START
  --profile-end PROFILE_END
  --profile-step PROFILE_STEP
  --input-excel INPUT_EXCEL
                        Path to the input Excel file

```  

The package is executed via the main script located in the `bin` directory. This script sets up the configuration, processes experimental data, performs parameter estimation, generates model simulations, and creates a comprehensive report.

### Running the Main Script

You can run the main script from the command line:

```bash
python bin/main.py --A-bound "0,100" --B-bound "0,100" --C-bound "0,100" --D-bound "0,100" --Ssite-bound "0,100" --Dsite-bound "0,100" --bootstraps 10 --input-excel "path/to/your/excel.xlsx"
```

The command-line arguments (such as parameter bounds, fixed parameters, bootstrapping iterations, and input file paths) are parsed by the configuration module. The main script then:
- Loads the experimental data.
- Logs the configuration and initializes output directories.
- Processes each gene in parallel using a ProcessPoolExecutor.
- Performs parameter estimation (toggling between sequential and normal modes as configured).
- Generates ODE simulations and various plots.
- Saves all results (including a global HTML report) in the designated output directory.

### Example

Here’s a brief overview of the execution flow:

1. **Configuration:**  
   - `config/config.py` and `config/constants.py` set up model options (e.g., `ODE_MODEL`, `ESTIMATION_MODE`), time points, file paths, and logging settings.
   - Command-line arguments are parsed to override default settings.

2. **Parameter Estimation:**  
   - Depending on the chosen estimation mode (sequential or normal), functions from `paramest/seqest.py` or `paramest/normest.py` are used.
   - The toggle functionality in `paramest/toggle.py` selects the appropriate routine.
   - Results are saved and passed for visualization.

3. **Model Simulation and Visualization:**  
   - The selected ODE model (from `models/`) is used to simulate system dynamics.
   - The `plotting` module generates plots (e.g., parallel coordinates, PCA, t-SNE, model fits, and sensitivity plots) to visually inspect the results.
   
4. **Reporting:**  
   - The `utils/display.py` and `utils/tables.py` modules save results and generate an HTML report summarizing the analysis.

## Modules

- **Config Module:**  
  - `config/constants.py`: Global constants (model settings, time points, directories, scoring weights, etc.).
  - `config/config.py`: Command-line argument parsing and configuration extraction.
  - `config/logconf.py`: Logging configuration with colored console output and rotating file logs.
  - `config/helpers/__init__.py`: Helper functions for generating parameter names, state labels, bounds, and clickable file links.

- **Models Module:**  
  Implements various ODE models:
  - `randmod.py`: Random model with vectorized state calculations.
  - `distmod.py`: Distributive model.
  - `succmod.py`: Successive model.
  - `weights.py`: Weighting schemes for parameter estimation.

- **Parameter Estimation Module:**  
  - `seqest.py`: Sequential (time-point-by-time-point) estimation.
  - `normest.py`: Normal (all timepoints at once) estimation.
  - `adapest.py`: Adaptive profile estimation.
  - `toggle.py`: Utility to switch between estimation modes.
  - `core.py`: Integrates estimation, ODE solving, error metrics, and visualization.

- **Steady-State Module:**  
  - `initdist.py`, `initrand.py`, `initsucc.py`: Compute steady-state initial conditions for each model type.

- **Sensitivity Module:**  
  - `analysis.py`: Implements Morris sensitivity analysis, including problem definition, sampling, analysis, and plotting of sensitivity indices.

- **Utils Module:**  
  - `display.py`: Helper functions for file/directory management, data loading, result saving, and report generation.
  - `tables.py`: Functions to generate, save, and compile data tables (LaTeX and CSV).

- **Bin Module:**  
  - `main.py`: The main entry point that orchestrates the entire workflow—from configuration and data loading to parameter estimation, simulation, visualization, and report generation.

## Customization

You can customize the package by:
- Adjusting model parameters and bounds in the config files.
- Choosing the ODE model type by modifying `ODE_MODEL` in `constants.py`.
- Setting the estimation mode (`ESTIMATION_MODE`) to "sequential" or "normal".
- Configuring output directories and file paths.
- Modifying the logging behavior in `logconf.py`.
- Tweaking the scoring function weights in `constants.py`.

## Conclusion

PhosKinTime is a flexible and powerful package for modeling phosphorylation kinetics. Its modular design allows researchers to simulate different mechanistic models, estimate kinetic parameters, analyze parameter sensitivity, and generate comprehensive visual and tabular reports. Whether you are exploring basic kinetic hypotheses or conducting in-depth sensitivity analysis, PhosKinTime offers the necessary tools for robust model-based analysis.

For more information, please refer to the individual module documentation and source code.

--- 

![Goal](static/images/goal_2.png)
