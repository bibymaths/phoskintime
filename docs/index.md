---
hide:
  - toc
---
 
# PhosKinTime Documentation

Welcome to the official documentation for **PhosKinTime**, an ODE-based modeling toolkit for phosphorylation kinetics
and transcriptional time-series analysis. This index page provides an overview of each package and submodule in the
project.
 
<img src="../images/logo_3.png" alt="Package Logo" width="200"/>     

---
## Acknowledgments

This project originated as part of my master's thesis work at Theoretical Biophysics group (
now, [Klipp-Linding Lab](https://www.klipp-linding.science/tbp/index.php/en/)), Humboldt Universität zu Berlin.

- **Conceptual framework and mathematical modeling** were developed under the supervision of **[Prof. Dr. Dr. H.C. Edda Klipp](https://www.klipp-linding.science/tbp/index.php/en/people/51-people/head/52-klipp)**.
- **Experimental datasets** were provided by the **[(Retd. Prof.) Dr. Rune Linding](https://www.klipp-linding.science/tbp/index.php/en/people/51-people/head/278-rune-linding)**.
- The subpackage `tfopt` is an optimized and efficient derivative
  of [original work](https://github.com/Normann-BPh/Transcription-Optimization) by my colleague **[Julius Normann](https://github.com/Normann-BPh)**, adapted with permission.

I am especially grateful
to [Ivo Maintz](https://rumo.biologie.hu-berlin.de/tbp/index.php/en/people/54-people/6-staff/60-maintz) for his generous
technical support, enabling seamless experimentation with packages and server setups.

- The package is built on the shoulders of giants, leveraging the power
  of [NumPy](https://numpy.org/), [SciPy](https://www.scipy.org/), [Matplotlib](https://matplotlib.org/),
  and [Pandas](https://pandas.pydata.org/) for numerical computations and data handling.
- The package also utilizes [Numba](https://numba.pydata.org/) for JIT compilation, enhancing performance for
  computationally intensive tasks.
- The package is designed to be compatible with [Python 3.8+](https://www.python.org/downloads/) and is tested on
  various platforms, including Windows, macOS, and Linux.

---

## Overview

PhosKinTime integrates:

- Parameter estimation 
- Mechanistic ODE models (distributive, successive, random)
- Steady-state computation
- Morris sensitivity analysis
- Static and interactive visualization
- Modular design for extensibility

PhosKinTime uses ordinary differential equations (ODEs) to model phosphorylation kinetics and supports multiple
mechanistic hypotheses, including:

- **Distributive Model:** Phosphorylation events occur independently.
- **Successive Model:** Phosphorylation events occur sequentially.
- **Random Model:** Phosphorylation events occur in a random manner.

---

## Core Modules

**config**

- Holds global constants, CLI parsing, and logging setup.

**models**

- Implements ODE systems for different phosphorylation hypotheses. 

**models**

- Parameter estimation routines for ODE models.

**steady**

- Computes steady-state initial conditions for each model.

**sensitivity**

- Morris sensitivity analysis for parameter sensitivity.

**plotting**

- Visualization tools for plotting results.

**utils**

- Helper functions for data loading, saving, and plotting. 

---
 
## Optimization Frameworks 

### kinopt

The **kinopt** package provides advanced optimization and post-processing:

#### evol

Global evolutionary optimization using pymoo (DE, NSGA-II):

- Problem formulation, data construction, exporter for Excel and plots

#### local

Local constrained optimization using SciPy solvers (SLSQP, TRUST-CONSTR) with Numba-accelerated objectives

#### optimality

Post-optimization analysis: feasibility checks, sensitivity reporting, LaTeX table generation, diagnostic plots

#### fitanalysis

Additional fit-evaluation utilities for residual and performance analysis

---

### tfopt

_Originally implemented by Julius Normann._

_This version has been modified and optimized by Abhinav Mishra._

The **tfopt** package estimates transcriptional regulation using mRNA and TF time-series data through constrained
optimization.

#### evol

Global evolutionary optimization using pymoo (NSGA-II, AGEMOEA, SMSEMOA):

- Multi-objective loss (fit error, α and β constraint violations)
- Parallel evaluation, Excel export, and HTML/plot reports

#### local

Local constrained optimization using SciPy solvers (SLSQP):

- Fast deterministic optimization under linear constraints
- Numba-accelerated objectives, identical output and reports as `evol`

#### objfn

Shared objective logic and prediction functions for both backends

#### optcon

Data construction and constraint generation from TF–mRNA interaction files

#### utils

Input parsing, Excel + plot output, and HTML report generation

---
 
## Command-Line Entry Point for the Phoskintime Pipeline

The `phoskintime` pipeline provides a command-line interface to execute various stages of the workflow,  
including preprocessing, optimization, and modeling. Below are the usage instructions and examples for running  
the pipeline.

Before running any commands, ensure you are in the working directory one level above the project root (where the project  
directory is visible).

### Run All Stages
Run the entire pipeline with the default (local) solver:
```bash
python phoskintime all
```

### Run Preprocessing Only
Execute only the preprocessing stage:
```bash
python phoskintime prep
```

### Run Transcription-Factor-mRNA Optimization (TFOPT)
Run TFOPT with the local solver:
```bash
python phoskintime tfopt --mode local
```

Run TFOPT with the evolutionary solver:
```bash
python phoskintime tfopt --mode evol
```

### Run Kinase-Phosphorylation Optimization (KINOPT)
Run KINOPT with the local solver:
```bash
python phoskintime kinopt --mode local
```

Run KINOPT with the evolutionary solver:
```bash
python phoskintime kinopt --mode evol
```

### Run the Model
Execute the modeling stage:
```bash
python phoskintime model
``` 

### Quick Start: Setting up environment

This guide provides clean setup instructions for running the `phoskintime` package on a new machine. Choose the scenario
that best fits your environment and preferences. 
 
Before proceeding, ensure you have the following prerequisites installed: 
 
- graphviz (for generating diagrams)   

```bash 
# For Debian/Ubuntu
sudo apt-get install graphviz   

# For Fedora
sudo dnf install graphviz    

# For MacOS
brew install graphviz 
``` 

- python 3.10 or higher 

```bash  
# Check python version 
python3 --version  

# If not installed, install python 3.10 or higher 

# For Debian/Ubuntu  
sudo apt-get install python3.10 
 
# For Fedora 
sudo dnf install python3.10 

# For MacOS
brew install python@3.10
``` 

- git (for cloning the repository) 
 
```bash  
# For Debian/Ubuntu 
sudo apt-get install git  

# For Fedora 
sudo dnf install git  

# For MacOS 
brew install git 
```

---

### Scenario 1: pip + virtualenv (Debian/Ubuntu/Fedora)

#### For **Debian/Ubuntu**

```bash
sudo apt update && sudo apt install -y python3 python3-pip python3-venv git
```

#### For **Fedora**

```bash
sudo dnf install -y python3 python3-pip python3-virtualenv git
```

#### Setup

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

### Scenario 2: Poetry + `pyproject.toml`

#### Install Poetry (all platforms)

```bash
curl -sSL https://install.python-poetry.org | python3 -
# Or: pip install poetry
```

#### Setup

```bash
git clone git@github.com:bibymaths/phoskintime.git
cd phoskintime

# Install dependencies
poetry install

# Optional: activate shell within poetry env
poetry shell
```

---

### Scenario 3: Using [`uv`](https://github.com/astral-sh/uv) (fast, isolated pip alternative)

#### Install `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Setup

```bash
git clone git@github.com:bibymaths/phoskintime.git
cd phoskintime

# Create virtual environment and install deps fast
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

---

### Scenario 4: Conda or Mamba (Anaconda/Miniconda users)

#### Setup

```bash
git clone git@github.com:bibymaths/phoskintime.git
cd phoskintime

# Create and activate conda environment
conda create -n phoskintime python=3.10 -y
conda activate phoskintime

# Install dependencies
pip install -r requirements.txt
```

Or if using `pyproject.toml`, add:

```bash
pip install poetry
poetry install
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

---