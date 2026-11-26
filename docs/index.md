---
hide:
  - toc
---
 
<img class="only-light hero-image" src="assets/images/16_9/light.png" alt="PhosKinTime overview - light mode">
<img class="only-dark hero-image" src="assets/images/16_9/dark.png" alt="PhosKinTime overview - dark mode">

# PhosKinTime Documentation

Welcome to the official documentation for **PhosKinTime**, an ODE-based modeling toolkit for phosphorylation kinetics
and transcriptional time-series analysis. This index page provides an overview of each package and submodule in the
project.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15351017.svg)](https://doi.org/10.5281/zenodo.15351017)   

---
## Acknowledgments

This project began from my master's thesis work in the Theoretical Biophysics group, now
the [Klipp-Linding Lab](https://rumo.biologie.hu-berlin.de/tbp/index.php/en/), at Humboldt-Universität zu Berlin. The
submitted thesis focused on `kinopt`, while related components of the broader modelling framework were developed in
parallel during that period and continued afterwards.

The implemented version of `kinopt` was developed as the core thesis-stage component and now lives in the standalone
KinOpt repository. The `tfopt` and `protwise`
components were developed by me during the same broader project period and subsequent continuation of the work. The
`networkmodel` component was later designed and implemented independently after that period as an extension of the
original modelling framework.

The initial distributive, successive, and combinatorial ODE formulations came from the thesis-stage modelling framework,
while the saturation model was added independently at a later stage as an extension of the system.

The former subpackage `tfopt`, now the standalone TFOpt tool, is an optimized and extended derivative
of [original work](https://github.com/Normann-BPh/Transcription-Optimization) by my
colleague [Julius Normann](https://github.com/Normann-BPh), adapted with permission.

I am grateful to [Ivo Maintz](https://rumo.biologie.hu-berlin.de/tbp/index.php/en/people/54-people/6-staff/60-maintz)
for generous technical support with server access, package experimentation, and computational setup.

---

## Educational notebooks

Executable educational notebooks are available under `notebooks/` and cover end-to-end dummy-data workflows for protein-wise ODE fitting and network-level modeling.

| Notebook | Module | What it demonstrates |
| -------- | ------ | -------------------- |
| `03_protwise_educational_workflow.ipynb` | `phoskintime.protwise` | Mode-aware protein-wise ODE fitting with Diffrax-based ODE solving and JAXopt parameter estimation. |
| `04_networkmodel_educational_workflow.ipynb` | `phoskintime.networkmodel` | Network-level multimodal loss handling, adjacency interpretation, JAX/Diffrax solving, multistart ranking, and exports. |

Run them interactively with `jupyter lab notebooks/` or as executable tests with `pytest --nbmake notebooks/*.ipynb`.


## No-code dashboard

Use the unified Streamlit dashboard to browse existing result directories, preview uploaded inputs, launch registered CLI/Pixi workflows, and download result archives. See [No-code dashboard documentation](Documentation/no_code_dashboard.md).

## Modeling backends (local vs global)

PhosKinTime provides two complementary modeling stacks:

1) **Local phosphorylation ODE models (per protein / per site)**  
   These implement the classic mechanistic hypotheses (distributive, successive, random) and are optimized against
   phosphoproteomics time series. They are intended for detailed fitting of individual proteins or phosphorylation sites. 

2) **Global coupled signaling–GRN model (`networkmodel`)**  
   This is a network-scale ODE system that couples kinase-driven phosphorylation dynamics to TF-mediated transcriptional
   regulation. It is intended for system-level simulations (e.g., global knockouts, functional influence propagation)
   and calibration against multi-omics time series.

For the full mathematical specification of the coupled global system (all equations), see:
[Global model documentation](Documentation/global/README.md)

Package mapping:
- **`networkmodel`**: coupled kinase-signaling + GRN simulation and optimization wrappers
- **KinOpt / TFOpt** (standalone tools, not part of this repository): produce the result workbooks that the model
  stages read as inputs; see [External tools](Documentation/external_tools.md)

Typical workflows:
- **Local phosphorylation fitting**: `prep → model → sensitivity → plotting` (reads KinOpt/TFOpt result workbooks)
- **Global network simulation**: `prep → networkmodel → simulate → measure → KO analysis` 
- **Global network optimization**: `prep → networkmodel → optimize → measure → KO analysis` 

---

## Core Modules

**config**

- Holds global constants, CLI parsing, and logging setup.

**models**

- Implements ODE systems for different phosphorylation hypotheses. 

**paramest**

- Parameter estimation routines for ODE models (currently using `paramest/normest.py`).

**steady**

- Computes steady-state initial conditions for each model.

**sensitivity**

- Morris sensitivity analysis for parameter sensitivity.

**plotting**

- Visualization tools for plotting results.

**utils**

- Helper functions for data loading, saving, and plotting. 

---
 
## KinOpt and TFOpt (standalone tools)

KinOpt (kinase-phosphosite optimization) and TFOpt (TF-mRNA optimization) were part of this repository and are now
separate tools. PhosKinTime does not run them. It reads their result workbooks (`kinopt_results.xlsx`,
`tfopt_results.xlsx`) through the file contract in `common/external_results.py`.

- **KinOpt**: <https://github.com/bibymaths/KinOpt> (no release tagged yet; refer to a commit).
- **TFOpt**: repository not yet public. TFOpt was originally implemented by Julius Normann and has been modified and
  optimized by Abhinav Mishra.

See [External tools](Documentation/external_tools.md) for how PhosKinTime uses these result files.

The `kinopt/` and `tfopt/` code was last included in PhosKinTime at commit `6eb04fa`.

---
 
## Command-Line Entry Point for the Phoskintime Pipeline

The `phoskintime` pipeline provides a command-line interface to execute various stages of the workflow,  
including preprocessing and modeling. Below are the usage instructions and examples for running  
the pipeline.

Before running any commands, ensure you are in the working directory one level above the project root (where the project  
directory is visible).

### Run All Stages
Run the entire pipeline with the default (local) solver:
```bash
python phoskintime all
```

> **Note:** The `all` command runs `prep → check KinOpt/TFOpt result files → model` only. It does not run KinOpt or
> TFOpt, and it stops before the model stage if their result files are missing or malformed. It does **not** invoke the
> global network simulation (`networkmodel`). To run the global model, use the separate entry point:
> ```bash
> phoskintime-global
> # or
> python -m networkmodel.runner
> ```
> The global model reads the KinOpt and TFOpt result workbooks named by `kinopt` and `tfopt` in the `[networkmodel]`
> section of `config.toml`. These files come from the standalone tools; see
> [External tools](Documentation/external_tools.md).

### Run Preprocessing Only
Execute only the preprocessing stage:
```bash
python phoskintime prep
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