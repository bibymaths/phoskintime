# Welcome to PhosKinTime

**PhosKinTime** is an ODE-based modeling package to explore phosphorylation and transcriptional time dynamics. It integrates robust tools for:
- parameter estimation,
- model simulation,
- sensitivity analysis,
- and comprehensive visualization of kinase-substrate interactions.

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

Install using Poetry:

```bash
git clone https://github.com/bibymaths/phoskintime.git
cd phoskintime
poetry install
poetry shell
```

Or with pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the main pipeline:

```bash
python bin/main.py --A-bound "0,100" --bootstraps 10 --input-excel path/to/data.xlsx
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

## Project Origin

This package was developed as part of a master's thesis at the [Klipp-Linding Lab](https://www.klipp-linding.science/tbp/index.php/en/) at Humboldt-Universität zu Berlin under:

- 🧠 **Prof. Dr. Dr. H.C. Edda Klipp** – supervision and mathematical modeling
- 🧪 **Dr. Rune Linding** – experimental data
- 🧑‍💻 **Julius Normann** – original transcription optimization logic (`tfopt`)
- 🛠️ **Ivo Maintz** – technical support & infrastructure

---