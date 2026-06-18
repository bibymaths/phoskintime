<div align="center">
  <img src="docs/assets/images/16_9/dark.png" alt="PhosKinTime Logo">

  [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15351017.svg)](https://doi.org/10.5281/zenodo.15351017) 
  ![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)
  ![JAX](https://img.shields.io/badge/JAX-accelerated-orange.svg)
  ![Coverage](docs/assets/coverage.svg)

</div> 

--- 

> 💡 **PhosKinTime** is an ODE-based modeling package for analyzing phosphorylation dynamics over time. It integrates
parameter estimation, sensitivity analysis, steady-state computation, and visualization tools to help researchers
explore kinase-substrate interactions in a temporal context. 

---
## [Documentation](https://bibymaths.github.io/phoskintime/)

---


<details>
<summary><strong>🚧 Work in Progress: Web Interface & Package Finalization</strong> (Click to expand)</summary>
<br>

I am actively working to make these powerful network dynamics tools as accessible and adaptable as possible for all researchers.

Currently, I am focusing on two major updates:

1. **A No-Code Web Interface:** I am building a frontend that will allow you to upload data, run parameter estimations, and generate interactive visualizations (PCA, t-SNE, network topology) directly from your browser—no command line required.
2. **Finalizing the Python Package:** The core ODE modeling and optimization logic is built. I am currently finalizing the documentation for experimental data preparation and refining the command-line entry points to make the package easily adaptable to your specific datasets.

*Note: Whether you prefer running Python scripts or using a graphical web app, be sure to **"Watch"** this repository (click the star/watch button at the top right) to be notified as soon as the complete documentation and frontend are released!*

</details>

---

## The Problem: Phosphorylation Cascades

In cellular signaling pathways, a series of proteins are phosphorylated in an activation cascade that drives cellular responses. Understanding these post-translational modifications is critical.

![Phosphorylation Cascade Concept](docs/assets/images/phoskintime_problem.png)

*Figure 1: Overview of protein post-translational modifications and the phosphorylation cascade mechanism.*

---

<details>
<summary>Acknowledgements (Click to expand) </summary>

This project originated as part of my master's thesis work at Theoretical Biophysics group (
now, [Klipp-Linding Lab](https://rumo.biologie.hu-berlin.de/tbp/index.php/en/)), Humboldt Universität zu Berlin.

- **Conceptual framework and mathematical modeling** were developed under the supervision of **[Prof. Dr. Dr. H.C. Edda Klipp](https://rumo.biologie.hu-berlin.de/tbp/index.php/en/people/51-people/head/52-klipp)**.
- **Experimental datasets** were provided by the **[(Retd. Prof.) Dr. Rune Linding](https://rumo.biologie.hu-berlin.de/tbp/index.php/en/people/51-people/head/278-rune-linding)**.
- The subpackage `tfopt` is an optimized and efficient derivative
  of [original work](https://github.com/Normann-BPh/Transcription-Optimization) by my colleague **[Julius Normann](https://github.com/Normann-BPh)**, adapted with permission.

I am especially grateful
to [Ivo Maintz](https://rumo.biologie.hu-berlin.de/tbp/index.php/en/people/54-people/6-staff/60-maintz) for his generous
technical support, enabling seamless experimentation with packages and server setups.

</details>

---

## Overview

PhosKinTime uses ordinary differential equations (ODEs) to model phosphorylation kinetics and supports multiple
mechanistic hypotheses, including:

- **Distributive Model:** Phosphorylation events occur independently.
- **Successive Model:** Phosphorylation events occur sequentially.
- **Random Model:** Phosphorylation events occur in a random manner.

The package is designed with modularity in mind. It consists of several key components:

- **Configuration:** Centralized settings (paths, parameter bounds, logging, etc.) are defined in the config module.
- **Models:** Different ODE models (distributive, successive, random) are implemented to simulate phosphorylation.
- **Parameter Estimation:** Normal estimation routines (`paramest/normest.py`) estimate kinetic parameters from
  experimental data.
- **Sensitivity Analysis:** Morris sensitivity analysis is used to evaluate the influence of each parameter on the model
  output.
- **Steady-State Calculation:** Functions compute steady-state initial conditions for ODE simulation.
- **Utilities:** Helper functions support file handling, data formatting, report generation, and more.
- **Visualization:** A comprehensive plotting module generates static and interactive plots to visualize model fits,
  parameter profiles, PCA, t-SNE, and sensitivity indices.

--- 

## Educational notebooks

The repository includes executable educational Jupyter notebooks that demonstrate the core modeling workflows with tiny deterministic dummy data. They are designed for learning, CI execution, and quick smoke-testing of the current JAX/JAXopt/Diffrax implementation.

| Notebook | Module | What it demonstrates |
| -------- | ------ | -------------------- |
| `notebooks/01_kinopt_educational_workflow.ipynb` | `kinopt` | Kinase/phosphosite-style input tables, KinOpt preprocessing arrays, alpha/beta constraints, local optimization, a ranked multistart solution ensemble, parameter export, and fit plots. |
| `notebooks/02_tfopt_educational_workflow.ipynb` | `tfopt` | TF-target regulatory networks, TF protein/phosphosite effect matrices, constrained local optimization, ranked multistart outputs, regulatory-effect visualization, and saved tables. |
| `notebooks/03_protwise_educational_workflow.ipynb` | `phoskintime.protwise` | Protein-wise ODE modeling with mRNA/protein/phosphosite modalities, mode-aware fitting logic, Diffrax-based ODE solving, JAXopt parameter estimation, multistart ranking, residual plots, and CSV/JSON exports. |
| `notebooks/04_networkmodel_educational_workflow.ipynb` | `phoskintime.networkmodel` | Network-level multimodal data handling, adjacency construction, alpha/beta projection utilities, missing-modality cases, Diffrax/JAX-based solving, local constrained optimization, ranked multistart outputs, mode exports, and network/parameter plots. |

Run the notebooks interactively:

```bash
jupyter lab notebooks/
```

---

## License

This package is distributed under the BSD 3-Clause License.  
See the [LICENSE](./LICENSE) file for full details.

---