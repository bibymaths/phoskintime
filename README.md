<div align="center">
  <img src="docs/assets/images/16_9/dark.png" alt="PhosKinTime Logo">

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15351017.svg)](https://doi.org/10.5281/zenodo.15351017)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)
![JAX](https://img.shields.io/badge/JAX-accelerated-orange.svg)
![Coverage](docs/assets/coverage.svg)

</div> 

--- 

> 💡 **PhosKinTime** is an ODE-based modeling package for analyzing phosphorylation dynamics over time. It integrates
> parameter estimation, sensitivity analysis, steady-state computation, and visualization tools to help researchers
> explore kinase-substrate interactions in a temporal context.

---

## [Documentation](https://bibymaths.github.io/phoskintime/)

---

## The Problem: Phosphorylation Cascades

In cellular signaling pathways, a series of proteins are phosphorylated in an activation cascade that drives cellular
responses. Understanding these post-translational modifications is critical.

![Phosphorylation Cascade Concept](docs/assets/images/phoskintime_problem.png)

*Figure 1: Overview of protein post-translational modifications and the phosphorylation cascade mechanism.*

---

<details>
<summary>Acknowledgements (Click to expand)</summary>

This project began from my master's thesis work in the Theoretical Biophysics group, now
the [Klipp-Linding Lab](https://rumo.biologie.hu-berlin.de/tbp/index.php/en/), at Humboldt-Universität zu Berlin. The
submitted thesis focused on `kinopt`, while related components of the broader modelling framework were developed in
parallel during that period and continued afterwards.

The implemented version of `kinopt` was developed as the core thesis-stage component. The `tfopt` and `protwise`
components were developed by me during the same broader project period and subsequent continuation of the work. The
`networkmodel` component was later designed and implemented independently after that period as an extension of the
original modelling framework.

The initial distributive, successive, and combinatorial ODE formulations came from the thesis-stage modelling framework,
while the saturation model was added independently at a later stage as an extension of the system.

The subpackage `tfopt` is an optimized and extended derivative
of [original work](https://github.com/Normann-BPh/Transcription-Optimization) by my
colleague [Julius Normann](https://github.com/Normann-BPh), adapted with permission.

I am grateful to [Ivo Maintz](https://rumo.biologie.hu-berlin.de/tbp/index.php/en/people/54-people/6-staff/60-maintz)
for generous technical support with server access, package experimentation, and computational setup.

</details>

---

## Overview

PhosKinTime uses ordinary differential equations (ODEs) to model phosphorylation kinetics and supports multiple
mechanistic hypotheses, including:

- **Distributive Model:** Phosphorylation events occur independently.
- **Successive Model:** Phosphorylation events occur sequentially.
- **Combinatorial Model:** Phosphorylation events occur in a hypercube lattice manner.
- **Saturation Model:** Phosphorylation dynamics follow saturating kinetics, where phosphorylation rates approach an
  upper limit as kinase input or substrate availability increases.

--- 

## Educational notebooks

The repository includes executable educational Jupyter notebooks that demonstrate the core modeling workflows with tiny
deterministic dummy data. They are designed for learning, CI execution, and quick smoke-testing of the current
JAX/JAXopt/Diffrax implementation.

| Notebook                                               | Module                     | What it demonstrates                                                                                                                                                                                                                                      |
|--------------------------------------------------------|----------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `notebooks/01_kinopt_educational_workflow.ipynb`       | `kinopt`                   | Kinase/phosphosite-style input tables, KinOpt preprocessing arrays, alpha/beta constraints, local optimization, a ranked multistart solution ensemble, parameter export, and fit plots.                                                                   |
| `notebooks/02_tfopt_educational_workflow.ipynb`        | `tfopt`                    | TF-target regulatory networks, TF protein/phosphosite effect matrices, constrained local optimization, ranked multistart outputs, regulatory-effect visualization, and saved tables.                                                                      |
| `notebooks/03_protwise_educational_workflow.ipynb`     | `phoskintime.protwise`     | Protein-wise ODE modeling with mRNA/protein/phosphosite modalities, mode-aware fitting logic, Diffrax-based ODE solving, JAXopt parameter estimation, multistart ranking, residual plots, and CSV/JSON exports.                                           |
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