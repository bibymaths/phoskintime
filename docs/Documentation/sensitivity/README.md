# Sensitivity Analysis

The Sensitivity module provides functionality for performing sensitivity analysis on the ODE-based phosphorylation
models in the PhosKinTime package. Using the Morris method from SALib, this module evaluates the influence of each model
parameter on the output of the system, thereby helping to identify the most critical parameters and potential nonlinear
interactions.

---

## Two Sensitivity Analysis Implementations

PhosKinTime includes **two separate** sensitivity analysis implementations for the two modeling stacks:

| | `sensitivity/analysis.py` | `global_model/sensitivity.py` |
|---|---|---|
| **Scope** | Local per-protein ODE models | Global coupled network ODE model |
| **Method** | Morris elementary effects (via SALib) | Morris elementary effects (trajectory-based perturbation) |
| **Workflow** | Local pipeline (`kinopt` → `model` → `sensitivity`) | Global pipeline (`global_model` → post-analysis) |
| **Input** | Per-protein fitted parameters, ODE definitions | Global model parameters, full network state vector |
| **Output** | Per-protein sensitivity indices (μ*, σ), diagnostic plots | Global sensitivity indices over full trajectory, perturbation cloud plots |
| **When to use** | Identifying critical parameters for individual protein/site ODE fits | Identifying critical global parameters driving network-scale dynamics |

> **Do not confuse the two.** They analyze different models and answer different questions.

---

## `sensitivity/analysis.py` — Local Sensitivity

### Overview

This module (primarily implemented in `analysis.py`) defines functions that:

- **Define the Sensitivity Problem:**  
  Two functions (`define_sensitivity_problem_rand` and `define_sensitivity_problem_ds`) generate the problem
  definition (number of variables, parameter names, and bounds) required for the Morris sensitivity analysis. The choice
  depends on whether the model is a random model (`randmod`) or a distributive/successive model.

- **Run Sensitivity Analysis:**  
  The `sensitivity_analysis` function:
    - Generates parameter samples using the Morris method.
    - Simulates the ODE system (via the package's `solve_ode` function) for each parameter set.
    - Computes a response metric (e.g., the sum of the phosphorylated states at the final time point).
    - Analyzes the sensitivity indices using SALib's `analyze` function.
    - Generates a suite of plots (bar plots, scatter, radial, CDF, and pie charts) to visually summarize the sensitivity
      of each parameter.

### Invocation

Run as part of the local pipeline via the `model` CLI stage or directly:

```bash
phoskintime model
```

---

## `global_model/sensitivity.py` — Global Sensitivity

### Overview

This module performs trajectory-based perturbation sensitivity analysis for the global coupled ODE model.

- **Elementary Effects (Morris method):** Perturbs each parameter independently across a grid of levels.
- **Parallel simulation:** Each perturbed trajectory is simulated using the full coupled ODE system.
- **Output:**
  - CSV file with per-parameter sensitivity indices (μ\* and σ)
  - Perturbation cloud visualizations showing parameter influence on the full trajectory

The sensitivity metric (how the trajectory is collapsed to a scalar) is controlled by
`sensitivity_metric` in `config.toml` (`total_signal`, `mean`, `variance`, `l2_norm`).

### Configuration (in `config.toml`)

```toml
[global_model]
sensitivity_analysis      = true
sensitivity_perturbation  = 0.05
sensitivity_trajectories  = 100
sensitivity_levels        = 40
sensitivity_top_curves    = 20
sensitivity_metric        = "total_signal"
```

### Invocation

Runs automatically after optimization if `sensitivity_analysis = true` in `config.toml`:

```bash
phoskintime-global
```

Or triggered from `global_model/runner.py` directly.
