# Parameter Estimation

This module provides the tools needed to estimate parameters for ODE‐based models of phosphorylation dynamics.

## Overview

The module is organized into several submodules:

- **`normest.py`** – Implements normal parameter estimation. This approach fits the entire time-series data in one step.
- **`toggle.py`** – Offers a single function (`estimate_parameters`) to pipe normal estimation based on a mode flag.
- **`core.py`** – Integrates the estimation methods, handling data extraction, calling the appropriate estimation (via
  the toggle), ODE solution, error calculation, and plotting.

## Features

- **Bootstrapping:**  
  Bootstrapping can be enabled to assess the variability of the parameter estimates.

- **Flexible Model Configuration:**  
  The module supports different ODE model types (e.g., Distributive, Successive, Random) through configuration
  constants. For example, when using the "randmod" (Random model), the parameter bounds are log-transformed and the
  optimizer works in log-space (with conversion back to the original scale).

- **Integration with Plotting:**  
  After estimation, the module calls plotting functions (via the `Plotter` class) to visualize the ODE solution,
  parameter profiles, and goodness-of-fit metrics.