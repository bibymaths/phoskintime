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

## Usage

### Estimation Mode Toggle

The function `estimate_parameters(mode, ...)` in `toggle.py` serves as the interface that selects the appropriate
routine and returns:

- `estimated_params`: A list of estimated parameter vectors.
- `model_fits`: A list of tuples containing the ODE solution and fitted data.
- `seq_model_fit`: A 2D array of model predictions with shape matching the measurement data.
- `errors`: Error metrics computed during estimation.

### Running the Estimation

The main script (`core.py`) extracts gene-specific data, sets up initial conditions, and calls `estimate_parameters` (
via the toggle) with appropriate inputs such as:

- Measurement data (`P_data`)
- Time points
- Model bounds and fixed parameter settings
- Bootstrapping iteration count

After estimation, the final parameter set is used to solve the full ODE system, and various plots (e.g., model fit, PCA,
t-SNE, profiles) are generated and saved.