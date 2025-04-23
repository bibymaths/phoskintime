# Parameter Estimation

This module provides the tools needed to estimate parameters for ODE‐based models of phosphorylation dynamics. It implements several estimation approaches and supports bootstrapping, adaptive profile estimation, and toggling between different estimation modes.

## Overview

The module is organized into several submodules:

- **`seqest.py`** – Implements sequential (time-point–wise) parameter estimation. This approach estimates parameters incrementally using data up to each time point.  
- **`normest.py`** – Implements normal (all timepoints at once) parameter estimation. This approach fits the entire time-series data in one step.  
- **`adapest.py`** – Provides adaptive estimation for profile generation. It uses data interpolation (via PCHIP) to generate a profile of parameter estimates over time.  
- **`toggle.py`** – Offers a single function (`estimate_parameters`) to switch between sequential and normal estimation modes based on a mode flag.  
- **`core.py`** – Integrates the estimation methods, handling data extraction, calling the appropriate estimation (via the toggle), ODE solution, error calculation, and plotting.

## Features

- **Estimation Modes:**  
  - **Sequential Estimation:** Parameters are estimated in a time-sequential manner, providing an evolving view of the fit.  
  - **Normal Estimation:** A single estimation over all-time points gives a comprehensive fit to the entire data set.
  
- **Adaptive Profile Estimation:**  
  Using interpolation and a weighted scheme, the module can generate time profiles of parameter estimates.

- **Bootstrapping:**  
  Bootstrapping can be enabled to assess the variability of the parameter estimates.

- **Flexible Model Configuration:**  
  The module supports different ODE model types (e.g., Distributive, Successive, Random) through configuration constants. For example, when using the "randmod" (Random model), the parameter bounds are log-transformed and the optimizer works in log-space (with conversion back to the original scale).

- **Integration with Plotting:**  
  After estimation, the module calls plotting functions (via the `Plotter` class) to visualize the ODE solution, parameter profiles, and goodness-of-fit metrics.

## Usage

### Estimation Mode Toggle

The global constant `ESTIMATION_MODE` (set in your configuration) controls which estimation approach is used:
- `"sequential"`: Uses the sequential estimation routine in `seqest.py`.
- `"normal"`: Uses the normal estimation routine in `normest.py`.

The function `estimate_parameters(mode, ...)` in `toggle.py` serves as the interface that selects the appropriate routine and returns:
- `estimated_params`: A list of estimated parameter vectors.
- `model_fits`: A list of tuples containing the ODE solution and fitted data.
- `seq_model_fit`: A 2D array of model predictions with shape matching the measurement data.
- `errors`: Error metrics computed during estimation.

### Running the Estimation

The main script (`core.py`) extracts gene-specific data, sets up initial conditions, and calls `estimate_parameters` (via the toggle) with appropriate inputs such as:
- Measurement data (`P_data`)
- Time points
- Model bounds and fixed parameter settings
- Bootstrapping iteration count

After estimation, the final parameter set is used to solve the full ODE system, and various plots (e.g., model fit, PCA, t-SNE, profiles) are generated and saved.