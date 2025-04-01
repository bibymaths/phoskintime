# Parameter Estimation Module

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

### **Tikhonov Regularization in ODE Parameter Estimation**

This project applies **Tikhonov regularization** (λ = 1e-3) to stabilize parameter estimates and improve identifiability in ODE-based model fitting.

#### What It Does
- Computes **unregularized estimates** and their **covariance matrix**.
- Applies Tikhonov regularization post hoc:
- **Regularized estimates**:  
  $$
  \theta_{\text{reg}} = \theta_{\text{fit}} - \lambda C \Gamma \theta_{\text{fit}}
  $$

- **Regularized covariance**:  
  $$
  C_{\text{reg}} = \left(C^{-1} + \lambda \Gamma \right)^{-1}
  $$
- Typically, `Γ` is the identity matrix.

#### Interpretation
- **Estimates are shrunk** toward zero (or prior).
- **Uncertainty (covariance)** is reduced, reflecting added prior information.
- Regularization improves **numerical stability** and reduces **overfitting**.

#### Post-Regularization Checks
- Compare `θ_fit` vs `θ_reg` and `C` vs `C_reg`.
- Assess model fit with regularized parameters.
- Examine parameter correlations and identifiability.
- Optionally test sensitivity to different `λ` values.

#### Note
This approach assumes the likelihood is locally quadratic—valid for most ODE-based models near optimum. 

## Dependencies

The module relies on several external packages:

- **Numerical and Scientific Libraries:**  
  `numpy`, `pandas`, `scipy`

- **Optimization and Integration:**  
  `scipy.optimize.curve_fit`, `scipy.integrate.odeint`

- **Performance:**  
  `numba` for Just-In-Time compilation (e.g., for early weighting computations)

- **Machine Learning:**  
  `sklearn` (for PCA and t-SNE)

- **Plotting:**  
  `matplotlib`, `seaborn`, `plotly`, and `adjustText`

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

### Example

```python
from paramest.toggle import estimate_parameters

# For a given gene, with data, initial conditions, etc.
mode = "normal"  # or "sequential"
model_fits, estimated_params, seq_model_fit, errors = estimate_parameters(
    mode, gene, P_data, init_cond, num_psites, time_points, bounds, fixed_params, bootstraps
)

# seq_model_fit will have the same shape as P_data: (num_psites, len(time_points))
```

## Customization

- **Parameter Bounds & Fixed Parameters:**  
  Bounds and fixed parameter settings are defined in your configuration files. These can be tuned based on the biological context.

- **Regularization:**  
  Tikhonov regularization is applied by concatenating a penalty term to the target vector; the regularization strength (`lambda_reg`) is configurable.

- **Bootstrapping:**  
  To assess uncertainty in the estimates, set the number of bootstrapping iterations as needed.

- **Model Type:**  
  The behavior for different ODE model types is controlled by the `ODE_MODEL` configuration constant. For example, for `"randmod"`, the parameters are estimated in log-space for `"normal"`estimation only.

## Conclusion

This parameter estimation module provides a flexible framework to fit ODE-based models to phosphorylation data. By offering multiple estimation approaches, adaptive profile generation, and robust error metrics, it is designed to support in-depth model analysis and validation. Combined with its integrated plotting utilities, the module facilitates both diagnostic assessment and presentation of results.
