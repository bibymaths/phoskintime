# Models Module

The **models** module provides the implementation of various ODE-based kinetic models used in the PhosKinTime package for phosphorylation dynamics. It is designed to support multiple model types, each corresponding to a different mechanistic hypothesis about how phosphorylation occurs.

## Overview

This module includes implementations of the following model types:

- **Random Model (`randmod.py`):**  
  Implements a vectorized and optimized ODE system using Numba. This model represents a random mechanism of phosphorylation, where transitions between phosphorylation states are computed based on binary representations. The module prepares vectorized arrays (e.g., binary states, phosphorylation/dephosphorylation targets) and defines the ODE system accordingly.

- **Distributive Model (`distmod.py`):**  
  Implements a distributive phosphorylation mechanism. In this model, a kinase adds phosphate groups in a manner where each phosphorylation event is independent, and the ODE system is defined with explicit state variables for the phosphorylated forms.

- **Successive Model (`succmod.py`):**  
  Implements a successive phosphorylation mechanism, where phosphorylation occurs in a sequential, stepwise manner. This model's ODE system is tailored to capture the sequential nature of the modification.

- **Weighting Functions (`weights.py`):**  
  Provides functions to compute various weighting schemes (e.g., early emphasis, inverse data, exponential decay) used during parameter estimation. These weights help tailor the fitting process to the dynamics of the observed data.

## Automatic Model Selection

The package’s `__init__.py` file in the models module automatically imports the correct model module based on the configuration constant `ODE_MODEL`. The selected module’s `solve_ode` function is then exposed as the default ODE solver for the package. This enables seamless switching between different mechanistic models without changing the rest of the code.

## Key Features

- **Vectorized Computation and JIT Optimization:**  
  For the random model, vectorized arrays and Numba’s `@njit` decorator are used to accelerate ODE evaluations.

- **Modular Design:**  
  Each model type is implemented in its own file, allowing easy extension or modification of the underlying kinetics without affecting the overall framework.

- **Flexible Integration:**  
  The models use `scipy.integrate.odeint` to numerically integrate the ODE system, ensuring robust and accurate simulation of phosphorylation dynamics.

- **Support for Multiple Phosphorylation Sites:**  
  All models are designed to handle an arbitrary number of phosphorylation sites, with appropriate state variable definitions and parameter extraction.

- **Customizable Weighting for Parameter Estimation:**  
  The weights module provides several functions for generating weights (e.g., early emphasis) to be used during the parameter estimation process, enhancing the fitting performance.

## Dependencies

- **NumPy & SciPy:** For numerical operations, ODE integration, and optimization.
- **Numba:** To accelerate performance-critical functions via just-in-time (JIT) compilation.
- **Other Dependencies:** The module works within the PhosKinTime package, leveraging configuration and logging utilities defined elsewhere in the package.

## Usage

Once configured (by setting `ODE_MODEL` appropriately in your configuration files), the package automatically imports the correct model:
  
```python
# In models/__init__.py
import importlib
from config.constants import ODE_MODEL

try:
    model_module = importlib.import_module(f'models.{ODE_MODEL}')
except ModuleNotFoundError as e:
    raise ImportError(f"Cannot import model module 'models.{ODE_MODEL}'") from e

solve_ode = model_module.solve_ode
```
---

### Units in the ODE Model

These ODE models supports **two interpretations** depending on whether quantities are scaled:

#### 1. **Dimensionless Model (Scaled)**
- All parameters and variables are **unitless**.
- Time and concentrations are **rescaled** to reference values (e.g., max input, steady state).
- Useful for qualitative dynamics, numerical stability, or fitting fold-change data.
- Interpretation:  
  - `A, B, C, D, S_rates[i], D_rates[i]` → **unitless**  
  - `y` (state vector: R, P, P_sites) → **unitless**

#### 2. **Dimensional (Mass-Action Style)**
- Variables represent **concentration** (e.g., μM), and time is in seconds.
- Parameters follow biochemical units:
  - `A` → concentration/time (e.g., μM/s)  
  - `B, C, D, S_rates[i], D_rates[i]` → 1/time (e.g., 1/s)  
  - `R, P, y[2+i]` → concentration (e.g., μM)
- Caveat: Dimensional consistency requires adjustment (e.g., replacing hardcoded `1.0` with a rate constant and scaling summed terms accordingly).

Here’s a concise and clear `README` section tailored for your **PhosKinTime** tool, explaining the normalization logic for fold change data:

---

### Fold Change Normalization in PhosKinTime

**PhosKinTime** supports modeling and parameter estimation of phosphorylation dynamics using time series data. Often, such experimental data is provided not in absolute concentration units but as **fold change (FC)** relative to a baseline (usually time point 0). To ensure accurate and biologically meaningful comparison between model output and experimental data, **PhosKinTime includes built-in support to normalize model output to fold change form.**

#### Why Normalize?

Experimental FC data is typically defined as:

$$
\text{FC}(t) = \frac{X(t)}{X(t_0)}
$$

where $X(t)$ is the measured signal (e.g., intensity or concentration) at time $t$, and $X(t_0)$ is the baseline (often the 0 min time point). It reflects **relative change**, not absolute concentration.

However, PhosKinTime's ODE models simulate **absolute concentrations** over time:

$$
Y(t) = \text{ODE solution}
$$

Directly comparing $Y(t)$ to FC data is **invalid**, as it compares mismatched scales and units. To bridge this gap, PhosKinTime transforms the model output into comparable fold change form by:

$$
\text{FC}_{\text{model}}(t) = \frac{Y(t)}{Y(t_0) + \epsilon}
$$

($\epsilon$ is a small constant to avoid division by zero.)

This transformation is applied per phosphorylation site (or species) independently, ensuring robust and interpretable parameter fitting.

### References

- Klipp, E., et al. (2016). *Systems Biology: A Textbook* (2nd ed.). Wiley-VCH.  
- Raue, A., et al. (2013). Lessons learned from quantitative dynamical modeling in systems biology. *PLoS ONE*, 8(9), e74335.  
- BioModels Documentation: [https://www.ebi.ac.uk/biomodels/docs/](https://www.ebi.ac.uk/biomodels/docs/)

---