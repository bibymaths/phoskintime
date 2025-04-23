# Models

The **models** module provides the implementation of various ODE-based kinetic models used in the PhosKinTime package for phosphorylation dynamics. It is designed to support multiple model types, each corresponding to a different mechanistic hypothesis about how phosphorylation occurs.

---

### **1. Distributive Model**

- $R$: mRNA concentration  
- $P$: protein concentration  
- $P_i$: phosphorylated site $i$  
- $A, B, C, D$: rate constants  
- $S_i$: phosphorylation rate for site $i$  
- $D_i$: dephosphorylation rate for site $i$  

Equations:

- $ \frac{dR}{dt} = A - B R $
- $ \frac{dP}{dt} = C R - (D + \sum_i S_i) P + \sum_i P_i $
- $ \frac{dP_i}{dt} = S_i P - (1 + D_i) P_i \quad \forall i $

---

### **2. Successive Model**

- $R, P, P_i$ as above  
- Phosphorylation proceeds in sequence  

Equations:

- $ \frac{dR}{dt} = A - B R $
- $ \frac{dP}{dt} = C R - D P - S_0 P + P_0 $

Phosphorylation sites:

- First site ($i = 0$):  
  $ \frac{dP_0}{dt} = S_0 P - (1 + S_1 + D_0) P_0 + P_1 $

- Intermediate sites ($0 < i < n-1$):  
  $ \frac{dP_i}{dt} = S_i P_{i-1} - (1 + S_{i+1} + D_i) P_i + P_{i+1} $

- Last site ($i = n - 1$):  
  $ \frac{dP_{n-1}}{dt} = S_{n-1} P_{n-2} - (1 + D_{n-1}) P_{n-1} $

---

### **3. Random Model**

- $X_j$: phosphorylated state $j$, total $2^n - 1$ states  
- $S_i$: phosphorylation rates for each site  
- $D_j$: degradation rate for state $j$  
- Binary transitions determine phosphorylation and dephosphorylation  

Equations:

- $ \frac{dR}{dt} = A - B R $
- $ \frac{dP}{dt} = C R - D P - (\sum_i S_i) P + \sum_{j \in \text{1-site}} X_j + \sum_{j \in \text{dephospho exit}} S_i X_j $

Each state $X_j$:

- $ \frac{dX_j}{dt} = \sum_{\text{phospho from}} S_i X_{src} - (\sum_i S_i + D_j) X_j + \sum_{\text{dephospho to}} S_i X_{src} $

--- 
  
## Weights 

### **Without Regularization**

Let:
- $x_i$: the data point at time $i$
- $t_i$: the time point index ($i = 1, 2, \dots$)
- $T$: total number of time points
- $w_i$: weight at time $i$

Basic schemes:

- **Inverse Data**:  
  $w_i = \frac{1}{|x_i| + \epsilon}$

- **Exponential Decay**:  
  $w_i = \exp(-0.5 \cdot x_i)$

- **Log Scale**:  
  $w_i = \frac{1}{\log(1 + |x_i|)}$

- **Time Difference**:  
  $w_i = \frac{1}{|x_i - x_{i-1}| + \epsilon}$

- **Moving Average Deviation**:  
  $w_i = \frac{1}{|x_i - \text{MA}_i| + \epsilon}$  
  where $\text{MA}_i$ is a moving average (e.g., over 3 points)

- **Sigmoid Time Decay**:  
  $w_i = \frac{1}{1 + \exp(t_i - 5)}$

- **Exponential Early Emphasis**:  
  $w_i = \exp(-0.5 \cdot t_i)$

- **Polynomial Decay**:  
  $w_i = \frac{1}{1 + 0.5 \cdot t_i}$

- **MS SNR Model (Signal-Noise Ratio)**:  
  $w_i = \frac{1}{\sqrt{|x_i| + \epsilon}}$

- **MS Inverse Variance Model**:  
  $w_i = \frac{1}{|x_i|^{0.7} + \epsilon}$

- **Flat Region Penalty**:  
  $w_i = \frac{1}{|\nabla x_i| + \epsilon}$

- **Steady State Decay**:  
  $w_i = \exp(-0.1 \cdot t_i)$

- **Combined Data and Time**:  
  $w_i = \frac{1}{|x_i| \cdot (1 + 0.5 \cdot t_i)}$

- **Inverse Sqrt Data**:  
  $w_i = \frac{1}{\sqrt{|x_i| + \epsilon}}$

- **Early Emphasis (moderate or steep decay)**:  
  $w_i = 1$ (or pre-defined stepwise decay vector)

- **Custom Early Emphasis**:  
  Based on:  
  $w_i = \frac{1}{(|x_i| + \epsilon)(\Delta t_i + \epsilon)}$ for early $t_i$, else $w_i = 1$

### **With Regularization**

Let $w_i$ be any of the above weights and $R$ be the number of regularization parameters:

- **Extended Weight Vector**:  
  $w = [w_1, w_2, \dots, w_T, 1, 1, \dots, 1]$  
  where the last $R$ entries are `1` (flat regularization penalty weights)

This simply appends a vector of ones of length equal to the number of regularization parameters to each weight vector. 
 
#### **Tikhonov Regularization in ODE Parameter Estimation**

This project applies **Tikhonov regularization** (λ = 1e-3) to stabilize parameter estimates and improve identifiability in ODE-based model fitting.

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
 
--- 

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