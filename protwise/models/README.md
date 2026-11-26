# Models

The **models** module provides the implementation of various ODE-based kinetic models used in the PhosKinTime package
for phosphorylation dynamics. It is designed to support multiple model types, each corresponding to a different
mechanistic hypothesis about how phosphorylation occurs.

---

### **1. Distributive Model**

- $R$: mRNA concentration
- $P$: protein concentration
- $P_i$: phosphorylated site $i$
- $A, B, C, D$: rate constants
- $S_i$: phosphorylation rate for site $i$
- $D_i$: degradation rate of the phosphorylated form at site $i$; dephosphorylation is the fixed rate 1 in
  $(1 + D_i)$ (KNOWN_ISSUES P9). The 1 is a fixed constant k = 1.0 min⁻¹, not a fitted parameter: in a test on
  three genes (ABL2, MAPK1, GRK2) a fitted k stayed near 1 (1.04–1.27), did not lower the objective, and was not
  identifiable from these data (flat directions of the objective's Hessian).

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

**Uncertanities from data**:  
They are interpreted as the **standard deviations** of the data points. The formula for the weighted least squares fit  
becomes:

$$
\chi^2 = \sum_{i=1}^N \left( \frac{y_i - f(x_i, \theta)}{\sigma_i} \right)^2
$$
where $y_i$ is the observed data point, $f(x_i, \theta)$ is the model function with parameters $\theta$, and $\sigma_i$
is the standard deviation (weights) for the $i^{th}$ data point.

The `absolute_sigma=True` argument ensures that the covariance matrix of the parameter estimates is scaled by the
provided  
$\sigma_i$ values, rather than being rescaled by the reduced chi-squared value. This means the uncertainties in the
data  
(as given by the weights) are treated as absolute.

When the estimations are not ideal, the weights can be adjusted to improve the fit. The `absoulute_sigma = False` for  
`curve_fit` will rescale the covariance matrix by the reduced chi-squared value, which can be useful when the
uncertainties  
in the data are not well characterized.

The following are some common weighting schemes:

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

This project applies **Tikhonov regularization** (λ = 1e-3) to stabilize parameter estimates and improve identifiability
in ODE-based model fitting.

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
  Implements a vectorized and optimized ODE system using Numba. This model represents a random mechanism of
  phosphorylation, where transitions between phosphorylation states are computed based on binary representations. The
  module prepares vectorized arrays (e.g., binary states, phosphorylation/dephosphorylation targets) and defines the ODE
  system accordingly.

- **Distributive Model (`distmod.py`):**  
  Implements a distributive phosphorylation mechanism. In this model, a kinase adds phosphate groups in a manner where
  each phosphorylation event is independent, and the ODE system is defined with explicit state variables for the
  phosphorylated forms.

- **Successive Model (`succmod.py`):**  
  Implements a successive phosphorylation mechanism, where phosphorylation occurs in a sequential, stepwise manner. This
  model's ODE system is tailored to capture the sequential nature of the modification.

- **Weighting Functions (`weights.py`):**  
  Provides functions to compute various weighting schemes (e.g., early emphasis, inverse data, exponential decay) used
  during parameter estimation. These weights help tailor the fitting process to the dynamics of the observed data.

## Automatic Model Selection

The package’s `__init__.py` file in the models module automatically imports the correct model module based on the
configuration constant `ODE_MODEL`. The selected module’s `solve_ode` function is then exposed as the default ODE solver
for the package. This enables seamless switching between different mechanistic models without changing the rest of the
code.

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
- Caveat: Dimensional consistency requires adjustment (e.g., replacing hardcoded `1.0` with a rate constant and scaling
  summed terms accordingly).

### Model output scale

The protwise fit compares the model states directly with the processed input values (fold changes); model
output is not rescaled. An earlier `normalize_model_output` option divided only the *reported* curves by the
initial conditions while the objective stayed unnormalised; it was removed (KNOWN_ISSUES P3). How the t = 0
baseline enters the model is set by the initial conditions (KNOWN_ISSUES P4).

### References

- Klipp, E., et al. (2016). *Systems Biology: A Textbook* (2nd ed.). Wiley-VCH.
- Raue, A., et al. (2013). Lessons learned from quantitative dynamical modeling in systems biology. *PLoS ONE*, 8(9),
  e74335.
- BioModels Documentation: [https://www.ebi.ac.uk/biomodels/docs/](https://www.ebi.ac.uk/biomodels/docs/)

---
### Random-order model: initial states (Round 2 item 14)

The randmod initial state follows an **independence assumption**: each site is phosphorylated independently of the
others, so the initial mass of subset state m is total · Π_{j∈m} occ_j · Π_{j∉m} (1 − occ_j) with occ_j = s_j / total
(s_j = site j value at t = 0). The total is chosen so that the unphosphorylated state (the randmod protein observable)
equals the protein value at t = 0: total · Π(1 − s_j/total) = P(0), a unique root with total > max s_j. The states sum
to the total; the protein value and every site value at t = 0 are reproduced exactly; no projection is needed.
