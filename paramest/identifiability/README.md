# Parameter Identifiability

This module provides a function to compute **confidence intervals (CIs)** and **significance statistics** for model parameters using **linear approximation** based on the covariance matrix from a nonlinear least squares fit.

---

## Purpose

Given:

- Best-fit parameter estimates (`popt`)
- Their covariance matrix (`pcov`)
- The observed data (`target`)

This function estimates:

- Standard errors
- t-statistics
- Two-sided p-values
- 95% confidence intervals (or another level via `alpha_val`)

---

## Mathematical Background

### 1. **Standard Error**

The standard error of each parameter is estimated as:

$$
\text{SE}(\beta_i) = \sqrt{ \text{Var}(\beta_i) } = \sqrt{ \text{diag}(\text{pcov})_i }
$$

Where `pcov` is the **covariance matrix** from the curve fitting routine (typically from `scipy.optimize.curve_fit`).

---

### 2. **Degrees of Freedom**

$$
\text{df} = n_{\text{obs}} - n_{\text{params}}
$$

Used to select the correct **t-distribution** for the confidence interval and p-value computation.

---

### 3. **t-Statistic**

For each parameter:

$$
t_i = \frac{\hat{\beta}_i}{\text{SE}(\hat{\beta}_i)}
$$

---

### 4. **p-Value**

Two-sided p-value from the Student’s t-distribution:

$$
p_i = 2 \cdot P(T > |t_i|) = 2 \cdot \text{sf}(|t_i|, \text{df})
$$

---

### 5. **Confidence Interval**

Using the t-critical value $t^*$:

$$
t^* = t_{1 - \alpha/2, \text{df}}
$$

Confidence bounds:

$$
\text{CI}_i = \left[ \max\left(0, \hat{\beta}_i - t^* \cdot \text{SE}(\hat{\beta}_i)\right),\ \hat{\beta}_i + t^* \cdot \text{SE}(\hat{\beta}_i) \right]
$$

> Lower bound is clipped to zero for non-negative parameters.

---

## When to Use

- To assess parameter certainty
- To report statistical significance and error bars