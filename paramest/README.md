### 📌 **Tikhonov Regularization in ODE Parameter Estimation**

This project applies **Tikhonov regularization** (λ = 1e-3) to stabilize parameter estimates and improve identifiability in ODE-based model fitting.

#### 🔍 What It Does
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

#### 📈 Interpretation
- **Estimates are shrunk** toward zero (or prior).
- **Uncertainty (covariance)** is reduced, reflecting added prior information.
- Regularization improves **numerical stability** and reduces **overfitting**.

#### ✅ Post-Regularization Checks
- Compare `θ_fit` vs `θ_reg` and `C` vs `C_reg`.
- Assess model fit with regularized parameters.
- Examine parameter correlations and identifiability.
- Optionally test sensitivity to different `λ` values.

#### 📎 Note
This approach assumes the likelihood is locally quadratic—valid for most ODE-based models near optimum.