# tfopt — Transcription Factor Optimization Framework

- **Originally implemented by Julius Normann.**  
- **This version has been modified and optimized for consistency & speed in submodules by Abhinav Mishra.**

`tfopt` provides a flexible architecture for estimating transcriptional regulatory influence using mRNA time series data, TF protein dynamics, and phosphorylation site signals.

The package contains two main submodules:

- **`tfopt/evol`** — global optimization via multi-objective evolutionary algorithms  
- **`tfopt/local`** — constrained optimization using SciPy solvers (e.g., SLSQP)

Both modules share a consistent data preparation pipeline and model formulation.

---

## Model Equation

For each mRNA (indexed by *i*), the measured time series is represented by:

$$
\mathbf{R}_i = \left([mRNA]_i(t_1), [mRNA]_i(t_2), \dots, [mRNA]_i(T)\right)
$$

Its predicted value is modeled as a weighted combination of the effects of transcription factors (TFs) that regulate it. Each TF (indexed by *j*) contributes in two ways:
- A **protein component** (when no phosphorylation site is reported) with time series \( TF_{i,j}(t) \)
- A **PSite component** (when phosphorylation sites are available) with time series \( PSite_{k,j}(t) \) for each site *k*

These contributions are modulated by two sets of parameters:
- **α-values**: For each mRNA, the impact of TF *j* is weighted by \( \alpha_{i,j} \)
- **β-values**: For each TF, a vector of weights:

  $$
  \beta_j = \left( \beta_{0,j}, \beta_{1,j}, \dots, \beta_{K_j,j} \right)
  $$

Here, $ \beta_{0,j} $ multiplies the raw TF protein signal, and the remaining terms multiply phosphorylation site contributions.

The full predicted mRNA time series is then:

$$
\hat{R}_i(t) = \sum_{j\in \mathcal{J}_i} \alpha_{i,j} \cdot TF_{i,j}(t) \left( \beta_{0,j} + \sum_{k=1}^{K_j} PSite_{k,j}(t) \cdot \beta_{k,j} \right)
$$

where $ \mathcal{J}_i $ is the set of TFs regulating gene *i* (according to the interaction data).

---

## Objective Function

To estimate the best set of weights, we minimize the difference between measured and predicted expression over all genes and time points:

$$
\min_{\{\alpha,\beta\}} \quad \sum_i \sum_t \left( R_i(t) - \hat{R}_i(t) \right)^2
$$

This formulation supports multiple loss types (MSE, MAE, soft L1, Cauchy, etc.) implemented in both submodules.

---

## Constraints

### α-constraints (for each mRNA *i*):

$$
\sum_{j\in \mathcal{J}_i} \alpha_{i,j} = 1, \quad 0 \le \alpha_{i,j} \le 1
$$

### β-constraints (for each TF *j*):

$$
\sum_{q=0}^{K_j} \beta_{q,j} = 1, \quad -2 \le \beta_{q,j} \le 2
$$

This ensures that weights are interpretable and stable.

---

## Optimization Problem Summary

The final optimization problem is:

$$
\min_{\{\alpha,\beta\}} \sum_i \sum_t \left( R_i(t) - \sum_{j\in \mathcal{J}_i} \alpha_{i,j} \cdot TF_{i,j}(t) \left( \beta_{0,j} + \sum_{k=1}^{K_j} PSite_{k,j}(t) \cdot \beta_{k,j} \right) \right)^2
$$

subject to the constraints above. This enables estimation of regulatory influences in a biologically meaningful and data-driven manner.

---

## Submodules

### `evol/` — Global Evolutionary Optimization

- Implements multi-objective optimization using `pymoo` (NSGA2, AGEMOEA, SMSEMOA)
- Evaluates tradeoffs between fit error, α-constraint violation, and β-constraint violation
- Outputs Excel summaries, static and interactive plots, and HTML reports

### `local/` — Constrained Local Optimization

- Implements deterministic solvers (e.g. SLSQP)
- Faster and more interpretable for small- to medium-scale systems
- Shares the same objective and constraint framework as `evol`
- Generates the same reports and plots as the global module

---

## Usage

From one level top of project root:

```bash
python -m phoskintime tfopt --mode evol
```

or

```bash
python -m phoskintime tfopt --mode local
```

Output will be saved in structured folders, including Excel files, plots, and an aggregated HTML report.