`Originally implemented by Julius Normann.` 

`This version has been modified and optimized for consistency & speed in submodules by Abhinav Mishra.`

## Model Equation

For each mRNA (indexed by *i*), the measured time series is represented by

$$
\mathbf{R}_i = \left([mRNA]_i(t_1), [mRNA]_i(t_2), \dots, [mRNA]_i(T)\right)
$$

Its predicted value is modeled as a weighted combination of the effects of transcription factors (TFs) that regulate it. Each TF (indexed by *j*) contributes in two ways:
- A **protein component** (when no phosphorylation site is reported) with time series \(TF_{i,j}(t)\).
- A **PSite component** (when a phosphorylation site is provided) with time series \(PSite_{k,j}(t)\) for each site \(k\).

These contributions are modulated by constant parameters:
- **α-values:** For each mRNA, the impact of TF *j* is given by a weight \(\alpha_{i,j}\).
- **β-values:** For each TF, the effect of its overall level (and its PSite contributions) is modeled by a vector 
  $$
   {\beta}_j = \bigl(\beta_{0,j}, \beta_{1,j}, \dots, \beta_{K_j,j}\bigr).
  $$
  Here, \(\beta_{0,j}\) multiplies the direct TF signal and the remaining β’s weight the PSite signals.

Thus, the predicted mRNA time series is given by

$$
\hat{R}_i(t) = \sum_{j\in \mathcal{J}_i} \alpha_{i,j} \; TF_{i,j}(t) \left( \beta_{0,j} + \sum_{k=1}^{K_j} PSite_{k,j}(t) \, \beta_{k,j} \right)
$$

where \(\mathcal{J}_i\) is the set of TFs (extracted from the interaction file, e.g. from *input4_reduced.csv*) that are linked to mRNA *i* (with corresponding protein data available in *input1_msgauss.csv*).

---

# Objective Function

The goal is to have the predicted mRNA time series $(\hat{R}_i(t))$ match the measured data $(R_i(t))$ as closely as possible. This can be achieved by minimizing the sum of squared differences over all mRNAs and time points:

$$
\min_{\{\alpha,\beta\}} \quad \sum_{i} \sum_{t} \Bigl( R_i(t) - \hat{R}_i(t) \Bigr)^2
$$

In this formulation, there is no explicit use of error vectors; the optimization directly minimizes the discrepancy between the measured and predicted values.

---

# Constraints

The parameters are subject to the following constraints:

1. **α-constraints (for each mRNA *i*):**

   $$
   \sum_{j\in \mathcal{J}_i} \alpha_{i,j} = 1,\qquad 0 \le \alpha_{i,j} \le 1.
   $$

2. **β-constraints (for each TF *j*):**

   $$
   \sum_{q=0}^{K_j} \beta_{q,j} = 1,\qquad -2 \le \beta_{q,j} \le 2.
   $$

---

In summary, the optimization problem is to find the α‐ and β‐parameters that minimize the squared error between the measured mRNA time series (LHS) and the predicted mRNA time series (RHS):

$$
\min_{\{\alpha,\beta\}} \quad \sum_{i} \sum_{t} \Bigl( R_i(t) - \sum_{j\in \mathcal{J}_i} \alpha_{i,j} \; TF_{i,j}(t) \left( \beta_{0,j} + \sum_{k=1}^{K_j} PSite_{k,j}(t) \, \beta_{k,j} \right) \Bigr)^2
$$

subject to the above constraints on \(\alpha\) and \(\beta\). This formulation enables estimation of the transcription factor and phosphorylation site contributions directly from the mRNA time series data.