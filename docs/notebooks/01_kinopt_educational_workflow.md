# KinOpt educational workflow: local kinase-effect fitting

**Purpose.** KinOpt estimates how upstream kinase activity explains observed phosphorylation-site dynamics. This notebook builds a tiny kinase–substrate example and walks through the real KinOpt preprocessing and objective-evaluation utilities, then performs a lightweight JAXopt local optimization to form a **ranked multistart solution ensemble**.

Scientifically, each substrate phosphorylation site is modeled as a weighted mixture of candidate kinase effects. Computationally, the workflow converts biological tables into fixed arrays, solves a **single-objective constrained optimization** problem, ranks starts by scalar loss, and exports parameters and plots.

## Mathematical problem

For substrate site \(i\), kinase \(k\), kinase phosphorylation/readout row \(r\), and time \(t\):

$$
M_k(t) = \sum_{r \in \mathcal{R}(k)} \beta_{kr} K_{r}(t), \qquad
\hat P_i(t) = \sum_{k \in \mathcal{K}(i)} \alpha_{ik} M_k(t)
$$

The scalar objective is mean squared error:

$$
L(\alpha,\beta)=\frac{1}{n}\sum_i\sum_t(P_i(t)-\hat P_i(t))^2.
$$

Here \(\alpha\) distributes influence among candidate kinases for each substrate site and \(\beta\) combines kinase-level measured rows. We enforce simplex-style constraints in the notebook optimizer so each \(\alpha\) block and each \(\beta\) block sums to one. The fitted result is interpreted as a local optimum of a single scalar loss, not as a multi-objective front.


```python

from pathlib import Path
import sys

REPO_ROOT = Path.cwd()
if not (REPO_ROOT / "README.md").exists():
    REPO_ROOT = Path.cwd().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT.parent))
import os
os.chdir(REPO_ROOT)

FAST_NOTEBOOK = True
RANDOM_SEED = 7

import json
from types import SimpleNamespace

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jaxopt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from kinopt.local.optcon.construct import (
    _build_P_initial, _build_K_data, _convert_to_sparse, _precompute_mappings,
    _compute_time_weights,
)
from kinopt.local.objfn.minfn import _objective
from kinopt.local.objfn import estimated_series
from kinopt.local.utils.params import extract_parameters, compute_metrics
from kinopt.local.exporter.plotout import export_outcomes_to_csv, plot_multistart_summary_runtime_overlay
from networkmodel.backend import project_alpha_blocks, project_beta_blocks

np.random.seed(RANDOM_SEED)
print("JAX float64 enabled:", jax.config.jax_enable_x64)

```

    JAX float64 enabled: True


## Dummy KinOpt input tables

KinOpt expects a phosphorylation table with `GeneID`, `Psite`, and time columns (`x1` … `x14`) plus an interaction table mapping each substrate site to candidate kinases. The synthetic data below has two substrate sites and two kinases. Fourteen columns are supplied because the current KinOpt preprocessing utilities expect that schema, even though this educational example uses only tiny smooth curves.


```python

time_cols = [f"x{i}" for i in range(1, 15)]
t = np.arange(14, dtype=float)

def curve(base, slope, wave=0.0):
    return base + slope * t + wave * np.sin(t / 2)

full_df = pd.DataFrame([
    {"GeneID": "SUB1", "Psite": "S10", **dict(zip(time_cols, curve(1.00, 0.035, 0.03)))},
    {"GeneID": "SUB2", "Psite": "T22", **dict(zip(time_cols, curve(0.90, 0.020, -0.02)))},
    {"GeneID": "KIN_A", "Psite": "Y100", **dict(zip(time_cols, curve(1.10, 0.040, 0.02)))},
    {"GeneID": "KIN_B", "Psite": "S200", **dict(zip(time_cols, curve(0.85, 0.015, 0.04)))},
])
interact_df = pd.DataFrame([
    {"GeneID": "SUB1", "Psite": "S10", "Kinase": ["KIN_A", "KIN_B"]},
    {"GeneID": "SUB2", "Psite": "T22", "Kinase": ["KIN_B"]},
])
display(full_df.head())
display(interact_df)

```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GeneID</th>
      <th>Psite</th>
      <th>x1</th>
      <th>x2</th>
      <th>x3</th>
      <th>x4</th>
      <th>x5</th>
      <th>x6</th>
      <th>x7</th>
      <th>x8</th>
      <th>x9</th>
      <th>x10</th>
      <th>x11</th>
      <th>x12</th>
      <th>x13</th>
      <th>x14</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>SUB1</td>
      <td>S10</td>
      <td>1.00</td>
      <td>1.049383</td>
      <td>1.095244</td>
      <td>1.134925</td>
      <td>1.167279</td>
      <td>1.192954</td>
      <td>1.214234</td>
      <td>1.234477</td>
      <td>1.257296</td>
      <td>1.285674</td>
      <td>1.321232</td>
      <td>1.363834</td>
      <td>1.411618</td>
      <td>1.461454</td>
    </tr>
    <tr>
      <th>1</th>
      <td>SUB2</td>
      <td>T22</td>
      <td>0.90</td>
      <td>0.910411</td>
      <td>0.923171</td>
      <td>0.940050</td>
      <td>0.961814</td>
      <td>0.988031</td>
      <td>1.017178</td>
      <td>1.047016</td>
      <td>1.075136</td>
      <td>1.099551</td>
      <td>1.119178</td>
      <td>1.134111</td>
      <td>1.145588</td>
      <td>1.155698</td>
    </tr>
    <tr>
      <th>2</th>
      <td>KIN_A</td>
      <td>Y100</td>
      <td>1.10</td>
      <td>1.149589</td>
      <td>1.196829</td>
      <td>1.239950</td>
      <td>1.278186</td>
      <td>1.311969</td>
      <td>1.342822</td>
      <td>1.372984</td>
      <td>1.404864</td>
      <td>1.440449</td>
      <td>1.480822</td>
      <td>1.525889</td>
      <td>1.574412</td>
      <td>1.624302</td>
    </tr>
    <tr>
      <th>3</th>
      <td>KIN_B</td>
      <td>S200</td>
      <td>0.85</td>
      <td>0.884177</td>
      <td>0.913659</td>
      <td>0.934900</td>
      <td>0.946372</td>
      <td>0.948939</td>
      <td>0.945645</td>
      <td>0.940969</td>
      <td>0.939728</td>
      <td>0.945899</td>
      <td>0.961643</td>
      <td>0.986778</td>
      <td>1.018823</td>
      <td>1.053605</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GeneID</th>
      <th>Psite</th>
      <th>Kinase</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>SUB1</td>
      <td>S10</td>
      <td>[KIN_A, KIN_B]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>SUB2</td>
      <td>T22</td>
      <td>[KIN_B]</td>
    </tr>
  </tbody>
</table>
</div>


## Preprocess into KinOpt arrays

The package preprocessing maps biological identifiers into compact numeric arrays: observed substrate matrix `P_array`, kinase readout matrix `K_array`, sparse kinase storage, and index vectors that locate each parameter block. This step is necessary because repeated objective calls must be fast and shape-stable.


```python

P_initial, P_array = _build_P_initial(full_df, interact_df)
K_index, K_array, beta_counts = _build_K_data(full_df, interact_df, estimate_missing=False)
K_sparse, K_data, K_indices, K_indptr = _convert_to_sparse(K_array)
(unique_kinases, gene_kinase_counts, gene_alpha_starts, gene_kinase_idx,
 total_alpha, kinase_beta_counts, kinase_beta_starts) = _precompute_mappings(P_initial, K_index)
t_max, P_dense, time_weights = _compute_time_weights(P_array, loss_type="mse")

print("unique_kinases:", unique_kinases)
print("P_array shape:", P_array.shape, "K_array shape:", K_array.shape)
print("total alpha parameters:", total_alpha, "total beta parameters:", int(sum(kinase_beta_counts)))
param_layout = pd.DataFrame({
    "substrate_site": [str(k) for k in P_initial.keys()],
    "candidate_kinases": [", ".join(v["Kinases"]) for v in P_initial.values()],
    "n_alpha": gene_kinase_counts,
})
display(param_layout)

```

    unique_kinases: ['KIN_A', 'KIN_B']
    P_array shape: (2, 14) K_array shape: (2, 14)
    total alpha parameters: 3 total beta parameters: 2



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>substrate_site</th>
      <th>candidate_kinases</th>
      <th>n_alpha</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>('SUB1', 'S10')</td>
      <td>KIN_A, KIN_B</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>('SUB2', 'T22')</td>
      <td>KIN_B</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>


## Model setup and local optimization

We evaluate the real KinOpt numba objective for diagnostics. For the educational local optimizer we use an equivalent JAX objective and JAXopt projected-gradient solver so this notebook follows the current constrained-local-optimization style used by the active JAX/JAXopt modules. Projection keeps each biological weight block interpretable as a convex contribution.


```python

# Block identifiers for simplex projection.
alpha_block_ids = np.concatenate([np.full(c, i) for i, c in enumerate(gene_kinase_counts)])
beta_block_ids = np.concatenate([np.full(c, i) for i, c in enumerate(kinase_beta_counts)])
n_beta = len(beta_block_ids)

P_j = jnp.asarray(P_dense)
K_j = jnp.asarray(K_array)
gene_starts_j = jnp.asarray(gene_alpha_starts)
gene_counts_j = jnp.asarray(gene_kinase_counts)
gene_kinase_idx_j = jnp.asarray(gene_kinase_idx)
kin_beta_starts_j = jnp.asarray(kinase_beta_starts)
kin_beta_counts_j = jnp.asarray(kinase_beta_counts)

def kinopt_jax_prediction(theta):
    alpha = theta[:total_alpha]
    beta = theta[total_alpha:]
    # Kinase effects M[k, t]
    M_rows = []
    for k in range(len(unique_kinases)):
        start = int(kinase_beta_starts[k]); count = int(kinase_beta_counts[k])
        M_rows.append(jnp.sum(beta[start:start+count, None] * K_j[start:start+count, :], axis=0))
    M = jnp.stack(M_rows)
    preds = []
    for i in range(P_dense.shape[0]):
        start = int(gene_alpha_starts[i]); count = int(gene_kinase_counts[i])
        kin_idx = gene_kinase_idx_j[start:start+count]
        preds.append(jnp.sum(alpha[start:start+count, None] * M[kin_idx, :], axis=0))
    return jnp.clip(jnp.stack(preds), 0.0)

def kinopt_objective_jax(theta):
    pred = kinopt_jax_prediction(theta)
    return jnp.mean((P_j - pred) ** 2)

def kinopt_projection(theta, _):
    a = project_alpha_blocks(theta[:total_alpha], alpha_block_ids)
    b = project_beta_blocks(theta[total_alpha:], beta_block_ids, lower=0.0, upper=1.0)
    return jnp.concatenate([a, b])

solver = jaxopt.ProjectedGradient(kinopt_objective_jax, kinopt_projection, maxiter=8 if FAST_NOTEBOOK else 100, tol=1e-6, stepsize=0.0)

def make_start(seed):
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0, 1, total_alpha + n_beta)
    return np.asarray(kinopt_projection(jnp.asarray(raw), None))

outcomes = []
for start_id, seed in enumerate([11, 12, 13] if FAST_NOTEBOOK else range(20)):
    theta0 = make_start(seed)
    params, state = solver.run(theta0, hyperparams_proj=None)
    params = np.asarray(kinopt_projection(params, None))
    package_loss = float(_objective(params, P_dense, t_max, P_dense.shape[0], gene_alpha_starts, gene_kinase_counts,
                                    gene_kinase_idx, total_alpha, kinase_beta_starts, kinase_beta_counts,
                                    K_data, K_indices, K_indptr, time_weights, 0))
    outcomes.append(SimpleNamespace(start_id=start_id, seed=seed, result=state, optimized_params=params,
                                    fun=package_loss, success=True, constr_violation=0.0, runtime_s=np.nan))

ranked = sorted(outcomes, key=lambda o: o.fun)
summary_df = pd.DataFrame([{"rank": r+1, "start_id": o.start_id, "seed": o.seed, "objective_loss": o.fun} for r, o in enumerate(ranked)])
display(summary_df)
best = ranked[0].optimized_params
print("Best start:", ranked[0].start_id, "loss:", ranked[0].fun)

```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rank</th>
      <th>start_id</th>
      <th>seed</th>
      <th>objective_loss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>13</td>
      <td>0.068975</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0</td>
      <td>11</td>
      <td>0.068976</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>12</td>
      <td>0.068976</td>
    </tr>
  </tbody>
</table>
</div>


    Best start: 2 loss: 0.06897509540280802


## Outputs and interpretation

The table above is a **ranked multistart solution ensemble**. The first row is the best local solution among starts because it has the lowest scalar loss. We now convert fitted parameters and fitted curves into human-readable tables.


```python

alpha_values, beta_values = extract_parameters(P_initial, gene_kinase_counts, total_alpha, unique_kinases, K_index, best)
param_rows = []
for site, vals in alpha_values.items():
    for kinase, value in vals.items():
        param_rows.append({"parameter": "alpha", "substrate_site": str(site), "kinase": kinase, "value": value})
for (kinase, psite), value in beta_values.items():
    param_rows.append({"parameter": "beta", "substrate_site": psite, "kinase": kinase, "value": value})
params_df = pd.DataFrame(param_rows)
P_est, residuals, mse, rmse, mae, mape, r2 = compute_metrics(best, P_dense, t_max, gene_alpha_starts, gene_kinase_counts,
                                                            gene_kinase_idx, total_alpha, kinase_beta_starts,
                                                            kinase_beta_counts, K_data, K_indices, K_indptr)
fit_df = pd.DataFrame(P_est, index=[str(k) for k in P_initial.keys()], columns=time_cols)
metrics_df = pd.DataFrame([{"mse": mse, "rmse": rmse, "mae": mae, "mape": mape, "r_squared": r2}])
display(params_df)
display(metrics_df)
display(fit_df.iloc[:, :6])

```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>parameter</th>
      <th>substrate_site</th>
      <th>kinase</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>alpha</td>
      <td>('SUB1', 'S10')</td>
      <td>KIN_A</td>
      <td>0.684868</td>
    </tr>
    <tr>
      <th>1</th>
      <td>alpha</td>
      <td>('SUB1', 'S10')</td>
      <td>KIN_B</td>
      <td>0.315132</td>
    </tr>
    <tr>
      <th>2</th>
      <td>alpha</td>
      <td>('SUB2', 'T22')</td>
      <td>KIN_B</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>beta</td>
      <td>Y100</td>
      <td>KIN_A</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>beta</td>
      <td>S200</td>
      <td>KIN_B</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mse</th>
      <th>rmse</th>
      <th>mae</th>
      <th>mape</th>
      <th>r_squared</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.004927</td>
      <td>0.070191</td>
      <td>0.04517</td>
      <td>4.155826</td>
      <td>0.778152</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x1</th>
      <th>x2</th>
      <th>x3</th>
      <th>x4</th>
      <th>x5</th>
      <th>x6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>('SUB1', 'S10')</th>
      <td>1.021217</td>
      <td>1.065949</td>
      <td>1.107593</td>
      <td>1.143819</td>
      <td>1.173621</td>
      <td>1.197567</td>
    </tr>
    <tr>
      <th>('SUB2', 'T22')</th>
      <td>0.850000</td>
      <td>0.884177</td>
      <td>0.913659</td>
      <td>0.934900</td>
      <td>0.946372</td>
      <td>0.948939</td>
    </tr>
  </tbody>
</table>
</div>


## Visualization and saving

KinOpt includes a multistart summary plotting function; we use it for the ranked loss diagnostic. We also create compact fitted-vs-observed and parameter plots for notebook readability.


```python

output_dir = REPO_ROOT / "notebooks" / "outputs" / "kinopt"
output_dir.mkdir(parents=True, exist_ok=True)
params_df.to_csv(output_dir / "fitted_parameters.csv", index=False)
fit_df.to_csv(output_dir / "fitted_timeseries.csv")
metrics_df.to_csv(output_dir / "fit_metrics.csv", index=False)
export_outcomes_to_csv(ranked, output_dir / "multistart_summary.csv")
plot_multistart_summary_runtime_overlay(output_dir / "multistart_summary.csv", out_path=output_dir / "multistart_fun_vs_rank_runtime.png")

fig, ax = plt.subplots(figsize=(6, 3.5))
for i, label in enumerate(fit_df.index):
    ax.plot(t[:6], P_dense[i, :6], "o--", label=f"observed {label}")
    ax.plot(t[:6], P_est[i, :6], "-", label=f"fitted {label}")
ax.set_xlabel("time index")
ax.set_ylabel("phosphorylation signal")
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(output_dir / "fitted_vs_observed.png", dpi=200)
plt.show()

fig, ax = plt.subplots(figsize=(6, 3))
params_df.plot.bar(x="kinase", y="value", ax=ax, legend=False)
ax.set_ylabel("fitted weight")
fig.tight_layout()
fig.savefig(output_dir / "parameter_weights.png", dpi=200)
plt.show()
print("Saved files:", sorted(p.name for p in output_dir.iterdir()))

```

    Saved files: ['.gitkeep', 'fit_metrics.csv', 'fitted_parameters.csv', 'fitted_timeseries.csv', 'fitted_vs_observed.png', 'multistart_fun_vs_rank_runtime.png', 'multistart_summary.csv', 'parameter_weights.png']


## End-to-end summary

- **Inputs:** substrate phosphorylation table plus kinase–substrate interaction table.
- **Preprocessing:** KinOpt converted biological labels into dense/sparse arrays and parameter blocks.
- **Model:** kinase effects were combined with \(\beta\), then assigned to substrates with \(\alpha\).
- **Solving:** a fast local JAXopt projected-gradient loop generated a ranked multistart solution ensemble and the KinOpt objective scored each solution.
- **Outputs:** parameter CSVs, fitted time series, fit metrics, and plots were saved under `notebooks/outputs/kinopt/`.
- **How KinOpt differs:** TFOpt uses a similar alpha/beta decomposition for TF regulation, whereas `phoskintime.protwise` and `networkmodel` fit ODE dynamics with Diffrax-based ODE solving.
