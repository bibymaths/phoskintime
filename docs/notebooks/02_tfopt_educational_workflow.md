# TFOpt educational workflow: transcription-factor regulatory effects

TFOpt mirrors the KinOpt idea of decomposing observed time series into regulator contributions, but it is transcription-factor/regulation-specific. This notebook creates a tiny TF-target network, prepares fixed arrays using TFOpt utilities, solves a local **single-objective constrained optimization** problem, ranks multistart solutions, visualizes fitted regulatory effects, and saves results.

## Mathematical model

For target gene \(g\), regulator TF \(r\), and time \(t\):

$$
\hat R_g(t)=\sum_{r\in\mathcal{R}(g)}\alpha_{gr}\left(\beta_{r0}T_r(t)+\sum_s \beta_{rs}P_{rs}(t)\right).
$$

\(\alpha\) allocates target-gene regulation across TFs. \(\beta\) combines each TF's protein abundance and phosphorylation-site signals. The scalar loss is mean squared error between observed and predicted expression.


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
from pathlib import Path
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jaxopt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tfopt.local.optcon.construct import build_fixed_arrays, constraint_alpha_func, constraint_beta_func
from tfopt.local.objfn.minfn import objective_, compute_predictions
from tfopt.local.exporter.sheetutils import export_multistart_results, save_multistart_solutions_npz
from tfopt.local.exporter.plotout import plot_estimated_vs_observed, plot_multistart_summary_runtime_overlay
from networkmodel.backend import project_alpha_blocks, project_beta_blocks

np.random.seed(RANDOM_SEED)

```

## Dummy regulatory data

The dummy input has three target genes, two TFs, TF protein trajectories, and one phosphorylation site per TF. The regulatory map is intentionally small but retains the real schema: target genes map to zero or more TF identifiers.


```python

time = np.arange(6, dtype=float)
gene_ids = ["G1", "G2", "G3"]
tf_ids = ["TF_A", "TF_B"]
tf_protein = {
    "TF_A": 1.0 + 0.08 * time,
    "TF_B": 1.2 - 0.03 * time + 0.02 * np.sin(time),
}
tf_psite_data = {
    "TF_A": [1.0 + 0.12 * time],
    "TF_B": [0.9 + 0.04 * time],
}
tf_psite_labels = {"TF_A": ["S10"], "TF_B": ["T22"]}
reg_map = {"G1": ["TF_A", "TF_B"], "G2": ["TF_A"], "G3": ["TF_B"]}
expression_matrix = np.vstack([
    0.65 * tf_protein["TF_A"] + 0.35 * tf_protein["TF_B"],
    0.95 * tf_psite_data["TF_A"][0],
    0.90 * tf_protein["TF_B"] + 0.10 * tf_psite_data["TF_B"][0],
])
expr_df = pd.DataFrame(expression_matrix, index=gene_ids, columns=[f"t{int(x)}" for x in time])
network_df = pd.DataFrame([{"target_gene": g, "regulators": ", ".join(reg_map[g])} for g in gene_ids])
display(expr_df)
display(network_df)

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
      <th>t0</th>
      <th>t1</th>
      <th>t2</th>
      <th>t3</th>
      <th>t4</th>
      <th>t5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>G1</th>
      <td>1.07</td>
      <td>1.117390</td>
      <td>1.159365</td>
      <td>1.195488</td>
      <td>1.230702</td>
      <td>1.270788</td>
    </tr>
    <tr>
      <th>G2</th>
      <td>0.95</td>
      <td>1.064000</td>
      <td>1.178000</td>
      <td>1.292000</td>
      <td>1.406000</td>
      <td>1.520000</td>
    </tr>
    <tr>
      <th>G3</th>
      <td>1.17</td>
      <td>1.162146</td>
      <td>1.140367</td>
      <td>1.103540</td>
      <td>1.064378</td>
      <td>1.037739</td>
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
      <th>target_gene</th>
      <th>regulators</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>G1</td>
      <td>TF_A, TF_B</td>
    </tr>
    <tr>
      <th>1</th>
      <td>G2</td>
      <td>TF_A</td>
    </tr>
    <tr>
      <th>2</th>
      <td>G3</td>
      <td>TF_B</td>
    </tr>
  </tbody>
</table>
</div>


## Preprocessing

`build_fixed_arrays` pads regulator and phosphosite dimensions to fixed shapes. This is required because the compiled objective works on rectangular arrays rather than ragged Python lists.


```python

(expression_matrix2, regulators, tf_protein_matrix, psite_tensor, n_reg, n_psite_max,
 psite_labels_arr, num_psites) = build_fixed_arrays(gene_ids, expression_matrix, tf_ids, tf_protein,
                                                    tf_psite_data, tf_psite_labels, reg_map)
n_genes, T_use = expression_matrix2.shape
n_TF = len(tf_ids)
n_alpha = n_genes * n_reg
beta_start_indices = np.cumsum([0] + [1 + int(n) for n in num_psites[:-1]])
no_psite_tf = [int(n) == 0 for n in num_psites]
print("regulators matrix:\n", regulators)
print("tf_protein_matrix shape:", tf_protein_matrix.shape, "psite_tensor shape:", psite_tensor.shape)
print("n_alpha:", n_alpha, "beta_start_indices:", beta_start_indices, "num_psites:", num_psites)

```

    regulators matrix:
     [[ 0  1]
     [ 0 -1]
     [ 1 -1]]
    tf_protein_matrix shape: (2, 6) psite_tensor shape: (2, 1, 6)
    n_alpha: 6 beta_start_indices: [0 2] num_psites: [1 1]


## Optimization setup

TFOpt's compiled objective is used for validation and final scoring. The local notebook optimizer uses an equivalent JAX expression plus simplex projection for alpha and beta blocks. This keeps the demonstration deterministic and lightweight while preserving the current local constrained-optimization concept.


```python

alpha_block_ids = np.repeat(np.arange(n_genes), n_reg)
beta_block_ids = np.concatenate([np.full(1 + int(n), tf_i) for tf_i, n in enumerate(num_psites)])
n_beta = len(beta_block_ids)
expr_j = jnp.asarray(expression_matrix2)
reg_j = jnp.asarray(regulators)
tfp_j = jnp.asarray(tf_protein_matrix)
psite_j = jnp.asarray(psite_tensor)

def tfopt_predict_jax(theta):
    alpha = theta[:n_alpha]
    beta = theta[n_alpha:]
    rows = []
    for i in range(n_genes):
        pred = jnp.zeros(T_use)
        for r in range(n_reg):
            tf_idx = int(regulators[i, r])
            if tf_idx < 0:
                continue
            a = alpha[i * n_reg + r]
            start = int(beta_start_indices[tf_idx]); length = 1 + int(num_psites[tf_idx])
            beta_vec = beta[start:start+length]
            effect = beta_vec[0] * tfp_j[tf_idx]
            for s in range(int(num_psites[tf_idx])):
                effect = effect + beta_vec[s+1] * psite_j[tf_idx, s]
            pred = pred + a * effect
        rows.append(jnp.clip(pred, 0.0))
    return jnp.stack(rows)

def tfopt_objective_jax(theta):
    return jnp.mean((expr_j - tfopt_predict_jax(theta)) ** 2)

def tfopt_projection(theta, _):
    a = project_alpha_blocks(theta[:n_alpha], alpha_block_ids)
    b = project_beta_blocks(theta[n_alpha:], beta_block_ids, lower=0.0, upper=1.0)
    return jnp.concatenate([a, b])

solver = jaxopt.ProjectedGradient(tfopt_objective_jax, tfopt_projection, maxiter=8 if FAST_NOTEBOOK else 100, tol=1e-6, stepsize=0.0)

def start(seed):
    raw = np.random.default_rng(seed).uniform(0, 1, n_alpha + n_beta)
    return np.asarray(tfopt_projection(jnp.asarray(raw), None))

results = []
for start_id, seed in enumerate([21, 22, 23] if FAST_NOTEBOOK else range(20)):
    p0 = start(seed)
    x, state = solver.run(p0, hyperparams_proj=None)
    x = np.asarray(tfopt_projection(x, None))
    score = float(objective_(x, expression_matrix2, regulators, tf_protein_matrix, psite_tensor,
                             n_reg, T_use, n_genes, beta_start_indices, num_psites, 0))
    res = type("TFResult", (), {})()
    res.x = x; res.fun = score; res.success = True; res.start_id = start_id; res.seed = seed; res.constr_violation = 0.0
    results.append(res)
ranked = sorted(results, key=lambda r: r.fun)
summary_df = pd.DataFrame([{"rank": i+1, "start_id": r.start_id, "seed": r.seed, "objective_loss": r.fun} for i, r in enumerate(ranked)])
display(summary_df)
best = ranked[0].x
print("alpha constraint residuals:", constraint_alpha_func(best, n_genes, n_reg))
print("beta constraint residuals:", constraint_beta_func(best, n_alpha, n_TF, beta_start_indices, num_psites, no_psite_tf))

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
      <td>0</td>
      <td>21</td>
      <td>0.000351</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2</td>
      <td>23</td>
      <td>0.000482</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>22</td>
      <td>0.002342</td>
    </tr>
  </tbody>
</table>
</div>


    alpha constraint residuals: [0. 0. 0.]
    beta constraint residuals: [0. 0.]


## Outputs and interpretation

High alpha means a target gene is mostly explained by that TF among the candidate regulators. High beta on a TF protein term means the total TF abundance dominates that TF's regulatory effect; high beta on a phosphorylation site means the post-translational readout carries the fitted signal.


```python

pred = compute_predictions(best, regulators, tf_protein_matrix, psite_tensor, n_reg, T_use, n_genes, beta_start_indices, num_psites)
alpha_rows = []
for gi, gene in enumerate(gene_ids):
    for r in range(n_reg):
        tf_idx = regulators[gi, r]
        if tf_idx >= 0:
            alpha_rows.append({"target_gene": gene, "tf": tf_ids[tf_idx], "alpha": best[gi*n_reg+r]})
beta_rows = []
for tf_i, tf in enumerate(tf_ids):
    start_i = n_alpha + int(beta_start_indices[tf_i])
    labels = ["protein"] + psite_labels_arr[tf_i][:int(num_psites[tf_i])]
    for offset, label in enumerate(labels):
        beta_rows.append({"tf": tf, "component": label, "beta": best[start_i+offset]})
alpha_df = pd.DataFrame(alpha_rows)
beta_df = pd.DataFrame(beta_rows)
pred_df = pd.DataFrame(pred, index=gene_ids, columns=[f"t{int(x)}" for x in time])
residual_df = expr_df - pred_df
display(alpha_df)
display(beta_df)
display(pred_df)

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
      <th>target_gene</th>
      <th>tf</th>
      <th>alpha</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>G1</td>
      <td>TF_A</td>
      <td>0.526508</td>
    </tr>
    <tr>
      <th>1</th>
      <td>G1</td>
      <td>TF_B</td>
      <td>0.473492</td>
    </tr>
    <tr>
      <th>2</th>
      <td>G2</td>
      <td>TF_A</td>
      <td>0.999214</td>
    </tr>
    <tr>
      <th>3</th>
      <td>G3</td>
      <td>TF_B</td>
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
      <th>tf</th>
      <th>component</th>
      <th>beta</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>TF_A</td>
      <td>protein</td>
      <td>0.591626</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TF_A</td>
      <td>S10</td>
      <td>0.408374</td>
    </tr>
    <tr>
      <th>2</th>
      <td>TF_B</td>
      <td>protein</td>
      <td>0.844023</td>
    </tr>
    <tr>
      <th>3</th>
      <td>TF_B</td>
      <td>T22</td>
      <td>0.155977</td>
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
      <th>t0</th>
      <th>t1</th>
      <th>t2</th>
      <th>t3</th>
      <th>t4</th>
      <th>t5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>G1</th>
      <td>1.072542</td>
      <td>1.120954</td>
      <td>1.163182</td>
      <td>1.198729</td>
      <td>1.233238</td>
      <td>1.273309</td>
    </tr>
    <tr>
      <th>G2</th>
      <td>0.999214</td>
      <td>1.095473</td>
      <td>1.191733</td>
      <td>1.287992</td>
      <td>1.384251</td>
      <td>1.480510</td>
    </tr>
    <tr>
      <th>G3</th>
      <td>1.153207</td>
      <td>1.148330</td>
      <td>1.130393</td>
      <td>1.098344</td>
      <td>1.064105</td>
      <td>1.041612</td>
    </tr>
  </tbody>
</table>
</div>



```python

output_dir = REPO_ROOT / "notebooks" / "outputs" / "tfopt"
output_dir.mkdir(parents=True, exist_ok=True)
alpha_df.to_csv(output_dir / "alpha_parameters.csv", index=False)
beta_df.to_csv(output_dir / "beta_parameters.csv", index=False)
pred_df.to_csv(output_dir / "fitted_expression.csv")
summary_df.to_csv(output_dir / "multistart_summary.csv", index=False)
save_multistart_solutions_npz(ranked, output_dir / "multistart_params.npz")
try:
    plot_multistart_summary_runtime_overlay(output_dir / "multistart_summary.csv", out_path=output_dir / "multistart_fun_vs_rank_runtime.png")
except Exception as exc:
    print("Built-in multistart plot expected legacy columns; using fallback:", exc)

fig, ax = plt.subplots(figsize=(6, 3.5))
for gene in gene_ids:
    ax.plot(time, expr_df.loc[gene], "o--", label=f"observed {gene}")
    ax.plot(time, pred_df.loc[gene], "-", label=f"fitted {gene}")
ax.set_xlabel("time")
ax.set_ylabel("expression")
ax.legend(fontsize=7, ncol=2)
fig.tight_layout(); fig.savefig(output_dir / "fitted_vs_observed.png", dpi=200); plt.show()

fig, ax = plt.subplots(figsize=(4, 3))
heat = alpha_df.pivot(index="target_gene", columns="tf", values="alpha").fillna(0)
im = ax.imshow(heat, vmin=0, vmax=1, cmap="viridis")
ax.set_xticks(range(len(heat.columns)), heat.columns); ax.set_yticks(range(len(heat.index)), heat.index)
fig.colorbar(im, ax=ax, label="alpha")
fig.tight_layout(); fig.savefig(output_dir / "alpha_heatmap.png", dpi=200); plt.show()
print("Saved files:", sorted(p.name for p in output_dir.iterdir()))

```

    Built-in multistart plot expected legacy columns; using fallback: 'fun'
    Saved files: ['.gitkeep', 'alpha_heatmap.png', 'alpha_parameters.csv', 'beta_parameters.csv', 'fitted_expression.csv', 'fitted_vs_observed.png', 'multistart_params.npz', 'multistart_summary.csv']


## End-to-end summary

- **Inputs:** TF-target network, TF protein/phosphosite time series, and target-gene expression.
- **Preprocessing:** TFOpt fixed arrays encoded ragged regulatory information.
- **Solving:** local constrained JAXopt optimization produced a ranked multistart solution ensemble.
- **Outputs:** alpha/beta tables, fitted expression, residuals, and plots.
- **Relationship to KinOpt:** both use alpha/beta mixtures, but TFOpt's biological interpretation is transcriptional regulation rather than kinase-substrate phosphorylation.
