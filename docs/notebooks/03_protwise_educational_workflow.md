# `phoskintime.protwise` educational workflow: protein-wise ODE fitting

This notebook demonstrates protein-wise / phosphorylation-site-wise modeling with tiny synthetic mRNA, protein, and phosphosite time-series data. It uses the current Diffrax/JAXopt implementation: Diffrax-based ODE solving supplies fitted trajectories, and JAXopt projected-gradient performs local single-objective constrained optimization.

The goal is not to infer biology from dummy data; it is to show the full computational workflow: data generation → mode-aware preprocessing → ODE setup → objective construction → local optimization → ranked multistart solution ensemble → visualization → saving → interpretation.

## Model and data modalities

For one protein with one phosphorylation site, the distributive local model has states approximately corresponding to mRNA \(R\), protein \(P\), and phosphosite \(P_1\). Parameters such as synthesis, degradation, translation, phosphorylation, and dephosphorylation rates determine:

$$
\frac{d y}{dt} = f(y(t), \theta).
$$

The objective compares simulated observables against whichever modalities are present:

$$
L(\theta)=L_{mRNA}+L_{protein}+L_{phospho}
$$

with unavailable modalities skipped. This matters scientifically because real experiments often measure only subsets of mRNA, protein, and phosphosite abundance.


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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import phoskintime.protwise as protwise_pkg
from protwise.models.diffrax_solver import solve_protwise_ode
from protwise.paramest.normest import _normalize_bounds, protwise_objective
from protwise.paramest.inference import InferenceContext, run_multistart
from protwise.plotting.plotting import Plotter
from networkmodel.backend import DataMode, optimize_scalar_objective
from config.constants import get_param_names

np.random.seed(RANDOM_SEED)
print("Imported phoskintime.protwise from:", protwise_pkg.__file__)

```

    Imported phoskintime.protwise from: /home/abhinav/Documents/phoskintime/protwise/__init__.py


## Generate small synthetic multimodal data

The notebook simulates a single protein with one phosphosite using known parameters, then treats the simulated values as observations. Tables are kept long/tidy because that is the easiest way to validate modality availability and adapt to real data.


```python

time_points = np.asarray([0.0, 0.5, 1.0, 2.0, 4.0, 8.0], dtype=float)
num_psites = 1
init_cond = np.asarray([1.0, 1.0, 1.0], dtype=float)
true_params = np.asarray([0.30, 0.20, 0.35, 0.10, 0.18, 0.12], dtype=float)
sol_true, flat_true = solve_protwise_ode(true_params, init_cond, num_psites, time_points, model_name="distmod")
sol_true = np.asarray(sol_true)
mrna_obs = sol_true[:, 0] * (1.0 + 0.01 * np.sin(time_points))
protein_obs = sol_true[:, 1] * (1.0 - 0.01 * np.cos(time_points))
phospho_obs = sol_true[:, 2]

mrna_df = pd.DataFrame({"protein": "PROT_A", "time": time_points, "fc": mrna_obs})
protein_df = pd.DataFrame({"protein": "PROT_A", "time": time_points, "fc": protein_obs})
phospho_df = pd.DataFrame({"protein": "PROT_A", "psite": "S10", "time": time_points, "fc": phospho_obs})
display(mrna_df)
display(protein_df)
display(phospho_df)

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
      <th>protein</th>
      <th>time</th>
      <th>fc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PROT_A</td>
      <td>0.0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>PROT_A</td>
      <td>0.5</td>
      <td>1.052604</td>
    </tr>
    <tr>
      <th>2</th>
      <td>PROT_A</td>
      <td>1.0</td>
      <td>1.099812</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PROT_A</td>
      <td>2.0</td>
      <td>1.175432</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PROT_A</td>
      <td>4.0</td>
      <td>1.265684</td>
    </tr>
    <tr>
      <th>5</th>
      <td>PROT_A</td>
      <td>8.0</td>
      <td>1.412894</td>
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
      <th>protein</th>
      <th>time</th>
      <th>fc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PROT_A</td>
      <td>0.0</td>
      <td>0.990000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>PROT_A</td>
      <td>0.5</td>
      <td>1.400140</td>
    </tr>
    <tr>
      <th>2</th>
      <td>PROT_A</td>
      <td>1.0</td>
      <td>1.654010</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PROT_A</td>
      <td>2.0</td>
      <td>1.960548</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PROT_A</td>
      <td>4.0</td>
      <td>2.317908</td>
    </tr>
    <tr>
      <th>5</th>
      <td>PROT_A</td>
      <td>8.0</td>
      <td>2.873488</td>
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
      <th>protein</th>
      <th>psite</th>
      <th>time</th>
      <th>fc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PROT_A</td>
      <td>S10</td>
      <td>0.0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>PROT_A</td>
      <td>S10</td>
      <td>0.5</td>
      <td>0.656900</td>
    </tr>
    <tr>
      <th>2</th>
      <td>PROT_A</td>
      <td>S10</td>
      <td>1.0</td>
      <td>0.482645</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PROT_A</td>
      <td>S10</td>
      <td>2.0</td>
      <td>0.357585</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PROT_A</td>
      <td>S10</td>
      <td>4.0</td>
      <td>0.353467</td>
    </tr>
    <tr>
      <th>5</th>
      <td>PROT_A</td>
      <td>S10</td>
      <td>8.0</td>
      <td>0.441930</td>
    </tr>
  </tbody>
</table>
</div>


## Mode-aware input handling

The same objective code can fit all modalities, only phosphosite data, only protein baseline/time series, or paired subsets. The table below makes modality choices explicit before optimization.


```python

def mode_from_arrays(mrna, protein, phospho):
    return {
        "fit_mrna": len(mrna) > 0,
        "fit_protein": len(protein) > 0,
        "fit_phospho": len(phospho) > 0,
        "n_rna": len(mrna),
        "scale_mrna": max(float(np.var(mrna)), 1.0),
        "scale_protein": max(float(np.var(protein)), 1.0),
        "scale_phospho": max(float(np.var(phospho)), 1.0),
    }

modalities = {
    "all_modalities": (mrna_obs, protein_obs, phospho_obs.reshape(-1)),
    "phospho_only": (np.asarray([]), np.asarray([]), phospho_obs.reshape(-1)),
    "protein_only": (np.asarray([]), protein_obs, np.asarray([])),
    "mrna_plus_phospho": (mrna_obs, np.asarray([]), phospho_obs.reshape(-1)),
}
mode_table = []
for name, (m, p, ph) in modalities.items():
    mode = mode_from_arrays(m, p, ph)
    mode_table.append({"case": name, "fit_mrna": mode["fit_mrna"], "fit_protein": mode["fit_protein"],
                       "fit_phospho": mode["fit_phospho"], "n_observations": len(m)+len(p)+len(ph)})
display(pd.DataFrame(mode_table))

```

    /home/abhinav/Documents/phoskintime/.pixi/envs/full/lib/python3.1/site-packages/numpy/_core/fromnumeric.py:4232: RuntimeWarning: Degrees of freedom <= 0 for slice
      return _methods._var(a, axis=axis, dtype=dtype, out=out, ddof=ddof,
    /home/abhinav/Documents/phoskintime/.pixi/envs/full/lib/python3.1/site-packages/numpy/_core/_methods.py:178: RuntimeWarning: invalid value encountered in divide
      arrmean = um.true_divide(arrmean, div, out=arrmean,
    /home/abhinav/Documents/phoskintime/.pixi/envs/full/lib/python3.1/site-packages/numpy/_core/_methods.py:211: RuntimeWarning: invalid value encountered in scalar divide
      ret = ret.dtype.type(ret / rcount)



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
      <th>case</th>
      <th>fit_mrna</th>
      <th>fit_protein</th>
      <th>fit_phospho</th>
      <th>n_observations</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>all_modalities</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>18</td>
    </tr>
    <tr>
      <th>1</th>
      <td>phospho_only</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>protein_only</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>mrna_plus_phospho</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>12</td>
    </tr>
  </tbody>
</table>
</div>


## Objective, bounds, and local optimization

The parameter vector uses the current `distmod` names from package configuration. Bounds are physical rate bounds. JAXopt projection enforces box constraints; the loss is a scalar sum of active modality errors. We use low iterations for CI but the cells are real optimization calls.


```python

bounds = {"A": (0.01, 1.0), "B": (0.01, 1.0), "C": (0.01, 1.0), "D": (0.01, 1.0), "S(i)": (0.01, 1.0), "D(i)": (0.01, 1.0)}
lower, upper = _normalize_bounds(bounds, "distmod", num_psites)
param_names = get_param_names(num_psites, "distmod")
theta0 = 0.5 * (lower + upper)

m, p, ph = modalities["all_modalities"]
mode = mode_from_arrays(m, p, ph)
target = {"mrna": jnp.asarray(m), "protein": jnp.asarray(p), "phospho": jnp.asarray(ph)}
objective = lambda x: protwise_objective(x, target, init_cond, num_psites, time_points, mode, "distmod")
initial_loss = float(objective(theta0))
best_params, opt_state, best_loss = optimize_scalar_objective(objective, theta0, lower, upper, maxiter=3 if FAST_NOTEBOOK else 100, tol=1e-6)
print("Initial loss:", initial_loss, "Best loss:", best_loss, "iterations:", getattr(opt_state, "iter_num", "unknown"))
param_df = pd.DataFrame({"parameter": param_names, "estimate": best_params, "lower": lower, "upper": upper})
display(param_df)

```

    [92m2026-06-09 11:02:32,960[0m - [93mnetworkmodel.backend[0m - [94mINFO[0m - [94m[Optimizer] Selected optimizer backend: jaxopt.ProjectedGradient[0m                                                                                [96m⏱ 15.80 sec[0m
    [92m2026-06-09 11:02:32,962[0m - [93mnetworkmodel.backend[0m - [94mINFO[0m - [94m[Constraints] Kinetic parameters use bound projection; fixed parameters are restored after each projection.[0m                                     [96m⏱ 15.80 sec[0m


    INFO: jaxopt.ProximalGradient: Iter: 1 Distance btw Iterates (stopping criterion): 2.1645195781340205 Stepsize:0.125 
    INFO: jaxopt.ProximalGradient: Iter: 2 Distance btw Iterates (stopping criterion): 1.3956946021590573 Stepsize:0.0625 
    INFO: jaxopt.ProximalGradient: Iter: 3 Distance btw Iterates (stopping criterion): 1.0936834904235042 Stepsize:0.03125 


    [92m2026-06-09 11:03:20,860[0m - [93mnetworkmodel.backend[0m - [94mINFO[0m - [94m[Optimizer] Convergence status: iterations=3 final_objective=0.10547857 grad_norm=1.1515[0m                                                        [96m⏱ 1.06 min[0m


    Initial loss: 1.3887147997393063 Best loss: 0.10547857135478621 iterations: 3



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
      <th>estimate</th>
      <th>lower</th>
      <th>upper</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A</td>
      <td>0.649784</td>
      <td>0.01</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>B</td>
      <td>0.358182</td>
      <td>0.01</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>C</td>
      <td>0.707908</td>
      <td>0.01</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>D</td>
      <td>0.309640</td>
      <td>0.01</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>S1</td>
      <td>0.415991</td>
      <td>0.01</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>D1</td>
      <td>0.462584</td>
      <td>0.01</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>


## Ranked multistart solution ensemble

A single local run can depend on initialization. The shared inference utility repeats the local optimization from deterministic starts and writes summary tables/plots. The best row is the lowest scalar objective among successful starts.


```python

output_dir = REPO_ROOT / "notebooks" / "outputs" / "protwise"
output_dir.mkdir(parents=True, exist_ok=True)
data_mode = DataMode(tuple(k for k, flag in (("mrna", mode["fit_mrna"]), ("protein", mode["fit_protein"]), ("phospho", mode["fit_phospho"])) if flag),
                     "+".join(k for k, flag in (("mrna", mode["fit_mrna"]), ("protein", mode["fit_protein"]), ("phospho", mode["fit_phospho"])) if flag),
                     mode["fit_mrna"], mode["fit_protein"], mode["fit_phospho"])
ctx = InferenceContext(objective, theta0, lower, upper, data_mode, output_dir, parameter_names=tuple(param_names), maxiter=2 if FAST_NOTEBOOK else 50)
ensemble = run_multistart(ctx, n_starts=2 if FAST_NOTEBOOK else 8, seed=RANDOM_SEED, max_workers=1)
display(ensemble["summary"].sort_values("final_objective"))
display(ensemble["best"])

```

    [92m2026-06-09 11:03:20,958[0m - [93mnetworkmodel.BayesianInference[0m - [94mINFO[0m - [94m[Inference] JAX/XLA multistart strategy: {'requested_workers': 1, 'effective_workers': 1, 'xla_flags': '--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1', 'omp_num_threads': '1'}[0m[96m⏱ 1.06 min[0m
    [92m2026-06-09 11:03:20,963[0m - [93mnetworkmodel.BayesianInference[0m - [94mINFO[0m - [94m[Optimizer] Selected optimizer backend: jaxopt.ProjectedGradient[0m                                                                      [96m⏱ 1.06 min[0m
    [92m2026-06-09 11:03:20,964[0m - [93mnetworkmodel.BayesianInference[0m - [94mINFO[0m - [94m[Constraints] Kinetic parameters use bound projection; fixed parameters are restored after each projection.[0m                           [96m⏱ 1.06 min[0m


    INFO: jaxopt.ProximalGradient: Iter: 1 Distance btw Iterates (stopping criterion): 2.1645195781340205 Stepsize:0.125 
    INFO: jaxopt.ProximalGradient: Iter: 2 Distance btw Iterates (stopping criterion): 1.3956946021590573 Stepsize:0.0625 


    [92m2026-06-09 11:04:07,382[0m - [93mnetworkmodel.BayesianInference[0m - [94mINFO[0m - [94m[Optimizer] Convergence status: iterations=2 final_objective=0.14939338 grad_norm=1.4473689[0m                                           [96m⏱ 1.83 min[0m
    [92m2026-06-09 11:04:07,390[0m - [93mnetworkmodel.BayesianInference[0m - [94mINFO[0m - [94m[Optimizer] Selected optimizer backend: jaxopt.ProjectedGradient[0m                                                                      [96m⏱ 1.83 min[0m
    [92m2026-06-09 11:04:07,391[0m - [93mnetworkmodel.BayesianInference[0m - [94mINFO[0m - [94m[Constraints] Kinetic parameters use bound projection; fixed parameters are restored after each projection.[0m                           [96m⏱ 1.84 min[0m


    INFO: jaxopt.ProximalGradient: Iter: 1 Distance btw Iterates (stopping criterion): 2.1922686866811816 Stepsize:0.0625 
    INFO: jaxopt.ProximalGradient: Iter: 2 Distance btw Iterates (stopping criterion): 0.9906125504026 Stepsize:0.03125 


    [92m2026-06-09 11:04:56,627[0m - [93mnetworkmodel.BayesianInference[0m - [94mINFO[0m - [94m[Optimizer] Convergence status: iterations=2 final_objective=0.12372695 grad_norm=0.85335853[0m                                          [96m⏱ 2.66 min[0m



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
      <th>start_id</th>
      <th>seed</th>
      <th>convergence_status</th>
      <th>success</th>
      <th>failure_reason</th>
      <th>final_objective</th>
      <th>active_loss_terms</th>
      <th>number_of_iterations</th>
      <th>solver_status</th>
      <th>optimizer_status</th>
      <th>runtime_seconds</th>
      <th>data_mode</th>
      <th>available_layers</th>
      <th>selected_best</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>8</td>
      <td>converged</td>
      <td>True</td>
      <td></td>
      <td>0.123727</td>
      <td>mrna_loss,protein_loss,phospho_loss</td>
      <td>2</td>
      <td>diffrax</td>
      <td>jaxopt.ProjectedGradient</td>
      <td>49.242441</td>
      <td>mrna+protein+phospho</td>
      <td>mrna,protein,phospho</td>
      <td>True</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>7</td>
      <td>converged</td>
      <td>True</td>
      <td></td>
      <td>0.149393</td>
      <td>mrna_loss,protein_loss,phospho_loss</td>
      <td>2</td>
      <td>diffrax</td>
      <td>jaxopt.ProjectedGradient</td>
      <td>46.426562</td>
      <td>mrna+protein+phospho</td>
      <td>mrna,protein,phospho</td>
      <td>False</td>
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
      <th>start_id</th>
      <th>seed</th>
      <th>success</th>
      <th>selected_best</th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>S1</th>
      <th>D1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>8</td>
      <td>True</td>
      <td>True</td>
      <td>0.712151</td>
      <td>0.834275</td>
      <td>0.829174</td>
      <td>0.123167</td>
      <td>0.259773</td>
      <td>0.862699</td>
    </tr>
  </tbody>
</table>
</div>


## Fitted trajectories and residuals

After fitting, Diffrax solves the ODE one more time with the best parameter vector. Fitted-vs-observed curves show systematic mismatch; residuals centered near zero suggest the model can reproduce the dummy data under the selected modality weights.


```python

sol_fit, flat_fit = solve_protwise_ode(best_params, init_cond, num_psites, time_points, model_name="distmod")
sol_fit = np.asarray(sol_fit)
fit_long = pd.DataFrame({
    "time": np.tile(time_points, 3),
    "state": np.repeat(["mRNA", "protein", "phospho_S10"], len(time_points)),
    "fitted": np.concatenate([sol_fit[:,0], sol_fit[:,1], sol_fit[:,2]]),
    "observed": np.concatenate([mrna_obs, protein_obs, phospho_obs]),
})
fit_long["residual"] = fit_long["fitted"] - fit_long["observed"]
display(fit_long.head(12))

fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
for state, sub in fit_long.groupby("state"):
    axes[0].plot(sub["time"], sub["observed"], "o--", label=f"obs {state}")
    axes[0].plot(sub["time"], sub["fitted"], "-", label=f"fit {state}")
    axes[1].plot(sub["time"], sub["residual"], "o-", label=state)
axes[0].set_title("Fitted vs observed"); axes[0].set_xlabel("time"); axes[0].set_ylabel("state value")
axes[1].axhline(0, color="black", lw=0.8); axes[1].set_title("Residuals"); axes[1].set_xlabel("time")
for ax in axes: ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(output_dir / "fitted_vs_observed_and_residuals.png", dpi=200); plt.show()

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
      <th>time</th>
      <th>state</th>
      <th>fitted</th>
      <th>observed</th>
      <th>residual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>mRNA</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.5</td>
      <td>mRNA</td>
      <td>1.133491</td>
      <td>1.052604</td>
      <td>0.080887</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>mRNA</td>
      <td>1.245094</td>
      <td>1.099812</td>
      <td>0.145282</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.0</td>
      <td>mRNA</td>
      <td>1.416401</td>
      <td>1.175432</td>
      <td>0.240969</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4.0</td>
      <td>mRNA</td>
      <td>1.619824</td>
      <td>1.265684</td>
      <td>0.354140</td>
    </tr>
    <tr>
      <th>5</th>
      <td>8.0</td>
      <td>mRNA</td>
      <td>1.767752</td>
      <td>1.412894</td>
      <td>0.354858</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.0</td>
      <td>protein</td>
      <td>1.000000</td>
      <td>0.990000</td>
      <td>0.010000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.5</td>
      <td>protein</td>
      <td>1.347053</td>
      <td>1.400140</td>
      <td>-0.053087</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1.0</td>
      <td>protein</td>
      <td>1.535865</td>
      <td>1.654010</td>
      <td>-0.118145</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2.0</td>
      <td>protein</td>
      <td>1.775419</td>
      <td>1.960548</td>
      <td>-0.185128</td>
    </tr>
    <tr>
      <th>10</th>
      <td>4.0</td>
      <td>protein</td>
      <td>2.147482</td>
      <td>2.317908</td>
      <td>-0.170426</td>
    </tr>
    <tr>
      <th>11</th>
      <td>8.0</td>
      <td>protein</td>
      <td>2.624122</td>
      <td>2.873488</td>
      <td>-0.249366</td>
    </tr>
  </tbody>
</table>
</div>


## Built-in visualization and exports

The protwise plotting module includes `Plotter.plot_parallel`, which is useful for viewing state trajectories across time. We save CSVs, a JSON summary, and plots under `notebooks/outputs/protwise/`.


```python

param_df.to_csv(output_dir / "fitted_parameters.csv", index=False)
fit_long.to_csv(output_dir / "fitted_timeseries_long.csv", index=False)
summary = {"model": "distmod", "data_mode": data_mode.data_mode, "best_loss": float(best_loss), "parameter_names": param_names}
(output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
plotter = Plotter("PROT_A", out_dir=str(output_dir))
plotter.plot_parallel(sol_fit, ["R", "P", "P_S10"])
print("Saved files:", sorted(p.name for p in output_dir.iterdir()))

```

    Saved files: ['.gitkeep', 'PROT_A_parallel_coordinates_.png', 'fitted_parameters.csv', 'fitted_timeseries_long.csv', 'fitted_vs_observed_and_residuals.png', 'optimization', 'plots', 'summary.json']


## End-to-end summary

- **Inputs:** synthetic mRNA, protein, and phosphosite time series for one protein.
- **Modes:** all modalities, phospho-only, protein-only, and paired modality logic were made explicit.
- **Model:** a protein-wise ODE model was solved with Diffrax-based ODE solving.
- **Optimization:** JAXopt local optimization and a ranked multistart solution ensemble estimated rate parameters.
- **Outputs:** parameter tables, fitted trajectories, residual plots, built-in parallel-coordinate plot, and JSON summary.
- **Adapting to real data:** replace the tidy data frames with measured values, set biologically defensible bounds, increase multistarts/iterations, and inspect residuals and parameter identifiability.
