# Metrics and Distance Measures

This page documents the metrics used in PhosKinTime to evaluate model fit quality.

---

## Fréchet Distance (`frechet/distance.py`)

### What it is

The **discrete Fréchet distance** is a measure of similarity between two ordered sequences of
points (curves). Informally, it is the minimum "leash length" required for a person walking along
one curve and a dog walking along the other — both moving forward only, never backward — to stay
connected.

For two curves $P = (p_0, p_1, \dots, p_n)$ and $Q = (q_0, q_1, \dots, q_m)$, the discrete
Fréchet distance is:

$$
d_F(P, Q) = \min_{\alpha, \beta} \max_{i} \, d(p_{\alpha(i)}, q_{\beta(i)})
$$

where $\alpha$ and $\beta$ are monotone traversals of the two curves.

### Implementation

`frechet/distance.py` provides a Numba JIT-compiled function:

```python
from frechet import frechet_distance
score = frechet_distance(true_coords, pred_coords)  # returns float
```

- Inputs: 2D arrays of shape `(T, D)` where `T` is the number of time points and `D` is the
  number of dimensions (observables per time point).
- Returns: a single `float64` scalar — the discrete Fréchet distance.
- Uses `@njit(parallel=True)` with Numba for JIT compilation and parallelized pairwise distance
  computation.

The dynamic programming recurrence:

```
cost[0, 0] = dist[0, 0]
cost[i, 0] = max(cost[i-1, 0], dist[i, 0])
cost[0, j] = max(cost[0, j-1], dist[0, j])
cost[i, j] = max(min(cost[i-1,j], cost[i,j-1], cost[i-1,j-1]), dist[i,j])
```

The final result is `cost[-1, -1]`.

### Where it is used

- **`global_model/runner.py`**: After optimization, Fréchet scores are computed between observed
  and predicted time-series trajectories for each gene. These scores are stored in the dashboard
  bundle (`frechet_scores` field) and displayed in the Streamlit dashboard.
- **`scripts/curve_similarity.py`**: Stand-alone script that computes per-row Fréchet distances
  between observed and estimated columns in `tfopt_results.xlsx` / `kinopt_results.xlsx`.

### Fréchet vs RMSE

| Property | RMSE | Discrete Fréchet |
|---|---|---|
| Sensitive to point-to-point order | No | Yes |
| Penalizes shape mismatch (timing shift) | Poorly | Well |
| Scale-dependent | Yes | Yes |
| Normalized | No (raw units) | No (raw units) |
| Interpretable threshold | No | No |
| Computational cost | O(T) | O(T²) |

Use Fréchet distance when **trajectory shape and timing** matter (e.g., detecting phase lags or
peak shifts). Use RMSE when you only care about average point-wise deviation.

### Normalization and Thresholds

The Fréchet distance values in PhosKinTime are **not normalized**. They are in the same units
as the input data. Use them for **ranking** fits (lower = more similar trajectory), not as
absolute quality judgments. Establish a baseline distribution (e.g., median across all proteins)
to contextualize individual scores.

### Performance

- Numba JIT compilation occurs on first call. Subsequent calls reuse the compiled binary.
- The `parallel=True` flag enables Numba's parallel loop (`prange`) for the pairwise distance
  computation.
- For large `T` (many time points), the O(T²) dynamic programming step dominates.
- Typical phosphoproteomics time series have T ≤ 15 time points, making the function fast.

---

## Loss Functions (Optimization)

Both local and global optimizers support configurable loss functions. See the
[Configuration Reference](configuration.md) for the integer codes.

| Code | Name | Use case |
|---|---|---|
| 0 | MSE | Standard squared error |
| 1 | MAE / Huber | Robust to outliers |
| 2 | Soft L1 / Pseudo-Huber | Smooth transition MSE↔MAE |
| 3 | Cauchy / Log-Cosh | Heavy-tail robust |
| 4 | Arctan | Bounded outlier penalty |
| 5 | Elastic Net | Sparsity + smoothness (default tfopt) |
| 6 | Tikhonov | L2 regularization |

---

## Sensitivity Metrics (`global_model/sensitivity.py`)

After global optimization, trajectory-based sensitivity is computed. The sensitivity metric
aggregates the model output into a scalar before computing Morris elementary effects:

| Metric | Description |
|---|---|
| `total_signal` | Sum of all state values at all time points |
| `mean` | Mean state value across time |
| `variance` | Variance of state values across time |
| `l2_norm` | L2 norm of the trajectory vector |

Set via `sensitivity_metric` in `config.toml` under `[global_model]`.
