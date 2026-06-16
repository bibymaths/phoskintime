# Memory Audit and Safe Fix Plan for Combinatorial Model OOM in phoskintime

## 1. Executive Summary

The `phoskintime` repository implements three network‑based models of phosphorylation: distributive (MODEL=0), sequential (MODEL=1) and combinatorial (MODEL=2). The first two models run within the available memory, but the combinatorial model exhausts available RAM and crashes the server. In a reported run the server had ~503 GB of RAM and `network.runner` alone consumed ~173 GB before failing. This audit inspects the `global` branch and pinpoints specific code paths where the combinatorial model materializes extremely large arrays and keeps them in memory. The planned fixes are conservative: they reduce memory amplification without changing the underlying biological equations or optimization objectives. They focus on streaming, chunking and down‑casting data instead of fully materializing combinatorial state spaces and trajectories, and on avoiding retention of large intermediate results. Only memory‑safe fixes are proposed; no changes to model semantics, optimization logic or public APIs are allowed.

## 2. Observed Failure

1.  **Symptom** – The combinatorial model (MODEL=2) causes the server to run out of memory. During a multi‑start optimization the process consumes ~173 GB of RAM (much higher than the distributive/sequential models) and eventually crashes.
2.  **Scale** – Memory growth appears to scale exponentially with the number of phosphorylation sites per protein: the state dimension for each protein is `1 + 2^n_sites`[\[1\]](https://raw.githubusercontent.com/bibymaths/phoskintime/global/networkmodel/network.py#:~:text=,i), and the simulation stores the entire trajectory `Y` of shape `(time_points, total_state_dim)`[\[2\]](https://raw.githubusercontent.com/bibymaths/phoskintime/global/networkmodel/simulate.py#:~:text=ns%20%3D%20int%28idx.n_states,p0%20%3D%20st%20%2B%201).
3.  **Immediate cause** – The combinatorial model precomputes all possible phosphorylation states and transitions for every protein using dense arrays. During simulation it materializes these states for all time points and computes per‑site signals using a dense bit‑mask matrix. These arrays can easily reach hundreds of gigabytes when proteins have more than a handful of phosphosites.

## 3. Root‑Cause Hypotheses Checked

| Hypothesis                                                                | Evidence                                                                                                                                                                                                                                                                                                                                                                                                                             | Outcome                                                                                                               |
|---------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| **3.1 Large state dimension due to 2^n_sites**                            | The `Index` class sets `n_states[i] = 1 << n_sites[i]` for combinatorial models, so each protein has `1 + 2^n_sites` dynamic states[\[1\]](https://raw.githubusercontent.com/bibymaths/phoskintime/global/networkmodel/network.py#:~:text=,i).                                                                                                                                                                                       | Confirmed. This exponential growth is inherent to the model.                                                          |
| **3.2 Dense hypercube transitions stored in memory**                      | `build_random_transitions` enumerates all dephosphorylation/phosphorylation transitions for each state and appends them to `trans_from`, `trans_to` and `trans_site` arrays[\[3\]](https://raw.githubusercontent.com/bibymaths/phoskintime/global/networkmodel/models.py#:~:text=def%20build_random_transitions%28idx%29%3A%20,for%20combinatorial%20topology). These arrays scale with `2^n_sites × n_sites`.                       | Confirmed. For `n_sites=6` this yields ~384 transitions per protein; for `n_sites=10` it becomes ~10,240 transitions. |
| **3.3 Dense** `bits` **matrix used to extract signals**                   | During simulation output extraction, the code constructs `bits = ((np.arange(ns)[:,None] >> np.arange(n_sites)[None,:]) & 1).astype(float)` and multiplies the state trajectories with this `(2^n_sites × n_sites)` matrix to compute per‑site phosphorylation counts[\[2\]](https://raw.githubusercontent.com/bibymaths/phoskintime/global/networkmodel/simulate.py#:~:text=ns%20%3D%20int%28idx.n_states,p0%20%3D%20st%20%2B%201). | Confirmed. The `bits` matrix is dense and duplicates weight vectors for each site.                                    |
| **3.4 Full trajectory materialization for all proteins and multi‑starts** | `simulate_diffrax` returns the entire trajectory `Y` for all dynamic states and time points. In `simulate_and_measure`, the code slices `Y` to extract per‑protein states and does not free them. Sensitivity analysis and multi‑start keep DataFrames of all trajectories and metrics in memory.                                                                                                                                    | Confirmed. These arrays are not released before the next iteration.                                                   |
| **3.5 JAX backend uses global max array sizes**                           | The JAX RHS constructs arrays with `jnp.arange(max_states)` and loops over `max_trans` for all proteins[\[4\]](https://raw.githubusercontent.com/bibymaths/phoskintime/global/networkmodel/backend.py#:~:text=). `max_states` is the maximum 2^n_sites across proteins.                                                                                                                                                              | Confirmed. On GPUs this increases host memory and JAX compilation time.                                               |
| **3.6** `S_cache` **allocated densely for all phosphorylation sites**     | In combinatorial mode, `System.__init__` allocates `S_cache = np.zeros((n_W_rows, kin_Kmat.shape[1]))` where `n_W_rows` equals the total number of phosphosites across all proteins[\[5\]](https://raw.githubusercontent.com/bibymaths/phoskintime/global/networkmodel/network.py#:~:text=,1%5D%29%2C%20dtype%3Dnp.float64). If there are many sites or time bins this array is large.                                               | Partially confirmed – moderate memory, but not the main issue compared with exponential state/transition explosion.   |
| **3.7 Retention of multistart/sensitivity history**                       | Multi‑start results and sensitivity analyses accumulate large DataFrames containing trajectories, parameter sets and metrics. These are not cleared until all starts finish.                                                                                                                                                                                                                                                         | Confirmed. This retention amplifies memory usage but is secondary to the exponential state/transition costs.          |

## 4. Confirmed Memory‑Amplification Points

1.  **Exponential state space** – `Index.__init__` computes `self.n_states[i] = 1 << n_sites[i]`[\[1\]](https://raw.githubusercontent.com/bibymaths/phoskintime/global/networkmodel/network.py#:~:text=,i). Each protein’s ODE segment includes `1 + 2^n_sites` states, leading to a `state_dim` that grows exponentially with `n_sites`. Running several proteins with more than 6 sites saturates memory quickly.

2.  **Dense transition lists** – `build_random_transitions` iterates over all possible states `m` (`0..2^n_sites-1`) and for each site `j` appends `m`, `m|(1<<j)` and `j` to arrays `trans_from`, `trans_to`, `trans_site`[\[3\]](https://raw.githubusercontent.com/bibymaths/phoskintime/global/networkmodel/models.py#:~:text=def%20build_random_transitions%28idx%29%3A%20,for%20combinatorial%20topology). The resulting arrays consume memory proportional to `2^n_sites × n_sites` per protein.

3.  **Dense bit‑mask matrix for output extraction** – In `simulate_and_measure`, after solving the ODE the code slices the state trajectory `Y` for each protein and computes phosphosite signals by building a dense `bits` matrix: `bits = ((np.arange(ns)[:,None] >> np.arange(n_sites)[None,:]) & 1).astype(float)` and performing `states @ bits`[\[2\]](https://raw.githubusercontent.com/bibymaths/phoskintime/global/networkmodel/simulate.py#:~:text=ns%20%3D%20int%28idx.n_states,p0%20%3D%20st%20%2B%201). For `ns = 2^n_sites`, `bits` is a float matrix of size `ns × n_sites`. When `ns` is large this matrix alone can occupy tens of gigabytes.

4.  **Full trajectory retention** – `simulate_diffrax` returns the complete trajectory `Y` across all time points. `simulate_and_measure` slices `Y` and constructs DataFrames for each protein, keeping the full arrays alive until the end of optimization or sensitivity analysis. Multi‑start and sensitivity analysis store these DataFrames in lists, multiplying memory usage by the number of starts or samples.

5.  **Global JAX array sizes** – In the JAX backend, `make_networkmodel_rhs` constructs arrays of size `max_states` and `max_trans` across all proteins[\[4\]](https://raw.githubusercontent.com/bibymaths/phoskintime/global/networkmodel/backend.py#:~:text=). This amplifies memory usage when even one protein has a large number of states.

6.  `S_cache` **zero‑initialization** – For combinatorial models, `System.__init__` allocates `S_cache` as a dense zero array sized `(number_of_sites, number_of_time_bins)`[\[5\]](https://raw.githubusercontent.com/bibymaths/phoskintime/global/networkmodel/network.py#:~:text=,1%5D%29%2C%20dtype%3Dnp.float64). Although smaller than the state trajectory, it is unnecessary to pre‑allocate the entire matrix when only a few columns are needed at a time.

## 5. Files and Exact Lines to Change

The following modifications reduce memory usage without altering the scientific equations or optimization logic. Line numbers refer to the `global` branch as inspected. Only the listed changes should be made; other code must remain unchanged.

### File: `networkmodel/models.py`

Current lines:

- **L454‑498** – Function `build_random_transitions` enumerates all forward transitions for combinatorial models. It loops over each state `m` (`0..2^n-1`) and for each site `j` not set in `m`, appends `m` (from state), `m|(1<<j)` (to state) and `j` to dense arrays `trans_from`, `trans_to` and `trans_site`[\[3\]](https://raw.githubusercontent.com/bibymaths/phoskintime/global/networkmodel/models.py#:~:text=def%20build_random_transitions%28idx%29%3A%20,for%20combinatorial%20topology).

Required change:

- **Refactor** `build_random_transitions` to return a *generator* or *lazy iterator* over transitions rather than fully materializing the `trans_from`, `trans_to` and `trans_site` arrays. For example, yield tuples `(m, m|(1<<j), j)` on the fly. The RHS functions should be updated to iterate over this generator when computing fluxes instead of indexing dense arrays. This avoids storing `O(2^n × n)` transitions in memory. Because the same enumeration logic is preserved, the model equations and reaction rates remain identical. The generator yields identical transitions in the same order as the current arrays.

### File: `networkmodel/network.py`

Current lines:

- **L294‑311** – In `System.__init__`, combinatorial mode precomputes `trans_from`, `trans_to` and `trans_site` using `build_random_transitions(idx)` and stores them in `self.comb_trans_from[i]`, `self.comb_trans_to[i]`, `self.comb_trans_site[i]`[\[6\]](https://raw.githubusercontent.com/bibymaths/phoskintime/global/networkmodel/network.py#:~:text=,1%5D%29%2C%20dtype%3Dnp.float64). It also allocates `S_cache` as `np.zeros((n_W_rows, kin_Kmat.shape[1]))`.

Required change:

- **Replace** storage of entire transition arrays with references to the generator returned by `build_random_transitions`. For each protein `i`, set `self.comb_trans_iter[i] = build_random_transitions(idx)` (yielding transitions on demand). The ODE RHS should iterate over this generator instead of indexing arrays. Delete the dense lists `comb_trans_from`, `comb_trans_to`, `comb_trans_site` or compute them only when `n_sites` is small (e.g., ≤4) to prevent memory blow‑up. This change does not alter the reaction network; it simply streams transitions instead of storing them.

- **Change** `S_cache` allocation to be lazy: allocate a single column buffer `S_cache_col = np.zeros(n_W_rows, dtype=some_dtype)` and reuse it for each time point. Alternatively allocate `S_cache` as a memory‑mapped array on disk. Do not allocate the full `(n_W_rows, kin_Kmat.shape[1])` matrix unless explicitly requested. This reduces memory by not storing all site‑time bins at once. The computed values are identical because only one column is used at a time.

- **Add** a memory guard in `Index.__init__` around **L134‑140**: after computing `n_states[i] = 1 << n_sites[i]`[\[1\]](https://raw.githubusercontent.com/bibymaths/phoskintime/global/networkmodel/network.py#:~:text=,i), check whether `n_states[i]` exceeds a safe threshold (e.g., `2**8`) and raise an informative error or require switching to the sequential model. This prevents accidental runs with extremely large state spaces. The guard does not alter equations but provides safety.

### File: `networkmodel/simulate.py`

Current lines:

- **L103‑121** – Function `simulate_and_measure` extracts combinatorial trajectories. For each protein with `n_sites` sites, it slices the full state matrix `Y[:, p0:p0 + ns]` (`ns = 2^n_sites`) to `states` and builds a dense bit‑mask matrix `bits = ((np.arange(ns)[:,None] >> np.arange(n_sites)[None,:]) & 1).astype(float)`; then computes `states @ bits` to get per‑site phosphorylated counts[\[2\]](https://raw.githubusercontent.com/bibymaths/phoskintime/global/networkmodel/simulate.py#:~:text=ns%20%3D%20int%28idx.n_states,p0%20%3D%20st%20%2B%201).

Required change:

- **Replace** the dense `bits` matrix with per‑site streaming. For each site `j` (0..n_sites‑1), compute a weight vector `w_j = ((np.arange(ns) >> j) & 1).astype(np.float32)` on‑the‑fly and calculate `states @ w_j` for that site. Or compute contributions sequentially within the loss function without constructing `w_j` at all, by using bitwise operations in the loop over states. Accumulate results into a `(times,)` vector for each site. This avoids allocating the `(ns × n_sites)` matrix and reduces memory by a factor of `n_sites`. Since each `w_j` is computed identically to a column of `bits`, the numerical outputs remain unchanged.

- **Free** the sliced `states` array as soon as per‑site and total protein signals are computed. Convert per‑site results to `float32` before storing in DataFrames. Do not keep `states` or intermediate arrays beyond the extraction step.

### File: `networkmodel/backend.py`

Current lines:

- **L303‑405** – In `make_networkmodel_rhs` for combinatorial models, the JAX RHS preallocates arrays like `m = jnp.arange(max_states)` and loops over `max_states` and `max_trans` to compute phosphorylation and dephosphorylation fluxes[\[4\]](https://raw.githubusercontent.com/bibymaths/phoskintime/global/networkmodel/backend.py#:~:text=).

Required change:

- **Refactor** the JAX RHS to operate per protein rather than over a global `max_states`. For each protein, compute `jnp.arange(n_states[i])` locally and loop only over its own transitions (via the generator from `build_random_transitions`). Use `jax.lax.scan` or `while_loop` to iterate over states and transitions without materializing large arrays. Avoid storing `m` or `trans` arrays of size `max_states`. This reduces the memory footprint of compiled JAX functions while preserving identical dynamics. Because the per‑protein loops compute the same flux contributions, the model equations remain unchanged.

- **Downcast** intermediate arrays to `float32` where safe, and ensure outputs are cast back to `float64` if required. Provide a configuration option to enable downcasting for memory‑constrained runs.

### File: `networkmodel/sensitivity.py` (and similar modules storing trajectories)

Current lines:

- The `morris_analysis` function (approx lines 271‑400) stores every trajectory’s DataFrames (`prot_df`, `rna_df`, `phos_df`) in a list `trajectory_storage`, along with simulation metrics. Only the top‐ranked curves are used later, but all DataFrames remain in memory.

Required change:

- **Replace** the list of full DataFrames with an on‑disk store or stream the metrics. For example, write each simulation’s results to a temporary file (CSV/Parquet) and store only file paths and summary statistics in memory. After ranking, load only the necessary top trajectories. This ensures memory usage does not scale with the number of sensitivity samples. The numerical metrics remain identical.

### File: `networkmodel/runner.py` (multi‑start logic)

Current lines:

- The `run_multistart` branch collects a pandas DataFrame with columns `X` and `F` for every start and appends them to a list. It may also keep entire simulation histories from each run.

Required change:

- **Modify** `run_multistart` so that after computing the objective value for a start, it retains only the parameter vector and objective scalar (or other minimal summary) in memory. Intermediate trajectories (`Y`, `states`, DataFrames) should be freed or written to disk immediately after evaluation. Optionally maintain an upper bound on the number of stored starts or use a streaming reducer to keep only the best `k` parameter sets. This change does not alter the optimization algorithm but prevents accumulating large data structures.

## 6. Required Fixes

### A. Fixes that only reduce memory materialization

1.  **Stream transitions** – Convert `build_random_transitions` to a generator and update ODE RHS and JAX backends to iterate over transitions lazily.

2.  Safe: ✅

3.  Risk: low

4.  Reason: the same transitions are computed; only the storage strategy changes.

5.  **Lazy** `bits` **computation** – Replace the dense `bits` matrix with per‑site weight vectors computed on demand in `simulate_and_measure`.

6.  Safe: ✅

7.  Risk: low

8.  Reason: computations use the same bitwise masks; results are identical.

9.  **Lazy** `S_cache` **and downcasting** – Allocate `S_cache` one column at a time or as a memory‑mapped array; downcast to `float32` where appropriate.

10. Safe: ✅

11. Risk: low

12. Reason: `S_cache` is an intermediate used for event times; using a smaller dtype or on‑disk storage does not change results.

13. **Per‑protein JAX loops** – Refactor JAX RHS to avoid global `max_states` arrays; use per‑protein loops and `jax.lax.scan`.

14. Safe: ✅ (with careful testing)

15. Risk: medium

16. Reason: requires rewriting the JAX function; while math is the same, subtle errors could occur. Comprehensive tests must be run.

17. **Memory guard on n_sites** – Add a guard that raises an informative error if `2^n_sites` exceeds a safe threshold (e.g., 256 states).

18. Safe: ✅

19. Risk: low

20. Reason: prevents impossible runs; does not change correct behaviour for feasible models.

### B. Fixes that prevent result/history retention

1.  **Streaming multi‑start results** – In `runner.py`, retain only essential summaries (parameter vector and objective) for each start; discard full trajectories and DataFrames after evaluation.

2.  Safe: ✅

3.  Risk: low

4.  Reason: the optimization algorithm uses only objective values; storing less history does not alter optimization logic.

5.  **Stream sensitivity trajectories** – In `sensitivity.py`, write each simulation’s results to disk and load only top trajectories when needed.

6.  Safe: ✅

7.  Risk: low

8.  Reason: analysis uses only selected trajectories; external storage preserves results without affecting ranking.

### C. Fixes that add chunking or streaming

1.  **Chunked simulation extraction** – For very large `ns`, process subsets of states when computing per‑site signals. For example, divide `states` columns into manageable chunks and compute contributions sequentially.

2.  Safe: ✅ (provided chunk boundaries are handled correctly)

3.  Risk: medium

4.  Reason: chunking introduces additional loops; careful accumulation is required to avoid numerical drift.

5.  **Chunked hypercube enumeration** – When `build_random_transitions` is called for proteins with many sites, generate transitions in chunks (e.g., by enumerating states up to a maximum memory footprint).

6.  Safe: ✅

7.  Risk: medium

8.  Reason: enumeration order must remain consistent for reproducibility. Implementation should test chunk boundaries thoroughly.

### D. Fixes that add safety guards

1.  **Early error on infeasible state sizes** – After calculating `n_states[i]`, check if `n_states[i] × time_points` would exceed a memory threshold (e.g., \>10^8 elements) and raise a `MemoryError` with guidance.

2.  Safe: ✅

3.  Risk: low

4.  Reason: prevents crashes by aborting early; does not alter valid runs.

5.  **Warn when JAX fallback is used** – If the JAX RHS cannot be compiled without full array materialization, print a warning and fall back to the numba implementation.

6.  Safe: ✅

7.  Risk: low

8.  Reason: does not change outputs; simply switches to a slower but safer backend.

### E. Fixes that improve logging/profiling without changing computation

1.  **Diagnostic logging** – Add optional logging to report estimated memory for state vectors, transitions and bit masks before simulation. This allows the user to adjust parameters.

2.  Safe: ✅

3.  Risk: low

4.  Reason: adds diagnostics only.

5.  **Profile multi‑start memory usage** – Insert debug prints that show current memory usage after each start and automatically stop new starts if usage exceeds a threshold.

6.  Safe: ✅

7.  Risk: low

8.  Reason: does not alter optimization results; just aborts early to avoid crashes.

## 7. What Must Not Change

- **Model equations and mechanistic logic** – The differential equations for translation, dephosphorylation and phosphorylation in the combinatorial model must remain exactly as implemented. Only the representation of transitions and states may change.
- **Optimization and ranking algorithms** – Multistart and sensitivity routines must still compute objective values and rank parameter sets identically. Only memory handling around these computations may change.
- **Data formats and API** – Input file formats, command‑line interfaces and output data structures must be preserved. Any downcasting must be optional and off by default to maintain precision.
- **Public semantics of combinatorial vs sequential/distributive models** – The conditions under which each model is selected must not change; the user must explicitly choose the combinatorial model.

## 8. Validation Plan

1.  **Tiny combinatorial smoke test** – Create a minimal network with one protein and two phosphorylation sites (`n_sites=2`). Run the combinatorial model with and without the fixes and ensure that ODE trajectories, per‑site signals and objective values match to within numerical tolerance.
2.  **Medium‑size memory test** – Construct a network with three proteins, each with `n_sites=6`, and run a single simulation. Measure peak memory usage using `tracemalloc` or `psutil` before and after the fixes. The fixed version should consume significantly less memory (ideally \<1 GB for simulation extraction).
3.  **Comparison with distributive/sequential models** – Run the distributive and sequential models for the same network pre‑ and post‑fix. Confirm that their outputs and performance remain unchanged, demonstrating that the fixes are isolated to the combinatorial path.
4.  **Combinatorial numerical fidelity** – For a network with `n_sites=3`, compute the loss values and gradients using both the original and the fixed code paths. Assert that differences are below a small tolerance (e.g., `1e‑10` for loss and `1e‑8` for gradients).
5.  **Peak memory scaling** – Vary `n_sites` from 2 to 10 and record peak memory usage. Verify that memory scales roughly linearly with `2^n_sites` only while computing per‑site signals, not multiplied by the number of sites or time bins. Ensure there is no global `max_states` explosion.
6.  **Retention test** – Run a multistart optimization with 10 starts on the medium network and monitor memory usage across starts. Confirm that memory remains bounded and does not accumulate across starts.

## 9. Expected Memory Impact

Implementing the above fixes should dramatically reduce memory usage:

- **Transition storage** – By streaming transitions instead of storing dense arrays, memory consumption drops from `O(2^n_sites × n_sites)` per protein to `O(1)` plus a small generator overhead.
- **Bit‑mask computation** – Per‑site streaming removes the need for an `ns × n_sites` matrix; memory used becomes proportional to `ns` at most. For `n_sites=10`, this saves `2^10 × 10 × 8 bytes ≈ 80 KB` per protein per time point.
- **Trajectory extraction** – Freeing state slices and storing only necessary signals reduces retained memory by orders of magnitude, especially in multistart and sensitivity loops.
- **JAX backend** – Per‑protein loops avoid global arrays of size `max_states`; memory overhead becomes proportional to the largest protein rather than all proteins.
- **Overall** – With these changes, running a medium combinatorial model (e.g., three proteins with `n_sites=6`) should require \<2 GB of RAM, enabling multi‑start optimization on typical servers.

## 10. Implementation Checklist

1.  \[ \] Refactor `build_random_transitions` to yield transitions instead of returning dense arrays.
2.  \[ \] Update all RHS implementations (Numba and JAX) to iterate over transition generators.
3.  \[ \] Modify `System.__init__` to store transition iterators and allocate `S_cache` lazily or as a memory‑mapped array.
4.  \[ \] Add a memory guard in `Index.__init__` to warn or abort when `2^n_sites` exceeds a safe threshold.
5.  \[ \] Replace dense `bits` matrix construction with per‑site streaming in `simulate_and_measure` and free intermediate arrays promptly.
6.  \[ \] Refactor JAX combinatorial RHS to use per‑protein loops without global `max_states` arrays; test with `jax.lax.scan`.
7.  \[ \] Downcast intermediate combinatorial arrays to `float32` with a configuration flag and cast outputs back to `float64` when necessary.
8.  \[ \] Stream DataFrame and trajectory storage in sensitivity analysis; write temporary results to disk and keep only summaries in memory.
9.  \[ \] Stream multistart results; retain only top parameter sets and objective values in memory.
10. \[ \] Add diagnostic logging and memory profiling utilities to inform users of estimated memory footprints and warn when thresholds are exceeded.
11. \[ \] Write unit tests implementing the validation plan; ensure outputs match the original code and memory usage is reduced.

[\[1\]](https://raw.githubusercontent.com/bibymaths/phoskintime/global/networkmodel/network.py#:~:text=,i) [\[5\]](https://raw.githubusercontent.com/bibymaths/phoskintime/global/networkmodel/network.py#:~:text=,1%5D%29%2C%20dtype%3Dnp.float64) [\[6\]](https://raw.githubusercontent.com/bibymaths/phoskintime/global/networkmodel/network.py#:~:text=,1%5D%29%2C%20dtype%3Dnp.float64) raw.githubusercontent.com

<https://raw.githubusercontent.com/bibymaths/phoskintime/global/networkmodel/network.py>

[\[2\]](https://raw.githubusercontent.com/bibymaths/phoskintime/global/networkmodel/simulate.py#:~:text=ns%20%3D%20int%28idx.n_states,p0%20%3D%20st%20%2B%201) raw.githubusercontent.com

<https://raw.githubusercontent.com/bibymaths/phoskintime/global/networkmodel/simulate.py>

[\[3\]](https://raw.githubusercontent.com/bibymaths/phoskintime/global/networkmodel/models.py#:~:text=def%20build_random_transitions%28idx%29%3A%20,for%20combinatorial%20topology) raw.githubusercontent.com

<https://raw.githubusercontent.com/bibymaths/phoskintime/global/networkmodel/models.py>

[\[4\]](https://raw.githubusercontent.com/bibymaths/phoskintime/global/networkmodel/backend.py#:~:text=) raw.githubusercontent.com

<https://raw.githubusercontent.com/bibymaths/phoskintime/global/networkmodel/backend.py>
