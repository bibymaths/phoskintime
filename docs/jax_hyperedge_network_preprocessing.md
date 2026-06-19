# Pure-JAX Triplet-Level Network Preprocessing Design

## 1. Purpose and non-goals

This document proposes an **independent, optional, pure-JAX preprocessing module** that discovers, scores, prunes, annotates, and exports kinase-site-substrate triplets before PhosKinTime model fitting. The module is intended to run between network/data loading and ODE/JAXopt fitting. It must not change existing default workflows: if users do not explicitly enable it, current `networkmodel`, `protwise`, `kinopt`, `tfopt`, dashboard, scripts, notebooks, and tests continue to behave as they do now.

The computational core must use `jax` and `jax.numpy`, enable and respect float64 precision, use `jax.jit`/`vmap`/batching where profitable, and keep NumPy/Pandas/Matplotlib/Seaborn/Plotly at I/O, reporting, plotting, and export boundaries only. The module must avoid SciPy, NetworkX-heavy runtime graph logic, pandas-in-the-core-loop, NumPy-in-the-core-loop, Numba, pymoo, evolutionary algorithms, and non-JAX optimization backends.

Non-goals for the first implementation:

- Do not implement new ODE kinetics.
- Do not replace `networkmodel` fitting or `protwise` fitting.
- Do not delete or shrink current loaders, output writers, plotting utilities, dashboard parsers, scripts, or tests.
- Do not require dense `O(n_kinases * n_sites * n_substrates)` tensor materialization.

## 2. Repository scan summary

A full repository scan was performed before this design was written. The relevant structure is summarized below.

### 2.1 Project and dependency context

| Area | Relevant files | Findings | Reuse / action |
|---|---|---|---|
| Package metadata and commands | `README.md`, `pixi.toml`, root `__init__.py`, root `__main__.py`, `config/cli.py` | The project is already JAX-oriented in current docs/notebooks, with Pixi tasks for tests, imports, notebooks, networkmodel, and dashboard. Dependencies include `jax`, `jaxopt`, `diffrax`, `pandas`, plotting packages, and `networkx`. | Reuse existing dependency set. Do not add SciPy/NetworkX as core requirements for this module. Add imports lazily where optional. |
| Result directory contract | `common/results.py`, `docs/result_directory_contract.md` | Standard output subdirectories are `tables`, `plots`, `logs`, `reports`, and `artifacts`; metadata and resolved config helpers already exist. | Reuse `common.results.ensure_result_dir()`, `write_metadata()`, `write_resolved_config()`, and `populate_standard_subdirs()` for module outputs. |
| Configuration | `config.toml`, `config_loader.py`, `networkmodel/config.py`, `config/config.py`, `config/constants.py` | Networkmodel runtime config is resolved through `config_loader.load_config_toml()` and `networkmodel.runner` parses CLI args after resolving `--conf`. | Add any future option as disabled-by-default config/CLI hook only; initial design doc requires no config edit. |
| Logging | `config.config.setup_logger()`, `networkmodel.runner.initialize_run_contract()` | Workflows already attach loggers to output directories. | Reuse workflow logger when integrated; module should accept an optional logger. |

### 2.2 Existing network, data-loading, and topology code

| Area | Relevant files | Findings | Reuse / action |
|---|---|---|---|
| Networkmodel data loading | `networkmodel/io.py` | `load_data(args)` reads kinase network CSV, normalizes columns, expands kinase sets in cells, uppercases proteins/kinases, merges optional KinOpt alpha/beta priors, reads TF network, merges optional TFOpt priors, and creates tidy protein/phospho/RNA data frames. It returns `df_kin_clean`, `df_tf_clean`, `df_prot`, `df_pho`, `df_rna`, `kin_beta_map`, and `tf_beta_map`. | Reuse this loader as the primary integration boundary. The new module should consume the cleaned frames rather than rereading input files. Extend only if a future input field is genuinely missing. |
| Matrix construction | `networkmodel/BuildMatrix.py` | `site_key()` normalizes phosphosite labels numerically. `build_W_parallel()` builds kinase-to-site SciPy CSR matrices from cleaned interactions and alpha weights. `build_tf_matrix()` builds TF regulatory matrices using alpha/beta and proxy-aware weights. | Reuse `site_key()` semantics for deterministic site ordering at I/O boundary. Do not call SciPy sparse constructors in the JAX core. The proposed sparse triplet tensor should be an indexed JAX/NumPy-compatible table. |
| Indexing and model universe | `networkmodel/network.py` | `Index` builds protein, kinase, site, state-offset, and proxy maps. It includes memory guards for combinatorial state explosion and logs counts. It handles orphan TF proxy redirection. | Reuse `Index` for downstream ODE construction. The preprocessing module may produce a pruned `df_kin`/triplet table that can be passed into `Index`, but must not mutate `Index` internals or change default construction. |
| Mode detection and JAX backend | `networkmodel/backend.py` | JAX float64 is enabled with `jax.config.update("jax_enable_x64", True)`. `DataMode`, `ensure_jax_float64()`, `detect_data_mode()`, and validation helpers already exist. | Reuse `ensure_jax_float64()` behavior or mirror it in the new module. Keep outputs compatible with JAXopt arrays and current scalar objective paths. |
| Loss-cache indexing | `networkmodel/cache.py` | Converts tidy observations into integer indexed arrays for fast loss evaluation. | Reuse as the conceptual pattern for converting Pandas frames to index arrays. Do not embed Pandas inside jitted kernels. |

### 2.3 ODE, optimization, inference, and parameter code

| Area | Relevant files | Findings | Reuse / action |
|---|---|---|---|
| Runner orchestration | `networkmodel/runner.py` | Main sequence: resolve config, initialize run contract, import runtime modules, load data, normalize/filter phospho, handle TF proxying, construct `Index`, build matrices, initialize parameters, run JAXopt/inference, write outputs. | Integrate only as an optional hook after `load_data()` and current mechanistic phospho filtering, before `Index`/matrix construction and before `init_raw_params()`/JAXopt fitting. Default must remain off. |
| Parameter packing | `networkmodel/params.py` | `init_raw_params()` builds flat theta, slices, bounds, and defaults for `A_i`, `B_i`, `C_i`, `D_i`, `E_i`, `c_k`, `tf_scale`, and `Dp_i`; alpha/beta are explicitly not optimized. | Identifiability preprocessing should output masks/projections/bounds recommendations compatible with this flat-slice layout, not add alpha/beta as optimized parameters. |
| Objective and inference | `networkmodel/OptimizationProblem.py`, `networkmodel/BayesianInference.py`, `networkmodel/PosteriorObjective.py` | Current fitting path is JAX/JAXopt/Diffrax oriented, with multistart/profile/posterior support. | The preprocessing output must be static before fitting and must not introduce non-JAX optimizers. Optional projected parameter masks should be applied before JAXopt. |
| Simulation/models | `networkmodel/models.py`, `networkmodel/simulate.py`, `networkmodel/InitialConditions.py`, `networkmodel/SteadyStateAnalysis.py` | ODE RHS and simulation are separate from network loading. Combinatorial logic has memory guards and streaming tests. | Do not touch RHS for preprocessing. Use triplet pruning to reduce model size before these modules are called. |
| Legacy local workflow | `protwise/runner/main.py`, `protwise/paramest/*`, `protwise/models/*`, `protwise/plotting/plotting.py` | Protein-wise workflow has its own data extraction, ODE solving, sensitivity, and plotting. | Do not integrate by default. A later optional adapter can consume per-gene site tables, but first target should be `networkmodel`. |

### 2.4 Output, plotting, dashboard, scripts, notebooks, and tests

| Area | Relevant files | Findings | Reuse / action |
|---|---|---|---|
| Output/export | `networkmodel/export.py`, `networkmodel/mode_outputs.py`, `common/utils/display.py`, `common/utils/tables.py`, `common/utils/latexit.py` | Networkmodel exports CSV/Excel tables, plots, residuals, parameter distributions, correlations, and S-rate reports. Common utilities handle reports/tables for legacy workflows. | The new module should write its own subfolder under the run output directory while following the standard directory contract. It should not overload existing fitted-output file names. |
| Plotting | `networkmodel/export.py`, `protwise/plotting/plotting.py`, dashboard components | Existing plots use Matplotlib/Seaborn/Plotly at output boundaries. Dashboard expects tables/plots/reports/artifacts. | Use Matplotlib/Seaborn/Plotly only outside JAX core. Save publication-quality PNG/SVG/HTML into `plots/` or a module subfolder. |
| CLI/dashboard | `config/cli.py`, `dashboard/*`, `run_dashboard.py`, `networkmodel/dashboard_bundle.py` | CLI and dashboard already discover workflow outputs and standard folders. | Add an optional CLI flag later, e.g. `--network-preprocessing`, disabled by default. Dashboard integration can be passive via generated tables/plots. |
| Scripts | `scripts/*.py` | Standalone analysis and visualization scripts read existing outputs. | Do not touch initially. Later scripts can read exported preprocessing CSVs. |
| Tests | `tests/*`, especially `tests/test_combinatorial_memory_safe.py`, `tests/test_result_contract.py`, networkmodel runner/inference tests, dashboard parser tests | Tests cover importability, result contract, runner config, inference workers, notebooks, memory-safe combinatorial behavior, and dashboard parsing. | Add focused tests for the new module plus regression tests proving current workflows still run unchanged when the hook is disabled. |
| Docs | `docs/phoskintime_multimodal_input_audit.md`, `docs/phoskintime_jaxopt_diffrax_networkmodel_protwise_audit.md`, `docs/Combinatorial_Model_Memory_Issue.md`, `docs/result_directory_contract.md` | Existing docs include audit-style developer blueprints and memory-safety guidance. | This document follows the same implementation-blueprint style. No runtime code is implemented here. |

### 2.5 What should not be touched

- Do not change `networkmodel/io.py` loader behavior for existing workflows unless adding an optional return/adapter later.
- Do not change default `networkmodel.runner.main()` behavior.
- Do not rewrite `networkmodel.network.Index`; pass pruned cleaned frames into it only when enabled.
- Do not change current `BuildMatrix.py` SciPy CSR behavior for default workflows.
- Do not alter `kinopt`, `tfopt`, `protwise`, notebooks, or dashboard default assumptions in the first implementation.
- Do not add NetworkX/SciPy/Pandas logic to kernels intended for JIT.

## 3. Proposed module name and location

Recommended path:

```text
network_preprocessing/
├── __init__.py
├── api.py
├── dataclasses.py
├── io_adapters.py
├── jax_kernels.py
├── scoring.py
├── motifs.py
├── identifiability.py
├── export.py
└── plotting.py
```

Rationale:

- A top-level `network_preprocessing` package mirrors existing top-level workflow packages (`networkmodel`, `protwise`, `processing`) without making it a hidden submodule of `networkmodel`.
- It remains optional and independent.
- `networkmodel` can import it lazily only when a disabled-by-default hook is enabled.
- The package can later support `protwise` or standalone CLI usage without depending on `networkmodel.runner`.

Alternative path `networkmodel/preprocessing/` was considered but rejected for the initial implementation because it would imply tighter coupling to the networkmodel ODE internals.

## 4. Public API design

### 4.1 Data structures

Use frozen dataclasses with JAX arrays in the computational layer and Pandas frames only at I/O/export boundaries.

```python
@dataclass(frozen=True)
class NetworkPreprocessingConfig:
    min_triplet_score: float = 0.0
    min_support_count: int = 1
    max_triplets: int | None = None
    batch_size: int = 65536
    dtype: str = "float64"
    enable_motifs: bool = True
    enable_identifiability: bool = True
    prune_self_loops: bool = True
    prune_missing_observations: bool = False
    score_weights: Mapping[str, float] = field(default_factory=dict)
    output_subdir: str = "network_preprocessing"
```

```python
@dataclass(frozen=True)
class EncodedNetwork:
    kinase_ids: jax.Array        # int32 [n_edges]
    substrate_ids: jax.Array     # int32 [n_edges]
    site_ids: jax.Array          # int32 [n_edges]
    edge_weight: jax.Array       # float64 [n_edges]
    support_count: jax.Array     # int32 [n_edges]
    kinase_labels: tuple[str, ...]
    substrate_labels: tuple[str, ...]
    site_labels: tuple[str, ...]
```

```python
@dataclass(frozen=True)
class TripletTable:
    kinase_ids: jax.Array        # int32 [n_triplets]
    site_ids: jax.Array          # int32 [n_triplets]
    substrate_ids: jax.Array     # int32 [n_triplets]
    score: jax.Array             # float64 [n_triplets]
    support_count: jax.Array     # int32 [n_triplets]
    flags: jax.Array             # uint32 bitmask [n_triplets]
```

```python
@dataclass(frozen=True)
class SparseThetaTensor:
    indices: jax.Array           # int32 [nnz, 3], columns k/site/substrate
    values: jax.Array            # float64 [nnz]
    shape: tuple[int, int, int]   # n_kinases, n_sites, n_substrates
```

```python
@dataclass(frozen=True)
class MotifTable:
    motif_type: jax.Array        # int16 [n_motifs]
    node_a: jax.Array            # int32 [n_motifs]
    node_b: jax.Array            # int32 [n_motifs]
    node_c: jax.Array            # int32 [n_motifs]
    edge_mask: jax.Array         # uint8 [n_motifs]
    score: jax.Array             # float64 [n_motifs]
```

```python
@dataclass(frozen=True)
class IdentifiabilityDiagnostics:
    retained_param_mask: jax.Array       # bool [n_candidate_params]
    group_id: jax.Array                  # int32 [n_candidate_params]
    local_rank_estimate: int
    redundancy_score: jax.Array          # float64 [n_candidate_params]
    design_column_norm: jax.Array        # float64 [n_candidate_params]
```

```python
@dataclass(frozen=True)
class NetworkPreprocessingResult:
    encoded: EncodedNetwork
    discovered: TripletTable
    pruned: TripletTable
    theta: SparseThetaTensor
    motifs: MotifTable | None
    identifiability: IdentifiabilityDiagnostics | None
    summary: Mapping[str, Any]
```

### 4.2 User-facing functions

```python
def preprocess_network(
    kinase_network: pd.DataFrame,
    *,
    phospho_observations: pd.DataFrame | None = None,
    protein_observations: pd.DataFrame | None = None,
    rna_observations: pd.DataFrame | None = None,
    tf_network: pd.DataFrame | None = None,
    config: NetworkPreprocessingConfig | None = None,
    output_dir: str | Path | None = None,
    logger: logging.Logger | None = None,
) -> NetworkPreprocessingResult:
    """Encode, discover, prune, detect motifs, build sparse theta, optionally export outputs."""
```

Primary adapter for `networkmodel.runner`:

```python
def preprocess_networkmodel_frames(
    df_kin: pd.DataFrame,
    df_tf: pd.DataFrame,
    df_prot: pd.DataFrame,
    df_pho: pd.DataFrame,
    df_rna: pd.DataFrame,
    *,
    config: NetworkPreprocessingConfig | None = None,
    output_dir: str | Path | None = None,
    logger: logging.Logger | None = None,
) -> tuple[pd.DataFrame, NetworkPreprocessingResult]:
    """Return a pruned df_kin compatible with existing networkmodel.Index plus diagnostics."""
```

Standalone export and plotting:

```python
def export_preprocessing_result(
    result: NetworkPreprocessingResult,
    output_dir: str | Path,
    *,
    labels: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Path]:
    """Write CSV/JSON/NPZ outputs under output_dir/network_preprocessing."""
```

```python
def plot_preprocessing_result(
    result: NetworkPreprocessingResult,
    output_dir: str | Path,
    *,
    style: str = "paper",
) -> dict[str, Path]:
    """Create diagnostic plots from exported or in-memory result arrays."""
```

Lower-level JAX API for tests and advanced users:

```python
def discover_hyperedges(encoded: EncodedNetwork, config: NetworkPreprocessingConfig) -> TripletTable

def prune_triplets(triplets: TripletTable, config: NetworkPreprocessingConfig) -> TripletTable

def detect_motifs(triplets: TripletTable, edge_index: jax.Array, config: NetworkPreprocessingConfig) -> MotifTable

def build_sparse_theta(triplets: TripletTable, shape: tuple[int, int, int]) -> SparseThetaTensor

def preprocess_identifiability(theta: SparseThetaTensor, design_features: jax.Array, config: NetworkPreprocessingConfig) -> IdentifiabilityDiagnostics
```

## 5. Internal algorithm design

### 5.1 I/O encoding boundary

1. Consume cleaned `df_kin` from `networkmodel.io.load_data()` whenever integrated.
2. Validate required columns: `protein`, `psite`, `kinase`; optional `alpha`, `beta`, `evidence`, `source`, `confidence`.
3. Normalize labels at boundary only:
   - protein/substrate labels: uppercase stripped strings,
   - kinase labels: uppercase stripped strings,
   - phosphosite labels: stripped strings sorted using `networkmodel.BuildMatrix.site_key()` when available.
4. Build categorical maps:
   - `kinase_label -> kinase_id`,
   - `(substrate_label, psite_label) -> site_id`,
   - `substrate_label -> substrate_id`.
5. Convert to fixed-width arrays: `int32` ids and `float64` weights.
6. Aggregate duplicate triplets with deterministic reductions:
   - support count: sum,
   - edge weight: max or weighted mean, configured but default `max(alpha)` to preserve strong prior evidence,
   - duplicate-source annotation: bitmask outside core.

No Pandas objects enter JIT-compiled functions.

### 5.2 Hyperedge discovery

A candidate hyperedge is `(kinase_id, site_id, substrate_id)` where the kinase is reported to regulate the substrate phosphosite. Existing `df_kin` rows already represent direct triplets after expansion by `networkmodel.io.load_data()`. Discovery therefore has two modes:

- **Direct mode**: canonicalize and deduplicate rows from the cleaned kinase network.
- **Augmented mode**: optionally infer additional candidate triplets from compatible modalities, such as kinase protein observations, phosphosite observations, and TF feedback context, without requiring them for operation.

Core scoring features should be array-based:

| Feature | Array source | Description |
|---|---|---|
| `prior_weight` | `edge_weight` | Alpha/confidence prior from KinOpt or raw network. |
| `support_log1p` | `support_count` | Duplicate/source support strength. |
| `observed_site` | `site_observed_mask[site_id]` | Whether the substrate-site appears in phospho observations. |
| `observed_kinase` | `kinase_observed_mask[kinase_id]` | Whether the kinase has protein/phospho evidence. |
| `temporal_assoc` | optional precomputed JAX arrays | Simple correlation-like or signed lag feature computed from aligned time arrays only when available. |
| `self_loop_penalty` | `kinase_to_substrate_id == substrate_id` | Penalize or remove self loops depending on config. |

Initial score formula:

```text
score = w_prior * log1p(prior_weight)
      + w_support * log1p(support_count)
      + w_site_obs * observed_site
      + w_kinase_obs * observed_kinase
      + w_temporal * temporal_assoc
      - w_self_loop * self_loop
```

The formula is deliberately simple, differentiable where useful, deterministic, and JAX-friendly. It can be extended without changing the output schema.

### 5.3 Network pruning

Pruning should be a boolean mask over candidate triplets, computed in JAX:

- score threshold: `score >= min_triplet_score`,
- support threshold: `support_count >= min_support_count`,
- optional self-loop removal,
- optional missing-observation removal,
- optional maximum triplet budget using stable top-k outside JIT or with static `k` if configured,
- optional per-kinase/per-substrate caps using segmented top-k in batches.

Pruning output must retain removed triplets with reason flags for export:

```text
FLAG_LOW_SCORE          = 1 << 0
FLAG_LOW_SUPPORT        = 1 << 1
FLAG_SELF_LOOP          = 1 << 2
FLAG_MISSING_SITE_OBS   = 1 << 3
FLAG_MISSING_KINASE_OBS = 1 << 4
FLAG_OVER_BUDGET        = 1 << 5
```

The pruned table contains only retained triplets, while export also writes a full annotated table with `retained` and `removal_reason` columns.

### 5.4 Motif detection

Motifs are detected on an indexed directed graph derived from retained triplets plus optional TF edges:

- kinase-to-substrate edge: `K -> S` for each retained triplet,
- substrate-site incidence retained as site annotation,
- optional TF edge: `TF -> target` from `df_tf`, encoded separately.

Target motifs:

1. **Feed-forward loop**: `A -> B`, `B -> C`, `A -> C`.
2. **Kinase cascade**: kinase `A` regulates kinase/substrate `B`, and `B` regulates `C`; optional closing edge `A -> C` distinguishes cascade vs feed-forward loop.
3. **Regulatory triangle**: any 3-node directed triangle with three retained regulatory edges.
4. **Kinase-site-substrate triangle**: kinase `K`, substrate `S`, and third regulator `R` where `K -> S`, `R -> K`, and `R -> S` or `S -> R` exists.

Implementation strategy:

- Build sparse edge lists `(src_id, dst_id, edge_type, score)` from retained triplets and optional TF edges.
- Sort lexicographically by `(src, dst)` outside JIT or with JAX sorting for fixed arrays.
- For each edge batch `(a, b)`, look up outgoing neighbors of `b` and test whether `(a, c)` exists using sorted-key binary search.
- Avoid dense adjacency matrices for large networks. Dense bitset adjacency may be allowed only for small `n_nodes <= config.dense_motif_node_limit` and must be disabled by default for large networks.
- Use `lax.scan` or chunked `vmap` over edge batches to avoid materializing all wedges.
- Return motif rows capped by a configured `max_motifs` with summary counts always computed.

Motif scoring:

```text
motif_score = geometric_mean(edge_scores) * motif_type_weight
```

For feed-forward loops and regulatory triangles, export node labels, edge labels, site annotations when present, motif type, and motif score.

### 5.5 Sparse theta tensor construction

The tensor is conceptually:

```text
theta[k, site, substrate] = interaction score or prior-scaled initial value
```

Do not allocate dense shape `(n_kinases, n_sites, n_substrates)` except in tiny tests. Use COO-like representation:

```python
indices = jnp.stack([kinase_ids, site_ids, substrate_ids], axis=1).astype(jnp.int32)
values = score.astype(jnp.float64)
shape = (n_kinases, n_sites, n_substrates)
```

Additional exports:

- `sparse_theta_indices.csv`: labelled `kinase_id`, `site_id`, `substrate_id`, labels, and value.
- `sparse_theta.npz`: compressed index/value arrays for downstream loading.
- Optional `theta_group_id` linking tensor entries to identifiable parameter groups.

Downstream JAXopt compatibility:

- `indices` and `values` are static inputs to model construction or objective setup.
- Any trainable parameter vector associated with triplets should be a length-`nnz` vector aligned to `indices`, not a dense tensor.
- If the first implementation only prunes topology and does not add trainable triplet parameters, `values` should be treated as fixed priors/weights.

### 5.6 Identifiability preprocessing

Goal: reduce the parameter space before JAXopt fitting without fitting the model.

Inputs:

- sparse theta indices/values,
- observation availability masks,
- current network degrees,
- optional design features derived from time-series availability,
- optional existing parameter slices from `networkmodel.params.init_raw_params()`.

Diagnostics:

1. **Structural observability mask**:
   - remove or freeze triplet parameters with no observed substrate/site and no path to observed proteins/RNA when `prune_missing_observations=True`.
2. **Duplicate column grouping**:
   - group triplets with identical `(kinase_id, substrate_id, site_observed_pattern, time_observed_pattern)` feature columns.
   - retain one representative or tie parameters via `group_id`.
3. **Rank approximation**:
   - build a compact design matrix from non-dense features: kinase one-hot hash, substrate/site incidence, observation masks, degree features, motif participation counts.
   - compute small Gram matrix `X.T @ X` only when feature dimension is bounded.
   - use `jnp.linalg.svd` or QR on the compact design matrix, not on a dense cubic tensor.
4. **Redundancy score**:
   - high for duplicate/near-duplicate feature columns,
   - high for zero-norm columns,
   - high for triplets that do not influence any observed state in a static reachability approximation.

Output can be consumed by downstream fitting in two ways:

- topology-only: pass pruned `df_kin` into existing model setup,
- parameter-projection: provide masks/groups to future JAXopt wrappers that optimize only retained independent variables and expand back to full theta via a pure-JAX projection.

Projection API:

```python
def make_parameter_projection(retained_mask: jax.Array, group_id: jax.Array):
    """Return project(reduced_theta)->full_theta and reduce(full_theta)->reduced_theta callables."""
```

## 6. JAX implementation plan

### 6.1 JAX setup

Every public entry point that uses JAX should call:

```python
jax.config.update("jax_enable_x64", True)
```

or reuse `networkmodel.backend.ensure_jax_float64()` when `networkmodel` is available. Dtypes:

- ids: `jnp.int32`, unless arrays exceed `2^31 - 1`, which should raise a clear error,
- scores/weights/diagnostics: `jnp.float64`,
- masks: `jnp.bool_`,
- flags: `jnp.uint32`.

### 6.2 JIT boundaries

JIT-compile:

- score computation from encoded arrays,
- pruning mask and reason flag computation,
- sparse theta construction from retained arrays,
- motif wedge scanning for fixed batch shapes,
- identifiability feature scoring/grouping kernels where shapes are static.

Do not JIT:

- Pandas loading/validation/label mapping,
- CSV/JSON/NPZ writing,
- Matplotlib/Seaborn/Plotly plotting,
- logger calls,
- variable-length text annotation,
- dynamic file path handling.

### 6.3 `vmap`, `lax.scan`, and batching

- Use `vmap` for per-triplet feature scoring and reason-flag construction.
- Use `lax.scan` over fixed-size batches for motif wedge enumeration and large degree/reachability passes.
- Use chunked host loops that call jitted batch kernels for arrays too large to fit in memory. Host loops may orchestrate batches but must not perform numerical core logic with NumPy/Pandas.
- Pad the final batch to `batch_size` and carry an explicit `valid_mask` to avoid shape-driven recompilation.
- Keep static arguments explicit: `batch_size`, `max_motifs_per_batch`, and feature dimensions should be `static_argnames`.

### 6.4 Avoiding JIT recompilation traps

- Do not pass Pandas frames, Python dicts of arbitrary labels, or variable-length tuples into jitted kernels.
- Convert config values used in JIT to a small frozen numeric dataclass or pass as scalar arguments.
- Bucket workloads by batch size rather than recompiling per network size.
- Keep optional features represented by zero-filled arrays and boolean enable flags rather than changing function signatures.

## 7. Integration points

### 7.1 Networkmodel integration sequence

Optional hook in `networkmodel.runner.main()` after loading and existing basic phospho filtering, before `Index` construction:

```python
df_kin, df_tf, df_prot, df_pho, df_rna, kin_beta_map, tf_beta_map = load_data(args)
# existing normalization/filtering remains
if args.enable_network_preprocessing:
    from network_preprocessing import preprocess_networkmodel_frames
    df_kin, prep_result = preprocess_networkmodel_frames(
        df_kin, df_tf, df_prot, df_pho, df_rna,
        config=NetworkPreprocessingConfig.from_args(args),
        output_dir=args.output_dir,
        logger=logger,
    )
# existing TF handling, Index(df_kin, ...), build_W_parallel(...), fitting continue
```

Default: `args.enable_network_preprocessing == False`.

### 7.2 Before ODE model construction

- The hook returns a pruned `df_kin` with the same schema expected by `Index` and `build_W_parallel()`.
- Existing `df_tf` remains unchanged unless a later optional TF pruning mode is enabled.
- `Index` sees fewer triplets/sites only when users opt in.

### 7.3 Before optimization and JAXopt fitting

- Run identifiability preprocessing before `init_raw_params()` where topology changes affect `idx.N`, `idx.total_sites`, and parameter vector sizes.
- If parameter projection is later added, apply it between `init_raw_params()` and `GlobalODEScalarObjective`/JAXopt setup.
- Keep alpha/beta as network weights, not optimized theta entries, preserving `networkmodel.params` expectations.

### 7.4 CLI, config, dashboard, scripts

Future disabled-by-default options:

```text
--enable-network-preprocessing
--network-preprocessing-min-score FLOAT
--network-preprocessing-min-support INT
--network-preprocessing-max-triplets INT
--network-preprocessing-batch-size INT
--network-preprocessing-disable-motifs
--network-preprocessing-disable-identifiability
```

Dashboard integration can be passive: generated tables and plots appear under standard result folders. No dashboard code is required for the first implementation unless a dedicated panel is desired later.

## 8. Output design

The module must independently save its own outputs in the current run output directory:

```text
<run_output_dir>/
├── tables/
│   ├── network_preprocessing_discovered_hyperedges.csv
│   ├── network_preprocessing_pruned_triplets.csv
│   ├── network_preprocessing_removed_triplets.csv
│   ├── network_preprocessing_motifs.csv
│   ├── network_preprocessing_sparse_theta.csv
│   ├── network_preprocessing_identifiability.csv
│   └── network_preprocessing_summary.csv
├── plots/
│   ├── network_preprocessing_hyperedge_score_distribution.png
│   ├── network_preprocessing_retained_removed_triplets.png
│   ├── network_preprocessing_motif_counts.png
│   ├── network_preprocessing_degree_distributions.png
│   ├── network_preprocessing_sparsity_summary.png
│   └── network_preprocessing_identifiability_diagnostics.png
├── reports/
│   └── network_preprocessing_report.md
└── artifacts/
    ├── network_preprocessing_sparse_theta.npz
    ├── network_preprocessing_encoded_network.npz
    └── network_preprocessing_config.json
```

CSV requirements:

- Always include integer ids and human-readable labels.
- Include `retained`, `score`, `support_count`, `prior_weight`, and `removal_reason` where applicable.
- Include enough provenance to reproduce mapping: kinase labels, substrate labels, site labels, shape, score weights, thresholds.
- Empty result sets should still write schema-correct empty CSVs and a summary explaining why no rows were retained.

Summary statistics:

- input triplet count,
- duplicate count,
- discovered triplet count,
- retained/removed counts,
- kinase/substrate/site counts before and after pruning,
- sparse tensor shape and density,
- motif counts by type,
- identifiability retained/frozen/grouped parameter counts,
- runtime and batch-size diagnostics,
- memory estimates for dense tensor avoided.

## 9. Plotting plan

Plotting uses exported Pandas tables or host-side arrays only. Recommended plots:

1. **Hyperedge score distributions**
   - histogram/KDE of all candidate scores,
   - vertical threshold line,
   - retained vs removed overlay.
2. **Retained vs removed triplets**
   - stacked bar by removal reason,
   - top kinases/substrates contributing removed triplets.
3. **Motif count summaries**
   - bar chart by motif type,
   - optional top motif score table rendered as HTML/CSV.
4. **Kinase/substrate degree distributions**
   - pre/post pruning in/out degree histograms,
   - log-scale option for large networks.
5. **Sparsity heatmaps or tensor summaries**
   - kinase-by-substrate nonzero count heatmap for top entities,
   - site-count distribution,
   - tensor density annotation.
6. **Identifiability diagnostics**
   - rank estimate and singular-value scree when compact SVD is available,
   - redundancy score distribution,
   - retained/frozen/grouped parameter counts.

All static plots should be saved at publication quality (`dpi >= 300`, tight layout, readable labels). Optional interactive Plotly HTML can be added later but should not be required for tests.

## 10. Testing plan

### 10.1 Unit tests

Add `tests/network_preprocessing/` with:

- `test_encoding.py`
  - duplicate aggregation,
  - deterministic label maps,
  - `site_key()`-compatible ordering,
  - malformed/missing required column errors.
- `test_hyperedge_discovery.py`
  - direct triplets from cleaned `df_kin`,
  - score formula on tiny arrays,
  - self-loop flags.
- `test_pruning.py`
  - threshold pruning,
  - support pruning,
  - reason flag combinations,
  - empty retained set schema.
- `test_sparse_theta.py`
  - COO indices/values match retained triplets,
  - no dense cubic allocation,
  - shape/density calculations.
- `test_motifs.py`
  - synthetic feed-forward loop,
  - kinase cascade,
  - regulatory triangle,
  - no false positives on acyclic two-edge graph.
- `test_identifiability.py`
  - duplicate feature columns grouped,
  - zero-norm columns flagged,
  - reduced/full projection round trip.

### 10.2 JAX-specific tests

- Verify `jax_enable_x64` is true after public API call.
- Assert score/value arrays are `float64` and ids are `int32`.
- Call jitted kernels twice with same padded shape and ensure outputs are stable.
- Use `jax.make_jaxpr()` or smoke tests to confirm kernels accept JAX arrays.
- Test batch padding with non-multiple-of-batch-size triplet counts.

### 10.3 Output tests

- Use a temporary result directory and call `export_preprocessing_result()`.
- Assert all expected CSV/NPZ/JSON files exist.
- Assert empty outputs write headers.
- Assert plots are created when plotting dependencies are available; mark as skip or use non-interactive backend.

### 10.4 Integration and regression tests

- Existing networkmodel tests must pass unchanged with preprocessing disabled.
- Add one tiny enabled networkmodel test that:
  1. loads small synthetic frames,
  2. runs preprocessing,
  3. passes pruned `df_kin` to `Index`,
  4. verifies `idx.total_sites` and `idx.kinases` are consistent.
- Add CLI parser regression only after CLI flags are implemented.
- Add dashboard parser regression proving generated tables/plots under standard folders do not break result browsing.

### 10.5 Large-scale safety tests

- Synthetic network with many kinases/sites/substrates but sparse triplets.
- Assert memory estimate remains proportional to `nnz`, not cubic tensor size.
- Assert dense motif mode is not used above limit.
- Assert configured `max_triplets` caps retained output deterministically.

## 11. Failure modes and safeguards

| Failure mode | Safeguard |
|---|---|
| Dense `O(n^3)` tensor blow-up | Never allocate dense theta except tiny tests. Export COO index/value arrays and explicit density diagnostics. |
| Motif wedge explosion | Use edge batching, motif caps, and summary-only fallback when candidate wedge count exceeds threshold. |
| JIT recompilation storm | Fixed batch sizes, padded final batches, static args, no labels/Pandas/dicts inside JIT. |
| Missing modalities | Treat protein/RNA/phospho observations as optional; set masks to zeros and continue unless config requires observations. |
| Malformed network columns | Validate at I/O boundary with clear expected column names and discovered columns. |
| Duplicated phosphosites/triplets | Deterministically aggregate support and weights; export duplicate counts. |
| Unsupported IDs or unparseable sites | Keep labels as strings; use numeric site sorting only when possible; fall back to lexical ordering with warnings outside JIT. |
| Empty candidate set | Write empty schema-correct outputs, return empty arrays, skip motif/rank kernels, and log summary. |
| Empty retained set after pruning | Do not crash exports; integration hook should raise a clear opt-in error before ODE construction unless user allows empty topology. |
| Float32 accidental use | Force `jax_enable_x64`; assert values dtype in tests and public validation. |
| Non-JAX operations in core loop | Code review and tests should keep Pandas/NumPy in adapters/export only. |
| Existing workflow behavior changes | Disabled-by-default integration; regression tests run existing workflows unchanged. |
| Over-pruning biologically important weak edges | Export removed triplets with reasons; provide conservative defaults (`min_triplet_score=0`, `min_support_count=1`). |
| Identifiability overconfidence | Label rank as preprocessing approximation; export feature basis and thresholds; do not silently delete parameters without diagnostics. |

## 12. Stepwise implementation roadmap

### Stage 0: Scan and map existing code

- Confirm loader schemas from `networkmodel.io.load_data()`.
- Confirm `Index` and `BuildMatrix` requirements.
- Confirm result directory utilities in `common.results`.
- Confirm JAX/JAXopt conventions in `networkmodel.backend`, `OptimizationProblem`, and `BayesianInference`.

### Stage 1: Add data structures and adapters

- Create `network_preprocessing/dataclasses.py`.
- Create `io_adapters.py` to convert cleaned Pandas frames to encoded JAX arrays.
- Add validation and deterministic label mapping.
- Add tests for encoding and malformed inputs.

### Stage 2: Add pure-JAX kernels

- Create `jax_kernels.py` with score, prune, sparse theta, degree, and utility kernels.
- Enable float64 and enforce dtypes.
- Add JIT and shape/dtype tests.

### Stage 3: Add scoring and pruning

- Implement configurable score weights.
- Implement reason flags and retained/removed split.
- Add conservative default behavior that preserves all valid triplets unless thresholds are changed.

### Stage 4: Add motif detection

- Implement sparse edge-list motif scan.
- Add tiny synthetic motif tests.
- Add batching/cap safeguards.

### Stage 5: Add sparse tensor export

- Implement `SparseThetaTensor` construction.
- Export CSV and NPZ with labelled rows.
- Add tests proving no dense cubic allocation is required.

### Stage 6: Add identifiability preprocessing

- Implement structural masks, duplicate grouping, compact design features, rank approximation, and projection helpers.
- Keep diagnostics conservative and transparent.
- Add round-trip projection tests.

### Stage 7: Add plotting/output layer

- Reuse `common.results.ensure_result_dir()` and standard subdirectories.
- Implement CSV/JSON/NPZ exports.
- Implement publication-quality static plots.
- Add output-writing tests with temporary directories.

### Stage 8: Add optional integration hook

- Add disabled-by-default config/CLI option.
- Lazily import `network_preprocessing` only when enabled.
- Insert hook before `Index` construction and before JAXopt setup.
- Preserve current behavior when disabled.

### Stage 9: Add integration tests and docs

- Add enabled tiny integration test.
- Run existing tests unchanged.
- Document CLI/config usage only after implementation exists.
- Update dashboard docs only if a dedicated panel is added.

## 13. Acceptance criteria for first implementation

- Existing workflows pass with preprocessing disabled.
- A small synthetic network discovers, scores, prunes, detects motifs, builds sparse theta, and exports outputs.
- All computational kernels use JAX/JAX NumPy and operate on JAX arrays.
- Float64 is enabled and tested.
- No dense cubic enumeration is used for production paths.
- Output files follow the repository result directory contract.
- Integration with `networkmodel` is optional, lazy, and non-breaking.
