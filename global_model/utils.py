"""
Utilities and Configuration Management Module.

This module provides essential helper functions and classes for:
1.  **Data Normalization:** Functions to clean column names (`_normcols`), calculate fold-changes
    relative to a baseline (`normalize_fc_to_t0`), and scale data for consistent modeling.
2.  **Configuration Loading:** The `load_config_toml` function parses the `config.toml` file
    into a structured `PhosKinConfig` dataclass, centralizing all run settings.
3.  **Dynamic Bound Calculation:** `calculate_bio_bounds` intelligently sets parameter limits
    based on the input data range and network topology, ensuring biological plausibility.
4.  **Math Helpers:** JIT-compiled kernels for common operations like `softplus` (for positive parameters)
    and `time_bucket` (for discrete input grids).


"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit
import tomllib
from config.config import setup_logger
from global_model.config import MODEL, RESULTS_DIR

logger = setup_logger(log_dir=RESULTS_DIR)


def _normcols(df):
    """Normalize DataFrame column names (lowercase, strip, replace spaces)."""
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def _base_idx(times, t0):
    """Find the index of the timepoint closest to t0."""
    return np.int32(int(np.argmin(np.abs(times - float(t0)))))


def _find_col(df, cands):
    """Return the first column name from `cands` that exists in `df`."""
    for c in cands:
        if c in df.columns:
            return c
    return None


def normcols(df):
    return _normcols(df)


def find_col(df, cands):
    return _find_col(df, cands)


def slen(s: slice) -> int:
    return int(s.stop) - int(s.start)


def normalize_fc_to_t0(df):
    """
    Normalizes a Fold-Change (FC) DataFrame relative to the value at time t=0.

    Logic:
    For each (protein, psite) group, find the FC at t=0.
    New_FC(t) = Raw_FC(t) / Raw_FC(0).
    """
    df = df.copy()

    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["fc"] = pd.to_numeric(df["fc"], errors="coerce")
    df = df.dropna(subset=["time", "fc"])

    has_psite = "psite" in df.columns
    if has_psite:
        df["psite"] = (
            df["psite"]
            .fillna("")
            .astype(str)
            .str.strip()
            .replace({"nan": "", "NaN": ""})
        )

    keys = ["protein"] + (["psite"] if has_psite else [])

    t0 = (
        df.loc[df["time"].eq(0.0), keys + ["fc"]]
        .drop_duplicates(subset=keys, keep="last")
        .set_index(keys)["fc"]
    )

    def _norm_row(r):
        k = (r["protein"], r["psite"]) if has_psite else r["protein"]
        base = t0.get(k, np.nan)
        if not np.isfinite(base) or base == 0.0:
            return np.nan
        return r["fc"] / base

    df["fc"] = df.apply(_norm_row, axis=1)
    return df.dropna(subset=["fc"])


def process_and_scale_raw_data(df, time_points, id_cols, scale_method='fc_start', epsilon=1e-3):
    """
    Scales time-series data ensuring ALL outputs remain non-negative (>= 0).



    This function handles the conversion from "Wide" format (columns x1, x2, ...) to "Long" format
    and applies the selected scaling transformation.

    Args:
        scale_method (str):
            - 'raw' / 'none': No scaling. Returns raw intensities/counts.
            - 'fc_start': Standard Fold-Change (x_t / x_0).
            - 'robust_fc': Fold-Change with noise floor (x_t / (x_0 + eps)).
            - 'max_scale': Normalizes to [0, 1] range (x_t / x_max).
            - 'mean_scale': Centers data around 1.0 (x_t / x_mean).
            - 'l2_norm': Unit vector scaling (x_t / ||x||).
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=id_cols + ['time', 'fc'])

    # --- A. Identify and Sort X-Columns ---
    x_cols = [c for c in df.columns if re.fullmatch(r"x\d+", str(c))]

    if not x_cols:
        logger.warning(f"No 'x' columns found. Returning empty.")
        return pd.DataFrame()

    x_cols.sort(key=lambda c: int(c[1:]))

    if len(x_cols) > len(time_points):
        x_cols = x_cols[:len(time_points)]

    t_map = {xc: tp for xc, tp in zip(x_cols, time_points)}

    # --- B. Scale Data (Non-Negative Matrix Operations) ---
    df_work = df.copy()

    # Ensure numeric
    for c in x_cols:
        df_work[c] = pd.to_numeric(df_work[c], errors='coerce')

    # 0. Raw / None (Just formatting)
    if scale_method in ['raw', 'none']:
        pass  # Do nothing to values

    # 1. Standard Fold-Change (x / x0)
    elif scale_method == 'fc_start':
        start_vals = df_work[x_cols[0]].replace(0, epsilon)
        df_work[x_cols] = df_work[x_cols].div(start_vals, axis=0)

    # 2. Robust Fold-Change
    elif scale_method == 'robust_fc':
        baseline = df_work[x_cols[0]] + epsilon
        df_work[x_cols] = df_work[x_cols].div(baseline, axis=0)

    # 3. Max Scaling (Peak Normalization)
    elif scale_method == 'max_scale':
        peaks = df_work[x_cols].max(axis=1).replace(0, epsilon)
        df_work[x_cols] = df_work[x_cols].div(peaks, axis=0)

    # 4. Mean Scaling
    elif scale_method == 'mean_scale':
        means = df_work[x_cols].mean(axis=1).replace(0, epsilon)
        df_work[x_cols] = df_work[x_cols].div(means, axis=0)

    # 5. L2 Norm Scaling
    elif scale_method == 'l2_norm':
        l2 = np.sqrt(np.square(df_work[x_cols]).sum(axis=1)).replace(0, epsilon)
        df_work[x_cols] = df_work[x_cols].div(l2, axis=0)

    # --- C. Melt to Tidy Format ---
    valid_ids = [c for c in id_cols if c in df_work.columns]

    melted = df_work[valid_ids + x_cols].melt(
        id_vars=valid_ids,
        value_vars=x_cols,
        var_name="xcol",
        value_name="fc"
    )

    melted["time"] = melted["xcol"].map(t_map)
    melted = melted.dropna(subset=["fc", "time"])

    # Text normalization
    if "protein" in melted.columns:
        melted["protein"] = melted["protein"].astype(str).str.strip().str.upper()
    if "psite" in melted.columns:
        melted["psite"] = melted["psite"].fillna("").astype(str).str.strip()

    logger.info(f"[Scaled] Processed {len(melted)} rows with {len(valid_ids)} IDs and {len(x_cols)} time points.")
    logger.info(f"[Scaled] Scaling method: {scale_method}")

    return melted[valid_ids + ["time", "fc"]]


@njit(cache=True, fastmath=True, nogil=True)
def _zero_vec(a):
    """JIT helper to zero out an array."""
    for i in range(a.size):
        a[i] = 0.0


@njit(cache=True, fastmath=True, nogil=True)
def time_bucket(t, grid):
    """
    Finds the index `j` such that `grid[j] <= t < grid[j+1]`.
    Used for piecewise-constant input interpolation.
    """
    if t <= grid[0]:
        return 0
    if t >= grid[-1]:
        return grid.size - 1
    j = np.searchsorted(grid, t, side="right") - 1
    if j < 0:
        j = 0
    if j >= grid.size:
        j = grid.size - 1
    return j


@njit(cache=True, fastmath=True, nogil=True)
def softplus(x):
    """
    Softplus activation: log(1 + exp(x)).
    Maps real numbers to positive numbers. Used for parameter transformation.
    """
    out = np.empty_like(x)
    for i in range(x.size):
        xi = x[i]
        if xi > 20.0:
            out[i] = xi
        else:
            out[i] = np.log1p(np.exp(xi))
    return out


@njit(cache=True, fastmath=True, nogil=True)
def inv_softplus(y):
    """Inverse Softplus: log(exp(y) - 1). Maps positive numbers to real numbers."""
    out = np.empty_like(y)
    for i in range(y.size):
        yi = y[i]
        if yi < 1e-12:
            yi = 1e-12
        out[i] = np.log(np.expm1(yi))
    return out


@njit(cache=True, fastmath=True, nogil=True)
def pick_best_lamdas(F, weights):
    """
    Selects the best solution from a Pareto front based on a weighted sum of normalized objectives.
    """
    F = np.asarray(F, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    n = F.shape[0]
    m = F.shape[1]

    F_min = np.empty(m, dtype=np.float64)
    F_ptp = np.empty(m, dtype=np.float64)

    # Normalize objectives to [0, 1] range
    for j in range(m):
        mn = F[0, j]
        mx = F[0, j]
        for i in range(1, n):
            v = F[i, j]
            if v < mn:
                mn = v
            if v > mx:
                mx = v
        F_min[j] = mn
        F_ptp[j] = (mx - mn) + 1e-12

    best_i = 0
    best_score = 0.0

    s0 = 0.0
    for j in range(m):
        s0 += ((F[0, j] - F_min[j]) / F_ptp[j]) * weights[j]
    best_score = s0

    for i in range(1, n):
        s = 0.0
        for j in range(m):
            s += ((F[i, j] - F_min[j]) / F_ptp[j]) * weights[j]
        if s < best_score:
            best_score = s
            best_i = i

    return best_i, float(best_score)


@dataclass(frozen=True)
class PhosKinConfig:
    """
    Immutable configuration object holding all settings for the simulation run.
    Loaded from `config.toml`.
    """
    kinase_net: str | Path
    tf_net: str | Path
    ms_data: str | Path
    rna_data: str | Path
    phospho_data: str | Path | None
    kinopt_results: str | Path
    tfopt_results: str | Path

    normalize_fc_steady: bool
    use_initial_condition_from_data: bool

    time_points_prot: np.ndarray
    time_points_rna: np.ndarray
    time_points_phospho: np.ndarray

    bounds_config: dict[str, tuple[float, float]]
    model: str

    use_custom_solver: bool
    ode_abs_tol: float
    ode_rel_tol: float
    ode_max_steps: int

    loss_mode: int
    maximum_iterations: int
    population_size: int
    seed: int
    cores: int
    refine: bool
    num_refine: int

    regularization_rna: float
    regularization_lambda: float
    regularization_phospho: float
    regularization_protein: float

    results_dir: str | Path

    app_name: str = "Phoskintime-Global"
    version: str = "0.1.0"
    parent_package: str = "phoskintime"
    citation: str = ""
    doi: str = ""
    github_url: str = ""
    docs_url: str = ""

    hyperparam_scan: bool = False

    optimizer: str = "pymoo"  # "optuna" or "pymoo"

    # Optuna-specific knobs
    study_name: str = ""
    sampler: str = "TPESampler"
    pruner: str = "MedianPruner"
    n_trials: int = 0

    # Data scaling & weighting
    scaling_method: str = "none"
    weighting_method_protein: str = "uniform"
    weighting_method_rna: str = "uniform"
    weighting_method_phospho: str = "uniform"

    # Sensitivity analysis
    sensitivity_analysis: bool = False
    sensitivity_perturbation: float = 0.2
    sensitivity_trajectories: int = 1000
    sensitivity_levels: int = 400
    sensitvity_top_curves: int = 50
    sensitivity_metric: str = "total_signal"

    # Models metadata
    available_models: tuple[str, ...] = ()


def load_config_toml(path: str | Path) -> PhosKinConfig:
    """
    Parses `config.toml` and returns a validated `PhosKinConfig` object.
    Includes fallbacks and default values for missing keys.
    """
    path = Path(path)

    with path.open("rb") as f:
        full_cfg = tomllib.load(f)

    cfg = (full_cfg or {}).get("global_model", {}) or {}

    # -------------------------
    # 0) Metadata
    # -------------------------
    app_name = cfg.get("app_name", "Phoskintime-Global")
    version = cfg.get("version", "0.1.0")
    parent_package = cfg.get("parent_package", "phoskintime")
    citation = cfg.get("citation", "")
    doi = cfg.get("doi", "")
    github_url = cfg.get("github_url", "")
    docs_url = cfg.get("docs_url", "")

    # -------------------------
    # 1) Inputs
    # -------------------------
    kinase_net = cfg.get("kinase_net", "data/input2.csv")
    tf_net = cfg.get("tf_net", "data/input4.csv")
    ms_data = cfg.get("ms", "data/input1.csv")
    rna_data = cfg.get("rna", "data/input3.csv")

    phospho_data = cfg.get("phospho", None)
    if not phospho_data:
        phospho_data = cfg.get("ms", "data/input1.csv")

    kinopt_res = cfg.get("kinopt", "data/kinopt_results.xlsx")
    tfopt_res = cfg.get("tfopt", "data/tfopt_results.xlsx")

    # -------------------------
    # 2) Output & Run Settings
    # -------------------------
    res_dir = cfg.get("output_dir", cfg.get("output_directory", "results_global"))
    cores = int(cfg.get("cores", 0))
    seed = int(cfg.get("seed", 42))
    refine = bool(cfg.get("refine", False))
    num_refine = int(cfg.get("num_refinements", cfg.get("num_refine", 0)))

    # -------------------------
    # 3) Data inference flags
    # -------------------------
    normalize_fc_steady = bool(cfg.get("normalize_fc_steady", False))
    use_initial_condition_from_data = bool(cfg.get("use_initial_condition_from_data", True))

    # -------------------------
    # 4) Timepoints
    # -------------------------
    tp_cfg = cfg.get("timepoints", {}) or {}
    tp_prot = tp_cfg.get("protein", []) or []
    tp_rna = tp_cfg.get("rna", []) or []
    tp_phospho = tp_cfg.get("phospho_protein", None)

    # If phospho timepoints are missing, fall back to protein timepoints (sane default)
    if not tp_phospho:
        tp_phospho = tp_prot

    time_points_prot = np.asarray(tp_prot, dtype=float)
    time_points_rna = np.asarray(tp_rna, dtype=float)
    time_points_phospho = np.asarray(tp_phospho, dtype=float)

    # -------------------------
    # 5) Model(s)
    # -------------------------
    models_cfg = cfg.get("models", {}) or {}
    model = models_cfg.get("default_model", cfg.get("model", "combinatorial"))
    available_models = tuple(models_cfg.get("available_models", []) or [])

    # -------------------------
    # 6) Optimizer selection + params
    # -------------------------
    optimizer = str(cfg.get("optimizer", "pymoo")).strip().lower()

    hyp_scan = bool(cfg.get("hyperparam_scan", False))

    # Pymoo-style knobs
    max_iter = int(cfg.get("n_gen", 200))
    pop_size = int(cfg.get("pop", 100))

    # Loss
    loss_mode = int(cfg.get("loss", 0))

    # Optuna knobs
    study_name = str(cfg.get("study_name", ""))
    sampler = str(cfg.get("sampler", "TPESampler"))
    pruner = str(cfg.get("pruner", "MedianPruner"))
    n_trials = int(cfg.get("n_trials", 0))

    # -------------------------
    # 7) Regularization (loss weights)
    # -------------------------
    # Support both the new flat keys and any legacy nested dicts if they exist.
    reg_cfg = cfg.get("regularization", {}) or {}
    reg_lambda = float(cfg.get("lambda_prior", reg_cfg.get("lambda", 0.01)))
    reg_protein = float(cfg.get("lambda_protein", reg_cfg.get("protein", 1.0)))
    reg_rna = float(cfg.get("lambda_rna", reg_cfg.get("rna", 1.0)))
    reg_phospho = float(cfg.get("lambda_phospho", reg_cfg.get("phospho", 1.0)))

    # -------------------------
    # 8) Solver
    # -------------------------
    sol_cfg = cfg.get("solver", {}) or {}
    use_custom_solver = bool(sol_cfg.get("use_custom_solver", False))
    ode_abs_tol = float(sol_cfg.get("absolute_tolerance", 1e-8))
    ode_rel_tol = float(sol_cfg.get("relative_tolerance", 1e-8))
    ode_max_steps = int(sol_cfg.get("max_timesteps", 200000))

    # -------------------------
    # 9) Bounds
    # -------------------------
    b = cfg.get("bounds", {}) or {}
    bounds_config: dict[str, tuple[float, float]] = {}
    for k, v in b.items():
        if not (isinstance(v, list) and len(v) == 2):
            raise ValueError(f"bounds.{k} must be a 2-element array [min, max], got: {v}")
        bounds_config[k] = (float(v[0]), float(v[1]))

    # -------------------------
    # 10) Scaling & weighting
    # -------------------------
    scaling_method = str(cfg.get("scaling_method", "none"))
    weighting_method_protein = str(cfg.get("weighting_method_protein", "uniform"))
    weighting_method_rna = str(cfg.get("weighting_method_rna", "uniform"))
    weighting_method_phospho = str(cfg.get("weighting_method_phospho", "uniform"))

    # -------------------------
    # 11) Sensitivity analysis
    # -------------------------
    sensitivity_analysis = bool(cfg.get("sensitivity_analysis", False))
    sensitivity_perturbation = float(cfg.get("sensitivity_perturbation", 0.2))
    sensitivity_trajectories = int(cfg.get("sensitivity_trajectories", 1000))
    sensitivity_levels = int(cfg.get("sensitivity_levels", 400))
    sensitivity_top_curves = int(cfg.get("sensitivity_top_curves", 50))  # key matches TOML
    sensitivity_metric = str(cfg.get("sensitivity_metric", "total_signal"))

    return PhosKinConfig(
        kinase_net=kinase_net,
        tf_net=tf_net,

        ms_data=ms_data,
        rna_data=rna_data,
        phospho_data=phospho_data,

        kinopt_results=kinopt_res,
        tfopt_results=tfopt_res,

        normalize_fc_steady=normalize_fc_steady,
        use_initial_condition_from_data=use_initial_condition_from_data,

        time_points_prot=time_points_prot,
        time_points_rna=time_points_rna,
        time_points_phospho=time_points_phospho,

        bounds_config=bounds_config,

        model=model,

        use_custom_solver=use_custom_solver,
        ode_abs_tol=ode_abs_tol,
        ode_rel_tol=ode_rel_tol,
        ode_max_steps=ode_max_steps,

        loss_mode=loss_mode,
        maximum_iterations=max_iter,
        population_size=pop_size,

        seed=seed,
        cores=cores,

        refine=refine,
        num_refine=num_refine,

        regularization_rna=reg_rna,
        regularization_lambda=reg_lambda,
        regularization_phospho=reg_phospho,
        regularization_protein=reg_protein,

        results_dir=res_dir,

        app_name=app_name,
        version=version,
        parent_package=parent_package,
        citation=citation,
        doi=doi,
        github_url=github_url,
        docs_url=docs_url,

        hyperparam_scan=hyp_scan,

        optimizer=optimizer,
        study_name=study_name,
        sampler=sampler,
        pruner=pruner,
        n_trials=n_trials,

        scaling_method=scaling_method,

        weighting_method_protein=weighting_method_protein,
        weighting_method_rna=weighting_method_rna,
        weighting_method_phospho=weighting_method_phospho,

        sensitivity_analysis=sensitivity_analysis,
        sensitivity_perturbation=sensitivity_perturbation,
        sensitivity_trajectories=sensitivity_trajectories,
        sensitivity_levels=sensitivity_levels,
        sensitvity_top_curves=sensitivity_top_curves,
        sensitivity_metric=sensitivity_metric,

        available_models=available_models,
    )


def get_parameter_labels(idx):
    """Generates descriptive labels for all parameters in the flattened decision vector."""
    labels = []

    # 1. Kinase Scaling (c_k)
    for k in idx.kinases:
        labels.append(f"c_k (Kinase: {k})")

    # 2. Protein-specific params (A, B, C, D, Dp, E)
    # The order MUST match params.py -> unpack_params
    for p_idx, p_name in enumerate(idx.proteins):
        labels.append(f"A_i (Synthesis) [{p_name}]")
        labels.append(f"B_i (Degradation) [{p_name}]")
        labels.append(f"C_i (Phos-Rate) [{p_name}]")
        labels.append(f"D_i (Dephos-Rate) [{p_name}]")

        # Site-specific dephosphorylation (Dp_i)
        n_sites = idx.n_sites[p_idx]
        if n_sites > 0:
            for s_idx in range(n_sites):
                site_name = idx.sites[p_idx][s_idx]
                labels.append(f"Dp_i (Site-Deg: {site_name}) [{p_name}]")
        else:
            # Even if 0 sites, params.py usually keeps a placeholder for some models
            # Check your unpack_params logic; if it's 1-per-protein, use:
            # labels.append(f"Dp_i (Dephos-2) [{p_name}]")
            pass

        labels.append(f"E_i (TF-Effect) [{p_name}]")

    # 3. Global TF Scale
    labels.append("Global TF Scale")

    return labels


def calculate_bio_bounds(idx, df_prot, df_rna, tf_mat, kin_in):
    """
    Dynamically calculates optimization bounds by analyzing network topology,
    data dynamic ranges, and kinetic equilibrium requirements.

    This ensures that the optimizer doesn't waste time searching parameter space
    regions that are mathematically impossible to yield the observed data
    (e.g., degradation rates that are too slow to match the observed decay).

    Returns:
        dict: Custom bounds for each parameter group.
    """
    logger.info("=" * 60)
    logger.info("[Bounds] CALCULATION: Performing Topological & Kinetic Analysis")

    # 1. ANALYZE DATA DYNAMIC RANGE
    # Cap the maximum Fold Change the model is ALLOWED to reach.
    max_prot_fc = df_prot['fc'].max() if (df_prot is not None and not df_prot.empty) else 5.0
    max_rna_fc = df_rna['fc'].max() if (df_rna is not None and not df_rna.empty) else 5.0

    # Safety buffer: Tighter than before (1.5x instead of 2.0x)
    safe_prot_max = max(2.0, max_prot_fc * 1.5)
    safe_rna_max = max(2.0, max_rna_fc * 1.5)

    # 2. mRNA KINETICS (A_i, B_i)
    # Bounds: 0.005 (~2.3h) to 0.1 (~7min)
    # We raise the floor to prevent 'stuck' mRNA
    b_min, b_max = 0.005, 0.15

    # A_i (Basal Synthesis)
    # Synthesis must balances Degradation at steady state (FC=1).
    # A_i approx B_i. We allow slight deviation for basal repression/activation.
    a_min = b_min * 0.1
    a_max = b_max * safe_rna_max

    # 3. PROTEIN KINETICS (C_i, D_i)
    # CRITICAL FIX: Raise the degradation floor.
    # If proteins degrade too slowly, errors accumulate over 960 minutes.
    # New Floor: 0.01 (approx 70 min half-life). Nothing lives forever.
    d_min, d_max = 0.01, 0.10

    # C_i (Translation)
    # Strict linkage: Synthesis cannot exceed Degradation * Max_Fold_Change
    # C_max = D_max * Max_FC.
    # If D=0.1 and MaxFC=5, C cannot exceed 0.5.
    c_min = d_min * 0.1
    c_max = d_max * safe_prot_max

    # 4. TOPOLOGICAL SENSITIVITY (E_i, tf_scale)
    # For large networks (200+ nodes), additive inputs can be huge.
    n_edges = tf_mat.nnz
    avg_density = n_edges / max(1, idx.N)

    if avg_density < 2.0:
        # Sparse (Sub-network): Allow higher gain
        e_max = 20.0
        tf_scale_min, tf_scale_max = 0.5, 5.0
    else:
        # Dense (Global): Clamp gain hard to prevent feedback explosions
        e_max = 5.0  # Reduced from 20.0
        tf_scale_min, tf_scale_max = 0.1, 2.5  # Reduced from 10.0

    # 5. SIGNALING VELOCITY (c_k, Dp_i)
    # Phospho is working well, keep these bounds wide to maintain signal fit.
    dp_min, dp_max = 0.1, 10.0

    # Check input variance. If inputs are flat, we need higher c_k.
    kin_variance = np.var(kin_in.Kmat)
    ck_max = 15.0 if kin_variance < 0.02 else 5.0

    bounds_dict = {
        "c_k": (0.01, ck_max),
        "A_i": (a_min, a_max),
        "B_i": (b_min, b_max),
        "C_i": (c_min, c_max),
        "D_i": (d_min, d_max),
        "Dp_i": (dp_min, dp_max),
        "E_i": (0.0, e_max),
        "tf_scale": (tf_scale_min, tf_scale_max)
    }

    # Model-specific adjustments
    # MODEL: 0 distributive, 1 sequential, 2 combinatorial, 4 saturating
    if MODEL == 1:
        # Sequential: tighter kinase gain, faster phospho turnover
        bounds_dict["Dp_i"] = (0.15, 8.0)
        ck_lo, ck_hi = bounds_dict["c_k"]
        bounds_dict["c_k"] = (ck_lo, max(3.0, 0.75 * ck_hi))

    elif MODEL == 2:
        # Combinatorial: clamp hard (many transitions per state)
        bounds_dict["Dp_i"] = (0.2, 3.0)
        ck_lo, ck_hi = bounds_dict["c_k"]
        bounds_dict["c_k"] = (ck_lo, min(2.5, ck_hi))

        # E_i is dephosph/backflow; keep smaller to avoid stiff cycling
        e_lo, e_hi = bounds_dict["E_i"]
        if avg_density >= 2.0:
            bounds_dict["E_i"] = (e_lo, min(e_hi, 2.5))
        else:
            bounds_dict["E_i"] = (e_lo, min(e_hi, 8.0))

    elif MODEL == 4:
        # Saturating: bounded forward flux allows more kinase gain and TF scale
        bounds_dict["Dp_i"] = (0.1, 8.0)
        ck_lo, ck_hi = bounds_dict["c_k"]
        bounds_dict["c_k"] = (ck_lo, min(10.0, 1.5 * ck_hi))

        tf_lo, tf_hi = bounds_dict["tf_scale"]
        if avg_density >= 2.0:
            bounds_dict["tf_scale"] = (tf_lo, max(tf_hi, 6.0))
        else:
            bounds_dict["tf_scale"] = (tf_lo, max(tf_hi, 10.0))

    logger.info("-" * 60)
    logger.info(f"{'Param':<10} | {'Min':<10} | {'Max':<10} | {'Constraint Logic'}")
    logger.info("-" * 60)
    logger.info(f"{'C_i':<10} | {c_min:<10.4f} | {c_max:<10.4f} | {'Linked to D_max * MaxFC'}")
    logger.info(f"{'D_i':<10} | {d_min:<10.4f} | {d_max:<10.4f} | {'Floor raised to 0.01'}")
    logger.info(f"{'tf_scale':<10} | {tf_scale_min:<10.3f} | {tf_scale_max:<10.3f} | {'Clamped for stability'}")
    logger.info("=" * 60)

    return bounds_dict


def _nondegenerate(mask_xl, mask_xu, eps=1e-14):
    return (mask_xu - mask_xl) > eps


def get_optimized_sets(idx, slices, xl, xu, eps=1e-14):
    """
    Identifies which entities (Proteins, Sites, Kinases) actually have
    free parameters being optimized (i.e., bounds are not collapsed).

    Returns:
      opt_proteins: set[str]  proteins with any free variable among A/B/C/D/E
      opt_sites: set[str]     site labels like 'EGFR_Y1173' where Dp_i is free
      opt_kinases: set[str]   kinases with free c_k
    """
    xl = np.asarray(xl, dtype=float)
    xu = np.asarray(xu, dtype=float)

    N = idx.N

    # Protein-level params (length N each)
    protein_free = np.zeros(N, dtype=bool)
    for key in ("A_i", "B_i", "C_i", "D_i", "E_i"):
        if key not in slices:
            continue
        sl = slices[key]  # slice in theta
        free = _nondegenerate(xl[sl], xu[sl], eps=eps)
        if free.size != N:
            raise ValueError(f"{key} slice size {free.size} != idx.N {N}")
        protein_free |= free

    opt_proteins = set(np.array(idx.proteins)[protein_free].tolist())

    # Kinase activities c_k (length nK)
    opt_kinases = set()
    if "c_k" in slices:
        sl = slices["c_k"]
        free = _nondegenerate(xl[sl], xu[sl], eps=eps)
        opt_kinases = set(np.array(idx.kinases)[free].tolist())

    # Site-level params Dp_i (length total_sites)
    opt_sites = set()
    if "Dp_i" in slices:
        sl = slices["Dp_i"]
        free = _nondegenerate(xl[sl], xu[sl], eps=eps)

        # Build flattened site labels in the same order as Dp_i indexing
        flat_labels = []
        for i, p in enumerate(idx.proteins):
            for s in idx.sites[i]:
                flat_labels.append(f"{p}_{s}")
        flat_labels = np.array(flat_labels)

        if free.size != flat_labels.size:
            raise ValueError(f"Dp_i slice size {free.size} != total_sites {flat_labels.size}")

        opt_sites = set(flat_labels[free].tolist())

    return opt_proteins, opt_sites, opt_kinases
