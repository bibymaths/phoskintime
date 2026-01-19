from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import sys
import pandas as pd
from numba import njit

# Python 3.11+ has tomllib, older versions need tomli
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


def _normcols(df):
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def _base_idx(times, t0):
    return np.int32(int(np.argmin(np.abs(times - float(t0)))))


def _find_col(df, cands):
    for c in cands:
        if c in df.columns:
            return c
    return None


def slen(s: slice) -> int:
    return int(s.stop) - int(s.start)


def normalize_fc_to_t0(df):
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


@njit(cache=True, fastmath=True, nogil=True)
def _zero_vec(a):
    for i in range(a.size):
        a[i] = 0.0


@njit(cache=True, fastmath=True, nogil=True)
def time_bucket(t, grid):
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
    out = np.empty_like(y)
    for i in range(y.size):
        yi = y[i]
        if yi < 1e-12:
            yi = 1e-12
        out[i] = np.log(np.expm1(yi))
    return out


@njit(cache=True, fastmath=True, nogil=True)
def pick_best_lamdas(F, weights):
    F = np.asarray(F, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    n = F.shape[0]
    m = F.shape[1]

    F_min = np.empty(m, dtype=np.float64)
    F_ptp = np.empty(m, dtype=np.float64)

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


def load_config_toml(path: str | Path) -> PhosKinConfig:
    path = Path(path)

    with path.open("rb") as f:
        full_cfg = tomllib.load(f)
        cfg = full_cfg.get("global_model", {})

    # 1. Inputs
    kinase_net = cfg.get("kinase_net", "data/input2.csv")
    tf_net = cfg.get("tf_net", "data/input4.csv")
    ms_data = cfg.get("ms", "data/input1.csv")
    rna_data = cfg.get("rna", "data/input3.csv")

    phospho_data = cfg.get("phospho", None)
    if not phospho_data:
        phospho_data = cfg.get("ms", "data/input1.csv")

    kinopt_res = cfg.get("kinopt", "data/kinopt_results.xlsx")
    tfopt_res = cfg.get("tfopt", "data/tfopt_results.xlsx")

    # 2. Output & Run Settings
    # FIX: Check 'output_dir' (TOML) first, then 'output_directory' (Legacy), then default
    res_dir = cfg.get("output_dir", cfg.get("output_directory", "results_global"))

    cores = cfg.get("cores", 0)
    seed = cfg.get("seed", 42)
    refine = cfg.get("refine", False)
    num_refine = cfg.get("num_refinements", 3)

    # 3. Data Config
    normalize_fc_steady = cfg["normalize_fc_steady"]
    use_initial_condition_from_data = cfg["use_initial_condition_from_data"]

    # 4. Timepoints
    tp_cfg = cfg.get("timepoints", {})
    tp_prot = tp_cfg.get("protein", [])
    tp_rna = tp_cfg.get("rna", [])
    tp_phospho = tp_cfg.get("phospho_protein", [])

    time_points_prot = np.asarray(tp_prot, dtype=float)
    time_points_rna = np.asarray(tp_rna, dtype=float)
    time_points_phospho = np.asarray(tp_phospho, dtype=float)

    # 5. Model
    model = cfg.get("models", {}).get("default_model", "combinatorial")

    # 6. Optimization
    opt_cfg = cfg.get("optimization", {})

    if "n_gen" in cfg:
        max_iter = cfg["n_gen"]
    else:
        max_iter = opt_cfg.get("max_iterations", 200)

    if "pop" in cfg:
        pop_size = cfg["pop"]
    else:
        pop_size = opt_cfg.get("population_size", 100)

    loss_mode = opt_cfg.get("loss", 0)

    # 7. Regularization
    reg_cfg = cfg.get("regularization", {})

    reg_lambda = cfg.get("lambda_prior", reg_cfg.get("lambda", 0.01))
    reg_protein = cfg.get("lambda_protein", reg_cfg.get("protein", 1.0))
    reg_rna = cfg.get("lambda_rna", reg_cfg.get("rna", 1.0))
    reg_phospho = cfg.get("lambda_phospho", reg_cfg.get("phospho", 1.0))

    # 8. Solver
    sol_cfg = cfg.get("solver", {})
    use_custom_solver = sol_cfg.get("use_custom_solver", True)
    ode_abs_tol = sol_cfg.get("absolute_tolerance", 1e-8)
    ode_rel_tol = sol_cfg.get("relative_tolerance", 1e-8)
    ode_max_steps = sol_cfg.get("max_timesteps", 200000)

    # 9. Bounds
    b = cfg.get("bounds", {})
    bounds_config = {}
    for k, v in b.items():
        if not (isinstance(v, list) and len(v) == 2):
            raise ValueError(f"bounds.{k} must be a 2-element array [min, max], got: {v}")
        bounds_config[k] = (float(v[0]), float(v[1]))

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
        results_dir=res_dir
    )