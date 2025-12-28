from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import tomllib

import pandas as pd
from numba import njit

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
    time_points_prot: np.ndarray
    time_points_rna: np.ndarray
    time_points_phospho: np.ndarray
    bounds_config: dict[str, tuple[float, float]]
    model: str
    use_custom_solver: bool
    ode_abs_tol: float
    ode_rel_tol: float
    ode_max_steps: int
    maximum_iterations: int
    population_size: int
    seed: int
    regularization_rna: float
    regularization_lambda: float
    regularization_phospho: float
    regularization_protein: float
    results_dir: str | Path

def load_config_toml(path: str | Path) -> PhosKinConfig:
    path = Path(path)

    with path.open("rb") as f:
        cfg = tomllib.load(f)

    tp_prot = cfg["timepoints"]["protein"]
    tp_rna = cfg["timepoints"]["rna"]
    tp_phospho = cfg["timepoints"]["phospho_protein"]

    time_points_prot = np.asarray(tp_prot, dtype=float)
    time_points_rna = np.asarray(tp_rna, dtype=float)
    time_points_phospho = np.asarray(tp_phospho, dtype=float)

    model = cfg["models"]["default_model"]

    use_custom_solver = cfg["solver"]["use_custom_solver"]
    ode_abs_tol = cfg["solver"]["absolute_tolerance"]
    ode_rel_tol = cfg["solver"]["relative_tolerance"]
    ode_max_steps = cfg["solver"]["max_timesteps"]

    max_iter = cfg["optimization"]["max_iterations"]
    pop_size = cfg["optimization"]["population_size"]
    seed = cfg["optimization"]["seed"]

    reg_protein = cfg["regularization"]["protein"]
    reg_rna = cfg["regularization"]["rna"]
    reg_phospho = cfg["regularization"]["phospho"]
    reg_lambda = cfg["regularization"]["lambda"]

    res_dir = cfg["general"]["output_directory"]

    b = cfg["bounds"]
    bounds_config: dict[str, tuple[float, float]] = {}
    for k, v in b.items():
        if not (isinstance(v, list) and len(v) == 2):
            raise ValueError(f"bounds.{k} must be a 2-element array [min, max], got: {v}")
        lo, hi = float(v[0]), float(v[1])
        if lo >= hi:
            raise ValueError(f"bounds.{k} invalid: min >= max ({lo} >= {hi})")
        bounds_config[k] = (lo, hi)

    return PhosKinConfig(
        time_points_prot=time_points_prot,
        time_points_rna=time_points_rna,
        time_points_phospho=time_points_phospho,
        bounds_config=bounds_config,
        model=model,
        use_custom_solver=use_custom_solver,
        ode_abs_tol=ode_abs_tol,
        ode_rel_tol=ode_rel_tol,
        ode_max_steps=ode_max_steps,
        maximum_iterations=max_iter,
        population_size=pop_size,
        seed=seed,
        regularization_rna=reg_rna,
        regularization_lambda=reg_lambda,
        regularization_phospho=reg_phospho,
        regularization_protein=reg_protein,
        results_dir=res_dir
    )
