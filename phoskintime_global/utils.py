from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import tomllib
from numba import njit

def _normcols(df):
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def _find_col(df, cands):
    for c in cands:
        if c in df.columns:
            return c
    return None

def normalize_fc_to_t0(df):
    df = df.copy()
    t0 = df[df["time"] == 0.0].set_index("protein")["fc"]
    df["fc"] = df.apply(lambda r: r["fc"] / t0.get(r["protein"], np.nan), axis=1)
    return df.dropna(subset=["fc"])

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

@njit(cache=True, fastmath=True, nogil=True, nopython=True)
def softplus(x):
    out = np.empty_like(x)
    for i in range(x.size):
        xi = x[i]
        if xi > 20.0:
            out[i] = xi
        else:
            out[i] = np.log1p(np.exp(xi))
    return out


@njit(cache=True, fastmath=True, nogil=True, nopython=True)
def inv_softplus(y):
    out = np.empty_like(y)
    for i in range(y.size):
        yi = y[i]
        if yi < 1e-12:
            yi = 1e-12
        out[i] = np.log(np.expm1(yi))
    return out

@njit(cache=True, fastmath=True, nogil=True, nopython=True)
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
    ode_abs_tol: float
    ode_rel_tol: float
    ode_max_steps: int

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
    ode_abs_tol = cfg["absolute_tolerance"]
    ode_rel_tol = cfg["relative_tolerance"]
    ode_max_steps = cfg["max_timesteps"]

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
        ode_abs_tol=ode_abs_tol,
        ode_rel_tol=ode_rel_tol,
        ode_max_steps=ode_max_steps
    )
