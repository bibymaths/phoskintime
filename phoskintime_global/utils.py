from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import tomllib

def softplus(x):
    return np.where(x > 20, x, np.log1p(np.exp(x)))


def inv_softplus(y):
    y = np.maximum(y, 1e-12)
    return np.log(np.expm1(y))


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

def pick_best_lamdas(F, weights):
    F = np.asarray(F, dtype=float)
    F_min = F.min(axis=0)
    F_ptp = np.ptp(F, axis=0) + 1e-12
    F_norm = (F - F_min) / F_ptp
    scores = (F_norm * weights).sum(axis=1)
    i = int(np.argmin(scores))
    return i, float(scores[i])

@dataclass(frozen=True)
class PhosKinConfig:
    time_points_prot: np.ndarray
    time_points_rna: np.ndarray
    time_points_phospho: np.ndarray
    bounds_config: dict[str, tuple[float, float]]
    model: str


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

    model = cfg["models"]

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
        model=model
    )
