import numpy as np


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
