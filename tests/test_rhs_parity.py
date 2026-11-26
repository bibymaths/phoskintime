"""S1/N1: the JAX right-hand side (networkmodel.backend.make_networkmodel_rhs) must match the numba
topology kernels in networkmodel.models for MODEL 0 (distributive), 1 (sequential) and 2 (combinatorial).

Parity is checked at the kernel level: the reference computes the kinase drive (the shared PCHIP drive,
networkmodel.kinase_drive.KinaseDrive in numpy; N17), S_all = W @ (K(t) * c_k), the TF input (TF @ P) / tf_deg and the kinase
driver mapping exactly as the JAX RHS does, then calls the numba kernel. This isolates the topology
equations from the known differences of the legacy System.rhs wrapper (N10: TF input squashed twice;
N17: kinase interpolation), which are not part of S1.

Error metric (fixed before running): per case, max_j |dy_jax - dy_ref|_j / max(||dy_ref||_inf, 1e-300),
required <= 1e-12 (float64); 24 fixed random cases per model, including t exactly on kinase knots.
"""
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp

N_SITES = [0, 1, 2, 3, 2]
PROTEINS = ["P0", "P1", "P2", "P3", "K1"]
KINASES = ["K0", "K1", "K2"]
KIN_GRID = np.array([0.0, 1.0, 2.0, 4.0, 8.0, 16.0])
N_CASES = 24
RTOL = 1e-12


def _make_sys(model_id, rng):
    n = len(PROTEINS)
    n_sites = np.asarray(N_SITES, dtype=np.int32)
    n_states = (2 ** n_sites).astype(np.int32)
    offset_y = np.zeros(n, dtype=np.int32)
    offset_s = np.zeros(n, dtype=np.int32)
    cy = cs = 0
    for i in range(n):
        offset_y[i], offset_s[i] = cy, cs
        cy += 1 + (n_states[i] if model_id == 2 else 1 + n_sites[i])
        cs += n_sites[i]
    n_tot_sites = int(n_sites.sum())
    W = rng.uniform(0.0, 1.0, size=(n_tot_sites, len(KINASES)))
    W[rng.uniform(size=W.shape) < 0.3] = 0.0
    TF = rng.uniform(-1.0, 1.0, size=(n, n))
    TF[rng.uniform(size=TF.shape) < 0.4] = 0.0
    idx = SimpleNamespace(
        N=n, proteins=PROTEINS, kinases=KINASES,
        p2i={p: i for i, p in enumerate(PROTEINS)}, k2i={k: i for i, k in enumerate(KINASES)},
        offset_y=offset_y, offset_s=offset_s, n_sites=n_sites, n_states=n_states, state_dim=cy,
    )
    kin = SimpleNamespace(grid=KIN_GRID.copy(), Kmat=rng.uniform(0.2, 3.0, size=(len(KINASES), KIN_GRID.size)))
    return SimpleNamespace(idx=idx, W_global=sp.csr_matrix(W), tf_mat=sp.csr_matrix(TF),
                           tf_deg=rng.uniform(0.5, 3.0, size=n), kin=kin)


def _random_params(sys, rng):
    n, n_tot = sys.idx.N, int(np.sum(sys.idx.n_sites))
    return {
        "c_k": rng.uniform(0.05, 3.0, len(KINASES)), "A_i": rng.uniform(0.01, 2.0, n),
        "B_i": rng.uniform(0.01, 1.0, n), "C_i": rng.uniform(0.01, 1.0, n), "D_i": rng.uniform(0.01, 0.5, n),
        "Dp_i": rng.uniform(0.01, 2.0, n_tot), "E_i": rng.uniform(0.0, 2.0, n), "tf_scale": rng.uniform(0.5, 5.0),
    }


def _reference_dy(model_id, sys, par, t, y):
    from networkmodel.models import combinatorial_rhs, distributive_rhs, sequential_rhs

    idx = sys.idx
    from networkmodel.kinase_drive import KinaseDrive

    Kt = KinaseDrive(sys.kin.grid, sys.kin.Kmat).evaluate(float(t)) * par["c_k"]
    S_all = sys.W_global.toarray() @ Kt
    P_vec = np.zeros(idx.N)
    for i, p in enumerate(idx.proteins):
        if p in idx.k2i:
            P_vec[i] = Kt[idx.k2i[p]]
        elif model_id == 2:
            P_vec[i] = y[idx.offset_y[i] + 1: idx.offset_y[i] + 1 + idx.n_states[i]].sum()
        else:
            P_vec[i] = y[idx.offset_y[i] + 1: idx.offset_y[i] + 2 + idx.n_sites[i]].sum()
    TF_inputs = (sys.tf_mat.toarray() @ P_vec) / np.maximum(sys.tf_deg, 1e-12)
    dy = np.zeros_like(y)
    common = (par["A_i"], par["B_i"], par["C_i"], par["D_i"], par["Dp_i"], par["E_i"], float(par["tf_scale"]),
              TF_inputs)
    if model_id == 2:
        combinatorial_rhs(y, dy, *common, S_all.copy(), idx.offset_y, idx.offset_s, idx.n_sites, idx.n_states)
    else:
        kernel = sequential_rhs if model_id == 1 else distributive_rhs
        kernel(y, dy, *common, S_all, idx.offset_y, idx.offset_s, idx.n_sites)
    return dy


def scaled_errors(model_id, monkeypatch, seed=20261126):
    import jax.numpy as jnp

    import networkmodel.config as nm_config
    from networkmodel.backend import ensure_jax_float64, make_networkmodel_rhs

    ensure_jax_float64()
    monkeypatch.setattr(nm_config, "MODEL", model_id)
    rng = np.random.default_rng(seed + model_id)
    sys = _make_sys(model_id, rng)
    rhs = make_networkmodel_rhs(sys)
    times = list(KIN_GRID) + list(rng.uniform(0.0, 20.0, N_CASES - KIN_GRID.size))
    errs = []
    for t in times:
        par = _random_params(sys, rng)
        y = rng.uniform(0.0, 2.0, sys.idx.state_dim)
        ref = _reference_dy(model_id, sys, par, float(t), y)
        got = np.asarray(rhs(jnp.float64(t), jnp.asarray(y), {k: jnp.asarray(v) for k, v in par.items()}))
        errs.append(np.max(np.abs(got - ref)) / max(np.max(np.abs(ref)), 1e-300))
    return np.asarray(errs)


@pytest.mark.parametrize(
    "model_id",
    [0, 1, 2],
    ids=["distributive", "sequential", "combinatorial"],
)
def test_jax_rhs_matches_numba_kernels(model_id, monkeypatch):
    errs = scaled_errors(model_id, monkeypatch)
    assert errs.size == N_CASES
    assert np.all(np.isfinite(errs))
    assert errs.max() <= RTOL, f"max scaled error {errs.max():.3e} (median {np.median(errs):.3e})"
